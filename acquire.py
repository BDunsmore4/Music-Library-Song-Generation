"""
Growing the library: fetching a real track, and generating one as a last resort.

Two ways a track can join the collection, with very different costs and so very
different limits.

  add_from_catalogue   Apple Music has it. Download the 30-second preview,
                       CLAP-embed it, append to the index. Costs a request.
                       Eight an hour.
  generate             Nobody has it, because it does not exist. Ask ElevenLabs
                       to compose thirty seconds, embed that, keep it. Costs
                       real money. One an hour, and the assistant is told to
                       say so when asked rather than quietly refusing.

Both limits are enforced from the append-only store log, not from memory, so
they survive a restart and are shared across every session. An in-process
counter would reset on each deploy and be per-user, which is not a limit.

Generated tracks are marked `source="generated"` and never pretend to be part
of the original collection. They are a different kind of object — nobody
recorded them, nobody chose them — and the map already distinguishes generated
points from library ones.
"""

import os
import time

ADD_PER_HOUR = int(os.environ.get("CRATE_ADD_PER_HOUR", 8))
GENERATE_PER_HOUR = int(os.environ.get("CRATE_GENERATE_PER_HOUR", 1))
WINDOW = 3600

ELEVEN_URL = "https://api.elevenlabs.io/v1/music/compose"
ELEVEN_MS = int(os.environ.get("CRATE_GENERATE_MS", 30000))
STOREFRONT = os.environ.get("APPLE_STOREFRONT", "gb")


def _limit(kind, cap):
    """(allowed, used, cap, seconds_until_next). Never raises."""
    try:
        import crate_store
        used = crate_store.count_recent(kind, WINDOW)
        if used < cap:
            return True, used, cap, 0
        oldest = crate_store.oldest_recent(kind, WINDOW)
        wait = int(max(0, WINDOW - (time.time() - oldest))) if oldest else WINDOW
        return False, used, cap, wait
    except Exception:
        return True, 0, cap, 0


def budget():
    """What is left in each allowance, for the assistant to quote honestly."""
    a_ok, a_used, a_cap, a_wait = _limit("add", ADD_PER_HOUR)
    g_ok, g_used, g_cap, g_wait = _limit("generate", GENERATE_PER_HOUR)
    return {
        "additions": {"used": a_used, "limit": a_cap,
                      "remaining": max(0, a_cap - a_used),
                      "minutes_until_next": round(a_wait / 60) if not a_ok else 0},
        "generation": {"used": g_used, "limit": g_cap,
                       "remaining": max(0, g_cap - g_used),
                       "minutes_until_next": round(g_wait / 60) if not g_ok else 0,
                       "note": "generation costs real money per call, which is "
                               "why it is once an hour and a last resort"},
    }


# ── a real track ─────────────────────────────────────────────────────────────

def search_catalogue(query, limit=5):
    """Apple Music song search. Returns candidates with previews only."""
    import requests
    try:
        import apple_token
        token = apple_token.get_developer_token()
    except Exception:
        return []
    try:
        r = requests.get("https://api.music.apple.com/v1/catalog/%s/search" % STOREFRONT,
                         headers={"Authorization": "Bearer %s" % token}, timeout=20,
                         params={"term": query, "types": "songs", "limit": max(limit, 10)})
        r.raise_for_status()
        out = []
        for s in (r.json().get("results", {}).get("songs", {}).get("data", [])):
            a = s.get("attributes", {})
            prev = (a.get("previews") or [{}])[0].get("url")
            if not prev:
                continue                      # no audio, nothing to embed
            out.append(dict(id="apple:" + str(s.get("id")), name=a.get("name", ""),
                            artists=a.get("artistName", ""), album=a.get("albumName", ""),
                            duration_ms=a.get("durationInMillis", ""),
                            preview_url=prev, isrc=a.get("isrc", ""),
                            release_date=a.get("releaseDate", ""),
                            genres="|".join(a.get("genreNames") or [])))
            if len(out) >= limit:
                break
        return out
    except Exception:
        return []


def add_from_catalogue(crate_model, query, pick=0):
    """Find a track, embed its preview, add it. Returns a report dict."""
    ok, used, cap, wait = _limit("add", ADD_PER_HOUR)
    if not ok:
        return {"added": False, "reason": "limit",
                "message": "That is %d additions this hour, which is the cap. "
                           "The next one is free in about %d minutes."
                           % (cap, round(wait / 60))}

    found = search_catalogue(query, limit=5)
    if not found:
        return {"added": False, "reason": "not_found",
                "message": "Nothing on Apple Music matches %r with a playable "
                           "preview." % query}
    chosen = found[min(max(pick, 0), len(found) - 1)]

    existing = crate_model._id_to_row.get(str(chosen["id"]))
    if existing is None and chosen.get("isrc"):
        match = crate_model.meta.index[
            crate_model.meta["isrc"].astype(str) == str(chosen["isrc"])]
        existing = int(match[0]) if len(match) else None
    if existing is not None:
        r = crate_model.meta.iloc[existing]
        return {"added": False, "reason": "already_present",
                "message": "Already in the library.",
                "track": {"track_id": str(r["id"]), "name": str(r["name"]),
                          "artist": str(r["artists"])}}

    import requests
    try:
        audio = requests.get(chosen["preview_url"], timeout=30)
        audio.raise_for_status()
        vec = crate_model.enc.embed_audio_bytes(audio.content)
    except Exception as e:
        return {"added": False, "reason": "embed_failed",
                "message": "Could not embed that preview (%s)." % str(e)[:80]}

    row, fresh = crate_model.add_track(chosen, vec, kind="added")
    try:
        import crate_store
        crate_store.log_acquisition("add", chosen["id"], chosen["name"])
    except Exception:
        pass
    left = max(0, cap - used - 1)
    return {"added": bool(fresh), "remaining_this_hour": left,
            "alternatives": [{"name": f["name"], "artist": f["artists"]}
                             for f in found[1:4]],
            "track": dict(
                {"track_id": chosen["id"], "name": chosen["name"],
                 "artist": chosen["artists"], "album": chosen["album"],
                 "preview_url": chosen["preview_url"],
                 "isolation": crate_model.isolation_pct(row),
                 "profile": crate_model.profile(row)},
                **crate_model.neighbour_position(row))}


# ── a track that does not exist yet ──────────────────────────────────────────

def generate(crate_model, prompt):
    """Compose thirty seconds with ElevenLabs, embed it, add it as generated."""
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not key:
        return {"generated": False, "reason": "not_configured",
                "message": "Generation is not configured on this server "
                           "(no ELEVENLABS_API_KEY)."}
    ok, used, cap, wait = _limit("generate", GENERATE_PER_HOUR)
    if not ok:
        return {"generated": False, "reason": "limit",
                "message": "Generation has already been used this hour. It costs "
                           "real money per track, so it is capped at %d an hour — "
                           "the next one is available in about %d minutes."
                           % (cap, round(wait / 60))}

    import requests
    try:
        r = requests.post(ELEVEN_URL, timeout=240,
                          headers={"xi-api-key": key, "Content-Type": "application/json"},
                          json={"prompt": prompt, "output_format": "mp3_44100_128",
                                "music_length_ms": ELEVEN_MS})
        if not r.ok:
            detail = ""
            try:
                d = r.json().get("detail", {})
                detail = d.get("message") if isinstance(d, dict) else str(d)
            except Exception:
                detail = r.text[:160]
            return {"generated": False, "reason": "provider_error",
                    "message": "ElevenLabs refused: %s" % (detail or r.status_code)}
        audio = r.content
        vec = crate_model.enc.embed_audio_bytes(audio)
    except Exception as e:
        return {"generated": False, "reason": "failed",
                "message": "Generation failed (%s)." % str(e)[:90]}

    url = ""
    try:
        import r2_client
        url = r2_client.upload_mp3(audio)
        r2_client.append_to_manifest({"label": prompt, "audio_url": url,
                                      "point_type": "generated"})
    except Exception as e:
        print("[acquire] generated track not persisted to R2: %s" % e)

    meta = {"id": "generated:%d" % int(time.time()), "name": prompt[:90],
            "artists": "Generated", "album": "", "duration_ms": ELEVEN_MS,
            "preview_url": url, "isrc": ""}
    row, _ = crate_model.add_track(meta, vec, kind="generated")
    try:
        import crate_store
        crate_store.log_acquisition("generate", meta["id"], prompt[:120])
    except Exception:
        pass
    return {"generated": True, "remaining_this_hour": max(0, cap - used - 1),
            "track": dict(
                {"track_id": meta["id"], "name": meta["name"],
                 "artist": "Generated", "preview_url": url, "generated": True,
                 "isolation": crate_model.isolation_pct(row),
                 "profile": crate_model.profile(row)},
                **crate_model.neighbour_position(row))}
