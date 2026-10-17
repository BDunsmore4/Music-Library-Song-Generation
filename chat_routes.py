"""
Crate-digging chat routes, mounted as a Flask blueprint alongside the others.

    POST /api/chat/session            -> {"session": "<id>"}
    POST /api/chat                    {"session","message"} -> reply + tracks + trail
    POST /api/chat/feedback           {"session","track_id","rating"}
    GET  /api/chat/config             -> axes, library size, whether it is live

Sessions are an in-memory dict keyed by id, the same single-gunicorn-worker
assumption `personal_routes` already makes. The Crate itself is built once,
lazily, on the first chat request rather than at import, so a deploy that
nobody chats to never pays for it and the Railway healthcheck is not racing a
model load.

Two production notes that matter more than the code:

* The trail point is taken from the TOP RESULT's precomputed map coordinates,
  not from `umap_reducer.transform(query)`. The transform runs its optimisation
  epochs on every call — seconds per request, which would dominate the whole
  endpoint — and a query's nearest neighbour sits essentially on top of it in
  the 2-D embedding anyway. Set CRATE_TRAIL=1 only if you want the true
  projection and can afford it.

* /api/chat spends your Anthropic key on behalf of whoever calls it. Set
  CHAT_SHARED_SECRET and have the frontend send it as X-Chat-Key, or leave the
  endpoint off a public origin.
"""

import json
import os
import time

from flask import Blueprint, Response, jsonify, request, stream_with_context

import llm

bp = Blueprint("chat", __name__)

SESSION_TTL = int(os.environ.get("CHAT_SESSION_TTL", 60 * 60))
MAX_SESSIONS = int(os.environ.get("CHAT_MAX_SESSIONS", 200))
SHARED_SECRET = os.environ.get("CHAT_SHARED_SECRET", "")
_HERE = os.path.dirname(os.path.abspath(__file__))
# Coordinates for the trail. `data/map_coords.json` is a {id: [x, y]} extract
# and is the one that ships: .dockerignore excludes Frontend/ and *.html (the
# frontend is served from Cloudflare Pages, not this service), so reading
# Frontend/data.json works locally and silently yields an empty trail in a
# Docker build. The extract is also ~10x smaller than the full map.
MAP_COORDS = os.environ.get("MAP_COORDS_JSON",
                            os.path.join(_HERE, "data", "map_coords.json"))
# 3-D positions for the stage. Same {id: [x,y,z]} extract trick, same reason.
MAP_COORDS_3D = os.environ.get("MAP_COORDS_3D_JSON",
                               os.path.join(_HERE, "data", "map_coords_3d.json"))
MAP_JSON = os.environ.get("FRONTEND_DATA_JSON",
                          os.path.join(_HERE, "Frontend", "data.json"))

_chat = None          # the shared CrateChat template (holds the Crate)
_coords = None        # {track_id: (x, y)} from the frontend map
_coords3d = None      # {track_id: (x, y, z)} for the 3-D stage
_sessions = {}        # session_id -> {"cc": CrateChat, "exp": epoch}


# ── lazy singletons ──────────────────────────────────────────────────────────

def _crate():
    """Build the Crate once. Imports are deferred so app boot stays cheap."""
    global _chat
    if _chat is None:
        import crate_chat
        _chat = crate_chat.CrateChat()
        print("[chat] crate ready: %d tracks, axes %s"
              % (_chat.crate.n, ",".join(_chat.crate.axis_names())))
    return _chat


def _map_coords():
    """Precomputed 2-D coordinates per track, for the trail."""
    global _coords
    if _coords is None:
        _coords = {}
        try:
            if os.path.exists(MAP_COORDS):
                with open(MAP_COORDS) as f:
                    _coords = dict((str(k), tuple(v))
                                   for k, v in json.load(f).items())
            else:
                with open(MAP_JSON) as f:
                    for p in json.load(f).get("points", []):
                        if p.get("id") is not None:
                            _coords[str(p["id"])] = (p.get("x"), p.get("y"))
            print("[chat] loaded %d map coordinates" % len(_coords))
        except Exception as e:
            print("[chat] no map coordinates (%s) — trail will be empty" % e)
    return _coords


def _map_coords_3d():
    global _coords3d
    if _coords3d is None:
        _coords3d = {}
        try:
            with open(MAP_COORDS_3D) as f:
                _coords3d = dict((str(k), tuple(v)) for k, v in json.load(f).items())
            print("[chat] loaded %d 3-D coordinates" % len(_coords3d))
        except Exception as e:
            print("[chat] no 3-D coordinates (%s)" % e)
    return _coords3d


def _position(tracks):
    """Where a query 'lands' on the stage: the relevance-weighted centroid of
    the tracks it actually returned.

    Deliberately not a projection of the query vector. UMAP is far too
    non-linear for a cheap 512->3 map to be honest, and its `transform` costs
    seconds. This is exact, free, and always consistent with what lights up —
    the query appears among the points it chose. It is a visualisation, not a
    claim about where the request "really" sits in 512 dimensions.
    """
    c3 = _map_coords_3d()
    num = [0.0, 0.0, 0.0]
    tot = 0.0
    for t in tracks or []:
        xyz = c3.get(str(t.get("track_id")))
        if not xyz:
            continue
        w = float(t.get("relevance") or 0.0) + 0.05
        tot += w
        for i in range(3):
            num[i] += w * float(xyz[i])
    if not tot:
        return None
    return [round(v / tot, 3) for v in num]


def _session(sid, create=False):
    now = time.time()
    for k in [k for k, v in _sessions.items() if v["exp"] < now]:
        _sessions.pop(k, None)
    s = _sessions.get(sid)
    if s:
        s["exp"] = now + SESSION_TTL
        return s["cc"]
    if not create:
        return None
    if len(_sessions) >= MAX_SESSIONS:
        oldest = min(_sessions, key=lambda k: _sessions[k]["exp"])
        _sessions.pop(oldest, None)
    import crate_chat
    base = _crate()
    cc = crate_chat.CrateChat(crate_model=base.crate)   # share the loaded model
    _sessions[sid] = {"cc": cc, "exp": now + SESSION_TTL}
    return cc


def _authorised():
    return (not SHARED_SECRET) or request.headers.get("X-Chat-Key") == SHARED_SECRET


def _public(results):
    """Strip internals before the browser sees them.

    Scores are deliberately not exposed: text-audio cosines are compressed by
    the modality gap, so a raw 0.31 reads as a bad match when the ranking is
    fine. The `fit` verdict is the honest version of that signal.
    """
    coords, c3 = _map_coords(), _map_coords_3d()
    results = results or []
    # Brightness, not a score. Normalised within the turn so the strongest match
    # reads as 1 and the wildcard — genuinely further out — visibly dimmer. The
    # raw cosine stays out of the payload: an absolute 0.31 misleads, a relative
    # intensity does not.
    scores = [float(r.get("score") or 0.0) for r in results]
    lo, hi = (min(scores), max(scores)) if scores else (0.0, 1.0)
    span = hi - lo
    # A catalogue answer or a direct play has no spread to normalise, so every
    # row is equally "relevant" — light them all fully rather than uniformly dim.
    uniform = span <= 1e-9
    span = span or 1.0
    out = []
    crate_model = _chat.crate if _chat else None
    for r, sc in zip(results, scores):
        xy = coords.get(str(r.get("track_id"))) or (None, None)
        xyz = c3.get(str(r.get("track_id")))
        # A track added since the map files were built is in neither lookup.
        # It still has a position — the mean of its nearest neighbours — and
        # throwing it away is what stopped added songs appearing on the map
        # and blocked navigation to them.
        if (xy[0] is None or xyz is None) and crate_model is not None \
                and r.get("row") is not None:
            try:
                near = crate_model.neighbour_position(int(r["row"]))
                if xy[0] is None and "x" in near:
                    xy = (near["x"], near["y"])
                if xyz is None and "x3" in near:
                    xyz = (near["x3"], near["y3"], near["z3"])
            except Exception:
                pass
        out.append(dict(track_id=r.get("track_id"), name=r.get("name"),
                        artist=r.get("artist"), album=r.get("album"),
                        preview_url=r.get("preview_url"),
                        wildcard=bool(r.get("wildcard")),
                        profile=r.get("profile") or {},
                        isolation=r.get("isolation"),
                        relevance=(1.0 if uniform
                                   else round(0.15 + 0.85 * (sc - lo) / span, 3)),
                        x=xy[0], y=xy[1],
                        x3=(xyz[0] if xyz else None),
                        y3=(xyz[1] if xyz else None),
                        z3=(xyz[2] if xyz else None)))
    return out


def _trail(cc):
    """Trail points, one per turn, positioned by each turn's top result."""
    coords = _map_coords()
    pts = []
    for t in cc.session.turns:
        ids = t.get("results") or []
        if not ids:
            continue
        xy = coords.get(str(ids[0]))
        if xy and xy[0] is not None:
            pts.append(dict(x=xy[0], y=xy[1], op=t.get("op"),
                            label=t.get("label")))
    return pts


# ── routes ───────────────────────────────────────────────────────────────────

@bp.route("/api/chat/config", methods=["GET"])
def chat_config():
    if not _authorised():
        return jsonify({"error": "unauthorised"}), 401
    try:
        cc = _crate()
    except Exception as e:
        return jsonify({"error": "crate unavailable: %s" % e}), 503
    dropped = sorted(n for n, r in cc.crate.axis_report.items()
                     if r.get("verdict") == "failed")
    info = llm.describe()
    return jsonify({
        "tracks": cc.crate.n,
        "axes": cc.crate.axis_names(),
        "axes_dropped": dropped,
        "adapter": cc.crate._adapter is not None,
        "live": bool(info.get("configured")),
        "backend": info.get("backend"),
        "model": info.get("model"),
        "base_url": info.get("base_url"),
        "reason": info.get("reason"),
        "workspace_id_set": info.get("workspace_id_set"),
    })


@bp.route("/api/chat/diagnostics", methods=["GET"])
def chat_diagnostics():
    """Is this deployment actually wired up? One call, no guessing.

    Exists because diagnosing "it cannot reach Apple Music" took several rounds
    of reading boot logs that did not contain the failure. Reports what works
    and the exact error when something does not — never key material, only
    whether a key is present and what it failed with.
    """
    if not _authorised():
        return jsonify({"error": "unauthorised"}), 401
    out = {}

    info = llm.describe()
    out["llm"] = {"backend": info.get("backend"), "model": info.get("model"),
                  "configured": info.get("configured"), "reason": info.get("reason")}

    # Apple, step by step, so a failure says which step
    apple = {"env_present": {}}
    for v in ("APPLE_TEAM_ID", "APPLE_KEY_ID", "APPLE_PRIVATE_KEY",
              "APPLE_PRIVATE_KEY_PATH"):
        raw = os.environ.get(v, "")
        apple["env_present"][v] = bool(raw)
        if v == "APPLE_PRIVATE_KEY" and raw:
            apple["private_key_looks_like_pem"] = "BEGIN" in raw and "PRIVATE KEY" in raw
            apple["private_key_chars"] = len(raw)
    try:
        import apple_token
        apple["is_configured"] = apple_token.is_configured()
        if apple["is_configured"]:
            try:
                tok = apple_token.get_developer_token()
                apple["token_mint"] = "ok (%d chars)" % len(tok)
                import requests
                r = requests.get(
                    "https://api.music.apple.com/v1/catalog/%s/search"
                    % os.environ.get("APPLE_STOREFRONT", "gb"),
                    headers={"Authorization": "Bearer %s" % tok}, timeout=15,
                    params={"term": "steely dan bodhisattva", "types": "songs",
                            "limit": 3})
                apple["live_search_status"] = r.status_code
                if r.status_code >= 400:
                    apple["live_search_body"] = r.text[:300]
                else:
                    songs = r.json().get("results", {}).get("songs", {}).get("data", [])
                    apple["live_search_results"] = len(songs)
                    apple["with_preview"] = sum(
                        1 for d in songs if (d.get("attributes", {}).get("previews") or []))
            except Exception as e:
                apple["token_or_search_error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
    except Exception as e:
        apple["import_error"] = "%s: %s" % (type(e).__name__, str(e)[:160])
    out["apple_music"] = apple

    out["elevenlabs"] = {"configured": bool(os.environ.get("ELEVENLABS_API_KEY"))}

    store_dir = os.environ.get("CRATE_STORE_DIR", "crate_store")
    store = {"dir": store_dir}
    try:
        if not os.path.isdir(store_dir):
            os.makedirs(store_dir)
        probe = os.path.join(store_dir, ".write-probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        store["writable"] = True
        store["persists_across_deploys"] = store_dir.startswith("/") and \
            not store_dir.startswith("/app")
    except Exception as e:
        store["writable"] = False
        store["error"] = str(e)[:160]
    out["store"] = store

    try:
        import acquire
        out["acquisition"] = acquire.budget()
    except Exception as e:
        out["acquisition"] = {"error": str(e)[:160]}

    try:
        c = _crate()
        out["library"] = {"tracks": c.crate.n,
                          "axes": len(c.crate.axes),
                          "enrichment": c.crate.year is not None,
                          "map_coords": len(_map_coords()),
                          "map_coords_3d": len(_map_coords_3d())}
    except Exception as e:
        out["library"] = {"error": str(e)[:160]}

    return jsonify(out)


@bp.route("/api/chat/session", methods=["POST"])
def chat_session():
    if not _authorised():
        return jsonify({"error": "unauthorised"}), 401
    import crate_store
    sid = crate_store.new_session_id()
    _session(sid, create=True)
    return jsonify({"session": sid})


@bp.route("/api/chat", methods=["POST"])
def chat():
    if not _authorised():
        return jsonify({"error": "unauthorised"}), 401
    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()
    sid = body.get("session")
    if not message:
        return jsonify({"error": "no message"}), 400
    info = llm.describe()
    if not info.get("configured"):
        return jsonify({"error": info.get("reason")
                        or "no LLM backend configured"}), 503

    cc = _session(sid) if sid else None
    if cc is None:
        import crate_store
        sid = crate_store.new_session_id()
        cc = _session(sid, create=True)

    try:
        import crate_store
        crate_store.log_message(cc.session.id, "user", message)
        reply = cc.say(message, verbose=False)
        crate_store.log_message(cc.session.id, "assistant", reply)
    except Exception as e:
        print("[chat] error: %s: %s" % (type(e).__name__, e))
        return jsonify({"error": str(e)}), 500

    print("[chat] %s" % cc.cache_summary())
    tracks = _public(cc.last_results)
    return jsonify({"session": sid, "reply": reply,
                    "tracks": tracks,
                    "steps": cc.steps,
                    "position": _position(tracks),
                    "trail": _trail(cc),
                    "usage": cc.usage})


@bp.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """Server-sent events, one per beat of the mechanism.

    POST rather than EventSource because EventSource is GET-only and cannot
    carry X-Chat-Key; the browser reads this with fetch + a stream reader.
    `stream_with_context` keeps the Flask request alive for the generator, the
    same way personal_routes streams a library build.
    """
    if not _authorised():
        return jsonify({"error": "unauthorised"}), 401
    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()
    if not message:
        return jsonify({"error": "no message"}), 400
    info = llm.describe()
    if not info.get("configured"):
        return jsonify({"error": info.get("reason") or "no LLM backend"}), 503

    sid = body.get("session")
    cc = _session(sid) if sid else None
    if cc is None:
        import crate_store
        sid = crate_store.new_session_id()
        cc = _session(sid, create=True)

    def events():
        import crate_store
        crate_store.log_message(cc.session.id, "user", message)
        yield "data: %s\n\n" % json.dumps({"type": "session", "session": sid})
        said = []
        try:
            for ev in cc.say_stream(message):
                if ev.get("type") == "tracks":
                    ev = dict(ev)
                    ev["results"] = _public(ev.get("results"))
                    ev["position"] = _position(ev["results"])
                if ev.get("type") == "reply":
                    said.append(ev.get("text") or "")
                yield "data: %s\n\n" % json.dumps(ev)
        except Exception as e:
            yield "data: %s\n\n" % json.dumps(
                {"type": "error", "message": "%s: %s" % (type(e).__name__, e)})
        if said:
            crate_store.log_message(cc.session.id, "assistant", "".join(said))
        print("[chat] %s" % cc.cache_summary())
        yield "data: %s\n\n" % json.dumps({"type": "end"})

    return Response(stream_with_context(events()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@bp.route("/api/chat/feedback", methods=["POST"])
def chat_feedback():
    """Explicit thumbs from the UI.

    Worth having separately from the conversation: a click is unambiguous where
    prose is not, and this is the data the learning in crate_store runs on. The
    rating is free-form in [0,1] so a slider can send a graded judgement rather
    than a binary one.
    """
    if not _authorised():
        return jsonify({"error": "unauthorised"}), 401
    body = request.get_json(silent=True) or {}
    sid = body.get("session")
    track_id = body.get("track_id")
    cc = _session(sid) if sid else None
    if cc is None or not track_id:
        return jsonify({"error": "unknown session or track"}), 400
    try:
        rating = float(body.get("rating", 1.0))
    except (TypeError, ValueError):
        return jsonify({"error": "rating must be a number in [0,1]"}), 400
    rating = max(0.0, min(1.0, rating))
    import crate_store
    crate_store.log_feedback(cc.session.id, track_id, rating,
                             body.get("source", "ui"))
    return jsonify({"ok": True})
