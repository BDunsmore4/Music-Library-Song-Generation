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

from flask import Blueprint, jsonify, request

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
MAP_JSON = os.environ.get("FRONTEND_DATA_JSON",
                          os.path.join(_HERE, "Frontend", "data.json"))

_chat = None          # the shared CrateChat template (holds the Crate)
_coords = None        # {track_id: (x, y)} from the frontend map
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
    coords = _map_coords()
    out = []
    for r in results or []:
        xy = coords.get(str(r.get("track_id"))) or (None, None)
        out.append(dict(track_id=r.get("track_id"), name=r.get("name"),
                        artist=r.get("artist"), album=r.get("album"),
                        preview_url=r.get("preview_url"),
                        wildcard=bool(r.get("wildcard")),
                        x=xy[0], y=xy[1]))
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
    })


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
    if not llm.describe().get("configured"):
        return jsonify({"error": "no LLM backend configured — set LLM_BACKEND "
                                 "and LLM_BASE_URL/LLM_API_KEY"}), 503

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

    return jsonify({"session": sid, "reply": reply,
                    "tracks": _public(cc.last_results),
                    "trail": _trail(cc)})


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
