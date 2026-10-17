"""
Conversation storage, and the one piece of learning it makes possible.

Every turn and every piece of feedback is appended to a JSONL log. Append-only
because the write path is the hot path and a crashed process should never cost
more than the line it was writing; plain JSON because the file has to stay
readable by hand when a result looks wrong six weeks later.

Query vectors and the tracks that came back are stored together, so accumulated
"that one worked" judgements can be turned into a correction. `fit_offset`
estimates the text-to-audio modality gap directly:

    delta = mean over liked pairs of ( unit(audio_vec) - unit(text_query_vec) )
    q'    = unit(q + eta * delta)

NEGATIVE RESULT, and the reason the gate below exists. On this library that
correction does not work. Simulated with 16 text queries where the liked track
sat at rank 8 (realistic — a correction has no room to help if feedback only
ever confirms the top hit), every strength made held-out rank *worse*, and
monotonically:

    eta    0.15   0.30   0.50   0.70   1.00     (no adapter: 9.2)
    rank    9.8   11.1   13.9   18.8   30.2

This agrees with the GAP_MODE experiment in crate.py, which found that
translating or centring text queries toward the audio cone degrades relevance.
Both are the same operation, and both say the same thing: the modality gap here
is a *calibration* problem, not a *ranking* problem. Absolute text-audio
cosines are low (mean 0.13), but the ordering they induce is good, and shifting
queries toward the audio centroid collapses them toward hub tracks that are
mildly close to everything.

So do not spend the logged feedback on the gap. The directions worth trying,
all fittable from `training_pairs`, are the ones that learn *this listener*
rather than the geometry:
  - per-user axis magnitudes: if "more upbeat" keeps following a result, the
    energy step is mis-scaled for them. Needs axis-critique data, which the
    log now captures via refine turns.
  - a diagonal re-weighting of the 512 dims (512 params, ~50 ratings).
  - low-rank W = I + UV' (~200 ratings).
Fine-tuning CLAP's text tower is not on that list: 110M parameters against a
few dozen ratings overfits immediately. The audio index is never touched either
way, so nothing here forces a re-embed of the library.

`fit_offset` refuses to save an adapter that does not improve leave-one-out
held-out rank, which is why none is installed. An unvalidated adapter is worse
than none: it silently degrades every query instead of visibly failing.

    python crate_store.py stats     # what has accumulated
    python crate_store.py fit       # fit + validate + save the offset adapter
"""

import base64
import json
import os
import sys
import time
import uuid

import numpy as np

STORE_DIR = os.environ.get("CRATE_STORE_DIR", "crate_store")
EVENTS = os.path.join(STORE_DIR, "events.jsonl")
ADAPTER = os.path.join(STORE_DIR, "adapter.npz")

# Below this many liked text-query pairs the offset is noise, not a correction.
MIN_PAIRS = 8
# Candidate strengths; the one with the best leave-one-out rank wins.
ETAS = (0.15, 0.3, 0.5, 0.7, 1.0)


def _b64(v):
    return base64.b64encode(np.asarray(v, dtype="float32").tobytes()).decode("ascii")


def _unb64(s):
    return np.frombuffer(base64.b64decode(s), dtype="float32")


def new_session_id():
    return uuid.uuid4().hex[:16]


def append(event):
    """Append one event. Never raises — losing a log line must not kill a reply."""
    try:
        if not os.path.isdir(STORE_DIR):
            os.makedirs(STORE_DIR)
        event.setdefault("ts", time.time())
        with open(EVENTS, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        sys.stderr.write("[crate_store] could not log: %s\n" % e)


def log_turn(session_id, op, label, query_vec, results, fit=None,
             wildcard=None, wildcard_rank=None):
    """Log one turn. The wildcard is recorded separately from the ranked
    results, because feedback on it is the high-information kind — it is a
    judgement about a track the ranking placed outside the top few."""
    append(dict(type="turn", session=session_id, op=op, label=label,
                q=_b64(query_vec), results=list(results), fit=fit,
                wildcard=wildcard, wildcard_rank=wildcard_rank))


def log_feedback(session_id, track_id, rating, source="keep"):
    """rating in [0,1]: 1.0 worked, 0.0 did not. Graded values are welcome."""
    append(dict(type="feedback", session=session_id, track_id=str(track_id),
                rating=float(rating), source=source))


def log_acquisition(kind, ref, detail=None):
    """Record that a track was added or generated.

    Written to the same append-only log as everything else so the rate limit
    survives a restart and is shared across sessions — an in-process counter
    would reset on every deploy and be per-user, which is not a limit at all.
    """
    append(dict(type="acquisition", kind=kind, ref=str(ref), detail=detail))


def count_recent(kind, seconds=3600, path=None):
    """How many acquisitions of this kind in the last `seconds`."""
    cutoff = time.time() - seconds
    return sum(1 for e in load_events(path)
               if e.get("type") == "acquisition" and e.get("kind") == kind
               and float(e.get("ts", 0)) >= cutoff)


def oldest_recent(kind, seconds=3600, path=None):
    """Timestamp of the earliest acquisition still inside the window, so the
    assistant can say when the next one becomes available rather than just
    refusing."""
    cutoff = time.time() - seconds
    ts = [float(e.get("ts", 0)) for e in load_events(path)
          if e.get("type") == "acquisition" and e.get("kind") == kind
          and float(e.get("ts", 0)) >= cutoff]
    return min(ts) if ts else None


def log_message(session_id, role, text):
    append(dict(type="message", session=session_id, role=role, text=text))


def load_events(path=None):
    p = path or EVENTS
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue          # a torn final line from a killed process
    return out


def training_pairs(crate_model, ops=("text",), path=None):
    """Assemble (query_vec, track_vec, rating, op) from the log.

    Feedback attaches to the most recent turn in the same session, which is how
    the chat layer emits it. Tracks that have since left the library are
    skipped rather than guessed at.
    """
    events = load_events(path)
    last_turn = {}
    pairs = []
    for e in events:
        s = e.get("session")
        if e.get("type") == "turn":
            last_turn[s] = e
        elif e.get("type") == "feedback":
            t = last_turn.get(s)
            if not t or (ops and t.get("op") not in ops):
                continue
            row = crate_model._id_to_row.get(str(e["track_id"]))
            if row is None:
                continue
            pairs.append((_unb64(t["q"]), crate_model.X[row],
                          float(e["rating"]), t.get("op")))
    return pairs


def wildcard_report(path=None):
    """How often the wildcard is the thing they pick.

    This is the number that says whether exploration is paying for itself. A
    hit rate near zero means the band is too far out; near the rate for ranked
    results means it is not exploring at all.
    """
    events = load_events(path)
    wilds, last = {}, {}
    shown = picked = rated = 0
    for e in events:
        if e.get("type") == "turn":
            last[e.get("session")] = e
            if e.get("wildcard"):
                shown += 1
                wilds[(e.get("session"), str(e["wildcard"]))] = e
        elif e.get("type") == "feedback":
            t = last.get(e.get("session")) or {}
            if t.get("wildcard") and str(t["wildcard"]) == str(e["track_id"]):
                rated += 1
                if float(e.get("rating", 0)) >= 0.7:
                    picked += 1
    ranks = [e.get("wildcard_rank") for e in wilds.values()
             if e.get("wildcard_rank") is not None]
    return dict(shown=shown, rated=rated, liked=picked,
                mean_rank=(round(float(np.mean(ranks)), 1) if ranks else None),
                hit_rate=(round(picked / float(rated), 2) if rated else None))


def fit_offset(crate_model, path=None, verbose=True):
    """Estimate the modality-gap offset from feedback and validate it.

    Returns (delta, eta, report) or (None, None, report) if it does not earn its
    place. Validation is leave-one-out: fit on every pair but one, then check
    whether the held-out liked track ranks higher under the corrected query.
    """
    pairs = [(q, x, r) for q, x, r, _ in
             training_pairs(crate_model, ops=("text",), path=path) if r >= 0.7]
    rep = dict(n_pairs=len(pairs), min_pairs=MIN_PAIRS)
    if len(pairs) < MIN_PAIRS:
        rep["verdict"] = ("not enough feedback yet — %d liked text queries, "
                          "need %d" % (len(pairs), MIN_PAIRS))
        if verbose:
            print(rep["verdict"])
        return None, None, rep

    u = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
    Q = u(np.stack([p[0] for p in pairs]))
    A = u(np.stack([p[1] for p in pairs]))
    X = crate_model.X

    def rank_of(q, target_vec):
        s = X @ u(q)
        return int((s > float(u(target_vec) @ u(q))).sum())

    base = float(np.mean([rank_of(Q[i], A[i]) for i in range(len(pairs))]))
    best = (None, None, base)
    for eta in ETAS:
        ranks = []
        for i in range(len(pairs)):
            keep = [j for j in range(len(pairs)) if j != i]
            d = (A[keep] - Q[keep]).mean(0)          # fitted without pair i
            ranks.append(rank_of(Q[i] + eta * d, A[i]))
        m = float(np.mean(ranks))
        if verbose:
            print("  eta=%.2f  mean held-out rank %.1f  (no adapter: %.1f)"
                  % (eta, m, base))
        if m < best[2]:
            best = (eta, m, m)
    rep["baseline_rank"] = round(base, 1)

    if best[0] is None:
        rep["verdict"] = ("no offset improved held-out rank (baseline %.1f) — "
                          "not saving one" % base)
        if verbose:
            print(rep["verdict"])
        return None, None, rep

    eta = best[0]
    delta = (A - Q).mean(0)
    rep.update(eta=eta, adapted_rank=round(best[1], 1),
               improvement=round(base - best[1], 1),
               verdict="offset improves mean held-out rank %.1f -> %.1f"
                       % (base, best[1]))
    if verbose:
        print(rep["verdict"])
    return delta.astype("float32"), eta, rep


def save_adapter(delta, eta, report, path=None):
    p = path or ADAPTER
    if not os.path.isdir(os.path.dirname(p) or "."):
        os.makedirs(os.path.dirname(p))
    np.savez(p, delta=delta, eta=np.array([eta], dtype="float32"),
             report=np.array([json.dumps(report)]))


def load_adapter(path=None):
    """Return (delta, eta) or (None, None). Applied only to text queries."""
    p = path or ADAPTER
    if not os.path.exists(p):
        return None, None
    try:
        z = np.load(p, allow_pickle=False)
        return z["delta"].astype("float32"), float(z["eta"][0])
    except Exception:
        return None, None


def stats(path=None):
    events = load_events(path)
    turns = [e for e in events if e.get("type") == "turn"]
    fb = [e for e in events if e.get("type") == "feedback"]
    sessions = set(e.get("session") for e in events)
    ops = {}
    for t in turns:
        ops[t.get("op")] = ops.get(t.get("op"), 0) + 1
    print("events       %d" % len(events))
    print("sessions     %d" % len(sessions))
    print("turns        %d  %s" % (len(turns), ops))
    print("feedback     %d  (%d positive)"
          % (len(fb), sum(1 for e in fb if e.get("rating", 0) >= 0.7)))
    w = wildcard_report(path)
    print("wildcards    %d shown, %d rated, %d liked (hit rate %s, mean rank %s)"
          % (w["shown"], w["rated"], w["liked"], w["hit_rate"], w["mean_rank"]))
    d, eta = load_adapter()
    print("adapter      %s" % ("eta=%.2f, ||delta||=%.3f"
                               % (eta, float(np.linalg.norm(d)))
                               if d is not None else "none fitted"))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if cmd == "stats":
        stats()
    elif cmd == "fit":
        import crate
        c = crate.Crate()
        d, eta, rep = fit_offset(c)
        if d is not None:
            save_adapter(d, eta, rep)
            print("saved -> %s" % ADAPTER)
    else:
        print(__doc__)
