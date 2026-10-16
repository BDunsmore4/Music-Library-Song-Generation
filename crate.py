"""
The crate-digging retrieval model: CLAP search, artist seeding and Rocchio
refinement over one person's library.

Nothing in here talks to an LLM. The chat layer (crate_chat.py) only ever calls
the four public operations on `Crate`, so the retrieval maths stays testable on
its own and the language model never sees a vector.

    crate = Crate()
    s = crate.new_session()
    crate.search_by_description(s, ["mellow acoustic guitar, soft vocals"])
    crate.search_by_artist(s, "Bonobo")
    crate.refine(s, adjust=[("groove", "+", "moderate")], reject=[track_id])

Three facts about CLAP drive the whole design, all measured on this library:

  * It does not know names. "This is a song by Beethoven" scored 0/5 against a
    library holding 184 Beethoven tracks. So artist and title requests must be
    resolved through metadata and then searched *audio-to-audio* — which is
    also tighter: audio-audio cosine averages 0.46 against 0.13 text-audio.

  * Text and audio sit in different cones: centroids 0.666 apart, cos 0.392.
    The obvious fix is to centre or translate text queries into the audio cone,
    and on this library that measurably makes retrieval *worse* — see GAP_MODE,
    which documents the comparison and why the default is to leave text alone.
    A Rocchio blend therefore does mix a text-cone component into an audio-cone
    query; DELTA is set low and uncalibrated for that reason.

  * It cannot read negation — "no drums" retrieves drums. So reductions are
    never phrased in prose; they are subtractions along a validated axis
    (taste_axes.py).
"""

import os
import re

import numpy as np

import taste_axes
from taste_axes import unit

EMB_NPY = taste_axes.EMB_NPY
META_CSV = taste_axes.META_CSV

CLAP_CHECKPOINT = os.environ.get("CLAP_CHECKPOINT", "laion/clap-htsat-unfused")
STOREFRONT = os.environ.get("APPLE_STOREFRONT", "gb")

# How a text query is handled against the audio index. Options: "none" (raw
# templated text vectors), "translate" (strip the text centroid, re-seat on the
# audio centroid), "centre_both" (centre both sides, compare in centred space).
#
# This was settled by eye on five queries, NOT by the coherence metric — and the
# two disagree, which is the whole lesson. Scoring neighbourhood coherence@20
# (how acoustically tight the retrieved set is) ranks them none 0.35 <
# translate 0.39 < centre_both 0.42, so coherence says centre. But inspecting
# the actual results reverses it:
#
#   "warm mellow acoustic guitar, soft vocals, slow tempo"
#      none        Ferenc Snetberger, Ferenc Snetberger, Luiz Eca   <- right
#      translate   Handel keyboard suite, Snetberger, Luiz Bonfa
#   "smooth jazz saxophone, walking bass, brushed drums"
#      none        Stevie Wonder, Alfa Mist, Alfa Mist              <- right
#      translate   Schulhoff 'Hot Sonate', Yoshimatsu, Miles Davis
#
# Coherence rewards a *tight* cluster, not a correct one: a tight clump of
# classical piano trios scores beautifully and answers the wrong question.
# Across the five queries "none" wins twice outright, centre_both once
# (AC/DC x3 for the distorted-guitar query) and they tie twice; "translate"
# never wins. So: raw templated text, and do not trust coherence as a proxy for
# relevance again. Keep the modes switchable — this is a property of this
# library, not of CLAP, and it is worth re-running on a different one.
GAP_MODE = os.environ.get("CLAP_GAP_MODE", "none")

# Rocchio weights. alpha keeps the conversation's momentum, beta pulls toward
# what they kept, gamma pushes off what they rejected, delta mixes in whatever
# they said in prose this turn.
#
# DELTA is deliberately the least trustworthy of the four and is not calibrated.
# With GAP_MODE="none" a text component is a text-cone vector sitting in an
# otherwise audio-cone query, and text-audio cosines are compressed (mean 0.13)
# next to audio-audio (mean 0.46) — so a unit of text contributes less ranking
# signal than a unit of audio, by an amount that depends on the query. If a
# follow-up is purely verbal ("actually I want something jazzier"), prefer a
# fresh search_by_description over refine(descriptions=...).
ALPHA, BETA, GAMMA, DELTA = 1.0, 0.8, 0.4, 0.5

# Axis magnitudes, calibrated by sweeping lambda on a seed track and watching
# top-10 churn: 0.2 barely moves, 0.5 lands on the right answer, 1.2 loses the
# original intent entirely.
MAGNITUDE = {"slight": 0.25, "moderate": 0.5, "strong": 0.8}

# Weight an artist reference carries when blended into the running query. Larger
# than MAGNITUDE because an artist is a *document* (a real point in audio space),
# not a direction, so it is directly comparable to the BETA term for kept
# tracks; "strong" deliberately overpowers ALPHA so "no, much more like this"
# can still take over the conversation.
ARTIST_W = {"slight": 0.5, "moderate": 0.9, "strong": 1.6}

# Cap on the combined axis push per query, for the same reason: past this much
# drift the query leaves the neighbourhood the user was talking about. Set to
# 1.2 rather than 1.0 so an ordinary two-axis critique ("groovier and less
# vocal" = 0.8 and 0.5, norm ~1.005) does not trip the cap and make the
# assistant report a clipped move on every turn. Three strong axes still clip.
MAX_DRIFT = 1.2

# The wildcard: one deliberate outlier appended to every set of results.
#
# This is an exploration device, not a garnish. Feedback on top-ranked results
# carries almost no information — in simulation, when the liked track was always
# the top hit, mean held-out rank was already 0.1 and no adapter could beat it
# (see crate_store). Learning needs judgements about tracks the ranking got
# *wrong but close*, and a pure exploit loop never produces one.
#
# BAND is the rank window it draws from: past the results proper, but still
# inside the top ~3% of 7,106 tracks, so it stays a plausible suggestion rather
# than noise. Within the band the pick is sampled with probability rising with
# distance from the presented cluster — the "sits slightly further out" part —
# and TEMP sets how random that is.
#
# TEMP was measured, not guessed: 60 draws against one fixed query, counting
# distinct tracks offered and where they land. The three matches for that query
# averaged cos 0.771 to the query and 0.948 to their own centroid.
#
#   TEMP   distinct/60   rank p10/med/p90   cos->query   cos->cluster
#   0.5        25         186/224/239          0.560        0.606
#   1.5        51          97/194/234          0.590        0.701
#   2.5        55          77/181/230          0.601        0.724
#
# At 0.5 the novelty weighting saturates: every draw is the furthest thing the
# band allows, so only 25 distinct tracks ever appear and the band's lower half
# is never used. At 2.5 it flattens toward uniform over the band and loses the
# distance bias that makes the pick interesting. 1.5 spreads across the band
# while staying far outside the presented cluster (0.70 against the matches'
# 0.95) and still plausibly relevant (0.59 against their 0.77).
WILDCARD_BAND = (15, 250)
WILDCARD_TEMP = 1.5

# A fixed, deliberately varied pool used once to estimate the text-cone centroid.
# It needs to span the kind of prose the assistant actually writes, which is why
# it is descriptive sentences rather than the bare genre words in anchors.json.
_TEXT_POOL = [
    "mellow acoustic guitar with soft vocals and a slow tempo",
    "aggressive distorted electric guitar with fast drums and shouting",
    "sparse solo piano, melancholic and quiet",
    "upbeat electronic dance music with a four on the floor kick",
    "smooth jazz saxophone with walking bass and brushed drums",
    "a hip hop beat with heavy bass and rapped vocals",
    "sweeping cinematic orchestral strings",
    "ambient synthesiser texture with no clear pulse",
    "bright female pop vocals over shimmering synths",
    "a slow mournful solo cello",
    "afrobeat with layered percussion and horns",
    "lo-fi bedroom indie with jangly guitar",
    "a string quartet playing a fast contrapuntal passage",
    "dub reggae with deep bass and spring reverb",
    "driving motorik krautrock drums with repetitive bass",
    "close-miked folk fingerpicking with breathy singing",
    "big band swing with muted trumpets",
    "distorted lo-fi drum machine and detuned synth bass",
    "gospel choir with organ and handclaps",
    "minimal techno with a dry kick and filtered hats",
    "a solo harpsichord playing a baroque dance",
    "country ballad with pedal steel and brushed snare",
    "noisy shoegaze wall of reverbed guitar",
    "soft bossa nova guitar with gentle vocals",
    "a male opera voice with full orchestra",
    "trap beat with rolling hi-hats and sub bass",
    "sitar and tabla in a slow raga",
    "bluegrass banjo and fiddle at speed",
    "a church organ playing slow sustained chords",
    "funk with slap bass and a tight horn section",
    "drum and bass with chopped breaks and heavy sub",
    "a lullaby on music box and soft strings",
    "field recording of rain with sparse piano",
    "punk played fast and loose with shouted vocals",
    "soul ballad with Rhodes piano and warm bass",
    "glitchy experimental electronics with no beat",
]


def _norm_name(s):
    """Lowercase and strip punctuation, for forgiving name matching."""
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


# ── encoders ─────────────────────────────────────────────────────────────────

class Encoders(object):
    """CLAP text and audio encoders, loaded once and lazily.

    If the web app's `embedder` module is already importable we reuse its loaded
    model rather than paying for a second copy — they must be the same
    checkpoint, since a text vector is only comparable to audio vectors
    produced by the same tower.
    """

    def __init__(self, checkpoint=CLAP_CHECKPOINT, audio_centroid=None):
        self.checkpoint = checkpoint
        self._model = None
        self._tok = None
        self._fe = None
        self._text_centroid = None
        self._audio_centroid = audio_centroid

    # Modules that may already hold a loaded CLAP, and the attributes they keep
    # it under. Two layouts exist across this project's history: `embedder.py`
    # exposes `_model`/`_feature_extractor`, while the deployed `app.py` loads
    # CLAP inline as `model`/`feature_extractor`. Finding either saves ~1GB of
    # RAM on a Railway dyno, and loading a second copy fails as a silent memory
    # ceiling rather than an error — so check both layouts.
    _HOSTS = (("embedder", "_model", "_feature_extractor"),
              ("app", "model", "feature_extractor"),
              ("app", "_model", "_feature_extractor"))

    def _borrow(self):
        """Reuse a CLAP already in this process. Only inspects modules that are
        ALREADY imported — importing `app` ourselves would re-enter the Flask
        module while it is still initialising."""
        import sys
        for mod, m_attr, fe_attr in self._HOSTS:
            host = sys.modules.get(mod)
            if host is None:
                continue
            m = getattr(host, m_attr, None)
            if m is not None:
                self._model = m
                self._fe = getattr(host, fe_attr, None)
                print("[crate] reusing the CLAP model already loaded by %s" % mod)
                return True
        return False

    def _load(self):
        if self._model is not None:
            return
        self._borrow()
        from transformers import ClapModel, AutoTokenizer, AutoFeatureExtractor
        if self._model is None:
            print("[crate] no host CLAP found — loading our own copy")
            self._model = ClapModel.from_pretrained(self.checkpoint)
            self._model.eval()
        if self._fe is None:
            self._fe = AutoFeatureExtractor.from_pretrained(self.checkpoint)
        self._tok = AutoTokenizer.from_pretrained(self.checkpoint)

    # -- text ----------------------------------------------------------------
    def _raw_text(self, texts):
        import torch
        self._load()
        batch = self._tok(list(texts), return_tensors="pt", padding=True,
                          truncation=True, max_length=77)
        with torch.no_grad():
            e = self._model.get_text_features(**batch)
        return unit(e.cpu().numpy().astype("float32"))

    def text_centroid(self):
        if self._text_centroid is None:
            pool = self._raw_text([taste_axes.TEMPLATE % t for t in _TEXT_POOL])
            self._text_centroid = unit(pool.mean(0))
        return self._text_centroid

    def embed_text(self, texts, centre=True, template=True):
        """Embed descriptions and (by default) move them into audio space.

        `centre=False` returns the raw text vectors, which is what the axis
        builder wants — a *difference* of two raw text vectors already has the
        modality offset cancelled out, so translating first would be wrong.
        """
        if isinstance(texts, str):
            texts = [texts]
        if template:
            texts = [taste_axes.TEMPLATE % t for t in texts]
        E = self._raw_text(texts)
        if not centre or GAP_MODE == "none":
            return E
        E = unit(E - self.text_centroid())
        if GAP_MODE == "translate" and self._audio_centroid is not None:
            E = unit(E + self._audio_centroid)
        return E

    # -- audio ---------------------------------------------------------------
    def embed_audio_bytes(self, audio_bytes):
        """Decode arbitrary audio bytes and return a unit 512-d CLAP vector."""
        import io
        import torch
        from pydub import AudioSegment
        self._load()
        seg = None
        for fmt in ("m4a", "mp3", "wav", "ogg"):
            try:
                seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
                break
            except Exception:
                continue
        if seg is None:
            raise ValueError("could not decode audio")
        seg = seg.set_frame_rate(48000).set_channels(1)
        x = np.array(seg.get_array_of_samples()).astype(np.float32)
        x = x / (2 ** (seg.sample_width * 8 - 1))
        if len(x) == 0:
            raise ValueError("empty audio")
        inputs = self._fe(x, return_tensors="pt", sampling_rate=48000)
        with torch.no_grad():
            e = self._model.get_audio_features(**inputs)
        return unit(e.squeeze().cpu().numpy().astype("float32"))


# ── Apple Music, for artists not already in the library ──────────────────────

def apple_top_songs(artist, limit=3, storefront=STOREFRONT):
    """Return [{name, artist, isrc, preview_url}] for an artist's top songs.

    Uses the same MusicKit developer-token mechanism as the rest of the app.
    Returns [] rather than raising if Apple is not configured, so the library
    path keeps working without credentials.
    """
    import requests
    try:
        import apple_token
        token = apple_token.get_developer_token()
    except Exception:
        return []
    h = {"Authorization": "Bearer %s" % token}
    base = "https://api.music.apple.com/v1/catalog/%s" % storefront
    try:
        r = requests.get(base + "/search", headers=h, timeout=15,
                         params={"term": artist, "types": "artists", "limit": 1})
        r.raise_for_status()
        data = r.json().get("results", {}).get("artists", {}).get("data", [])
        if not data:
            return []
        aid = data[0]["id"]
        resolved = data[0].get("attributes", {}).get("name", artist)
        r = requests.get(base + "/artists/%s/view/top-songs" % aid,
                         headers=h, timeout=15, params={"limit": max(limit, 5)})
        r.raise_for_status()
        out = []
        for s in r.json().get("data", [])[:limit]:
            at = s.get("attributes", {})
            prev = (at.get("previews") or [{}])[0].get("url")
            if prev:
                out.append(dict(name=at.get("name", "?"),
                                artist=at.get("artistName", resolved),
                                isrc=at.get("isrc"), preview_url=prev))
        return out
    except Exception:
        return []


# ── conversation state ───────────────────────────────────────────────────────

class Session(object):
    """One conversation: the live query vector, the feedback, and the trail.

    `trail` is the sequence of (x, y, label) the frontend draws on the map —
    one point per turn, so the dialogue is visible as a path through the space.
    """

    def __init__(self, session_id=None, seed=None):
        self.id = session_id
        # Per-session RNG: the wildcard is meant to be unpredictable between
        # sessions but reproducible within one when a result needs explaining.
        self.seed = seed if seed is not None else int(np.random.SeedSequence()
                                                      .generate_state(1)[0])
        self.rng = np.random.default_rng(self.seed)
        self.q = None
        self.liked = []       # library row indices
        self.disliked = []
        self.seen = set()
        self.trail = []
        self.turns = []       # [{"op":…, "detail":…}] for the write-up

    def reset(self):
        self.__init__()


# ── the model ────────────────────────────────────────────────────────────────

class Crate(object):
    def __init__(self, emb_npy=None, meta_csv=None, encoders=None, axes=None):
        import pandas as pd
        self.X = unit(np.load(emb_npy or EMB_NPY).astype("float32"))
        self.meta = pd.read_csv(meta_csv or META_CSV)
        self.n = len(self.X)
        self.centroid = unit(self.X.mean(0))
        self.enc = encoders or Encoders(audio_centroid=self.centroid)
        if getattr(self.enc, "_audio_centroid", None) is None:
            self.enc._audio_centroid = self.centroid
        # Only axes that survived validation are exposed. `tempo` and
        # `brightness` fail on this library — they do not track the waveform
        # feature their names imply — so the chat model is never offered them.
        if axes is not None:
            self.axes, self.axis_report = axes, {}
        else:
            self.axes, self.axis_report = taste_axes.trusted()
        self._id_to_row = {}
        for i, tid in enumerate(self.meta["id"].astype(str)):
            self._id_to_row.setdefault(tid, i)
        self._load_enrichment()
        self._artist_norm = self.meta["artists"].astype(str).map(_norm_name)
        self._name_norm = self.meta["name"].astype(str).map(_norm_name)
        self._reducer = None
        self._axis_pct = None
        self._iso = None
        self._libclust = None
        self._nn_ref = None
        self.last_fit = None
        self.last_results = []
        self.last_filter = None
        # The offset learned from accumulated feedback, applied to text queries
        # only — that is where the modality gap lives. None until enough
        # feedback has been logged AND it beat held-out rank (crate_store).
        try:
            import crate_store
            self._adapter, self._adapter_eta = crate_store.load_adapter()
        except Exception:
            self._adapter, self._adapter_eta = None, None

    # -- where a track sits on each axis, as a library percentile -----------
    def _axis_percentiles(self):
        """Projection of every track onto every validated axis, plus a sorted
        copy so a percentile is one searchsorted.

        8 axes x 7,106 tracks is a quarter of a megabyte, built once. This is
        what lets the assistant describe a track from measurement rather than
        from whatever it happens to remember about the recording.
        """
        if self._axis_pct is None:
            self._axis_pct = {}
            for name, d in self.axes.items():
                proj = self.X @ d
                self._axis_pct[name] = (proj, np.sort(proj))
        return self._axis_pct

    def profile(self, row):
        """{axis: percentile 0-100} for one library row."""
        out = {}
        for name, (proj, srt) in self._axis_percentiles().items():
            out[name] = int(round(100.0 * np.searchsorted(srt, proj[row]) / len(srt)))
        return out

    # -- how well does the library actually answer this query? ---------------
    def _reference(self, kind, sample=800, seed=0):
        """Top-1 similarities a query of this kind *typically* achieves here.

        This has to be per kind or it is worse than useless. The three
        distributions are nowhere near each other on this library:

            audio (track to its nearest neighbour)   p5 0.846  median 0.947
            text  (a canned description's best hit)  p5 0.556  median 0.684
            blend (an axis-perturbed track)          computed below

        Scoring a text query against the audio reference labels every text
        search "poor" — which is exactly what the first version of this did,
        calling Tracy Chapman's "Fast Car" a poor answer to "warm mellow
        acoustic guitar with soft vocals".
        """
        if self._nn_ref is None:
            self._nn_ref = {}
        if kind in self._nn_ref:
            return self._nn_ref[kind]
        rng = np.random.default_rng(seed)

        if kind == "text":
            pool = self.enc.embed_text(_TEXT_POOL)
            ref = np.sort((pool @ self.X.T).max(1))
        elif kind == "seed3":
            # Artist queries average three tracks, and a mean of three vectors
            # is smoother than any one of them, so its best match is lower by
            # construction. Matching the reference to that shape is what makes
            # Fela Kuti ("close", and the library does hold afrobeat-adjacent
            # material) separable from Burial ("poor", and it holds no dubstep).
            by_artist = {}
            for i, a in enumerate(self._artist_norm.values):
                by_artist.setdefault(a, []).append(i)
            trips = [v for v in by_artist.values() if len(v) >= 3]
            rng.shuffle(trips)
            tops = []
            for rows in trips[:240]:
                pick = list(rng.choice(rows, 3, replace=False))
                q = unit(self.X[pick].mean(0))
                s = self.X @ q
                s[pick] = -np.inf
                tops.append(s.max())
            ref = np.sort(np.array(tops)) if tops else self._reference("audio")
        else:
            idx = rng.choice(self.n, min(sample, self.n), replace=False)
            S = self.X[idx] @ self.X.T
            S[np.arange(len(idx)), idx] = -np.inf
            ref = np.sort(S.max(1))

        self._nn_ref[kind] = ref
        return ref

    def fit(self, top_score, kind="audio"):
        """Place a query's best score against what this *kind* of query achieves.

        Returns {"top", "percentile", "kind", "verdict"}. The verdict is what the
        assistant repeats to the user; the number is for debugging only.
        """
        ref = self._reference(kind)
        pct = 100.0 * float((ref < top_score).mean())
        if pct >= 25:
            v = "close — the library has genuinely similar material"
        elif pct >= 5:
            v = "loose — roughly in the right area, not a close match"
        else:
            v = "poor — nothing in this library is really like that"
        return dict(top=round(float(top_score), 4), percentile=round(pct, 1),
                    kind=kind, verdict=v)

    # -- helpers -------------------------------------------------------------
    def new_session(self, session_id=None, seed=None):
        if session_id is None:
            try:
                import crate_store
                session_id = crate_store.new_session_id()
            except Exception:
                session_id = None
        return Session(session_id, seed=seed)

    def axis_names(self):
        return sorted(self.axes)

    def _rows_for_ids(self, ids):
        return [self._id_to_row[str(i)] for i in (ids or []) if str(i) in self._id_to_row]

    def _project(self, q):
        """UMAP coords for the query, for the trail. Silently skipped if absent.

        OFF BY DEFAULT, and deliberately so. umap's `transform` runs its
        optimisation epochs on every call — seconds per search, plus an 11s
        one-off load of the 50MB reducer — and the web path does not even read
        the result: chat_routes builds its trail from each turn's top result and
        the precomputed coordinates in data/map_coords.json, which is free and
        visually equivalent. Leaving this on cost a multi-second UMAP transform
        per turn for a value nobody looked at.

        Set CRATE_TRAIL=1 to get the true projection of the query vector, which
        is what `session.trail` is for if you want it from the CLI.
        """
        if os.environ.get("CRATE_TRAIL", "0") not in ("1", "true", "yes"):
            return None
        if self._reducer is None:
            try:
                import joblib
                self._reducer = joblib.load(
                    os.environ.get("UMAP_REDUCER_PATH", "umap_reducer.pkl"))
            except Exception:
                self._reducer = False
        if self._reducer is False:
            return None
        try:
            xy = self._reducer.transform(np.atleast_2d(q.astype("float32")))[0]
            return float(xy[0]), float(xy[1])
        except Exception:
            return None

    def _results(self, q, k, session, per_artist=1, exclude=(), allow=None):
        """Rank the library against q, deduped by artist and by what's been shown.

        Artist dedup matters more than it sounds: the tightest neighbourhoods
        here run to cos 0.93, so an undeduped top-10 is often four tracks from
        the seed's own album and reads as though refinement did nothing.
        """
        s = self.X @ q
        order = np.argsort(-s)
        if allow is not None:
            order = order[allow[order]]
        drop = set(exclude) | session.seen
        out, used = [], {}
        for j in order:
            j = int(j)
            if j in drop:
                continue
            a = str(self.meta.artists[j]).split(",")[0].strip().lower()
            if used.get(a, 0) >= per_artist:
                continue
            used[a] = used.get(a, 0) + 1
            row = self.meta.iloc[j]
            out.append(dict(track_id=str(row["id"]), name=str(row["name"]),
                            artist=str(row["artists"]), album=str(row.get("album", "")),
                            preview_url=(None if str(row.get("preview_url")) == "nan"
                                         else str(row.get("preview_url"))),
                            score=round(float(s[j]), 4), row=j,
                            profile=self.profile(j),
                            isolation=self.isolation_pct(j)))
            if len(out) >= k:
                break
        for r in out:
            session.seen.add(r["row"])
        return out

    def _wildcard(self, session, q, shown_rows, exclude=(), allow=None):
        """One track from the rank band, biased toward distance from the cluster.

        Returns a result dict with wildcard=True, or None if the band is empty.
        Relevance is guaranteed by the band (it is still a top-few-percent
        match); the sampling only decides *which* of those plausible-but-not-top
        tracks to offer, favouring ones unlike what was just presented.
        """
        lo, hi = WILDCARD_BAND
        s = self.X @ q
        order = np.argsort(-s)
        drop = set(int(j) for j in exclude) | set(int(j) for j in shown_rows)             | set(int(j) for j in session.disliked)
        if allow is not None:
            order = order[allow[order]]
        band = [int(j) for j in order[lo:hi] if int(j) not in drop]
        if not band:
            return None

        # Distance from the centroid of what was just shown. With nothing shown
        # yet, fall back to distance from the query itself.
        ref = unit(self.X[shown_rows].mean(0)) if len(shown_rows) else q
        nov = 1.0 - (self.X[band] @ ref)
        if nov.std() > 1e-6:
            z = (nov - nov.mean()) / nov.std()
        else:
            z = np.zeros(len(band))
        w = np.exp(z / max(WILDCARD_TEMP, 1e-3))
        w = w / w.sum()
        j = int(session.rng.choice(band, p=w))

        row = self.meta.iloc[j]
        return dict(track_id=str(row["id"]), name=str(row["name"]),
                    artist=str(row["artists"]), album=str(row.get("album", "")),
                    preview_url=(None if str(row.get("preview_url")) == "nan"
                                 else str(row.get("preview_url"))),
                    score=round(float(s[j]), 4), row=j, wildcard=True,
                    profile=self.profile(j), isolation=self.isolation_pct(j),
                    rank=int(np.where(order == j)[0][0]))

    def _commit(self, session, q, label, k, exclude=(), kind="audio", op=None,
                wildcard=True, allow=None):
        session.q = unit(q)
        xy = self._project(session.q)
        if xy:
            session.trail.append(dict(x=xy[0], y=xy[1], label=label))
        res = self._results(session.q, k, session, exclude=exclude, allow=allow)
        session.turns.append(dict(op=(op or kind or "refine"), label=label,
                                  results=[r["track_id"] for r in res]))
        # kind=None means "not calibratable" — a refined query sits off the
        # manifold by an amount that depends on which axes were pushed and how
        # hard, so no fixed reference describes it. Better to report nothing
        # than a confident label that is wrong.
        self.last_fit = (self.fit(res[0]["score"], kind)
                         if (res and kind) else None)

        # Appended after `fit`, so the wildcard never affects the confidence
        # reading, and last so res[0] stays the genuine best match.
        wc = self._wildcard(session, session.q, [r["row"] for r in res],
                            exclude=exclude, allow=allow) if (wildcard and res) else None
        if wc:
            session.seen.add(wc["row"])
            res = res + [wc]

        self.last_results = res
        if session.id:
            try:
                import crate_store
                crate_store.log_turn(
                    session.id, op or kind or "refine", label, session.q,
                    [r["track_id"] for r in res if not r.get("wildcard")],
                    self.last_fit,
                    wildcard=(wc["track_id"] if wc else None),
                    wildcard_rank=(wc["rank"] if wc else None))
            except Exception:
                pass
        return res

    # -- operation 1: free-text description ----------------------------------
    def search_by_description(self, session, descriptions, k=5, wildcard=True,
                              year_from=None, year_to=None, genres=None):
        """Average 1-3 sound descriptions into one query and search.

        Averaging is done on the unit vectors *after* translation into audio
        space, so each description contributes equally regardless of how
        confidently CLAP placed it.
        """
        if isinstance(descriptions, str):
            descriptions = [descriptions]
        descriptions = [d for d in descriptions if str(d).strip()]
        if not descriptions:
            raise ValueError("no descriptions given")
        E = self.enc.embed_text(descriptions)
        q = unit(E.mean(0))
        if self._adapter is not None:
            q = unit(q + self._adapter_eta * self._adapter)
        mask, note = self.filter_rows(year_from, year_to, genres)
        allow = mask if (year_from or year_to or genres) else None
        self.last_filter = dict(note=note, candidates=int(mask.sum())) if allow is not None else None
        label = "text: " + "; ".join(descriptions)[:90] + (" [%s]" % note if note else "")
        return self._commit(session, q, label, k, kind="text", op="text",
                            wildcard=wildcard, allow=allow)

    # -- the enrichment: release year and genre tags ------------------------
    def _load_enrichment(self):
        """Join data/library_enriched.csv on track id, if it has been built.

        Optional by design: without it `year` and `genres` filters simply
        report themselves unavailable rather than failing. Run
        `python enrich_metadata.py` to build it (~282 Apple calls, 98.8%
        coverage on this library).
        """
        import pandas as pd
        path = os.environ.get("LIBRARY_ENRICHED",
                              os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "data", "library_enriched.csv"))
        self.year = None
        self.genres = None
        if not os.path.exists(path):
            return
        try:
            e = pd.read_csv(path, dtype={"track_id": str})
            e = e.set_index("track_id")
            ids = self.meta["id"].astype(str)
            yr = pd.to_numeric(e["year"].reindex(ids).values, errors="coerce")
            self.year = yr.astype("float32")
            g = e["genres"].reindex(ids).fillna("").astype(str).str.lower().values
            self.genres = np.array(g, dtype=object)
            print("[crate] enrichment: %d/%d tracks have a year"
                  % (int(np.isfinite(self.year).sum()), self.n))
        except Exception as ex:
            print("[crate] enrichment unavailable (%s)" % ex)
            self.year = None
            self.genres = None

    def filter_rows(self, year_from=None, year_to=None, genres=None):
        """Boolean mask over the library. Returns (mask, note).

        `year` is the RELEASE date of the recording, not when the music was
        written — see enrich_metadata.py. Fine for "80s dance", wrong for
        "Baroque", which is what the genre tags are for ("Baroque Era",
        "Romantic Era" are among Apple's tags).
        """
        mask = np.ones(self.n, dtype=bool)
        notes = []
        if (year_from or year_to) and self.year is None:
            return mask, "no year data — run enrich_metadata.py"
        if year_from is not None:
            mask &= np.isfinite(self.year) & (self.year >= float(year_from))
            notes.append("from %d" % year_from)
        if year_to is not None:
            mask &= np.isfinite(self.year) & (self.year <= float(year_to))
            notes.append("to %d" % year_to)
        if genres:
            if self.genres is None:
                return mask, "no genre data — run enrich_metadata.py"
            want = [str(g).lower().strip() for g in genres if str(g).strip()]
            if want:
                gm = np.zeros(self.n, dtype=bool)
                for i, tags in enumerate(self.genres):
                    if tags and any(w in tags for w in want):
                        gm[i] = True
                mask &= gm
                notes.append("genre " + "/".join(want))
        return mask, (", ".join(notes) if notes else None)

    # -- metadata, answered as metadata -------------------------------------
    def lookup(self, artists=None, tracks=None, albums=None, limit=40):
        """Pure metadata query. No CLAP, no neighbours, no wildcard.

        "What Paul Simon have I got" is a question about the catalogue, not a
        similarity search, and answering it with four acoustic neighbours and an
        outlier is answering a different question. This also powers the
        knowledge-then-audio pattern: the assistant names the artists it would
        expect for a genre CLAP cannot read, this says which are actually here,
        and the search is then seeded from real audio.
        """
        # Word-boundary matching, not bare substring. "Ott" as a substring
        # matches Travis Scott and Ottorino Respighi, which turned a check for
        # seven psytrance artists into 158 false positives — and the assistant
        # would have believed it.
        hits, seen, per = [], set(), {}
        def pat(t):
            return r"\b" + re.escape(t) + r"\b"
        def add(rows, why):
            per[why] = per.get(why, 0) + len(rows)
            for j in rows:
                j = int(j)
                if j in seen:
                    continue
                seen.add(j)
                r = self.meta.iloc[j]
                hits.append(dict(track_id=str(r["id"]), name=str(r["name"]),
                                 artist=str(r["artists"]),
                                 album=str(r.get("album", "")),
                                 preview_url=(None if str(r.get("preview_url")) == "nan"
                                              else str(r.get("preview_url"))),
                                 matched=why, row=j, score=1.0,
                                 profile=self.profile(j),
                                 isolation=self.isolation_pct(j)))
        for a in (artists or []):
            t = _norm_name(a)
            per.setdefault(a, 0)
            if t:
                add(np.where(self._artist_norm.str.contains(pat(t), regex=True).values)[0], a)
        for t_ in (tracks or []):
            t = _norm_name(t_)
            per.setdefault(t_, 0)
            if t:
                add(np.where(self._name_norm.str.contains(pat(t), regex=True).values)[0], t_)
        if albums:
            col = self.meta["album"].astype(str).map(_norm_name)
            for al in albums:
                t = _norm_name(al)
                per.setdefault(al, 0)
                if t:
                    add(np.where(col.str.contains(pat(t), regex=True).values)[0], al)
        return hits[:limit], len(hits), per

    def play(self, query):
        """Resolve one named track and hand it back by itself.

        "Play Graceland" wants Graceland, not things that sound like it.
        """
        rows = self._rows_for_ids([query])
        if not rows:
            t = _norm_name(query)
            m = self._name_norm.str.contains(re.escape(t), regex=True)
            rows = list(np.where(m.values)[0])
            if not rows:
                toks = [w for w in t.split() if len(w) > 2]
                if toks:
                    m = self._name_norm.str.contains("|".join(re.escape(w) for w in toks))
                    rows = list(np.where(m.values)[0])
        if not rows:
            return None, 0
        j = int(rows[0])
        r = self.meta.iloc[j]
        return dict(track_id=str(r["id"]), name=str(r["name"]),
                    artist=str(r["artists"]), album=str(r.get("album", "")),
                    preview_url=(None if str(r.get("preview_url")) == "nan"
                                 else str(r.get("preview_url"))),
                    profile=self.profile(j), row=j), len(rows)

    # -- how unusual is a track? -------------------------------------------
    def _isolation(self, k=10, chunk=512):
        """Mean cosine to each track's k nearest neighbours. Lower = stranger.

        Deliberately k-NN rather than the single nearest neighbour, which is
        the obvious choice and the fragile one: a track with a live version,
        a remaster or a double entry has a neighbour at cos ~1.0 and can never
        rank as unusual however strange it is. Averaging over ten also stops a
        pair of mutually-weird tracks hiding each other — Romitelli's
        movements only surface as outliers under k-NN, because under nearest-
        neighbour they are each other's company.

        Chunked because the full 7,106 x 7,106 similarity matrix is 200MB; this
        holds 512 rows at a time. Computed once, lazily.
        """
        if getattr(self, "_iso", None) is not None:
            return self._iso
        iso = np.empty(self.n, dtype="float32")
        kk = min(k, self.n - 1)
        for a in range(0, self.n, chunk):
            b = min(a + chunk, self.n)
            S = self.X[a:b] @ self.X.T
            S[np.arange(b - a), np.arange(a, b)] = -np.inf   # not its own neighbour
            # partition on -S puts the kk largest cosines first; the leading
            # minus restores them to actual cosines. Negating again here is
            # what inverted the whole ranking first time round.
            top = -np.partition(-S, kk, axis=1)[:, :kk]
            iso[a:b] = top.mean(1)
        self._iso = iso
        return iso

    def isolation_pct(self, row):
        """0-100, where 100 is the strangest thing in the library."""
        iso = self._isolation()
        return int(round(100.0 * float((iso > iso[row]).mean())))

    def outliers(self, limit=8, typical=False):
        """The strangest tracks in the library — or, with typical=True, the most
        representative. Answers "surprise me with the most unusual track", which
        previously had no tool and had to be improvised with a search wildcard.
        """
        iso = self._isolation()
        order = np.argsort(iso if not typical else -iso)[:max(1, min(limit, 50))]
        return dict(measure="mean cosine to the 10 nearest neighbours",
                    ranking="most unusual first" if not typical
                            else "most representative first",
                    tracks=[dict(track_id=str(self.meta.id[int(j)]),
                                 name=str(self.meta.name[int(j)]),
                                 artist=str(self.meta.artists[int(j)]),
                                 isolation=self.isolation_pct(int(j)),
                                 nearest_cos=round(float(iso[int(j)]), 3))
                            for j in order])

    # -- does a set of tracks split into distinct sounds? -------------------
    def cluster(self, artist=None, track_ids=None, max_k=5, session=None):
        """Find whether a set of tracks is one sound or several, and name the
        difference.

        This is what lets "more like Radiohead" stop being a blurry average.
        Averaging an artist's catalogue into one centroid silently assumes they
        are coherent; many are not, and for those the mean is a point no track
        of theirs actually occupies. Clustering first lets the assistant ask
        which end is meant.

        With no artist and no ids, clusters the whole library and marks the
        groups this conversation has not drawn from — which is a far better
        answer to "something completely different" than distance alone, because
        distance from a jazz query is just a random other genre.

        Returns {n, splits, silhouette, groups:[{size, tracks, higher, lower}]}.
        """
        from sklearn.cluster import KMeans, MiniBatchKMeans
        from sklearn.metrics import silhouette_score

        if artist:
            rows = self._artist_rows(artist)
            scope = artist
        elif track_ids:
            rows = self._rows_for_ids(track_ids)
            scope = "%d tracks" % len(rows)
        else:
            rows = list(range(self.n))
            scope = "the whole library"
        rows = [int(r) for r in rows]
        if len(rows) < 6:
            return dict(scope=scope, n=len(rows), splits=False,
                        note="too few tracks to cluster meaningfully",
                        groups=[])

        whole = artist is None and not track_ids
        if whole and self._libclust is not None:
            k, sil, lab = self._libclust          # the library does not change
        else:
            V = self.X[rows]
            big = len(rows) > 2000
            best = (1, -1.0, None)
            for k in range(2, min(max_k, len(rows) - 1) + 1):
                km = (MiniBatchKMeans(n_clusters=k, n_init=3, random_state=0,
                                      batch_size=1024)
                      if big else KMeans(n_clusters=k, n_init=6, random_state=0))
                lab = km.fit_predict(V)
                if len(set(lab)) < 2:
                    continue
                # silhouette_score builds the full pairwise distance matrix, so
                # on the whole library it is O(n^2) — 50 million pairs, per k.
                # Sampling makes it a fraction of a second at no real cost to
                # the ranking, which is all it is used for.
                sc = float(silhouette_score(V, lab, metric="cosine",
                                            sample_size=min(1500, len(rows)),
                                            random_state=0))
                if sc > best[1]:
                    best = (k, sc, lab)
            k, sil, lab = best
            if whole:
                self._libclust = (k, sil, lab)
        # 0.10 is low by textbook standards and deliberately so: these are
        # dense audio embeddings, not well-separated blobs, and a real stylistic
        # split in one artist's work shows up as a modest but consistent score.
        splits = lab is not None and sil >= 0.10
        if not splits:
            return dict(scope=scope, n=len(rows), splits=False,
                        silhouette=round(sil, 3) if lab is not None else None,
                        note="one coherent sound; no useful split", groups=[])

        seen_rows = set(session.seen) if session is not None else set()
        groups = []
        for c in sorted(set(lab)):
            idx = [rows[i] for i in range(len(rows)) if lab[i] == c]
            others = [rows[i] for i in range(len(rows)) if lab[i] != c]
            cen = unit(self.X[idx].mean(0))
            reps = [idx[i] for i in np.argsort(-(self.X[idx] @ cen))[:3]]
            # name the difference using the validated axes, so the assistant can
            # say *how* the groups differ instead of just that they do
            diffs = []
            for a, d in self.axes.items():
                mine = float((self.X[idx] @ d).mean())
                rest = float((self.X[others] @ d).mean()) if others else 0.0
                sd = float((self.X[rows] @ d).std()) or 1.0
                diffs.append((a, (mine - rest) / sd))
            diffs.sort(key=lambda t: -abs(t[1]))
            top = [t for t in diffs if abs(t[1]) >= 0.35][:3]
            groups.append(dict(
                size=len(idx),
                tracks=[dict(track_id=str(self.meta.id[j]),
                             name=str(self.meta.name[j]),
                             artist=str(self.meta.artists[j])) for j in reps],
                higher=[a for a, v in top if v > 0],
                lower=[a for a, v in top if v < 0],
                visited=bool(set(idx) & seen_rows)))
        return dict(scope=scope, n=len(rows), splits=True,
                    silhouette=round(sil, 3), groups=groups)

    def present(self, session, track_ids):
        """Set exactly what the user is shown, in this order. No wildcard.

        Searching and presenting used to be the same act: whatever the last
        tool returned was rendered, so the assistant could tell you a result
        was wrong and still have it sitting on the shelf underneath — which is
        precisely what happened with "happy 80s dance tracks". It judged the
        acoustic matches correctly, named six genuinely present 80s tracks
        instead, and the UI went on displaying the four it had just rejected.

        An empty list presents nothing, which is the right answer when a search
        found nothing worth showing.
        """
        out = []
        for tid in (track_ids or []):
            row = self._id_to_row.get(str(tid))
            if row is None:
                continue
            r = self.meta.iloc[row]
            out.append(dict(track_id=str(r["id"]), name=str(r["name"]),
                            artist=str(r["artists"]), album=str(r.get("album", "")),
                            preview_url=(None if str(r.get("preview_url")) == "nan"
                                         else str(r.get("preview_url"))),
                            score=1.0, profile=self.profile(row), row=row,
                            isolation=self.isolation_pct(row)))
            session.seen.add(row)
        self.last_results = out
        return out

    def overview(self, top=12):
        """Orientation: how big the library is and who is most of it."""
        counts = self.meta["artists"].astype(str).str.split(",").str[0].str.strip()
        vc = counts.value_counts()
        return dict(tracks=int(self.n), distinct_lead_artists=int(vc.size),
                    most_represented=[{"artist": a, "tracks": int(c)}
                                      for a, c in vc.head(top).items()],
                    axes=sorted(self.axes))

    # -- artist resolution, shared by both artist operations -----------------
    def _artist_rows(self, artist):
        """Every library row whose artist string contains this name."""
        target = _norm_name(artist)
        if not target:
            return []
        hit = self._artist_norm.str.contains(re.escape(target), regex=True)
        return list(np.where(hit.values)[0])

    def _resolve_artist(self, artist, n_seed=2):
        """Turn an artist name into seed vectors, however we can get them.

        Metadata first: if they are in the library, seed from the tracks closest
        to that artist's own centroid — the most *typical* of them, which beats
        an arbitrary handful. Otherwise fall through to Apple Music's top songs
        and embed those previews live.

        `exclude` is every row by that artist, not just the seeds. Naming an
        artist is a request for things *like* them, so their own tracks are the
        one answer that is never interesting — returning them looks like the
        system has understood nothing.

        Returns (seeds (m,512) or None, info, exclude set).
        """
        rows = self._artist_rows(artist)
        if rows:
            V = self.X[rows]
            c = unit(V.mean(0))
            pick = [rows[i] for i in np.argsort(-(V @ c))[:n_seed]]
            return (self.X[pick],
                    dict(source="library", resolved=artist, n_in_library=len(rows),
                         seeds=[dict(track_id=str(self.meta.id[j]),
                                     name=str(self.meta.name[j]),
                                     artist=str(self.meta.artists[j])) for j in pick]),
                    set(int(j) for j in rows))

        songs = apple_top_songs(artist, limit=n_seed)
        if not songs:
            return None, dict(source="none", resolved=artist, n_in_library=0, seeds=[],
                              error="not in the library and no Apple Music match"), set()
        import requests
        vecs, got = [], []
        for s in songs:
            try:
                r = requests.get(s["preview_url"], timeout=25)
                r.raise_for_status()
                vecs.append(self.enc.embed_audio_bytes(r.content))
                got.append(s)
            except Exception:
                continue
        if not vecs:
            return None, dict(source="none", resolved=artist, n_in_library=0, seeds=[],
                              error="previews could not be embedded"), set()
        # An Apple-sourced artist may still appear in the library as a guest or
        # under a different primary credit; exclude those too.
        return (np.stack(vecs),
                dict(source="apple_music", resolved=got[0]["artist"], n_in_library=0,
                     seeds=[dict(track_id=None, name=s["name"], artist=s["artist"])
                            for s in got]),
                set(int(j) for j in self._artist_rows(got[0]["artist"])))

    # -- operation 2: "like this artist" -------------------------------------
    def search_by_artist(self, session, artist, k=5, n_seed=3):
        """Search from one artist, replacing the current query."""
        seeds, info, exclude = self._resolve_artist(artist, n_seed=n_seed)
        if seeds is None:
            return [], info
        q = unit(seeds.mean(0))
        for j in exclude:
            session.seen.add(j)
        res = self._commit(session, q, "artist: %s" % artist, k, exclude=exclude,
                           kind="seed3", op="artist")
        return res, info

    # -- operation 2b: artists as a weight on the running query --------------
    def add_artists(self, session, artists, k=5, weight="moderate", n_seed=2):
        """Blend one or more artist references *into* the conversation so far.

        This is the difference between "show me Bonobo" and "what we were
        talking about, but pulled toward Bonobo and Four Tet". Each artist
        contributes the mean of their own seed vectors, the artists are then
        averaged so two references get equal say regardless of how many seeds
        each yielded, and the result enters the query as one more Rocchio
        document:

            q <- ALPHA*q + w * mean_over_artists( mean(seeds_for_that_artist) )

        If there is no query yet this behaves like a plain multi-artist search.
        Every named artist's own tracks are excluded from the results.
        """
        if isinstance(artists, str):
            artists = [artists]
        names = [a for a in (artists or []) if str(a).strip()]
        if not names:
            raise ValueError("no artists given")

        had_q = session.q is not None
        per, infos, exclude = [], [], set()
        for a in names:
            seeds, info, ex = self._resolve_artist(a, n_seed=n_seed)
            infos.append(info)
            exclude |= ex
            if seeds is not None:
                per.append(unit(seeds.mean(0)))
        if not per:
            return [], dict(artists=infos, applied=False,
                            error="none of those artists could be resolved")

        pull = unit(np.stack(per).mean(0))
        w = ARTIST_W.get(str(weight).lower(), ARTIST_W["moderate"])
        q = unit(ALPHA * session.q + w * pull) if had_q else pull
        for j in exclude:
            session.seen.add(j)
        label = "artists: " + ", ".join(names)
        res = self._commit(session, q, label, k, exclude=exclude,
                           kind=(None if had_q else "seed3"), op="add_artists")
        return res, dict(artists=infos, applied=True, weight=str(weight).lower(),
                         w=w, resolved=[i.get("resolved") for i in infos],
                         blended_into_existing_query=had_q)

    # -- operation 3: "more like this track" ---------------------------------
    def search_by_track(self, session, query, k=5):
        """Resolve a track by name (and optional artist) then search audio-to-audio."""
        rows = self._rows_for_ids([query])
        if not rows:
            t = _norm_name(query)
            m = self._name_norm.str.contains(re.escape(t), regex=True)
            rows = list(np.where(m.values)[0])
            if not rows:
                toks = [w for w in t.split() if len(w) > 2]
                if toks:
                    m = self._name_norm.str.contains("|".join(re.escape(w) for w in toks))
                    rows = list(np.where(m.values)[0])
        if not rows:
            return [], dict(resolved=None, error="no track in the library matches %r" % query)
        j = int(rows[0])
        q = self.X[j]
        session.seen.add(j)
        res = self._commit(session, q, "track: %s" % str(self.meta.name[j])[:60], k,
                           exclude={j}, op="track")
        return res, dict(resolved=dict(track_id=str(self.meta.id[j]),
                                       name=str(self.meta.name[j]),
                                       artist=str(self.meta.artists[j])),
                         n_candidates=len(rows))

    # -- operation 4: refine on feedback -------------------------------------
    def refine(self, session, adjust=None, keep=None, reject=None,
               descriptions=None, k=5):
        """One Rocchio step, generalised from documents to documents + directions.

            q <- a*q + b*mean(kept) - c*mean(rejected) + d*mean(text) + sum(lam*sign*axis)

        The axis term is what lets a critique like "more groovy" act on the
        query at all: it is a direction, not a document, and there is nothing in
        the library to point at. Its combined magnitude is capped at MAX_DRIFT.

        Returns (results, applied) — `applied` names every component that
        actually took effect, including axes that were requested but are not in
        the validated set, so the assistant can admit it rather than pretending.
        """
        if session.q is None:
            if not descriptions:
                raise ValueError("nothing to refine — search first")
            return self.search_by_description(session, descriptions, k=k), {}

        q = ALPHA * session.q
        applied = dict(axes=[], unknown_axes=[], kept=0, rejected=0, descriptions=0)

        # Feedback is logged before it is applied, so a crash mid-update still
        # keeps the judgement — it is the scarce thing here, not the vector.
        if session.id:
            try:
                import crate_store
                for t in (keep or []):
                    crate_store.log_feedback(session.id, t, 1.0, "keep")
                for t in (reject or []):
                    crate_store.log_feedback(session.id, t, 0.0, "reject")
            except Exception:
                pass

        keep_rows = self._rows_for_ids(keep)
        if keep_rows:
            session.liked.extend(keep_rows)
            q = q + BETA * unit(self.X[session.liked].mean(0))
            applied["kept"] = len(keep_rows)

        rej_rows = self._rows_for_ids(reject)
        if rej_rows:
            session.disliked.extend(rej_rows)
            q = q - GAMMA * unit(self.X[session.disliked].mean(0))
            applied["rejected"] = len(rej_rows)

        if descriptions:
            if isinstance(descriptions, str):
                descriptions = [descriptions]
            descriptions = [d for d in descriptions if str(d).strip()]
            if descriptions:
                q = q + DELTA * unit(self.enc.embed_text(descriptions).mean(0))
                applied["descriptions"] = len(descriptions)

        push = np.zeros_like(q)
        for item in (adjust or []):
            name, sign, mag = _parse_adjust(item)
            if name not in self.axes:
                applied["unknown_axes"].append(name)
                continue
            lam = MAGNITUDE.get(str(mag).lower(), MAGNITUDE["moderate"])
            s = -1.0 if str(sign).strip() in ("-", "less", "down", "neg") else 1.0
            push = push + s * lam * self.axes[name]
            applied["axes"].append(dict(axis=name, sign=("+" if s > 0 else "-"),
                                        magnitude=str(mag).lower(), lam=lam))
        nrm = float(np.linalg.norm(push))
        if nrm > MAX_DRIFT:
            push = push * (MAX_DRIFT / nrm)
            applied["drift_capped"] = True
        q = q + push

        label = "refine: " + (", ".join("%s%s" % (a["sign"], a["axis"])
                                        for a in applied["axes"]) or "feedback")
        res = self._commit(session, q, label, k, exclude=set(session.disliked),
                           kind=None, op="refine")
        return res, applied


def _parse_adjust(item):
    """Accept ("groove","+","strong") or {"axis":…,"sign":…,"magnitude":…}."""
    if isinstance(item, dict):
        return (item.get("axis"), item.get("sign", "+"),
                item.get("magnitude", "moderate"))
    item = list(item)
    while len(item) < 3:
        item.append("moderate" if len(item) == 2 else "+")
    return item[0], item[1], item[2]
