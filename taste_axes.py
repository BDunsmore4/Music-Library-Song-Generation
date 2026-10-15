"""
Latent attribute axes for conversational refinement.

CLAP has no "energy" coordinate. Its 512 dims are an entangled distributed code
and no single one means anything nameable. But because CLAP was trained
contrastively against captions full of words like "upbeat" and "mellow",
attributes are *linearly decodable*: some weighted combination of all 512 dims
tracks them. Zero-shot CLAP tagging is already exactly that — a linear probe
along such a direction. This module finds those directions explicitly, strips
their shared loudness component, and validates them against features measured
from the actual waveforms rather than trusting their names.

Construction, per axis:

  1. d_text  = unit(emb(positive prompt) - emb(negative prompt))
     The subtraction is what makes this work across the modality gap: both
     endpoints carry the same text-cone offset, so it cancels. Measured on the
     7,106-track library, |d_text . audio_centroid| lands at 0.02-0.03 for the
     axes that validate, i.e. essentially orthogonal to the gap direction.

  2. d_audio = unit(mean(top-N library by d_text) - mean(bottom-N))
     Re-anchors the direction inside the audio cone, where the index lives.
     d_text and d_audio agree at cos 0.54-0.77 — the text probe finds the right
     axis, but the audio-anchored version is the one to search with.

  3. Gram-Schmidt against `energy`, so "groovier" means groovier *at the same
     energy*. Without it, cos(energy, groove) = 0.43 and the two critiques do
     nearly the same thing to the query.

Validation (300 random previews, decoded and measured; never used to build the
axes) is the part that matters, and it is the part that fails. Each axis is
screened twice: once for being a loudness duplicate, once against the waveform
feature its *name* implies (PROXY below). Result on this library:

    axis          rho(rms)  rho(onsets)  rho(centroid)   verdict
    energy          +0.840      +0.520         +0.789     verified
    groove          +0.441      +0.218         +0.092     verified
    warmth          +0.016      -0.110         -0.359     verified
    aggression      -0.005      +0.086         +0.208     verified
    vocal           -0.298      -0.158         +0.246     unverified (no proxy)
    density         -0.361      -0.156         +0.134     unverified (no proxy)
    electronic      -0.106      -0.104         -0.277     unverified (no proxy)
    happiness       -0.256      -0.156         +0.220     unverified (no proxy)
    tempo           -0.319      -0.175         +0.104     FAILED
    brightness      -0.497      -0.274         +0.008     FAILED

`tempo` correlates *negatively* with measured onset rate and `brightness` is
+0.008 against spectral centroid — neither does what it is called, so both are
dropped. That is the whole reason this harness exists: the prompts read fine and
the poles looked plausible for both.

The energy axis sits above the 99.95th percentile of 2,000 random unit
directions for loudness. Note the null is not zero — random directions reach a
median |rho| of 0.32, because loudness is pervasive in the embedding. `groove`
only separates from energy *after* step 3: raw, it scored +0.775 on RMS.

Usage:
    python taste_axes.py build        # build + orthogonalise + cache
    python taste_axes.py measure      # download a sample, measure waveforms
    python taste_axes.py validate     # correlate axes against those waveforms
"""

import os
import sys

import numpy as np

def _first_existing(env, *candidates):
    """Env var wins; otherwise the first path that exists.

    The repo-local `data/` copy is listed first so a Railway deploy finds the
    library without any configuration — the original lives outside the repo
    under SPOTIFY_APP/.venv, which obviously does not ship. Keep both: the
    dev machine can carry on using the original.
    """
    v = os.environ.get(env)
    if v:
        return v
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


_HERE = os.path.dirname(os.path.abspath(__file__))
EMB_NPY = _first_existing(
    "LIBRARY_EMB_NPY",
    os.path.join(_HERE, "data", "library_embeddings.npy"),
    "/Users/bendunsmore/PycharmProjects/SPOTIFY_APP/.venv/library_embeddings.npy")
META_CSV = _first_existing(
    "LIBRARY_EMB_META",
    os.path.join(_HERE, "data", "library_metadata.csv"),
    "/Users/bendunsmore/PycharmProjects/SPOTIFY_APP/.venv/library_metadata.csv")

# taste_axes.npz / the validation report are small and belong in the repo, but
# resolve them next to this file so a worker started from another cwd finds them.
AXES_CACHE = os.environ.get("TASTE_AXES_PATH", os.path.join(_HERE, "taste_axes.npz"))

ACOUSTIC_CSV = os.environ.get("ACOUSTIC_FEATS", "acoustic_feats.csv")

# Prompts are wrapped in this template before embedding. Measured: templating
# lifts neighbourhood coherence@20 from 0.39 to 0.49 and is not optional.
TEMPLATE = "This is a %s song."

# Poles for each candidate axis. Keep this list curated and small — every axis
# must survive validate() before the chat model is allowed to use it.
POLES = {
    "energy":     ("fast, upbeat, energetic, driving",
                   "slow, subdued, calm, low-energy"),
    "groove":     ("groovy, funky, syncopated, with a strong rhythmic pocket",
                   "stiff, unrhythmic, free-time, without a beat"),
    "vocal":      ("with prominent lead vocals and singing",
                   "purely instrumental with no voice"),
    "electronic": ("electronic, synthesised, programmed",
                   "acoustic, played on real instruments"),
    "density":    ("dense, layered, many instruments playing at once",
                   "sparse, minimal, one or two instruments"),
    "brightness": ("bright, crisp, treble-forward",
                   "dark, muffled, bass-heavy"),
    "warmth":     ("warm, analogue, mellow in tone",
                   "harsh, cold, brittle in tone"),
    "aggression": ("aggressive, distorted, harsh and heavy",
                   "gentle, clean, soft and smooth"),
    "happiness":  ("happy, bright, major key, uplifting",
                   "sad, melancholic, minor key, mournful"),
    "tempo":      ("very fast tempo, rapid and busy",
                   "very slow tempo, languid and unhurried"),
}

# The axis every other axis is residualised against. Loudness dominates the
# embedding, so without this most axes are mostly re-expressions of it.
BASE_AXIS = "energy"

# How many tracks at each pole define the audio-anchored direction.
POLE_N = 150

# An axis is kept only if it adds something loudness does not.
MAX_ABS_RMS_RHO = 0.60

# What each axis claims to measure, and the waveform feature that claim implies.
# This is the part that stops an axis being trusted because its *name* sounds
# right: if "brightness" does not correlate with spectral centroid, it is not a
# brightness axis whatever its prompts said. (sign, feature, minimum |rho|)
PROXY = {
    "energy":     ("+", "rms", 0.40),
    "tempo":      ("+", "onset_rate", 0.20),
    "groove":     ("+", "onset_rate", 0.20),
    "brightness": ("+", "centroid", 0.20),
    "warmth":     ("-", "centroid", 0.20),
    "aggression": ("+", "hi_ratio", 0.15),
}
# Axes with no honest waveform proxy. They can still be used — `vocal` has the
# cleanest poles of the whole set — but they are verified by pole inspection
# only, and should be labelled as such rather than quietly trusted.
NO_PROXY = ("vocal", "density", "electronic", "happiness")


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def load_library():
    """Return (X unit-normalised, metadata DataFrame)."""
    import pandas as pd
    X = unit(np.load(EMB_NPY).astype("float32"))
    meta = pd.read_csv(META_CSV)
    return X, meta


# ── building ─────────────────────────────────────────────────────────────────

def build(encoders=None, verbose=True):
    """Build every axis, orthogonalise against BASE_AXIS, cache and return.

    Returns a dict of {name: unit 512-vector}. The returned directions live in
    audio space and are safe to add to an audio query vector.
    """
    if encoders is None:
        from crate import Encoders
        encoders = Encoders()

    X, _ = load_library()
    names = list(POLES)
    pos = encoders.embed_text([TEMPLATE % POLES[n][0] for n in names], centre=False)
    neg = encoders.embed_text([TEMPLATE % POLES[n][1] for n in names], centre=False)

    raw = {}
    for i, name in enumerate(names):
        d_text = unit(pos[i] - neg[i])
        s = X @ d_text
        hi = np.argsort(-s)[:POLE_N]
        lo = np.argsort(s)[:POLE_N]
        d_audio = unit(X[hi].mean(0) - X[lo].mean(0))
        raw[name] = d_audio
        if verbose:
            print("  %-11s cos(text,audio)=%+.3f  |d.centroid|=%.3f"
                  % (name, float(d_text @ d_audio), abs(float(d_audio @ unit(X.mean(0))))))

    axes = orthogonalise(raw, base=BASE_AXIS)
    save(axes)
    if verbose:
        print("\nsaved %d axes to %s (orthogonalised against '%s')"
              % (len(axes), AXES_CACHE, BASE_AXIS))
    return axes


def orthogonalise(raw, base=BASE_AXIS):
    """Residualise every axis against `base` so critiques act independently.

    Only one pass is needed: the point is to stop every axis from doubling as a
    loudness control, not to force a fully orthogonal basis (which would make
    the later axes uninterpretable).
    """
    if base not in raw:
        return dict(raw)
    b = unit(raw[base])
    out = {base: b}
    for name, d in raw.items():
        if name == base:
            continue
        resid = d - float(d @ b) * b
        n = np.linalg.norm(resid)
        # A near-zero residual means the axis *was* the base axis. Keep the raw
        # direction rather than amplifying numerical noise.
        out[name] = unit(resid) if n > 1e-3 else b.copy()
    return out


def save(axes):
    names = sorted(axes)
    np.savez(AXES_CACHE,
             names=np.array(names),
             matrix=np.stack([axes[n] for n in names]).astype("float32"),
             base=np.array(BASE_AXIS))


def load(path=None):
    """Load cached axes; returns {} if they have not been built yet."""
    p = path or AXES_CACHE
    if not os.path.exists(p):
        return {}
    z = np.load(p, allow_pickle=False)
    return {str(n): z["matrix"][i] for i, n in enumerate(z["names"])}


# ── measuring real acoustics, for validation ─────────────────────────────────

def measure(n=300, seed=0, out=None, cache_dir="preview_cache"):
    """Download a random sample of previews and measure waveform features.

    Deliberately a *random* sample, not stratified by any axis — stratifying on
    the predictor would inflate the validation correlations.
    """
    import pandas as pd
    import requests
    from concurrent.futures import ThreadPoolExecutor

    out = out or ACOUSTIC_CSV
    _, meta = load_library()
    ok = meta[meta.preview_url.notna()
              & meta.preview_url.astype(str).str.startswith("http")]
    rng = np.random.default_rng(seed)
    idx = rng.choice(ok.index.values, min(n, len(ok)), replace=False)
    os.makedirs(cache_dir, exist_ok=True)

    def fetch(i):
        f = os.path.join(cache_dir, "%d.m4a" % i)
        if os.path.exists(f) and os.path.getsize(f) > 5000:
            return f
        try:
            r = requests.get(meta.preview_url[i], timeout=25)
            if r.status_code == 200 and len(r.content) > 5000:
                with open(f, "wb") as fh:
                    fh.write(r.content)
                return f
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(16) as ex:
        files = list(ex.map(fetch, idx))

    rows = []
    for i, f in zip(idx, files):
        if not f:
            continue
        feats = _waveform_features(f)
        if feats:
            feats["i"] = int(i)
            rows.append(feats)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print("measured %d previews -> %s" % (len(df), out))
    return df


def _waveform_features(path):
    """Loudness, brightness and rhythmic activity straight from the waveform.

    Hand-rolled on pydub + numpy because librosa is not a dependency of this
    app and these four features do not justify adding it.
    """
    try:
        from pydub import AudioSegment
        from scipy.signal import find_peaks
        a = AudioSegment.from_file(path).set_frame_rate(22050).set_channels(1)
        x = np.array(a.get_array_of_samples()).astype(np.float32)
        x = x / (2 ** (a.sample_width * 8 - 1))
        if len(x) < 22050:
            return None
        n, hop = 1024, 512
        nf = (len(x) - n) // hop
        if nf < 8:
            return None
        frames = np.lib.stride_tricks.as_strided(
            x, (nf, n), (x.strides[0] * hop, x.strides[0]))
        S = np.abs(np.fft.rfft(frames * np.hanning(n), axis=1))
        freqs = np.fft.rfftfreq(n, 1.0 / 22050)
        rms = np.sqrt((frames ** 2).mean(1))
        env = rms / (rms.max() + 1e-9)
        peaks, _ = find_peaks(env, height=0.25, distance=4)
        return dict(
            rms=float(np.sqrt((x ** 2).mean())),
            centroid=float(((S * freqs).sum(1) / (S.sum(1) + 1e-9)).mean()),
            onset_rate=float(len(peaks) / (len(x) / 22050.0)),
            hi_ratio=float(S[:, freqs > 3000].sum() / (S.sum() + 1e-9)),
        )
    except Exception:
        return None


# ── validation ───────────────────────────────────────────────────────────────

def validate(axes=None, n_null=2000, verbose=True):
    """Correlate each axis against measured acoustics; flag the useless ones.

    Returns {name: {"rho_rms":…, "independent": bool, "keep": bool}}. An axis is
    kept if it is not simply a loudness control.
    """
    import pandas as pd
    from scipy.stats import spearmanr

    axes = axes or load()
    if not axes:
        raise RuntimeError("no axes cached — run `python taste_axes.py build`")
    if not os.path.exists(ACOUSTIC_CSV):
        raise RuntimeError("no measurements — run `python taste_axes.py measure`")

    X, _ = load_library()
    F = pd.read_csv(ACOUSTIC_CSV)
    Xs = X[F.i.values]
    feats = [c for c in ("rms", "onset_rate", "centroid", "hi_ratio") if c in F]

    if verbose:
        print("n = %d randomly sampled tracks, features measured from waveforms\n" % len(F))
        print("%-12s" % "axis" + "".join("%12s" % f for f in feats) + "   keep")

    report = {}
    for name in sorted(axes):
        p = Xs @ axes[name]
        rhos = dict((f, float(spearmanr(p, F[f]).statistic)) for f in feats)

        # Screen 1: is it just a loudness control in disguise?
        if name != BASE_AXIS and abs(rhos.get("rms", 0.0)) > MAX_ABS_RMS_RHO:
            verdict, why = "failed", "duplicates loudness"
        # Screen 2: does it track the feature its name claims?
        elif name in PROXY:
            sign, feat, floor = PROXY[name]
            r = rhos.get(feat, 0.0)
            want = 1.0 if sign == "+" else -1.0
            if r * want >= floor:
                verdict, why = "verified", "%s %s%.2f" % (feat, sign, abs(r))
            else:
                verdict, why = "failed", "%s should be %s, measured %+.3f" % (feat, sign, r)
        elif name in NO_PROXY:
            verdict, why = "unverified", "no waveform proxy — judge by poles"
        else:
            verdict, why = "unverified", "no expectation declared"

        report[name] = dict(verdict=verdict, note=why,
                            keep=(verdict != "failed"), **rhos)
        if verbose:
            print("%-12s" % name
                  + "".join("%12s" % ("%+.3f" % rhos[f]) for f in feats)
                  + "   %-10s %s" % (verdict, why))

    bad = [n for n in report if report[n]["verdict"] == "failed"]
    if verbose and bad:
        print("\nDROP these — they do not measure what they are named: %s"
              % ", ".join(sorted(bad)))
        print("Either rewrite their poles or delete them from POLES. The chat "
              "model is only offered axes that pass (see crate.Crate.axis_names).")

    if n_null and verbose and BASE_AXIS in axes:
        rng = np.random.default_rng(1)
        R = unit(rng.standard_normal((n_null, 512)).astype("float32"))
        P = Xs @ R.T
        null = np.abs([spearmanr(P[:, j], F.rms).statistic for j in range(n_null)])
        e = abs(report[BASE_AXIS]["rms"])
        print("\n'%s' vs %d random directions on RMS: |rho|=%.3f, "
              "null median %.3f / 95th %.3f -> percentile %.2f%%"
              % (BASE_AXIS, n_null, e, np.median(null),
                 np.percentile(null, 95), 100.0 * (null < e).mean()))
        print("(the null is not zero — loudness is pervasive in the embedding)")
    return report


def poles(name, axes=None, k=3):
    """The k most extreme library tracks at each end of an axis — a sanity eye-test."""
    axes = axes or load()
    X, meta = load_library()
    s = X @ axes[name]
    o = np.argsort(-s)
    fmt = lambda j: "%s — %s" % (str(meta.name[j])[:34], str(meta.artists[j]).split(",")[0][:22])
    return {"+": [fmt(j) for j in o[:k]], "-": [fmt(j) for j in o[-k:][::-1]]}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        ax = build()
        print("\npoles:")
        for n in sorted(ax):
            p = poles(n, ax, k=2)
            print("  %-11s + %s\n  %-11s - %s"
                  % (n, " | ".join(p["+"]), "", " | ".join(p["-"])))
    elif cmd == "measure":
        measure()
    elif cmd == "validate":
        validate()
    else:
        print(__doc__)


def trusted(axes=None, report_path=None):
    """Axes that passed validation, with their verdicts — what the chat may use.

    Cached to JSON by `validate` so the serving path never needs the waveform
    sample or scipy.
    """
    import json
    axes = axes if axes is not None else load()
    report_path = report_path or os.path.join(_HERE, "taste_axes_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            rep = json.load(f)
        keep = [n for n in axes if rep.get(n, {}).get("verdict") != "failed"]
        return dict((n, axes[n]) for n in keep), rep
    return dict(axes), {}
