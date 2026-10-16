"""
Fill in the metadata the library is missing, keyed on ISRC.

The chat could never answer "play some 80s dance tracks" because there is no
year in the library — only id, name, artists, album, duration, preview_url and
isrc. CLAP cannot hear a decade either: asked for 1980s dance it returned 2020s
techno, because it heard "upbeat, dance, synths" and matched exactly that.

So this is a data problem, not a model problem. Every track has an ISRC, and
Apple's catalogue will resolve one to a release date and genre tags. 7,046
unique ISRCs, batched 25 to a request, is ~282 calls.

    python enrich_metadata.py            # resumable; re-run after a failure
    python enrich_metadata.py --report    # what came back, and how good it is

Writes data/apple_cache.json (raw, keyed by ISRC, so a re-run costs nothing)
and data/library_enriched.csv (isrc, release_date, year, genres, …).

IMPORTANT CAVEAT, and it decides how `year` may be used. An ISRC identifies a
*recording*, and Apple returns the release date of the version it matched. For
popular music that is usually what you want: Shoot to Thrill comes back
1980-07-25. For classical it is the date the performance was released, not when
the music was written — Kabalevsky's 24 Preludes resolve to 2000-01-01, and
there are 1,000+ classical tracks here. So:

    year  = when THIS RECORDING came out.    Reliable, and what "80s dance"
                                             means for popular music.
    NOT   = when the music was composed.     Never use it for repertoire era.

Genre tags partly cover the gap — classical recordings do come back tagged
"Classical" — but they are commercial categories, not musicology, and they are
sometimes very coarse (Graceland is filed under "Worldwide").
"""

import json
import os
import sys
import time

BATCH = 25                      # Apple accepts filter[isrc] as a comma list
WORKERS = 6
STOREFRONT = os.environ.get("APPLE_STOREFRONT", "gb")
_HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(_HERE, "data", "library_metadata.csv")
CACHE = os.path.join(_HERE, "data", "apple_cache.json")
OUT = os.path.join(_HERE, "data", "library_enriched.csv")


def _chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def fetch(verbose=True):
    import pandas as pd
    import requests
    from concurrent.futures import ThreadPoolExecutor
    import apple_token

    meta = pd.read_csv(META)
    want = sorted({str(x).strip() for x in meta["isrc"].dropna()
                   if str(x).strip() and str(x).strip().lower() != "nan"})

    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            cache = json.load(f)
    todo = [i for i in want if i not in cache]
    if verbose:
        print("%d unique ISRCs | %d cached | %d to fetch (%d calls)"
              % (len(want), len(want) - len(todo), len(todo),
                 (len(todo) + BATCH - 1) // BATCH))
    if not todo:
        return cache

    token = apple_token.get_developer_token()
    url = "https://api.music.apple.com/v1/catalog/%s/songs" % STOREFRONT
    session = requests.Session()
    session.headers["Authorization"] = "Bearer %s" % token
    done = {"n": 0}

    def one(batch):
        for attempt in range(4):
            try:
                r = session.get(url, params={"filter[isrc]": ",".join(batch)},
                                timeout=40)
                if r.status_code == 429:
                    time.sleep(3 * (attempt + 1))
                    continue
                r.raise_for_status()
                out = {}
                for song in r.json().get("data", []):
                    a = song.get("attributes") or {}
                    isrc = a.get("isrc")
                    if not isrc:
                        continue
                    # Several releases can share an ISRC (reissues,
                    # compilations). Keep the earliest date — it is the closest
                    # thing to "when this recording came out".
                    prev = out.get(isrc)
                    if prev and (prev.get("releaseDate") or "9999") <= (a.get("releaseDate") or "9999"):
                        continue
                    out[isrc] = {"releaseDate": a.get("releaseDate"),
                                 "genreNames": a.get("genreNames") or [],
                                 "albumName": a.get("albumName"),
                                 "composerName": a.get("composerName"),
                                 "name": a.get("name"),
                                 "artistName": a.get("artistName")}
                # record misses so a re-run does not retry them for ever
                for i in batch:
                    out.setdefault(i, None)
                return out
            except Exception as e:
                if attempt == 3:
                    sys.stderr.write("batch failed: %s\n" % str(e)[:120])
                    return {}
                time.sleep(2 * (attempt + 1))
        return {}

    batches = list(_chunks(todo, BATCH))
    with ThreadPoolExecutor(WORKERS) as ex:
        for got in ex.map(one, batches):
            cache.update(got)
            done["n"] += 1
            if verbose and done["n"] % 25 == 0:
                print("  %d/%d calls, %d resolved"
                      % (done["n"], len(batches),
                         sum(1 for v in cache.values() if v)))
    with open(CACHE, "w") as f:
        json.dump(cache, f)
    if verbose:
        hit = sum(1 for i in want if cache.get(i))
        print("resolved %d/%d ISRCs (%.1f%%)" % (hit, len(want), 100.0 * hit / len(want)))
    return cache


def build(cache=None, verbose=True):
    """Join the cache back onto the library and write library_enriched.csv."""
    import pandas as pd
    if cache is None:
        with open(CACHE) as f:
            cache = json.load(f)
    meta = pd.read_csv(META)
    rows = []
    for _, r in meta.iterrows():
        isrc = str(r.get("isrc") or "").strip()
        a = cache.get(isrc) or {}
        date = a.get("releaseDate") or ""
        year = ""
        if len(date) >= 4 and date[:4].isdigit():
            year = int(date[:4])
        rows.append({"track_id": str(r["id"]), "isrc": isrc,
                     "release_date": date, "year": year,
                     "genres": "|".join(a.get("genreNames") or []),
                     "apple_album": a.get("albumName") or "",
                     "composer": a.get("composerName") or ""})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    if verbose:
        have = (df.year != "").sum()
        print("wrote %s: %d rows, %d with a year (%.1f%%)"
              % (OUT, len(df), have, 100.0 * have / len(df)))
    return df


def report():
    import pandas as pd
    df = pd.read_csv(OUT)
    yr = pd.to_numeric(df.year, errors="coerce").dropna()
    print("tracks            %d" % len(df))
    print("with a year       %d (%.1f%%)" % (len(yr), 100.0 * len(yr) / len(df)))
    print("with genre tags   %d" % df.genres.fillna("").str.len().gt(0).sum())
    print("\nby decade:")
    for dec, n in yr.floordiv(10).mul(10).astype(int).value_counts().sort_index().items():
        print("  %ds  %5d  %s" % (dec, n, "#" * int(60.0 * n / len(yr))))
    print("\ncommonest genre tags:")
    tags = df.genres.fillna("").str.split("|").explode()
    tags = tags[(tags != "") & (tags != "Music")]
    for t, n in tags.value_counts().head(14).items():
        print("  %-26s %5d" % (t, n))
    print("\nReminder: `year` is when this RECORDING was released, not when the")
    print("music was written — classical tracks here date from their performance.")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        build(fetch())
        print()
        report()
