# Deployment & setup runbook

Architecture after the rearchitecture:

- **Frontend** (`Frontend/`, static on Cloudflare Pages): `index.html` fetches
  `data.json` (the map) and the R2 `generated.json` manifest on load. No API key
  in the browser. Generating a track POSTs a prompt to the backend.
- **Backend** (`app.py` on Railway): holds the ElevenLabs key, calls ElevenLabs,
  CLAP-embeds the audio, uploads the mp3 to Cloudflare R2, and appends to the
  `generated.json` manifest. Routes: `/`, `/embed`, `/embed_blob`, `/generate`,
  `/health`.
- **Storage** (Cloudflare R2): generated mp3s + `generated.json` manifest.

## 1. Cloudflare R2 (dashboard)

1. **Create a bucket**, e.g. `spotify-app-audio`.
2. **Enable public access** — turn on the managed `r2.dev` URL (or attach a
   custom domain). Note the public base, e.g. `https://pub-xxxx.r2.dev`.
3. **Create an R2 API token** (R2 → Manage API Tokens) with **Object Read &
   Write** on the bucket. Save the Access Key ID + Secret. Your Account ID is on
   the R2 overview page.
4. **Configure CORS** on the bucket so the Pages site can `fetch()` the manifest:
   - AllowedOrigins: your Pages URL(s) (e.g. `https://<project>.pages.dev`, any
     custom domain, and `http://localhost:*` for local testing)
   - AllowedMethods: `GET`, `HEAD`
   - AllowedHeaders: `*`
   - (mp3 `<audio>` playback isn't CORS-gated; only the manifest fetch is.)

## 2. Railway secrets (service → Variables)

- `ELEVENLABS_API_KEY` — **a freshly rotated key**. The old one leaked in the
  shipped HTML and git history; revoke it in the ElevenLabs dashboard.
- `R2_ACCOUNT_ID`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`
- `R2_BUCKET` (e.g. `spotify-app-audio`)
- `R2_PUBLIC_BASE_URL` (no trailing slash, e.g. `https://pub-xxxx.r2.dev`)

Existing `UMAP_REDUCER_PATH`, `HTML_FILE`, `PORT` are unchanged.

## 3. Frontend config

In `Frontend/index.html`, set `R2_PUBLIC_BASE` (near the top of the `<script>`)
to your R2 public base URL — it currently says `https://REPLACE-ME.r2.dev`.

## 4. Seed the 7 existing tracks (one-time, run locally)

With the same `R2_*` env vars exported locally:

```bash
python seed_r2.py
```

This uploads `track1..7.mp3` from
`/Users/bendunsmore/PycharmProjects/Spotify_app_for_cloudflare/generated`
and writes the initial `generated.json`. Override the source folder with
`SEED_AUDIO_DIR=...` if needed.

## 5. Deploy

- Push to GitHub → Railway rebuilds the backend (now installs `boto3`, and
  `nixpacks.toml` pins `ffmpeg`).
- Cloudflare Pages redeploys the `Frontend/` folder (now `index.html` + `data.json`).

## 6. Verify

```bash
SERVER=https://music-library-song-generation-production.up.railway.app

curl $SERVER/health                       # {"status":"ok"}
curl -X POST $SERVER/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a calm ambient piano piece"}'   # {x,y,label,audio_url} (~tens of s)
curl $R2_PUBLIC_BASE_URL/generated.json   # list now includes the new record
```

Then load the Pages site: the map renders from `data.json`, generated points
render from the manifest, generating works, and a **hard refresh keeps the new
point** (the core fix).

## Notes / limits

- `gunicorn --workers 1` blocks during a generation (tens of seconds); `/health`
  can stall meanwhile. Optionally add `--threads 2 --worker-class gthread` (the
  manifest write is already lock-guarded).
- R2 is public: all generated mp3s and every prompt in `generated.json` are
  world-readable. Keep prompts free of anything sensitive.

---

# Crate-digging chat (Railway)

Blueprint `chat_routes.bp`, registered in `app.py` beside `personal_routes`.

    POST /api/chat/session     -> {"session": id}
    POST /api/chat             {"session","message"} -> {"reply","tracks","trail"}
    POST /api/chat/feedback    {"session","track_id","rating"}   rating in [0,1]
    GET  /api/chat/config      -> axes, library size, "live": is the key set

`/api/chat/config` returns `live: false` when `ANTHROPIC_API_KEY` is unset, so
the frontend can hide the panel rather than offering a button that 503s.

## 1. Railway variables

| Variable | Why |
|---|---|
| `ANTHROPIC_API_KEY` | required; without it `/api/chat` returns 503 |
| `CHAT_SHARED_SECRET` | **set this.** `/api/chat` spends your key for whoever calls it. The frontend sends it as `X-Chat-Key`. |
| `CRATE_STORE_DIR` | path on a **persistent volume** — see §3 |
| `CRATE_TRAIL` | leave unset (defaults off in the web path) |
| `CRATE_MODEL` | optional, default `claude-sonnet-5` |
| `APPLE_TEAM_ID`, `APPLE_KEY_ID`, `APPLE_PRIVATE_KEY` | already set for MusicKit; `add_artists` reuses them to look up artists you do not own. Without them that path degrades to library-only. |

## 2. The library data ships in the repo

`data/library_embeddings.npy` (14 MB) and `data/library_metadata.csv` (2.3 MB).
`taste_axes.py` resolves `data/` first and falls back to the original path under
`SPOTIFY_APP/.venv/` for local work, so no configuration is needed either side.
Also committed, and all small: `taste_axes.npz`, `taste_axes_report.json`.

`acoustic_feats.csv` is only needed to re-run `python taste_axes.py validate`.
It can ship or not; serving never touches it.

## 3. Storage must be a volume — Railway's filesystem is ephemeral

Attach a Railway Volume and point `CRATE_STORE_DIR` at its mount path, e.g.
`/data/crate_store`. Without this the conversation log is wiped on every deploy
and restart, which silently destroys the only training data the system produces
— the whole point of `crate_store`. Volumes bind to a single service and do not
support multiple writers; that is fine here because the Procfile already runs
`--workers 1`.

## 4. Memory and boot

CLAP is loaded once by `embedder.py` at import, and `crate.Encoders` reuses that
same in-memory model rather than loading a second copy. The Crate adds ~16 MB
for the embedding matrix. Peak is dominated by torch + CLAP, which this service
already paid for before the chat existed.

The Crate is built lazily on the first chat request, so a deploy nobody chats to
never pays for it. The first request after a restart is slower than the rest: it
builds the reference distributions and embeds 36 canned prompts to estimate the
text centroid. Hit `/api/chat/config` once after deploy to warm it.

`gunicorn --timeout 300` already covers a cold tool loop.

## 5. Checks after deploy

```bash
curl -s $BASE/api/chat/config -H "X-Chat-Key: $CHAT_SHARED_SECRET"
# expect: tracks 7106, 8 axes, axes_dropped [brightness, tempo], live true

SID=$(curl -s -X POST $BASE/api/chat/session -H "X-Chat-Key: $CHAT_SHARED_SECRET" | jq -r .session)
curl -s -X POST $BASE/api/chat -H "X-Chat-Key: $CHAT_SHARED_SECRET" \
     -H 'Content-Type: application/json' \
     -d "{\"session\":\"$SID\",\"message\":\"something mellow for a rainy afternoon\"}" | jq .
```

`axes_dropped` being `[brightness, tempo]` confirms the validated axis set
loaded; an empty list means `taste_axes_report.json` did not ship and every
candidate axis is being offered, including the two that do not work.

## 6. Learning loop (run locally, against the volume's log)

```bash
railway run python crate_store.py stats      # turns, feedback, wildcard hit rate
railway run python crate_store.py fit        # fit + leave-one-out validate
```

`fit` refuses to install an adapter that does not improve held-out rank, and on
the current data it correctly refuses — see the negative result documented at
the top of `crate_store.py`.

## 7. Choosing the model (open weights or Claude)

The chat is not tied to a vendor. `llm.py` speaks two protocols, picked by
`LLM_BACKEND`:

- `openai` — any OpenAI-compatible `/chat/completions` endpoint **with tool
  support**. Covers Groq, Together, Fireworks, DeepInfra, OpenRouter, vLLM,
  llama.cpp's server and Ollama. Needs no SDK; it is plain `requests`.
- `anthropic` — the Claude Messages API. Needs `anthropic` from
  requirements.txt; drop that line if you are never using it.

Hosted open weights (cheap, no GPU of your own):

    LLM_BACKEND=openai
    LLM_BASE_URL=https://api.groq.com/openai/v1
    LLM_API_KEY=...
    LLM_MODEL=llama-3.3-70b-versatile

Fully local, nothing billed (dev machine, not Railway — a CPU dyno cannot serve
a 14B model at conversational speed):

    LLM_BACKEND=openai
    LLM_BASE_URL=http://localhost:11434/v1
    LLM_MODEL=qwen3:14b

Model names move around — check your provider's current list. The only hard
requirement is **tool calling**; without it the design collapses, because the
assistant's entire job is choosing tool calls.

`/api/chat/config` reports `backend`, `model` and `live`, so you can confirm
which one answered without reading logs.

### Before you switch, check the prompt

The system prompt is ~2,500 tokens and is resent on every turn, so it dominates
the bill far more than the model's output does. Enabling prompt caching on the
Anthropic path is a one-line change and cuts repeat turns by roughly 90% — do
that before concluding the model is the expensive part.

### What degrades on a smaller model

Open-weights models are weaker at obeying a long list of "never do X" rules,
and this prompt has several. Expect artist names leaking into `descriptions`
and negation in prose ("not too fast"), both of which quietly produce bad
retrieval rather than an error. Three defences are already in place:

- `force_tool` requires a tool call on the opening turn instead of hoping for
  one, so the model cannot burn the turn asking "what mood are you in?".
- If it still will not call a tool, the chat falls back to a deterministic
  `search_by_description` on the raw message, so the user always gets tracks.
- Malformed tool arguments are returned to the model as an error it can retry,
  rather than raising. Smaller models need this occasionally.

An axis the model invents is already reported back as `unknown_axes` rather
than silently ignored, so a hallucinated "danceability" shows up instead of
doing nothing.

### Prompt caching (on by default, Anthropic path)

The static prefix is ~2,500 tokens of system prompt plus ~1,200 of tool
schemas, and it is resent on every **API call** — not every user turn. The tool
loop makes two to four calls for a single thing the user says, so without
caching that prefix is billed two to four times per message.

Two `cache_control` breakpoints are set per call:

1. the last system block — and because the cached prefix runs
   `tools -> system -> messages`, this also captures the tool schemas;
2. the end of the conversation so far, so call N of a tool loop reuses call
   N-1's prefix rather than only the static header.

Two of the four available breakpoints, so there is room if you add more.

Verify it rather than assuming. Every `/api/chat` response carries a `usage`
object and the server logs a line per turn:

    [chat] 3 calls | 180 uncached in, 3700 cache write, 7400 cache read
           (87% of input served from cache) | 210 out

`cache_read` is the only proof caching is live. **If it stays 0, the
breakpoints are not matching and you are paying full price on every call** —
the most likely causes are a prefix below the model's minimum cacheable length,
or something non-deterministic in the system prompt (it is built from the axis
list, which is stable, so this should not happen).

Controls:

    LLM_CACHE=0            disable entirely, to compare bills
    LLM_CACHE_MESSAGES=0   keep the static header cached, drop breakpoint 2

Cache writes cost more than ordinary input and reads cost much less, so this
pays off as soon as a prefix is read once — immediate inside a tool loop. The
static header is where nearly all the saving is; breakpoint 2 is marginal and
switchable for that reason. Entries are the default 5-minute TTL, which
comfortably covers a conversation but not a gap between sessions, so the first
call of each new conversation writes rather than reads.

## 8. The chat page

`Frontend/crate.html` — a standalone page in the same visual language as the
map. Deliberately just a conversation and a track list: no map, no sidebar.
Each turn shows three matches plus the wildcard (thin amber rule, "further
out" tag), every row has a 30-second preview and thumbs up/down.

Two things to set at the top of the `<script>`:

    const API = "https://<your-service>.up.railway.app";
    const CHAT_KEY = "";   // see the warning below

The thumbs post straight to `/api/chat/feedback`, so a click is recorded even
if the user never says anything about it in prose — that is the cleanest
training signal the system gets, and the wildcard is where it is most
informative.

**`CHAT_KEY` in a static page is obfuscation, not security.** Anything shipped
to a browser is readable, so a shared secret only deters casual traffic. The
options, in order of how much work they are:

1. Set a **monthly spend cap on the Anthropic account**. Do this regardless —
   it is the only backstop that cannot be bypassed.
2. Proxy `/api/chat` through a Cloudflare Pages Function and inject the secret
   server-side, so it never reaches the browser. You are already on Cloudflare,
   so this is the proper fix and not much work.
3. Rate-limit per IP in Flask.

## 9. Testing the branch on Railway without touching main

You do not need a separate repo. In the **same Railway project**, add a second
service from the same GitHub repo and set its branch to `crate-chat`. It gets
its own URL and its own variables; the live service keeps running from `main`
untouched. Point `API` in `crate.html` at the staging URL to try it.

(Railway can also create a deployment per pull request if you enable PR
environments on the repo, which does the same thing automatically.)

Four things to get right on a staging service:

- **Give it its own Volume**, with its own `CRATE_STORE_DIR`. Sharing one means
  test conversations land in the same training log as real ones.
- **R2 is shared** if you copy the `R2_*` variables across. The chat endpoints
  never write to R2, so chat-only testing is safe — but `/generate` and
  `/embed_track` append to the live `generated.json` / `searched.json`
  manifests, so do not exercise those on staging with production credentials.
- It loads **its own copy of CLAP**, so it costs memory and money while it is
  running. Sleep or delete it when you are done.
- OAuth redirect URIs are per-URL, so Spotify/Apple login will not work on the
  staging URL until you add it to those apps. Irrelevant for the chat.

When you are happy, merge `crate-chat` into `main` and delete the staging
service — the live service picks the change up on its next deploy.

### If the key is an organisation-level one

Anthropic keys created at the organisation level are not bound to a workspace,
and the API rejects the request:

    400 invalid_request_error — This API key is not scoped to a workspace, so
    this request must include the anthropic-workspace-id header

Two ways out, and the first is better:

1. **Create a workspace-scoped key** in the Anthropic Console (open the
   workspace, then create the key inside it) and use that. Nothing else to set,
   and it keeps this project's spend attributable to one workspace.
2. Keep the org-level key and set `ANTHROPIC_WORKSPACE_ID` (or
   `LLM_WORKSPACE_ID`) to the workspace ID. The backend then sends the
   `anthropic-workspace-id` header on every call.

`/api/chat/config` reports `workspace_id_set`, so you can tell which path is
in effect without reading logs.

## 10. The stage (Frontend/stage.html)

Two acts. It opens as an ordinary warm-toned chat with nothing to suggest there
is anything else there. On the codeword the assistant says "Enjoy the show" and
from then on each prompt plays as an explainer inside the 3-D library, then
returns you to the chat with the tracks listed.

The stage is a **demonstration, not a place to live**, and its job is
educational: show what actually happened. The caption panel is the lesson; the
point field is the illustration.

| # | Step | What it shows |
|---|---|---|
| 1 | you typed | your words, unaltered |
| 2 | claude structured it | the literal `search_by_description(descriptions=[…])` call |
| 3 | what clap is given, verbatim | `This is a sparse solo piano, slow and quiet song.` — the template applied, then `2 vectors, averaged · L2-normalised · 512-d` |
| 4 | where that lands | a halo over the region, wide establishing shot, then a scripted pan in |
| 5 | nearest neighbours in 512-d | the matches ignite and label themselves with name and artist |

Step 3 is the point of the whole thing: it prints the exact string handed to
CLAP's text encoder, template included, rather than a paraphrase. The template
itself comes from the backend (`taste_axes.TEMPLATE`) so it cannot drift out of
date.

Step 5 says "in 512-d" deliberately, and the note under step 4 says the view is
a projection. The matches sometimes will *not* look adjacent on screen, and
that is the lesson rather than a glitch — the neighbours are chosen by cosine
distance in the full space, and the 3-D layout is a UMAP shadow of it.

The field is rendered with the original `space.html` parameters — `setHSL(hue,
0.55, 0.6)`, point size 3.4, `FogExp2(…, 0.00055)` — so the constellation looks
as it did there. An earlier version dimmed it to make highlights pop and lost
the constellation; don't do that again. The camera is a **scripted tween**, not
a free orbit: wide on the whole field, hold, then ease in to the neighbourhood,
because the shot is part of the explanation.

The codeword is `CODEWORD` near the top of the `<script>`, matched forgivingly
(case and punctuation are ignored). **It is in client source, so it is a
surprise, not a secret** — check it server-side if it must be unguessable.
A "show the mechanism: on/off" toggle appears in the header once unlocked, so
the explainer can be skipped without reloading.

`crate.html` is untouched and still the plain version.

### What the stage needs from the backend

- `POST /api/chat/stream` — SSE, one event per step, from a `say_stream()`
  generator over the same tool loop. Streamed rather than returned at the end
  so the explanation unfolds *while* the model works.
- `structuring` carries `descriptions`, `embedded` (the templated strings),
  `template`, `dims` and `averaged`.
- `tracks[].relevance` — 0–1 within the turn. Brightness, not a score.
- `tracks[].x3/y3/z3` and `position`, from `data/map_coords_3d.json`.
- `steps[]` on the non-streaming `/api/chat` for anything that wants the
  mechanism without the theatre.

`position` is the relevance-weighted centroid of the tracks returned, not a
projection of the query vector. A 512→3 linear fit would be dishonest (UMAP is
far too non-linear) and `umap.transform` costs seconds, which would wreck the
timing. This is exact, free and always consistent with what lights up — and the
caption says so on screen.

### Grounded descriptions

Claude only ever sees `track_id`, `name`, `artist` — never the audio or the
embedding — so for a long time every sonic adjective in its replies came from
its own knowledge of the recording. Usually right, never verified, and it would
describe a track it had never heard of from the title with identical
confidence.

Every result now carries a `profile`: that track's percentile in this library on
each validated axis, from the audio embedding. One dot product per axis, cached.

The division of labour matters and is easy to get backwards:

- **What Claude knows is usually more accurate for what the music *is*** —
  instrumentation, genre, era, performer. A solo piano record is a solo piano
  record and no projection overrules that.
- **The profile is better for where a track sits relative to this library** and
  to what was just played: calmer than the last set, unusually warm, an
  outlier. Knowledge cannot supply that, because a percentile is a position in
  this particular collection.

The output is one coherent description that uses each for what it is good at.
The assistant is explicitly told never to mention a profile, percentile or axis
to the user and never to stage a disagreement with itself — that is the system
talking about itself instead of about the music. Where the two genuinely clash,
it leads with what it knows and makes no claim on that dimension.

Per-axis weighting in the prompt is generated from `taste_axes_report.json`, so
it cannot drift from what was validated.

`taste_axes.py stability` adds a label-free screen for the axes `validate()`
cannot test: an axis measuring something real varies less within one artist's
catalogue than across the library. All eight pass (energy 0.36 tightest,
aggression 0.74 weakest). Worth re-running after any change to POLES.

### Autonomy and the wider toolset

The assistant now decides what to do rather than running the same retrieval
after every message. The forced tool call on the opening turn is gone — it did
get tracks on screen fast, but it made it impossible to answer "what Paul Simon
have I got" without an acoustic search, or to ask a clarifying question when one
was genuinely needed. The prompt steers toward acting over interviewing; the
choice is the model's.

Eight tools:

| tool | for |
|---|---|
| `search_by_description` | acoustic search; `wildcard` is now a parameter |
| `search_by_artist` / `add_artists` | seed or weight by artist |
| `search_by_track` | neighbours of one track |
| `refine` | Rocchio over feedback and axes |
| **`library_lookup`** | metadata only — "what X have I got", "do I own any of these" |
| **`play`** | one named track, by itself |
| **`library_overview`** | size, distinct artists, who dominates |

Output shape follows the request: one track for a play, a list for a catalogue
question, three or four for a search. No wildcard on a lookup or a play.

**The knowledge-then-audio pattern.** CLAP cannot read a genre name, but the
model knows which artists belong to one. So it names candidates from its own
knowledge, `library_lookup` says which are actually present, and the search is
then seeded from real audio:

    "got any psytrance?"
      -> library_lookup(artists=["Infected Mushroom","Astrix","Shpongle", ...])
      -> whichever exist here
      -> add_artists on those

`library_lookup` matches on **word boundaries**, not bare substrings. That
matters more than it sounds: as a substring, "Ott" matches Travis Scott and
Ottorino Respighi, which turned a seven-artist psytrance check into 158 false
positives the assistant would have believed. It also returns `per_query` counts
and an `absent` list, so "you have none of the usual psytrance names, shall I
find the nearest thing by sound?" is answerable truthfully.

### Searching is not recommending

The user sees `last_results`, which used to be a side effect of whichever tool
ran last. So the assistant could judge a search correctly, say so, recommend
better tracks in prose — and the UI would go on displaying the ones it had just
rejected.

That is exactly what happened on "can you play some happy 80s dance tracks".
The acoustic search returned Jeroen Search, SkyZlimit and The Chainsmokers;
the model recognised they were wrong, named six genuinely present 80s tracks
instead (all of them verified real — Michael Jackson, Donna Summer, Eurythmics,
Wham!, Earth Wind & Fire), and the shelf underneath still showed the modern
electronic ones. Its judgement was right and had no way to reach the screen.

`present(track_ids)` fixes it: the assistant sets exactly what is displayed, in
its own order, from ids returned by any earlier tool. An empty list shows
nothing, which is the honest answer when a search found nothing worth showing.
The prompt now forbids leaving a rejected result on screen, and requires that
anything it recommends be confirmed present with `library_lookup` first — its
knowledge proposes, the metadata confirms, `present` decides.

## 11. Metadata enrichment (era and genre)

`python enrich_metadata.py` — resolves every ISRC against Apple's catalogue for
a release date and genre tags. 7,046 unique ISRCs, batched 25 to a request, so
~282 calls and about a minute. Resumable: raw responses are cached in
`data/apple_cache.json`, so a re-run after a failure costs nothing. Output is
`data/library_enriched.csv`, joined on track id at Crate init.

Coverage on this library: **7,024 of 7,106 tracks (98.8%)** have a year and
genre tags. By decade: 734 in the 1980s, 1,070 in the 90s, 1,843 in the 2010s.
Commonest tags: Classical 4,112, Pop 713, Rock 503, Jazz 463, Dance 332.

`search_by_description` now takes `year_from`, `year_to` and `genres`, which
filter the library *before* acoustic ranking, and the result carries a `filter`
block with the surviving candidate count so the assistant can admit when a
filter left almost nothing. The wildcard respects the filter too — an outlier
that breaks the requested era is not an outlier, it is a mistake.

This is what "happy 80s dance tracks" needed. Unfiltered it returned 2020s
trance, because CLAP hears "upbeat, dance, synths" and matches exactly that;
the decade is inaudible to it. Filtered to 1980–1989 with Dance/Pop tags it
returns Eurythmics, The Alan Parsons Project and Paul Simon.

**`year` is when the RECORDING was released, not when the music was written.**
For popular music those coincide. For the 4,000+ classical tracks here it is
the date of the performance — Kabalevsky's 24 Preludes resolve to 2000. So
never use a year range for a repertoire era; use the genre tags, which include
"Baroque Era" and "Romantic Era". Filtering `genres=["Baroque Era"]` correctly
returns Handel and the Goldberg Variations.

Genre tags are commercial categories, not musicology, and sometimes very
coarse — Graceland is filed under "Worldwide". Treat them as a usable filter,
not a taxonomy.

### cluster(): when an artist is not one sound

Averaging a catalogue assumes the artist is coherent. Plenty are not, and for
those the mean is a point none of their tracks occupies — so "more like X" is a
blurry query and the results are a compromise nobody asked for.

`cluster(artist=…)` k-means over that artist's embeddings, picks k by
silhouette, and names each group by its three most representative tracks plus
the validated axes on which it differs from the rest. The threshold is 0.10,
low by textbook standards and deliberately: dense audio embeddings are not
well-separated blobs, and a real stylistic split shows up as a modest but
consistent score.

It finds musically real divisions:

| artist | n | silhouette | split |
|---|---|---|---|
| Beethoven | 184 | 0.571 | string quartets (82) vs piano sonatas and symphonies (102) |
| AC/DC | 10 | 0.469 | full-throttle (8) vs Hells Bells, Rock and Roll Ain't Noise Pollution |
| Fergus McCreadie | 27 | 0.394 | instrumental trio (21) vs the more vocal tracks (6) |
| Alfa Mist | 34 | 0.377 | five groups, separating the jazz-hop from the electronic |
| Michael Jackson | 4 | — | declines: too few tracks to cluster |

With no argument it clusters the whole library and marks which groups the
conversation has not drawn from. That is the honest answer to "play me
something completely different": distance alone just returns a random other
genre, whereas an unvisited dense region is somewhere they actually have music
and have not been. After a solo-piano search, three of the four groups come
back unvisited.

### outliers(): "surprise me" had no tool

Asked for "the most unusual track in my library" the agent had nothing to call,
so it improvised — a text search for "avant-garde, eerie" with the wildcard on
— and then, challenged, correctly described what it had actually done and why
that was a weaker claim than the question implied. The gap was real: a wildcard
is unusual *relative to one query*, not isolated in the collection.

`outliers()` measures mean cosine to each track's **ten** nearest neighbours.
Not the single nearest, which is the obvious choice and the fragile one: a
track with a live version, a remaster or a duplicate entry has a neighbour at
cos ~1.0 and can never rank as unusual however strange it is. Averaging over
ten also stops a pair of mutually-odd tracks hiding each other — Romitelli's
movements only surface under k-NN, because under nearest-neighbour they are
each other's company.

    most unusual                                     nn cos
      Sufres Al Verme — Brayan Hino, Quixosis         0.313
      An Index Of Metals: IV — Fausto Romitelli       0.331
      An Index Of Metals: II, V                       0.424, 0.429
      Walton: Symphony No. 2, Introduction            0.430
    most representative
      Nocturne No. 14 in C — John Field               0.984

Chunked over 512 rows because the full similarity matrix is 200MB, computed
once and cached: 0.49s cold, free thereafter.

Every result now also carries `isolation`, 0–100, so how unusual a track is is
known in every context rather than only when asked. Golijov's "Muerte" — the
track the agent originally offered — comes back at 92: genuinely odd, but 8% of
the library is stranger, which is roughly what it guessed from the `fit`
verdict alone.

### cluster() on the whole library was O(n²)

`silhouette_score` builds the full pairwise distance matrix, so library-wide
clustering computed 50 million pairs per candidate k. Now sampled at 1,500
points, with MiniBatchKMeans above 2,000 tracks, and the library clustering
cached since the library does not change between calls — only the `visited`
flags do.
