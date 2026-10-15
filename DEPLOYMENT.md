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
