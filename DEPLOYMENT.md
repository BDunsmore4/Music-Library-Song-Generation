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
