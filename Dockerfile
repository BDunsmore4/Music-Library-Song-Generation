# Pin Python 3.8 directly via the official image — no mise/Railpack Python install,
# so the GitHub-attestation build failures can't occur. 3.8 + the exact pinned
# library versions are what let umap_reducer.pkl unpickle.
FROM python:3.8-slim

# ffmpeg is required by pydub to decode mp3/m4a/ogg audio previews.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code (incl. umap_reducer.pkl at repo root).
COPY . .

# Railway injects $PORT at runtime; default to 8765 locally.
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-8765} --workers 1 --timeout 300"]
