import io
import os
import numpy as np
import requests
import torch
import joblib
from flask import Flask, request, jsonify, make_response, send_file
from pydub import AudioSegment
from transformers import ClapModel, AutoFeatureExtractor

import r2_client

PORT               = int(os.environ.get("PORT", 8765))
UMAP_REDUCER       = os.environ.get("UMAP_REDUCER_PATH", "umap_reducer.pkl")
HTML_FILE          = os.environ.get("HTML_FILE", "taste_map_with_words.html")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_URL     = "https://api.elevenlabs.io/v1/music/compose"

print("Loading CLAP model...")
model             = ClapModel.from_pretrained("laion/clap-htsat-unfused")
feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
model.eval()
print("  CLAP ready")

print("Loading UMAP reducer...")
reducer = joblib.load(UMAP_REDUCER)
print("  UMAP reducer ready")

app = Flask(__name__)

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


# ── Shared embedding helpers ────────────────────────────────────────────────
def decode_audio_to_samples(audio_bytes):
    """Decode audio bytes (mp3/m4a/wav/ogg) into 48kHz mono float32 samples."""
    audio = None
    for fmt in ["mp3", "m4a", "wav", "ogg"]:
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
            break
        except Exception:
            continue
    if audio is None:
        raise ValueError("Could not decode audio — unsupported format")

    audio   = audio.set_frame_rate(48000).set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples = samples / (2 ** (audio.sample_width * 8 - 1))
    if len(samples) == 0:
        raise ValueError("Empty audio")
    return samples


def embed_samples(samples):
    """Run CLAP on normalised samples and return the embedding vector."""
    inputs = feature_extractor(samples, return_tensors="pt", sampling_rate=48000)
    with torch.no_grad():
        emb = model.get_audio_features(**inputs)
    return emb.squeeze().numpy()


def embed_bytes_to_coords(audio_bytes):
    """Decode + embed audio bytes, project to 2D, return (x, y) floats."""
    emb    = embed_samples(decode_audio_to_samples(audio_bytes))
    coords = reducer.transform([emb])[0]
    return float(coords[0]), float(coords[1])


# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def serve_html():
    return send_file(HTML_FILE)

@app.route("/embed", methods=["POST"])
def embed():
    body  = request.get_json()
    url   = body.get("url")
    label = body.get("label", "Generated")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        print(f"Embedding: {label}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        x, y = embed_bytes_to_coords(resp.content)
        print(f"  -> ({x:.3f}, {y:.3f})")
        return jsonify({"x": x, "y": y, "label": label})
    except Exception as e:
        print(f"  Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/embed_blob", methods=["POST"])
def embed_blob():
    print("  Received request")
    label = request.form.get("label", "Generated")

    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_bytes = request.files["audio"].read()
    print(f"  Audio size: {len(audio_bytes)} bytes")

    try:
        x, y = embed_bytes_to_coords(audio_bytes)
        print(f"  Embedded -> ({x:.3f}, {y:.3f})")
        return jsonify({"x": x, "y": y, "label": label})
    except Exception as e:
        print(f"  Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/generate", methods=["POST"])
def generate():
    """Generate audio via ElevenLabs (server-side key), embed it, persist to
    R2, and return the plotted point."""
    body   = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    if not ELEVENLABS_API_KEY:
        return jsonify({"error": "Server not configured for generation"}), 500

    try:
        print(f"Generating: {prompt}")
        el_resp = requests.post(
            ELEVENLABS_URL,
            headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
            json={
                "prompt": prompt,
                "output_format": "mp3_44100_128",
                "music_length_ms": 30000,
            },
            timeout=240,
        )
        if not el_resp.ok:
            try:
                detail = el_resp.json().get("detail", {})
                message = detail.get("message") if isinstance(detail, dict) else str(detail)
            except Exception:
                message = el_resp.text[:200]
            return jsonify({"error": f"ElevenLabs error: {message or el_resp.status_code}"}), 502

        mp3_bytes = el_resp.content
        print(f"  Generated {len(mp3_bytes)} bytes, embedding...")

        x, y      = embed_bytes_to_coords(mp3_bytes)
        audio_url = r2_client.upload_mp3(mp3_bytes)
        record    = {"x": x, "y": y, "label": prompt,
                     "audio_url": audio_url, "point_type": "generated"}
        r2_client.append_to_manifest(record)
        print(f"  Stored -> ({x:.3f}, {y:.3f}) {audio_url}")
        return jsonify(record)

    except Exception as e:
        print(f"  Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/search_track", methods=["POST"])
def search_track():
    """Look up a song on the iTunes Search API (no auth) and return candidate
    matches with 30s preview URLs, so the user can confirm the right version
    before embedding via /embed."""
    body  = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        r = requests.get(
            "https://itunes.apple.com/search",
            params={"term": query, "entity": "song", "limit": 6},
            timeout=15,
        )
        r.raise_for_status()
        results = []
        for item in r.json().get("results", []):
            if not item.get("previewUrl"):
                continue
            results.append({
                "name":        item.get("trackName"),
                "artist":      item.get("artistName"),
                "album":       item.get("collectionName"),
                "preview_url": item.get("previewUrl"),
                "artwork":     item.get("artworkUrl100"),
            })
        return jsonify({"results": results})
    except Exception as e:
        print(f"  Search error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print(f"\nCLAP server running on http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
