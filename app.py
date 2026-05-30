import io
import os
import numpy as np
import requests
import torch
import joblib
from flask import Flask, request, jsonify, make_response, send_file
from pydub import AudioSegment
from transformers import ClapModel, AutoFeatureExtractor

PORT          = int(os.environ.get("PORT", 8765))
UMAP_REDUCER  = os.environ.get("UMAP_REDUCER_PATH", "umap_reducer.pkl")
HTML_FILE     = os.environ.get("HTML_FILE", "taste_map_with_words.html")

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

@app.route('/')
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
        emb    = embed_audio_url(url)
        coords = reducer.transform([emb])[0]
        print(f"  -> ({coords[0]:.3f}, {coords[1]:.3f})")
        return jsonify({"x": float(coords[0]), "y": float(coords[1]), "label": label})
    except Exception as e:
        print(f"  Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/embed_blob", methods=["POST"])
def embed_blob():
    print("  Received request")
    label = request.form.get("label", "Generated")

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_bytes = request.files['audio'].read()
    print(f"  Audio size: {len(audio_bytes)} bytes")

    try:
        audio = None
        for fmt in ["mp3", "m4a", "wav", "ogg"]:
            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
                break
            except Exception:
                continue
        if audio is None:
            return jsonify({"error": "Could not decode audio"}), 400

        audio   = audio.set_frame_rate(48000).set_channels(1)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        samples = samples / (2 ** (audio.sample_width * 8 - 1))

        inputs = feature_extractor(samples, return_tensors="pt", sampling_rate=48000)
        with torch.no_grad():
            emb = model.get_audio_features(**inputs)

        coords = reducer.transform([emb.squeeze().numpy()])[0]
        print(f"  Embedded -> ({coords[0]:.3f}, {coords[1]:.3f})")
        return jsonify({"x": float(coords[0]), "y": float(coords[1]), "label": label})

    except Exception as e:
        print(f"  Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

def embed_audio_url(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    audio = None
    for fmt in ["m4a", "mp3", "wav"]:
        try:
            audio = AudioSegment.from_file(io.BytesIO(response.content), format=fmt)
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

    inputs = feature_extractor(samples, return_tensors="pt", sampling_rate=48000)
    with torch.no_grad():
        emb = model.get_audio_features(**inputs)
    return emb.squeeze().numpy()

if __name__ == "__main__":
    print(f"\nCLAP server running on http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)