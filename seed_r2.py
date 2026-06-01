"""
One-time seed: upload the 7 existing generated tracks to Cloudflare R2 and
write the initial generated.json manifest.

Run locally (NOT on Railway) with the R2_* env vars set, e.g.:



Idempotent: re-running overwrites the same keys + manifest.
"""

import os

import r2_client

# Local folder holding track1.mp3 … track7.mp3
AUDIO_DIR = os.environ.get(
    "SEED_AUDIO_DIR",
    "/Users/bendunsmore/PycharmProjects/Spotify_app_for_cloudflare/generated",
)

# (x, y, label) for each track, copied from visualisation_redesign.py
GENERATED_TRACKS = [
    (6.6139817237854,   11.485816955566406, "a piece for orchestra with driving strings and weighty percussion with no regular time signature"),
    (-1.5788573026657104, -0.2519499659538269, "an afrobeat track which puts a smile on the face"),
    (-1.514056921005249,  -0.27723804116249084, "an afrobeat track which puts a smile on the face v2"),
    (-0.5337150692939758, -0.17413343489170074, "a synthetic upbeat pop song"),
    (-2.523174285888672,  -0.031268637627363205, "an experimental electronic track inspired by music concrète"),
    (1.166, 3.649, "collective improvisation in the style of new orleans jazz"),
    (-3.347, 1.763, "the most groovy, funky song imaginable"),
]


def main():
    records = []
    for i, (x, y, label) in enumerate(GENERATED_TRACKS, start=1):
        mp3_path = os.path.join(AUDIO_DIR, f"track{i}.mp3")
        with open(mp3_path, "rb") as f:
            audio_bytes = f.read()
        key       = f"generated/seed-track{i}.mp3"
        audio_url = r2_client.upload_mp3(audio_bytes, key=key)
        print(f"  uploaded {mp3_path} -> {audio_url}")
        records.append({
            "x": x, "y": y, "label": label,
            "audio_url": audio_url, "point_type": "generated",
        })

    r2_client.write_manifest(records)
    print(f"\nWrote generated.json with {len(records)} records.")


if __name__ == "__main__":
    main()
