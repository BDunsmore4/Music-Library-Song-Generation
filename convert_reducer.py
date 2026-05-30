"""
Run this once locally with Python 3.8 (your existing venv):
    python convert_reducer.py

Extracts the numpy arrays from the UMAP reducer pkl and saves them
as plain .npy files that can be loaded on any Python version.
"""
import os
import joblib
import numpy as np

PKL = os.path.join(os.path.dirname(__file__), ".venv", "umap_reducer.pkl")

print("Loading reducer...")
reducer = joblib.load(PKL)

print(f"  raw_data shape:  {reducer._raw_data.shape}")
print(f"  embedding shape: {reducer.embedding_.shape}")

np.save("umap_raw_data.npy",  reducer._raw_data.astype(np.float32))
np.save("umap_embedding.npy", reducer.embedding_.astype(np.float32))

print("Saved umap_raw_data.npy and umap_embedding.npy")
print("Commit and push both files to GitHub.")
