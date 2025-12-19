import faiss
import pickle
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_PATH = os.path.join(DATA_DIR, "spotify.index")
META_PATH = os.path.join(DATA_DIR, "spotify_meta.pkl")

def load_resources():
    print("Loading resources...")
    
    # 1. Load Index
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Index not found at {INDEX_PATH}")
    
    # Workaround: FAISS C++ read_index also fails with non-ASCII paths on Windows.
    # Copy to a temp file with ASCII path, read it, then delete temp.
    try:
        fd, temp_path = tempfile.mkstemp(suffix=".index")
        os.close(fd)
        
        # Copy original index to temp path
        shutil.copy2(INDEX_PATH, temp_path)
        
        # Read from safe path
        index = faiss.read_index(temp_path)
        print(f"Index loaded. Total vectors: {index.ntotal}")
        
    finally:
        # Cleanup
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
    
    # 2. Load Metadata
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata not found at {META_PATH}")
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    print(f"Metadata loaded. Total records: {len(metadata)}")
    
    # 3. Load Model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    return index, metadata, model

def search(query, index, metadata, model, k=5):
    print(f"\nScanning for: '{query}'")
    
    # 1. Embed query
    query_vec = model.encode([query])
    query_vec = np.array(query_vec).astype('float32')
    
    # 2. Normalize (Important! We did this during build)
    faiss.normalize_L2(query_vec)
    
    # 3. Search
    D, I = index.search(query_vec, k)
    
    # 4. Display results
    print(f"{'Score':<10} | {'Track Name':<30} | {'Artist':<20} | {'Genre'}")
    print("-" * 80)
    
    found_indices = I[0]
    scores = D[0]
    
    for score, idx in zip(scores, found_indices):
        if idx == -1: continue # Should not happen if k < ntotal
        
        meta = metadata[idx]
        print(f"{score:.4f}     | {meta['track_name'][:28]:<30} | {meta['artists'][:18]:<20} | {meta['track_genre']}")

def main():
    try:
        index, metadata, model = load_resources()
        
        # Test Queries
        queries = [
            "sad song for rainy day",
            "high energy workout music",
            "relaxing piano music"
        ]
        
        for q in queries:
            search(q, index, metadata, model)
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
