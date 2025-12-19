import pandas as pd
import numpy as np
import faiss
import os
import pickle
import tempfile
import shutil
from sentence_transformers import SentenceTransformer

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAG_DOC_PATH = os.path.join(DATA_DIR, "spotify_rag_doc.csv")
INDEX_PATH = os.path.join(DATA_DIR, "spotify.index")
META_PATH = os.path.join(DATA_DIR, "spotify_meta.pkl")
SONGS_CSV_PATH = os.path.join(DATA_DIR, "songs.csv")

def build_index():
    print("--- Building FAISS Index ---")
    
    # 1. Load Data
    if not os.path.exists(RAG_DOC_PATH):
        print(f"Error: {RAG_DOC_PATH} not found.")
        return

    print("Loading text data...")
    df = pd.read_csv(RAG_DOC_PATH)
    
    # Clean data
    df = df.dropna(subset=['track_id', 'rag_doc']).drop_duplicates(subset=['track_id'])
    
    # Optimization: Filter to only songs in the app's songs.csv (similar to what we did for Chroma)
    # This keeps the index small (~2000 vs 114k) and fast
    if os.path.exists(SONGS_CSV_PATH):
        print("Filtering to match songs.csv...")
        df_songs = pd.read_csv(SONGS_CSV_PATH)
        target_ids = set(df_songs["track_id"].astype(str))
        original_len = len(df)
        df = df[df["track_id"].astype(str).isin(target_ids)]
        print(f"Filtered from {original_len} to {len(df)} records.")
    
    # Limit for quick testing as requested
    df = df.head(2000)
    print(f"Limiting processing to first {len(df)} records.") 

    documents = df['rag_doc'].astype(str).tolist()
    ids = df['track_id'].astype(str).tolist()
    
    # Prepare Metadata Map (FAISS only stores vectors, we need a way to look up info by ID)
    # We will use the index integer ID as the key to look up track_id and metadata
    metadata_map = []
    
    # Pre-extract metadata columns
    print("Preparing metadata...")
    meta_cols = ['track_name', 'artists', 'album_name', 'track_genre']
    for idx, row in df.iterrows():
        meta = {
            'track_id': str(row['track_id']),
            'rag_doc': str(row['rag_doc']),
            'track_name': str(row.get('track_name', 'Unknown')),
            'artists': str(row.get('artists', 'Unknown')),
            'album_name': str(row.get('album_name', 'Unknown')),
            'track_genre': str(row.get('track_genre', 'Unknown'))
        }
        metadata_map.append(meta)

    # 2. Generate Embeddings
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Generating embeddings for {len(documents)} documents...")
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    # Normalize for cosine similarity (FAISS uses L2, but normalized vectors + L2 = Cosine)
    faiss.normalize_L2(embeddings)
    
    # 3. Build Index
    d = embeddings.shape[1] # Dimension (384 for MiniLM)
    print(f"Building IndexFlatIP (Inner Product) for dimension {d}...")
    
    # IndexFlatIP corresponds to Cosine Similarity if vectors are normalized
    index = faiss.IndexFlatIP(d) 
    index.add(embeddings)
    
    print(f"Index built with {index.ntotal} vectors.")
    
    # 4. Save
    print(f"Saving index to {INDEX_PATH}...")
    # faiss.write_index(index, INDEX_PATH)
    
    # Workaround: FAISS on Windows often fails with non-ASCII paths (UnicodeEncodeError/Illegal byte sequence)
    # Write to a clean temp path first, then move using Python's robust path handling.
    try:
        # Create a temp file to get a safe path
        fd, temp_path = tempfile.mkstemp(suffix=".index")
        os.close(fd) # Close immediately, we just wanted a name
        
        # Write index to the temp path
        faiss.write_index(index, temp_path)
        
        # Move it to the final destination
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
        shutil.move(temp_path, INDEX_PATH)
        print("Index saved successfully (via temp file).")
        
    except Exception as e:
        print(f"Failed to save index: {e}")
        # Clean up temp file if it exists and wasn't moved
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise e
    
    print(f"Saving metadata to {META_PATH}...")
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata_map, f)
        
    print("Done!")

if __name__ == "__main__":
    build_index()
