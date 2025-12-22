import pandas as pd
import json
import os
import shutil

# Paths (using relative paths since we run from root usually, but script will handle cwd)
SONGS_CSV_PATH = 'data/songs.csv'
BIG_JSON_PATH = 'data/sim_items_json/precomputed_similar_songs.json'
BACKUP_JSON_PATH = 'data/sim_items_json/precomputed_similar_songs.json.bak'
OUTPUT_JSON_PATH = 'data/sim_items_json/precomputed_similar_songs.json'

def main():
    print(f"Reading {SONGS_CSV_PATH}...")
    try:
        df_songs = pd.read_csv(SONGS_CSV_PATH)
        active_track_ids = set(df_songs['track_id'].unique())
        print(f"Found {len(active_track_ids)} unique track IDs in CSV.")
    except Exception as e:
        print(f"Error reading songs.csv: {e}")
        return

    if not os.path.exists(BIG_JSON_PATH):
        print(f"Error: {BIG_JSON_PATH} not found.")
        return

    print(f"Loading {BIG_JSON_PATH} (this might take a moment)...")
    try:
        with open(BIG_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items from JSON.")
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    # Filter
    print("Filtering data...")
    filtered_data = []
    
    for item in data:
        if item['track_id'] in active_track_ids:
            filtered_data.append(item)

    print(f"Filtered down to {len(filtered_data)} items.")

    # Backup
    if os.path.exists(OUTPUT_JSON_PATH):
        if not os.path.exists(BACKUP_JSON_PATH):
            print(f"Creating backup at {BACKUP_JSON_PATH}...")
            shutil.copy2(OUTPUT_JSON_PATH, BACKUP_JSON_PATH)
        else:
            print(f"Backup already exists at {BACKUP_JSON_PATH}, skipping backup creation.")

    # Save
    print(f"Saving filtered JSON to {OUTPUT_JSON_PATH}...")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    main()
