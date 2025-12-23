import pandas as pd
import numpy as np
from collections import Counter
import os
import json
import chromadb
import openai
from dotenv import load_dotenv

load_dotenv()
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials


def analyze_persona(history_list):
    """
    Analyzes a list of tracks (history) to derive persona traits.
    Returns a dictionary of traits.
    """
    if not history_list:
        return {
            "top_genres": [],
            "avg_popularity": 0,
            "favorite_artists": []
        }

    genres = [track.get('track_genre', 'unknown') for track in history_list]
    artists = [track.get('artists', 'unknown') for track in history_list]
    popularities = [track.get('popularity', 0) for track in history_list]

    top_genres = [g for g, c in Counter(genres).most_common(3)]
    top_artists = [a for a, c in Counter(artists).most_common(3)]
    avg_pop = sum(popularities) / len(popularities) if popularities else 0

    return {
        "top_genres": top_genres,
        "avg_popularity": avg_pop,
        "top_artists": top_artists,
        "history_count": len(history_list)
    }


def get_recommendations(df_songs, context_song, persona_traits, top_k=3):
    """
    Simulates the RAG retrieval and generation process.
    1. Retrieval: Find songs similar to context_song (same genre, similar tempo).
    2. Filtering: Filter based on persona traits (e.g. popularity match).
    3. Ranking: Random sample for demo.
    """
    # 1. Retrieval (Simple filtering for demo)
    # Find songs with same genre
    candidates = df_songs[df_songs['track_genre']
                          == context_song['track_genre']].copy()

    # If not enough, relax constraint
    if len(candidates) < 10:
        candidates = df_songs.copy()

    # Calculate simple similarity (distance in tempo)
    candidates['tempo_diff'] = abs(candidates['tempo'] - context_song['tempo'])

    # Sort by tempo similarity
    retrieved = candidates.sort_values('tempo_diff').head(20)

    # 2. Filtering (Agentic Step)
    # Example: If persona likes popular music, filter out low popularity
    # For demo, we just return the retrieved list and let the UI visualize the "filtering"

    # 3. Final Selection
    final_recs = retrieved.sample(min(top_k, len(retrieved)))

    return retrieved, final_recs


def load_precomputed_sims(path="data/sim_items_json/precomputed_similar_songs.json"):
    """
    Loads precomputed similarity data from a JSON file.
    Returns a dictionary for fast lookup by track_id.
    """
    import os
    import json

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Convert list to dict for O(1) lookup: {track_id: {data}}
            return {item['track_id']: item for item in data}
    return {}


def get_precomputed_candidates(song_id, precomputed_data):
    """
    Retrieves pre-computed similar songs for a given song_id.
    Returns a list of candidate dictionaries or an empty list.
    """
    if not precomputed_data or song_id not in precomputed_data:
        return []

    # Get the item data
    item_data = precomputed_data[song_id]

    # Return top_n_similar_songs list, filtering out the query song itself
    # Filter by ID AND Name/Artist to catch different versions of the same song
    similar_songs = item_data.get('top_n_similar_songs', [])

    # Get query details
    query_name = None
    query_artist = None
    # We need to find the query song's name/artist from the precomputed data or pass it in.
    # Since we only have song_id here, we look up the item_data's own metadata if possible,
    # but item_data is the *result* list usually? No, item_data is the source node.
    # Let's assume precomputed_data[song_id] has keys like 'track_name' too?
    # Checking file structure from earlier view or inference...
    # Usually it's {id: {top_n: [...]}}.
    # Safe bet: We filter duplicates within the list and any that match the *song_id*.
    # AND, we filter any that have identical Name+Artist to each other or the query?
    # We don't have query name here easily.
    # Let's deduplicate the OUTPUT list first by Name+Artist.

    seen = set()
    unique_candidates = []

    for s in similar_songs:
        # 1. Filter exact ID match (Self)
        if s['track_id'] == song_id:
            continue

        # 2. Filter duplicates in the result list (e.g. same song appears twice)
        # Create a signature: Name + Artist
        sig = (s.get('track_name', '').strip().lower(),
               s.get('artists', '').strip().lower())

        if sig in seen:
            continue
        seen.add(sig)

        unique_candidates.append(s)

    return unique_candidates


def generate_explanation(rec_song, context_song, persona_traits):
    """
    Generates a template-based explanation.
    """
    reasons = [
        f"這首歌的 {rec_song['track_genre']} 風格與您剛聽的歌曲相似。",
        f"它的節奏 ({int(rec_song['tempo'])} BPM) 與您的當前狀態非常契合。",
        f"這符合您對 {persona_traits['top_genres'][0] if persona_traits['top_genres'] else '音樂'} 的喜好。"
    ]

    return f"推薦理由：{reasons[0]} 且{reasons[2]}"


def plot_pca_visualization(df_songs, context_song, recommended_songs, user_history=[], step1_cands=[], step2_cands=[], df_pca=None):
    """
    Performs PCA to reduce song features to 2D and plots the distribution.
    Highlights:
    1. User History
    2. Pre-computed Candidates
    3. FAISS Candidates
    4. Re-ranked Recommendations
    5. Current Playing (Context)

    If df_pca is provided with precomputed PCA columns, it will be used instead of recomputing.
    """
    # print("===== df_songs =====")
    # print(df_songs.head())
    # print("===== context_song =====")
    # print(context_song.head())
    # print("===== recommended_songs =====")
    # print(recommended_songs.head())
    # print("===== user_history =====")
    # print(user_history[:5])
    # print("===== step1_cands =====")
    # print(step1_cands[:5])
    # print("===== step2_cands =====")
    # print(step2_cands[:5])
    # print("===== df_pca =====")
    # print(df_pca.head())

    import plotly.express as px
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    plot_df = None
    precomputed_used = False

    if df_pca is not None:
        # Accept both PC1/PC2 and PC_1/PC_2 naming conventions
        pc1_col = 'PC1' if 'PC1' in df_pca.columns else 'PC_1' if 'PC_1' in df_pca.columns else None
        pc2_col = 'PC2' if 'PC2' in df_pca.columns else 'PC_2' if 'PC_2' in df_pca.columns else None

        if pc1_col and pc2_col and 'track_id' in df_pca.columns:
            precomputed_used = True
            songs_with_id = df_songs.copy()
            songs_with_id['track_id'] = songs_with_id['track_id'].astype(str)
            pca_copy = df_pca.rename(
                columns={pc1_col: 'PC1', pc2_col: 'PC2'}).copy()
            pca_copy['track_id'] = pca_copy['track_id'].astype(str)

            plot_df = songs_with_id.merge(
                pca_copy[['track_id', 'PC1', 'PC2']], on='track_id', how='left')

            # If merge leaves missing components, fallback to recomputing
            if plot_df[['PC1', 'PC2']].isnull().any().any():
                precomputed_used = False
                plot_df = None

    if not precomputed_used:
        # 1. Feature Engineering
        # Select numerical features
        features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                    'instrumentalness', 'liveness', 'valence', 'tempo']

        # Check if 'key' and 'mode' exist for advanced encoding
        if 'key' in df_songs.columns:
            df_songs['key_sin'] = np.sin(2 * np.pi * df_songs['key'] / 12)
            df_songs['key_cos'] = np.cos(2 * np.pi * df_songs['key'] / 12)
            features.extend(['key_sin', 'key_cos'])

        if 'mode' in df_songs.columns:
            features.append('mode')

        # Prepare data
        X = df_songs[features].fillna(0)

        # Standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 2. PCA Reduction
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)

        # Create DataFrame for plotting
        plot_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
        plot_df['track_id'] = df_songs['track_id'].values
        plot_df['track_name'] = df_songs['track_name'].values
        plot_df['artists'] = df_songs['artists'].values
        plot_df['track_genre'] = df_songs['track_genre'].values

    plot_df['track_id'] = plot_df['track_id'].astype(str)
    plot_df['Type'] = 'Other'
    plot_df['Size'] = 1  # base size before scaling

    # Helper to clean ID lists (some might be dicts or strings)
    def extract_ids(source):
        if source is None:
            return []
        if isinstance(source, pd.DataFrame):
            return [] if source.empty else [str(x) for x in source['track_id'].values]
        if not source:
            return []  # Safe for lists
        # Check if list of dicts or list of strings
        if isinstance(source[0], dict):
            return [str(x.get('track_id')) for x in source]
        return [str(x) for x in source]

    hist_ids = extract_ids(user_history)
    step1_ids = extract_ids(step1_cands)
    # Helper to clean ID lists (some might be dicts or strings)
    def extract_ids(source):
        if source is None:
            return []
        if isinstance(source, pd.DataFrame):
            return [] if source.empty else [str(x) for x in source['track_id'].values]
        if not source:
            return []  # Safe for lists
        # Check if list of dicts or list of strings
        if isinstance(source[0], dict):
            return [str(x.get('track_id')) for x in source]
        return [str(x) for x in source]

    hist_ids = extract_ids(user_history)
    step1_ids = extract_ids(step1_cands)
    step2_ids = extract_ids(step2_cands)
    rec_ids = extract_ids(recommended_songs)
    context_id = str(context_song['track_id'])
    all_ids_in_plot = set(plot_df['track_id'])

    # Layer 1: User History
    plot_df.loc[plot_df['track_id'].isin(hist_ids), 'Type'] = 'User History'
    plot_df.loc[plot_df['track_id'].isin(hist_ids), 'Size'] = 4

    # Layer 2: Pre-computed
    plot_df.loc[plot_df['track_id'].isin(
        step1_ids), 'Type'] = 'Pre-computed Similar Candidates'
    plot_df.loc[plot_df['track_id'].isin(step1_ids), 'Size'] = 6

    # Layer 3: FAISS
    plot_df.loc[plot_df['track_id'].isin(
        step2_ids), 'Type'] = 'FAISS Search Candidates'
    plot_df.loc[plot_df['track_id'].isin(step2_ids), 'Size'] = 6

    # Layer 4: Re-ranked Recommendations (Winner)
    plot_df.loc[plot_df['track_id'].isin(
        rec_ids), 'Type'] = 'Re-ranked Recommendations'
    plot_df.loc[plot_df['track_id'].isin(rec_ids), 'Size'] = 8

    # Layer 5: Current Playing (Top)
    plot_df.loc[plot_df['track_id'] == context_id, 'Type'] = 'Current Playing'
    plot_df.loc[plot_df['track_id'] == context_id, 'Size'] = 12

    # If some highlight IDs are missing from plot_df (e.g., not in df_songs), try to append them using df_pca
    def append_missing_from_pca(id_list, type_label, size_value):
        nonlocal plot_df
        if df_pca is None:
            return
        # Accept either PC1/PC2 or PC_1/PC_2 naming
        pc1_col = None
        pc2_col = None
        if 'PC1' in df_pca.columns and 'PC2' in df_pca.columns:
            pc1_col, pc2_col = 'PC1', 'PC2'
        elif 'PC_1' in df_pca.columns and 'PC_2' in df_pca.columns:
            pc1_col, pc2_col = 'PC_1', 'PC_2'
        if pc1_col is None or pc2_col is None or 'track_id' not in df_pca.columns:
            return
        missing_ids = [tid for tid in id_list if tid not in all_ids_in_plot]
        if not missing_ids:
            return
        df_pca_local = df_pca.copy()
        df_pca_local['track_id'] = df_pca_local['track_id'].astype(str)
        extra_rows = df_pca_local[df_pca_local['track_id'].isin(
            missing_ids)].copy()
        if extra_rows.empty:
            return
        extra_rows = extra_rows.rename(
            columns={pc1_col: 'PC1', pc2_col: 'PC2'})
        extra_rows = extra_rows[['track_id', 'track_name',
                                 'artists', 'track_genre', 'PC1', 'PC2']]
        extra_rows['Type'] = type_label
        extra_rows['Size'] = size_value
        plot_df = pd.concat([plot_df, extra_rows], ignore_index=True)
        all_ids_in_plot.update(extra_rows['track_id'].tolist())

    append_missing_from_pca(hist_ids, 'History', 4)
    append_missing_from_pca(step1_ids, 'Step 1 (Similar)', 6)
    append_missing_from_pca(step2_ids, 'Step 2 (Semantic)', 6)
    append_missing_from_pca(rec_ids, 'Step 3 (Re-ranked)', 8)
    append_missing_from_pca([context_id], 'Now Playing', 12)

    # If still missing highlights (not in df_songs or df_pca), create dummy points to keep categories visible
    def add_dummy_points(id_list, type_label, size_value):
        nonlocal plot_df
        missing_ids = [tid for tid in id_list if tid not in all_ids_in_plot]
        if not missing_ids:
            return
        dummy_rows = pd.DataFrame({
            'track_id': missing_ids,
            'track_name': [type_label] * len(missing_ids),
            'artists': [type_label] * len(missing_ids),
            'track_genre': [type_label] * len(missing_ids),
            'PC1': [0.0] * len(missing_ids),
            'PC2': [0.0] * len(missing_ids),
            'Type': [type_label] * len(missing_ids),
            'Size': [size_value] * len(missing_ids),
        })
        plot_df = pd.concat([plot_df, dummy_rows], ignore_index=True)
        all_ids_in_plot.update(missing_ids)

    add_dummy_points(hist_ids, 'History', 4)
    add_dummy_points(step1_ids, 'Step 1 (Similar)', 6)
    add_dummy_points(step2_ids, 'Step 2 (Semantic)', 6)
    add_dummy_points(rec_ids, 'Step 3 (Re-ranked)', 8)
    add_dummy_points([context_id], 'Now Playing', 12)

    # Intelligent Sampling: Keep all highlights, sample background
    highlights = plot_df[plot_df['Type'] != 'Other'].copy()
    others = plot_df[plot_df['Type'] == 'Other'].copy()

    # Defaults for 'Other'
    others['Size'] = 1
    if len(others) > 2000:
        others = others.sample(2000, random_state=42)

    final_plot_df = pd.concat([others, highlights])
    final_plot_df = final_plot_df.dropna(subset=['PC1', 'PC2'])

    # 3. Plotting
    category_orders = {
        "Type": [
            'History',                      # 1. 使用者收聽紀錄
            'Now Playing',                   # 2. 現在收聽歌曲
            'Step 1 (Similar)',   # 3. STEP 1 結果
            'Step 2 (Semantic)',           # 4. STEP 2 結果
            'Step 3 (Re-ranked)',         # 5. STEP 3 前六筆
            'Other'
        ]
    }

    fig = px.scatter(
        final_plot_df,
        x='PC1',
        y='PC2',
        color='Type',
        size='Size',
        symbol='Type',
        hover_name='track_name',
        hover_data=['artists', 'track_genre'],
        category_orders=category_orders,
        color_discrete_map={
            'Other': '#888888',       # Lighter gray for visibility
            'History': '#9d4edd',                # Purple
            'Step 1 (Similar)': '#ff9e00',  # Orange
            'Step 2 (Semantic)': '#3a86ff',     # Blue
            'Step 3 (Re-ranked)': '#00FFFF',   # Cyan
            'Now Playing': '#FF0000'              # Red
        },
        symbol_map={
            'Other': 'circle',
            'History': 'circle',
            'Step 1 (Similar)': 'circle',
            'Step 2 (Semantic)': 'circle',
            'Step 3 (Re-ranked)': 'circle',
            'Now Playing': 'x'
        },
        title="Song Embedding Space (PCA 2D Projection)",
        template="plotly_dark",
        opacity=0.9,
        size_max=14
    )

    fig.update_layout(
        paper_bgcolor='#121212',  # Force dark background, no transparency
        plot_bgcolor='#121212',
        height=700,  # Fixed height to match UI iframe
        # width=900,   # REMOVE fixed width to allow responsiveness
        autosize=True, # Enable autosize
        margin=dict(l=20, r=20, t=50, b=100),  # Increase bottom margin for legend
        font=dict(
            family="'Circular', 'Helvetica Neue', Helvetica, Arial, sans-serif",
            size=12,
            color="white"
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            orientation="h",   # Horizontal legend
            yanchor="top",
            y=-0.1,            # Position below the plot
            xanchor="center",
            x=0.5,
            title=None         # Remove title to prevent overlap
        ),
        hoverlabel=dict(
            bgcolor="#121212",
            font_color="white",
            bordercolor="#555555"       
        )
    )

    # Hide background points by default; users can toggle via legend
    # Hide background points by default; users can toggle via legend
    for trace in fig.data:
        if getattr(trace, "name", "") == "Other":
            trace.visible = "legendonly"
            trace.legendrank = 1000  # Force 'Other' to the end
        else:
            trace.legendrank = 1     # Others appear first

    return fig


def get_chroma_collection(path):
    """
    Initializes and returns the ChromaDB collection.
    """
    try:
        if not os.path.exists(path):
            print(f"ChromaDB path not found: {path}")
            return None

        client = chromadb.PersistentClient(path=path)
        collection = client.get_collection(name="spotify_songs")
        return collection
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        return None


def query_chromadb(collection, song_id, n_results=5):
    """
    Queries ChromaDB for similar songs based on the embeddings of the input song_id.
    1. Get embedding for the input song_id.
    2. Query the collection for nearest neighbors.
    3. Return formatted results.
    """
    if collection is None:
        return []

    try:
        # 1. Get embedding for the query song
        # We fetch the song by ID to retrieve its embedding (if it exists in Chroma)
        # Note: In a real scenario, if the song isn't in DB, we'd need to compute embedding on the fly.
        # Here we assume the song exists in the pre-computed DB.
        result = collection.get(
            ids=[song_id],
            include=["embeddings"]
        )

        if not result['embeddings']:
            print(f"Embedding not found for song_id: {song_id}")
            return []

        query_embedding = result['embeddings'][0]

        # 2. Query for nearest neighbors
        # We ask for n_results + 1 because the query song itself will be returned as the top match (dist=0)
        query_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results + 1,
            include=["metadatas", "documents", "distances"]
        )

        # 3. Format results
        candidates = []
        # Result structure is list of lists (for batched queries). We only have one query.
        ids = query_results['ids'][0]
        metadatas = query_results['metadatas'][0]
        documents = query_results['documents'][0]
        distances = query_results['distances'][0]

        for i in range(len(ids)):
            # Skip the query song itself
            if ids[i] == song_id:
                continue

            cand = {
                'track_id': ids[i],
                'track_name': metadatas[i].get('track_name', 'Unknown'),
                'artists': metadatas[i].get('artists', 'Unknown'),
                'album_name': metadatas[i].get('album_name', 'Unknown'),
                'explanation': documents[i],  # The LLM generated description
                'distance': distances[i]
            }
            candidates.append(cand)

            # Stop if we have enough results (since we requested n+1)
            if len(candidates) >= n_results:
                break

        return candidates

    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return []


def rerank_candidates(df_songs, step1_cands, step2_cands, context_song, persona_traits, top_k=3):
    """
    Combines candidates from Step 1 and Step 2, applies filters/scoring, and returns top_k recommendations.

    Scoring Logic:
    1. Genre Match: High score if matches context song or persona's top genres.
    2. BPM Match: Higher score for closer tempo.
    3. Artist Match: Bonus if artist is in persona's top artists.
    """
    # 1. Collect all Candidate IDs
    cand_ids = set()
    if step1_cands:
        cand_ids.update([str(c['track_id']) for c in step1_cands])
    if step2_cands:
        cand_ids.update([str(c['track_id']) for c in step2_cands])

    if not cand_ids:
        # Fallback if no candidates found
        return df_songs.sample(min(top_k, len(df_songs)))

    # 2. Get DataFrame for candidates
    # Ensure IDs are strings for matching
    df_cands = df_songs[df_songs['track_id'].astype(
        str).isin(list(cand_ids))].copy()

    if df_cands.empty:
        return df_songs.sample(min(top_k, len(df_songs)))

    # 3. Calculate Scores

    # helper for genre score
    target_genre = context_song['track_genre']
    persona_genres = persona_traits.get('top_genres', [])
    persona_artists = persona_traits.get('top_artists', [])
    target_tempo = context_song['tempo']

    def calculate_score(row):
        score = 0

        # Genre Match (Max 40 pts)
        if row['track_genre'] == target_genre:
            score += 40
        elif row['track_genre'] in persona_genres:
            score += 20

        # BPM Similarity (Max 30 pts)
        # Assuming acceptable range +/- 30 bpm. Clamping difference.
        diff = abs(row['tempo'] - target_tempo)
        if diff <= 10:
            score += 30
        elif diff <= 30:
            score += 15
        elif diff <= 50:
            score += 5

        # Artist Affinity (Max 20 pts)
        if row['artists'] in persona_artists:
            score += 20

        # Popularity bonus (Max 10 pts) - Normalized 0-100 -> 0-10
        score += row['popularity'] / 10

        return score

    df_cands['ranking_score'] = df_cands.apply(calculate_score, axis=1)

    # 4. Sort and Select
    final_recs = df_cands.sort_values(
        'ranking_score', ascending=False).head(top_k)

    return final_recs


def llm_rerank_candidates(df_songs, step1_cands, step2_cands, context_song, persona_traits, top_k=3, model_name="gpt-4o", provider="openai"):
    """
    Uses an LLM to re-rank candidates based on persona and context.
    Supports 'openai' and 'openrouter' providers.
    Falls back to rules if API fails.
    """
    try:
        api_key = None
        base_url = None

        if provider == "openrouter":
            api_key = os.getenv("OPEN_ROUTER_API_KEY")
            base_url = "https://openrouter.ai/api/v1"
            if not api_key:
                print("Warning: OPEN_ROUTER_API_KEY not found.")
        else:  # default to openai
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY")
        
        if not api_key:
            print(f"Warning: API Key for {provider} not found. Falling back to rule-based reranking.")
            return rerank_candidates(df_songs, step1_cands, step2_cands, context_song, persona_traits, top_k), "Rule-based fallback (No API Key)"

        # 1. Prepare Candidates
        # Collect unique candidate IDs from both steps
        candidate_ids = set()
        if step1_cands:
            for c in step1_cands:
                candidate_ids.add(c['track_id'])
        if step2_cands:
            for c in step2_cands:
                candidate_ids.add(c['track_id'])

        # Filter df_songs for these candidates
        candidates_df = df_songs[df_songs['track_id'].isin(candidate_ids)].copy()

        # If not enough candidates, add random ones
        if len(candidates_df) < top_k:
            remaining = top_k - len(candidates_df)
            random_filler = df_songs[~df_songs['track_id'].isin(
                candidate_ids)].sample(min(remaining, len(df_songs)))
            candidates_df = pd.concat([candidates_df, random_filler])

        # Convert candidates to compact JSON for Prompt
        candidates_list = []
        for _, row in candidates_df.iterrows():
            # Explicitly cast to native types to avoid JSON serialization TypeError (int64/float32)
            candidates_list.append({
                "track_id": str(row['track_id']),
                "track_name": str(row['track_name']),
                "artists": str(row['artists']),
                "genre": str(row['track_genre']),
                "bpm": f"{row['tempo']:.0f}",
                "features": f"Energy:{row.get('energy',0):.2f}, Valence:{row.get('valence',0):.2f}"
            })
        candidates_str = json.dumps(candidates_list, ensure_ascii=False)

        # 2. Construct Prompt
        # Simplify persona traits for prompt
        persona_summary = f"Top Genres: {', '.join(persona_traits.get('top_genres', []))}. " \
                          f"Top Artists: {', '.join(persona_traits.get('top_artists', []))}. " \
                          f"Preferred Properties: {persona_traits.get('audio_properties', {})}"

        current_song_str = f"{context_song['track_name']} by {context_song['artists']} (Genre: {context_song['track_genre']}, BPM: {context_song['tempo']:.0f})"

        prompt = f"""
        You are an expert music recommender system (Agentic RAG).
        
        Context:
        - User Persona: {persona_summary}
        - Current Song: {current_song_str}
        
        Task:
        Select the best {top_k} songs from the CANDIDATES list below to recommend next.
        Rank them by relevance to the User Persona and continuity with the Current Song.
        
        CANDIDATES:
        {candidates_str}
        
        Output Format:
        Return ONLY a raw JSON string (no markdown formatting) with this structure:
        {{
            "recommendations": [
                {{
                    "track_id": "...",
                    "reason": "Write a short, engaging reason (in Traditional Chinese) citing specific features (BPM, Genre, Mood) why this fits."
                }},
                ...
            ],
            "explanation": "Briefly summarize your overall recommendation strategy and how these choices fit the user persona (in Traditional Chinese)."
        }}
        """

        # 3. Call LLM
        if base_url:
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = openai.OpenAI(api_key=api_key)
            
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful music recommendation assistant. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
            timeout=60
        )
        content = response.choices[0].message.content

        parsed = json.loads(content)
        recommendations = parsed.get("recommendations", [])
        overall_explanation = parsed.get(
            "explanation", "Designed by AI Agent.")

        if not recommendations:
            # Try fallback parsing if "recommendations" key is missing but list exists
            if isinstance(parsed, list):
                recommendations = parsed

        # 4. Process Results
        final_recs = []
        for rec in recommendations:
            track_id = rec.get("track_id")
            reason = rec.get("reason", "AI Recommended")

            # Find the song row (ensure type matching for ID)
            # Normalize ID to str for reliable lookup
            song_row = candidates_df[candidates_df['track_id'].astype(str) == str(track_id)]
            if not song_row.empty:
                song_data = song_row.iloc[0].to_dict()
                song_data['reason'] = reason
                # Assign a high fake score to preserve order
                song_data['ranking_score'] = 99.0 - len(final_recs)
                final_recs.append(song_data)

        # Convert to DataFrame
        if final_recs:
            return pd.DataFrame(final_recs).head(top_k), overall_explanation
        else:
            print("LLM returned no valid recommendations. Using rule-based.")
            return rerank_candidates(df_songs, step1_cands, step2_cands, context_song, persona_traits, top_k), "LLM returned no valid recommendations."

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"LLM Rerank Failed: {e}. Using rule-based.")
        return rerank_candidates(df_songs, step1_cands, step2_cands, context_song, persona_traits, top_k), f"LLM Rerank Failed: {e}"

def get_google_sheet_resource():
    """
    Returns (client, spreadsheet_url) if secrets are configured, else (None, None).
    """
    if "gsheets" not in st.secrets.get("connections", {}):
        return None, None
        
    try:
        secrets = st.secrets["connections"]["gsheets"]
        spreadsheet_url = secrets.get("spreadsheet")
        
        # Construct credentials dict, removing generic keys if needed, 
        # but from_service_account_info ignores extras usually.
        # We need to ensure we pass a dict, secrets object behaves like one.
        creds_dict = dict(secrets)
        
        # Define scopes
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=scopes
        )
        
        client = gspread.authorize(creds)
        return client, spreadsheet_url
    except Exception as e:
        print(f"GSpread Auth Error: {e}")
        return None, None


def save_vote(vote_data, csv_filepath="data/user_votes.csv"):
    """
    Saves a vote record. Tries Google Sheets first, falls back to CSV.
    """
    # 1. Try Google Sheets
    client, sheet_url = get_google_sheet_resource()
    if client and sheet_url:
        try:
            sh = client.open_by_url(sheet_url)
            worksheet = sh.get_worksheet(0) # First sheet
            
            # Prepare row data
            columns = ["timestamp", "persona", "selected_song_name", "vote_reason", "vote_song"]
            row_values = [vote_data.get(col, "") for col in columns]
            
            # Check if sheet is empty (to add headers)
            # Efficient check: if first row is empty, it needs headers.
            first_row = worksheet.row_values(1)
            if not first_row:
                worksheet.append_row(columns)
                
            # Append new row
            worksheet.append_row(row_values)
            return True
        except Exception as e:
            print(f"Google Sheets Save Error (gspread): {e}")
            # Fallback
            
    # 2. Fallback to Local CSV
    print("Falling back to local CSV for vote storage.")
    try:
        file_exists = os.path.isfile(csv_filepath)
        df = pd.DataFrame([vote_data])
        
        # Ensure consistent columns for CSV if possible, but pandas handles dicts well.
        # But for 'columns' above, let's keep it robust.
        # Just use the dict as is for CSV.
        df.to_csv(csv_filepath, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')
        return True
    except Exception as e:
        print(f"CSV Save Error: {e}")
        return False


@st.cache_data(ttl=300, show_spinner=False)
def load_votes(csv_filepath="data/user_votes.csv"):
    """
    Loads voting data from Google Sheets (preferred) or CSV.
    """
    # 1. Try Google Sheets
    client, sheet_url = get_google_sheet_resource()
    if client and sheet_url:
        try:
            sh = client.open_by_url(sheet_url)
            worksheet = sh.get_worksheet(0)
            
            # Get all values as list of lists
            # data = worksheet.get_all_records() # This fails if no proper header
            all_values = worksheet.get_all_values()
            
            if not all_values:
                return pd.DataFrame()

            # Smart detection: does first row look like headers?
            # Our expected headers: "timestamp", "persona", ...
            expected_headers = ["timestamp", "persona", "selected_song_name", "vote_reason", "vote_song"]
            
            # Normalize first row to lowercase for comparison
            first_row_lower = [str(x).lower() for x in all_values[0]]
            
            if "timestamp" in first_row_lower and "persona" in first_row_lower:
                # Has headers
                if len(all_values) > 1:
                    return pd.DataFrame(all_values[1:], columns=all_values[0])
                else:
                    # Only headers, no data
                    return pd.DataFrame(columns=all_values[0])
            else:
                # No headers, assume all is data (e.g. user voted on empty sheet before we added header logic)
                return pd.DataFrame(all_values, columns=expected_headers)
                
        except Exception as e:
            print(f"Google Sheets Load Error (gspread): {e}")

    # 2. Fallback to CSV
    if not os.path.exists(csv_filepath):
        return None
    
    try:
        return pd.read_csv(csv_filepath, encoding='utf-8-sig')
    except Exception as e:
        print(f"Error loading votes from CSV: {e}")
        return None
