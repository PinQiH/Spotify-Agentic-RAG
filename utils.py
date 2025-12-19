import pandas as pd
import numpy as np
from collections import Counter
import os
import json
import chromadb
import openai
from dotenv import load_dotenv

load_dotenv()

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
    candidates = df_songs[df_songs['track_genre'] == context_song['track_genre']].copy()
    
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

def load_precomputed_similarities(path="data/sim_items_json/precomputed_similar_songs.json"):
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
        sig = (s.get('track_name', '').strip().lower(), s.get('artists', '').strip().lower())
        
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

def plot_pca_visualization(df_songs, context_song, recommended_songs):
    """
    Performs PCA to reduce song features to 2D and plots the distribution.
    Highlights the context song and recommended songs.
    """
    import plotly.express as px
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # 1. Feature Engineering
    # Select numerical features
    features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                'instrumentalness', 'liveness', 'valence', 'tempo']
    
    # Check if 'key' and 'mode' exist for advanced encoding
    if 'key' in df_songs.columns:
        # Sin/Cos encoding for Key (Circular feature)
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
    plot_df['track_name'] = df_songs['track_name']
    plot_df['artists'] = df_songs['artists']
    plot_df['track_genre'] = df_songs['track_genre']
    plot_df['Type'] = 'Other' # Default type
    plot_df['Size'] = 1       # Default size
    
    # Highlight Context Song
    context_idx = df_songs[df_songs['track_id'] == context_song['track_id']].index
    if not context_idx.empty:
        plot_df.loc[context_idx, 'Type'] = 'Now Playing'
        plot_df.loc[context_idx, 'Size'] = 5
        
    # Highlight Recommended Songs
    rec_ids = recommended_songs['track_id'].values
    rec_indices = df_songs[df_songs['track_id'].isin(rec_ids)].index
    if not rec_indices.empty:
        plot_df.loc[rec_indices, 'Type'] = 'Recommended'
        plot_df.loc[rec_indices, 'Size'] = 3
        
    # 3. Plotting
    fig = px.scatter(
        plot_df, 
        x='PC1', 
        y='PC2', 
        color='Type',
        size='Size',
        hover_data=['track_name', 'artists', 'track_genre'],
        color_discrete_map={
            'Other': '#282828',       # Dark Gray for background
            'Now Playing': '#1DB954', # Spotify Green
            'Recommended': '#00FFFF'  # Cyan for recommendations
        },
        title="Song Embedding Space (PCA 2D Projection)",
        template="plotly_dark",
        opacity=0.8
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend_title_text=''
    )
    
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
                'explanation': documents[i], # The LLM generated description
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
    df_cands = df_songs[df_songs['track_id'].astype(str).isin(list(cand_ids))].copy()
    
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
    final_recs = df_cands.sort_values('ranking_score', ascending=False).head(top_k)
    
    return final_recs

def llm_rerank_candidates(df_songs, step1_cands, step2_cands, context_song, persona_traits, top_k=3):
    """
    Uses GPT-4o to re-rank candidates based on persona and context.
    Falls back to rules if API fails.
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found. Falling back to rule-based reranking.")
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
        random_filler = df_songs[~df_songs['track_id'].isin(candidate_ids)].sample(min(remaining, len(df_songs)))
        candidates_df = pd.concat([candidates_df, random_filler])

    # Convert candidates to compact JSON for Prompt
    candidates_list = []
    for _, row in candidates_df.iterrows():
        candidates_list.append({
            "track_id": row['track_id'],
            "track_name": row['track_name'],
            "artists": row['artists'],
            "genre": row['track_genre'],
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
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful music recommendation assistant. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        
        parsed = json.loads(content)
        recommendations = parsed.get("recommendations", [])
        overall_explanation = parsed.get("explanation", "Designed by AI Agent.")
        
        if not recommendations:
             # Try fallback parsing if "recommendations" key is missing but list exists
             if isinstance(parsed, list):
                 recommendations = parsed

        # 4. Process Results
        final_recs = []
        for rec in recommendations:
            track_id = rec.get("track_id")
            reason = rec.get("reason", "AI Recommended")
            
            # Find the song row
            song_row = candidates_df[candidates_df['track_id'] == track_id]
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
        print(f"LLM Rerank Failed: {e}. Using rule-based.")
        return rerank_candidates(df_songs, step1_cands, step2_cands, context_song, persona_traits, top_k), f"LLM Rerank Failed: {e}"
