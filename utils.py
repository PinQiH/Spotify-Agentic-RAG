import pandas as pd
import numpy as np
from collections import Counter

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
