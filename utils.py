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
