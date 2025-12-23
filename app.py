import streamlit as st
import pandas as pd
import time
import json
import os
import utils
import faiss
import pickle
import shutil
import tempfile
import numpy as np
import plotly.io as pio
from sentence_transformers import SentenceTransformer
import streamlit.components.v1 as components

# Set page config MUST be the first Streamlit command
st.set_page_config(
    page_title="Spotify Agentic RAG",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_PATH = os.path.join(DATA_DIR, "spotify.index")
META_PATH = os.path.join(DATA_DIR, "spotify_meta.pkl")

# Custom CSS for Spotify Dark Theme & Layout Fixes
st.markdown("""
<style>
    /* 1. Global Background & Font */
    .stApp {
        background-color: #121212; /* Deep Gray */
        color: #FFFFFF;
        font-family: 'Circular', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* 2. Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #000000; /* Pure Black */
        border-right: 1px solid #282828;
    }
    section[data-testid="stSidebar"] * {
        color: #B3B3B3 !important; /* Light Gray Text */
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important; /* White Headers */
    }

    /* 3. Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* 4. Circular Buttons (Green Icon) */
    .stButton > button {
        background-color: #000000; /* Solid Black Background */
        color: #1DB954; /* Spotify Green */
        border: 2px solid #1DB954;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        padding: 0;
        font-size: 18px;
        line-height: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s ease;
        margin: 0 auto; /* Center in column */
    }
    .stButton > button:hover {
        background-color: #1DB954;
        color: #000000;
        transform: scale(1.1);
        box-shadow: 0 0 15px rgba(29, 185, 84, 0.6);
        border-color: #1DB954;
    }
    .stButton > button:disabled {
        border: 2px solid #1DB954 !important;
        color: #1DB954 !important;
        opacity: 0.5;
        cursor: not-allowed;
    }
    /* Pagination buttons should be normal pills */
    div[data-testid="column"] .stButton > button {
        /* Reset for non-grid buttons if possible, but Streamlit CSS is global. 
           We'll stick to circular for selection, maybe adjust for pagination below if needed.
           Actually, let's make specific buttons circular by context if we could, 
           but for now let's make ALL buttons pill-shaped EXCEPT the grid ones? 
           Hard to differentiate in CSS without custom classes. 
           Let's keep the circular style for the main action and maybe tweak pagination.
        */
    }
    
    /* 5. Card Styling (Hover Effect) */
    div[data-testid="column"]:has(iframe) {
        background-color: #181818; /* Surface */
        border-radius: 8px;
        padding: 16px;
        transition: all 0.3s ease; /* Animate all properties */
        border: 1px solid transparent;
    }
    div[data-testid="column"]:has(iframe):hover {
        background-color: #282828; /* Lighter on hover */
        border: 1px solid #333;
        transform: translateY(-5px); /* Pop up effect */
        box-shadow: 0 10px 20px rgba(0,0,0,0.5);
    }
    /* Linked Hover: Change button when card is hovered */
    div[data-testid="column"]:has(iframe):hover button {
        background-color: #1DB954;
        color: #000000;
        border-color: #1DB954;
        transform: scale(1.1);
        box-shadow: 0 0 15px rgba(29, 185, 84, 0.6);
    }
    
    /* Force center alignment for buttons in song cards */
    div[data-testid="column"]:has(iframe) .stButton {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    
    /* 6. Terminal Style (Agentic Thinking) */
    div[data-testid="stStatusWidget"] {
        background-color: #000000 !important;
        border: 2px solid #00FF41 !important;
        border-radius: 4px !important;
        font-family: 'Consolas', 'Courier New', monospace !important;
        box-shadow: 0 0 15px #00FF41, inset 0 0 10px rgba(0, 255, 65, 0.2) !important;
        min-height: 100px;
        position: relative;
    }
    
    /* Header/Summary Styling */
    div[data-testid="stStatusWidget"] > div:first-child {
        background-color: #000000 !important;
    }
    
    div[data-testid="stStatusWidget"] label {
        color: #00FF41 !important; /* Neon Green Title */
        font-weight: bold !important;
        font-family: 'Consolas', 'Courier New', monospace !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Content Styling */
    div[data-testid="stStatusWidget"] div[data-testid="stMarkdownContainer"] p {
        color: #00FFFF !important; /* Cyan Content */
        font-family: 'Consolas', 'Courier New', monospace !important;
        font-size: 14px !important;
    }
    
    /* Blinking Cursor Animation */
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0; }
        100% { opacity: 1; }
    }
    div[data-testid="stStatusWidget"]::after {
        content: " █";
        color: #00FF41;
        animation: blink 1s infinite;
        font-weight: bold;
        position: absolute;
        bottom: 10px;
        right: 10px;
    }

    /* Improve st.info visibility */
    div[data-testid="stAlert"] {
        color: #FFFFFF;
        border: 1px solid #1DB954;
    }
    div[data-testid="stAlert"] p {
        color: #FFFFFF !important;
    }
    /* Remove border for alerts in sidebar */
    section[data-testid="stSidebar"] div[data-testid="stAlert"] {
        border: none;
    }
    
    /* 7. Layout Fixes */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
        max-width: 1200px;
    }
    header[data-testid="stHeader"] {
        display: none;
    }
    
    /* 8. RWD Optimization */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem !important;
        }
        iframe { width: 100% !important; }
    }
</style>
""", unsafe_allow_html=True)


def load_data():
    if not os.path.exists("data/songs.csv"):
        st.error("找不到資料。請先執行 scripts/download_data.py。")
        return None
    return pd.read_csv("data/songs.csv")


@st.cache_data
def load_pca_data():
    """Loads precomputed PCA projections if available."""
    path = "data/df_pca.csv"
    if not os.path.exists(path):
        st.warning("找不到 df_pca.csv，將改以即時計算 PCA。")
        return None
    return pd.read_csv(path)


@st.cache_data
def load_personas():
    personas = {}
    history_dir = "persona_listening_histories"
    if os.path.exists(history_dir):
        for f in os.listdir(history_dir):
            if f.endswith(".json"):
                name = f.replace("_history.json", "").replace("_", " ").title()
                with open(os.path.join(history_dir, f), "r", encoding="utf-8") as file:
                    personas[name] = json.load(file)
    return personas


@st.cache_data
def load_persona_summaries():
    """Loads persona summaries from the JSON file."""
    path = "data/persona_listening_histories/persona_listening_histories/persona_summaries.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_resource
def load_faiss_resources():
    """Loads FAISS index, metadata, and embedding model."""
    print("Loading FAISS resources...")

    # 1. Load Index (with Windows workaround)
    if not os.path.exists(INDEX_PATH):
        st.error(f"Index not found at {INDEX_PATH}")
        return None, None, None

    try:
        # Workaround: FAISS C++ read_index fails with non-ASCII paths on Windows.
        # Copy to a temp file with ASCII path, read it, then delete temp.
        fd, temp_path = tempfile.mkstemp(suffix=".index")
        os.close(fd)
        shutil.copy2(INDEX_PATH, temp_path)

        index = faiss.read_index(temp_path)

    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        return None, None, None
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

    # 2. Load Metadata
    if not os.path.exists(META_PATH):
        st.error(f"Metadata not found at {META_PATH}")
        return None, None, None

    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    # 3. Load Model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    return index, metadata, model


@st.cache_data
def load_precomputed_data():
    """Loads precomputed similarity data (cached)."""
    return utils.load_precomputed_sims()


def spotify_embed(track_id, height=80):
    """Embeds a Spotify player for the given track_id."""
    url = f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0"
    components.iframe(url, height=height)


def render_landing_page(personas, persona_summaries):
    """Renders the initial persona selection landing page."""
    st.title("Choose Your Persona...")
    # st.markdown("### 請選擇一個角色以開始體驗 (Select a persona to start)")
    # st.divider()

    # Custom CSS for Landing Page Cards
    st.markdown("""
    <style>
        /* Target the column itself using the marker */
        div[data-testid="column"]:has(.persona-card-marker) {
            background-color: #181818;
            border: 1px solid #282828;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            transition: all 0.3s ease;
            height: 100%;
            min-height: 250px; /* Ensure consistent height */
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        div[data-testid="column"]:has(.persona-card-marker):hover {
            transform: translateY(-10px);
            box-shadow: 0 10px 30px rgba(29, 185, 84, 0.3);
            border-color: #1DB954;
        }
        
        .persona-title {
            color: #1DB954;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 12px;
        }
        .persona-desc {
            color: #B3B3B3;
            font-size: 16px;
            margin-bottom: 24px;
            flex-grow: 1;
            text-align: left;
        }
        /* Center the button */
        div[data-testid="column"]:has(.persona-card-marker) .stButton {
            width: 100%;
            display: flex;
            justify-content: center;
            margin-top: auto;
        }
    </style>
    """, unsafe_allow_html=True)

    # PERSONA_DESCRIPTIONS (Removed hardcoded dict)

    # Convert to list for easier indexing
    persona_items = list(personas.items())

    for i in range(0, len(persona_items), 2):
        cols = st.columns(2)
        batch = persona_items[i:i+2]

        for idx, (name, history) in enumerate(batch):
            with cols[idx]:
                # Inject marker for CSS targeting
                st.markdown(
                    '<div class="persona-card-marker" style="display:none;"></div>', unsafe_allow_html=True)

                desc = persona_summaries.get(name, "一位熱愛音樂的用戶。")

                # Truncate to 50 chars if longer
                if len(desc) > 40:
                    desc = desc[:40] + "..."

                # Replace newlines with <br> for HTML rendering
                desc = desc.replace("\n", "<br>")

                st.markdown(
                    f'<div class="persona-title">{name}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="persona-desc">{desc}</div>', unsafe_allow_html=True)

                if st.button(f"✔", key=f"select_{name}", use_container_width=True):
                    st.session_state.selected_persona = name
                    st.rerun()


def render_main_app(df_songs, df_pca, personas, persona_summaries, precomputed_data, faiss_index, faiss_metadata, embedding_model):
    """Renders the main application interface."""
    # Sidebar: Persona Selection
    with st.sidebar:
        # st.title("🎧 Spotify Agentic RAG")
        st.header("用戶角色 (User Persona)")

        # Default to the selected persona from landing page
        default_index = list(personas.keys()).index(
            st.session_state.selected_persona)
        selected_persona_name = st.selectbox(
            "選擇角色", list(personas.keys()), index=default_index)

        # Update session state if changed via sidebar
        if selected_persona_name != st.session_state.selected_persona:
            st.session_state.selected_persona = selected_persona_name
            # Clear caches that depend on Persona
            if 'llm_results' in st.session_state:
                st.session_state.llm_results = {}
            if 'viz_cache' in st.session_state:
                st.session_state.viz_cache = {}
            # Step 1 and Step 2 caches are pure Song-Song similarity, so they can persist?
            # Actually Step 2 might be pure song embedding query. Yes.
            # But to be safe and clean, let's clear everything or keep purely song-based stuff?
            # Keep song-based (Step1/2) for performance if user switches persona back and forth for same song.
            st.rerun()

        # Persona Descriptions

        # Show Description
        desc = persona_summaries.get(selected_persona_name, "一位熱愛音樂的用戶。")
        # Ensure it renders as markdown lists correctly if it contains bullets
        # st.info interprets markdown, but sometimes needs double newlines for strict markdown
        # However, for pure display, let's keep it as is, usually st.info handles \n as a line break if it's md.
        # But if user says it doesn't work, maybe it was the landing page.
        # Let's assume landing page was the main issue (HTML div).
        # But just in case, let's fix sidebar too if needed.
        # Actually, if the JSON has "- ...", markdown needs a newline before the list starts if it follows text.
        # The JSON values start with "- ", so it should be fine as a list.
        # Let's just fix the landing page first as that is definitely broken (HTML ignoring \n).
        st.info(f"📝 **角色描述:**\n{desc}")

        # Show mini profile
        history = personas[selected_persona_name]
        traits = utils.analyze_persona(history)
        st.caption(f"喜好風格: {', '.join(traits['top_genres'][:2])}")
        st.caption(f"最愛藝人: {traits['top_artists'][0]}")

        st.divider()

        # Listening History (Embeds)
        st.subheader("📜 最近收聽 (History)")
        for track in history[:5]:  # Show top 5 recent
            spotify_embed(track['track_id'], height=80)

        st.divider()

        if st.button("⏻"):
            st.session_state.selected_persona = None
            st.session_state.selected_song = None
            st.session_state.analysis_done = False
            st.rerun()

    # Main Content

    # Section 1: Music Library (Grid)
    st.title("🎧 Spotify Agentic RAG")

    # Search & Pagination State
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1

    def reset_page():
        st.session_state.current_page = 1

    # Search Bar
    search_col, _ = st.columns([2, 1])
    with search_col:
        search_query = st.text_input(
            "Search", placeholder="搜尋歌曲或藝人 (Search)", on_change=reset_page, label_visibility="collapsed")

    # Filter Logic
    if search_query:
        filtered_songs = df_songs[
            df_songs['track_name'].str.contains(search_query, case=False) |
            df_songs['artists'].str.contains(search_query, case=False)
        ]
    else:
        filtered_songs = df_songs

    # Pagination Logic
    ITEMS_PER_PAGE = 12
    total_songs = len(filtered_songs)
    total_pages = max(1, (total_songs + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)

    # Ensure current page is valid
    if st.session_state.current_page > total_pages:
        st.session_state.current_page = total_pages

    start_idx = (st.session_state.current_page - 1) * ITEMS_PER_PAGE
    end_idx = start_idx + ITEMS_PER_PAGE
    display_songs = filtered_songs.iloc[start_idx:end_idx]

    # Grid Display
    if display_songs.empty:
        st.info("找不到符合的歌曲。")
    else:
        for i in range(0, len(display_songs), 4):
            cols = st.columns(4)
            batch = display_songs.iloc[i:i+4]
            for idx, (_, row) in enumerate(batch.iterrows()):
                with cols[idx]:
                    with st.container():
                        # Embed Player
                        spotify_embed(row['track_id'], height=80)
                        # Selection Button
                        # Use track_id for unique key across pages
                        if st.button("▶", key=f"btn_{row['track_id']}"):
                            st.session_state.selected_song = row
                            st.session_state.analysis_done = True  # Auto-start analysis
                            st.session_state.scroll_to_now_playing = True # Trigger scroll
                            st.rerun()

        # Pagination Controls
        st.write("")
        col_prev, col_info, col_next = st.columns([1, 10, 1])
        with col_prev:
            if st.button("❮", disabled=st.session_state.current_page == 1):
                st.session_state.current_page -= 1
                st.rerun()
        with col_info:
            st.markdown(
                f"<div style='text-align: center; padding-top: 10px;'>Page {st.session_state.current_page} of {total_pages}</div>", unsafe_allow_html=True)
        with col_next:
            if st.button("❯", disabled=st.session_state.current_page == total_pages):
                st.session_state.current_page += 1
                st.rerun()

    st.divider()

    # Section 2: Now Playing & Analysis
    if st.session_state.selected_song is not None:
        # Auto-scroll anchor
        st.markdown('<div id="now-playing-section"></div>',
                    unsafe_allow_html=True)

        selected_song = st.session_state.selected_song

        # Re-fetch persona traits for the current selection (in case it changed)
        selected_persona_name = st.session_state.selected_persona
        history = personas[selected_persona_name]
        
        # Cache persona traits
        if 'persona_traits_cache' not in st.session_state:
            st.session_state.persona_traits_cache = {}
            
        if selected_persona_name in st.session_state.persona_traits_cache:
            traits = st.session_state.persona_traits_cache[selected_persona_name]
        else:
            traits = utils.analyze_persona(history)
            st.session_state.persona_traits_cache[selected_persona_name] = traits

        st.title("🎵 Now Playing")

        # Inject Auto-scroll JS if analysis just started (One-time trigger)
        if st.session_state.get('scroll_to_now_playing', False):
            components.html(
                f"""
                <script>
                    window.parent.document.getElementById('now-playing-section').scrollIntoView({{behavior: 'smooth'}});
                </script>
                """,
                height=0
            )
            # Reset flag so it doesn't scroll again on next interaction
            st.session_state.scroll_to_now_playing = False

        col_hero_1, col_hero_2 = st.columns([3, 1])
        with col_hero_1:
            spotify_embed(selected_song['track_id'], height=152)

        # Analysis Section (Auto-triggered)
        if st.session_state.analysis_done:
            st.divider()
            st.title("🧠 Agentic Thinking")

            # Step 1: Pre-computed Candidates
            current_song_id = selected_song['track_id']
            if 'step1_cache' not in st.session_state:
                st.session_state.step1_cache = {}
            
            is_step1_cached = current_song_id in st.session_state.step1_cache
            step1_label = "[STEP 1: Candidates from Pre-computed Similar Items: CACHED]" if is_step1_cached else "[STEP 1: Candidates from Pre-computed Similar Items: PROCESSING...]"
            
            # Auto-expand only if processing (not cached)
            with st.status(step1_label, expanded=not is_step1_cached) as status:
                step1_candidates = []
                
                if is_step1_cached:
                    step1_candidates = st.session_state.step1_cache[current_song_id]
                    st.markdown(
                        f"<span style='font-family: Consolas, monospace;'>&gt;&gt; Cached results found. Loaded <span style='color:#00FF41'>[{len(step1_candidates)}]</span> candidates.</span>", unsafe_allow_html=True)
                else:
                    time.sleep(0.5)
                    st.markdown(
                        "<span style='font-family: Consolas, monospace;'>&gt;&gt; Accessing <span style='color:#00FF41'>[Co-occurrence_DB]</span> for high-confidence pairs...</span>", unsafe_allow_html=True)

                    # Retrieve pre-computed candidates
                    raw_candidates = utils.get_precomputed_candidates(
                        selected_song['track_id'], precomputed_data)

                    # Filter out candidates that match the current song's Name AND Artist (ignoring case)
                    q_name = selected_song['track_name'].strip().lower()
                    q_artist = selected_song['artists'].strip().lower()

                    step1_candidates = [
                        c for c in raw_candidates
                        if not (c['track_name'].strip().lower() == q_name and c['artists'].strip().lower() == q_artist)
                    ]
                    
                    # Cache the results
                    st.session_state.step1_cache[current_song_id] = step1_candidates

                    if step1_candidates:
                        st.markdown(
                            f"<span style='font-family: Consolas, monospace;'>&gt;&gt; Retrieved <span style='color:#00FF41'>[{len(step1_candidates)}]</span> pre-computed candidates based on audio similarity.</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f"<span style='font-family: Consolas, monospace;'>&gt;&gt; No pre-computed candidates found for this track. Using fallback retrieval.</span>", unsafe_allow_html=True)

                # Render Results (Always)
                if step1_candidates:
                    for i, cand in enumerate(step1_candidates):
                        score = cand.get('similarity_score', 0)
                        st.markdown(
                            f"<span style='font-family: Consolas, monospace; margin-left: 20px;'>* #{i+1} <span style='color:#00FFFF'>{cand['track_name']}</span> by {cand['artists']} (Score: {score:.4f})</span>", unsafe_allow_html=True)

                status.update(
                    label="[STEP 1: Candidates from Pre-computed Similar Items: OK]", state="complete", expanded=False)

            # Step 2: Semantic Search (FAISS)
            # Step 2: Candidates from FAISS Semantic Search
            if 'step2_cache' not in st.session_state:
                st.session_state.step2_cache = {}
            
            is_step2_cached = current_song_id in st.session_state.step2_cache

            step2_label = "[STEP 2: Candidates from FAISS Semantic Search: CACHED]" if is_step2_cached else "[STEP 2: Candidates from FAISS Semantic Search: PROCESSING...]"

            # Auto-expand only if processing (not cached)
            with st.status(step2_label, expanded=not is_step2_cached) as status:
                step2_candidates = []
                
                if is_step2_cached:
                    step2_candidates = st.session_state.step2_cache[current_song_id]
                    st.markdown(
                        f"<span style='font-family: Consolas, monospace;'>&gt;&gt; Cached results found. Loaded <span style='color:#00FF41'>[{len(step2_candidates)}]</span> candidates.</span>", unsafe_allow_html=True)
                
                elif faiss_index and faiss_metadata and embedding_model:
                    time.sleep(0.5)
                    st.markdown(
                        "<span style='font-family: Consolas, monospace;'>&gt;&gt; Converting query to vector embedding... <span style='color:#00FF41'>[All-MiniLM-L6-v2]</span></span>", unsafe_allow_html=True)
                    st.markdown(
                        "<span style='font-family: Consolas, monospace;'>&gt;&gt; Querying <span style='color:#00FF41'>[FAISS]</span> for semantic similarity...</span>", unsafe_allow_html=True)

                    # Construct query: Track Name + Artist + Genre
                    query = f"{selected_song['track_name']} {selected_song['artists']} {selected_song['track_genre']}"

                    # Embed
                    query_vec = embedding_model.encode([query])
                    query_vec = np.array(query_vec).astype('float32')
                    faiss.normalize_L2(query_vec)

                    # Search
                    D, I = faiss_index.search(query_vec, k=10)

                    # Process Results
                    step2_candidates = []
                    found_indices = I[0]
                    scores = D[0]

                    # Deduplication set: Current song + Step 1 Candidates
                    existing_ids = set()
                    existing_keys = set()
                    
                    def normalize_key(name, artist):
                        return (name.strip().lower(), artist.strip().lower())

                    # Add current song
                    existing_ids.add(selected_song['track_id'])
                    existing_keys.add(normalize_key(selected_song['track_name'], selected_song['artists']))

                    if 'step1_candidates' in locals():
                        for c in step1_candidates:
                            existing_ids.add(c['track_id'])
                            existing_keys.add(normalize_key(c['track_name'], c['artists']))

                    for score, idx in zip(scores, found_indices):
                        if idx == -1:
                            continue
                        meta = faiss_metadata[idx]
                        
                        # Filter duplicates
                        current_key = normalize_key(meta['track_name'], meta['artists'])
                        if meta['track_id'] in existing_ids or current_key in existing_keys:
                            continue
                            
                        # Add to existing to prevent duplicates within Step 2 itself
                        existing_ids.add(meta['track_id'])
                        existing_keys.add(current_key)

                        step2_candidates.append({
                            'track_id': meta['track_id'],
                            'track_name': meta['track_name'],
                            'artists': meta['artists'],
                            'score': float(score),
                            'rag_doc': meta['rag_doc']
                        })
                    
                    # Cache the results
                    st.session_state.step2_cache[current_song_id] = step2_candidates

                    st.markdown(
                        f"<span style='font-family: Consolas, monospace;'>&gt;&gt; Vector search complete. Retrieved <span style='color:#00FF41'>[{len(step2_candidates)}]</span> semantic candidates.</span>", unsafe_allow_html=True)



                else:
                    st.error("FAISS resources not loaded.")
                    step2_candidates = []

                # Render Results (Always)
                for i, cand in enumerate(step2_candidates):
                    score = cand['score']
                    expl_desc = cand['rag_doc'][:70] + \
                        "..." if len(cand['rag_doc']
                                        ) > 60 else cand['rag_doc']
                    st.markdown(
                        f"<span style='font-family: Consolas, monospace; margin-left: 20px;'>* #{i+1} <span style='color:#00FFFF'>{cand['track_name']}</span> by {cand['artists']} (Score: {score:.4f}) <br>  <span style='color:#AAAAAA; font-size: 0.8em; margin-left: 20px;'>Desc: {expl_desc}</span></span>", unsafe_allow_html=True)

                # Keep expanded so user sees the list
                # Update status
                # If was cached, stay collapsed. If just processed, auto-expand (or user can close).
                # Actually, better UX: always collapse when "complete" to avoid taking up huge space on re-runs
                # But initial request said "keep expanded so user sees list". 
                # Compromise: expand if we just ran it (not cached), collapse if we just loaded cache.
                status.update(
                    label="[STEP 2: Candidates from FAISS Semantic Search: OK]", state="complete", expanded=not is_step2_cached)

            # Step 3: Re-ranking & Filtering
            current_song_id = selected_song['track_id']
            is_cached = current_song_id in st.session_state.get('llm_results', {})
            step3_label = "[STEP 3: Re-ranked and Filtered Recommendations: CACHED]" if is_cached else "[STEP 3: Re-ranked and Filtered Recommendations: PROCESSING...]"
            
            # Auto-expand only if processing (not cached)
            with st.status(step3_label, expanded=not is_cached) as status:
                time.sleep(0.5)
                st.markdown(
                    f"<span style='font-family: Consolas, monospace;'>&gt;&gt; Loading persona profile: <span style='color:#00FF41'>['{selected_persona_name}']</span></span>", unsafe_allow_html=True)
                st.markdown(
                    f"<span style='font-family: Consolas, monospace;'>&gt;&gt; Applying filters: Name=<span style='color:#00FF41'>[{selected_song['track_name']}]</span></span>", unsafe_allow_html=True)

                # Combine and Rerank (LLM-based)
                if 'step1_candidates' not in locals():
                    step1_candidates = []

                # Prepare Parallel Execution
                # Prepare Parallel Execution
                import concurrent.futures
                from streamlit.runtime.scriptrunner import add_script_run_ctx
                
                # Helper function for parallel execution
                def fetch_recommendations(model_name, provider, col_name):
                    return utils.llm_rerank_candidates(
                        df_songs, step1_candidates, step2_candidates, selected_song, traits, top_k=20,
                        model_name=model_name, provider=provider
                    )

                # LLM Comparison Columns
                col_gpt, col_gemini, col_grok = st.columns(3)
                
                # Display Headers immediately
                with col_gpt:
                    st.markdown("### 🤖 GPT-4o")
                    st.caption("(Active via OpenAI)")
                    st.markdown(f"<span style='font-family: Consolas, monospace;'>&gt;&gt; Sending candidates to <span style='color:#FF00FF'>GPT-4o</span>...</span>", unsafe_allow_html=True)
                with col_gemini:
                    st.markdown("### ⚡ Gemini 2.0 Flash")
                    st.caption("(Active via OpenRouter)")
                    st.markdown(f"<span style='font-family: Consolas, monospace;'>&gt;&gt; Sending candidates to <span style='color:#FFD700'>Gemini 2.0 Flash</span>...</span>", unsafe_allow_html=True)
                with col_grok:
                    st.markdown("### 🚀 Grok 4.1 Fast")
                    st.caption("(Active via OpenRouter)")
                    st.markdown(f"<span style='font-family: Consolas, monospace;'>&gt;&gt; Sending candidates to <span style='color:#00BFFF'>Grok 4.1 Fast</span>...</span>", unsafe_allow_html=True)
                
                # Execute in parallel or load from cache
                
                if 'llm_results' not in st.session_state:
                    st.session_state.llm_results = {}
                
                # Check cache
                cached_res = st.session_state.llm_results.get(current_song_id)
                
                if cached_res:
                    st.success("✅ Loaded cached recommendations.")
                    gpt_res = cached_res['gpt']
                    gemini_res = cached_res['gemini']
                    grok_res = cached_res['grok']
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                        future_gpt = executor.submit(add_script_run_ctx(fetch_recommendations), "gpt-4o", "openai", "GPT-4o")
                        future_gemini = executor.submit(add_script_run_ctx(fetch_recommendations), "google/gemini-2.0-flash-001", "openrouter", "Gemini")
                        future_grok = executor.submit(add_script_run_ctx(fetch_recommendations), "x-ai/grok-4.1-fast", "openrouter", "Grok")
                        
                        # Wait for results
                        gpt_res = future_gpt.result()
                        gemini_res = future_gemini.result()
                        grok_res = future_grok.result()
                    
                    # Cache results
                    st.session_state.llm_results[current_song_id] = {
                        'gpt': gpt_res,
                        'gemini': gemini_res,
                        'grok': grok_res
                    }

                # Render GPT Results
                with col_gpt:
                    final_recs, llm_explanation = gpt_res
                    if "Rule-based fallback" in llm_explanation or "LLM Rerank Failed" in llm_explanation:
                        st.warning(f"⚠️ {llm_explanation}")

                    st.markdown(
                        f"<span style='font-family: Consolas, monospace;'>&gt;&gt; Re-ranking complete. <span style='color:#00FF41'>[{len(final_recs)}]</span> candidates.</span>", unsafe_allow_html=True)

                    for i, (_, row) in enumerate(final_recs.iterrows()):
                        reason = row.get('reason') if pd.notna(
                            row.get('reason')) else utils.generate_explanation(row, selected_song, traits)
                        st.markdown(
                            f"<span style='font-family: Consolas, monospace; margin-left: 20px;'>* #{i+1} <span style='color:#00FFFF'>{row['track_name']}</span> by {row['artists']} <br> <span style='color:#AAAAAA; font-size: 0.8em; margin-left: 20px;'> Reason: {reason}</span></span>", unsafe_allow_html=True)

                # Render Gemini Results
                with col_gemini:
                    final_recs_gemini, gemini_explanation = gemini_res
                    if "Rule-based fallback" in gemini_explanation or "LLM Rerank Failed" in gemini_explanation:
                        st.warning(f"⚠️ {gemini_explanation}")

                    st.markdown(
                        f"<span style='font-family: Consolas, monospace;'>&gt;&gt; Re-ranking complete. <span style='color:#00FF41'>[{len(final_recs_gemini)}]</span> candidates.</span>", unsafe_allow_html=True)

                    for i, (_, row) in enumerate(final_recs_gemini.iterrows()):
                        reason = row.get('reason') if pd.notna(row.get('reason')) else "AI Recommended"
                        st.markdown(
                            f"<span style='font-family: Consolas, monospace; margin-left: 20px;'>* #{i+1} <span style='color:#00FFFF'>{row['track_name']}</span> by {row['artists']} <br> <span style='color:#AAAAAA; font-size: 0.8em; margin-left: 20px;'> Reason: {reason}</span></span>", unsafe_allow_html=True)

                # Render Grok Results
                with col_grok:
                    final_recs_grok, grok_explanation = grok_res
                    if "Rule-based fallback" in grok_explanation or "LLM Rerank Failed" in grok_explanation:
                        st.warning(f"⚠️ {grok_explanation}")
                    
                    st.markdown(
                        f"<span style='font-family: Consolas, monospace;'>&gt;&gt; Re-ranking complete. <span style='color:#00FF41'>[{len(final_recs_grok)}]</span> candidates.</span>", unsafe_allow_html=True)
                    
                    for i, (_, row) in enumerate(final_recs_grok.iterrows()):
                        reason = row.get('reason') if pd.notna(row.get('reason')) else "AI Recommended"
                        st.markdown(
                            f"<span style='font-family: Consolas, monospace; margin-left: 20px;'>* #{i+1} <span style='color:#00FFFF'>{row['track_name']}</span> by {row['artists']} <br> <span style='color:#AAAAAA; font-size: 0.8em; margin-left: 20px;'> Reason: {reason}</span></span>", unsafe_allow_html=True)

                status.update(
                    label="[STEP 3: Re-ranked and Filtered Recommendations: OK]", state="complete", expanded=False)

            # Step 4: LLM Explanation
            with st.status("[STEP 4: LLM's Overall Explanation: PROCESSING...]", expanded=True) as status:
                st.markdown(
                    "<span style='font-family: Consolas, monospace;'>&gt;&gt; Synthesizing reasoning context... <span style='color:#00FF41'>[Done]</span></span>", unsafe_allow_html=True)
                st.markdown(
                    "<span style='font-family: Consolas, monospace;'>&gt;&gt; Generating natural language explanations via <span style='color:#00FF41'>[LLM]</span>... <span style='color:#00FF41'>[Done]</span></span>", unsafe_allow_html=True)
                
                exp_col1, exp_col2, exp_col3 = st.columns(3)
                
                with exp_col1:
                    st.markdown("**GPT-4o Reasoning**")
                    st.info(llm_explanation)
                
                with exp_col2:
                    st.markdown("**Gemini 2.0 Flash Reasoning**")
                    st.info(gemini_explanation)
                
                with exp_col3:
                    st.markdown("**Grok 4.1 Fast Reasoning**")
                    st.info(grok_explanation)

                status.update(
                    label="[STEP 4: LLM's Overall Explanation: OK]", state="complete", expanded=True)

            # Step 3: Generation
            st.divider()
            st.title("🎧 Recommended for You")

            rec_count = 0
            display_limit = 6

            # Iterate in chunks of 3 for grid layout
            rec_tabs = st.tabs(["GPT-4o", "Gemini 2.0 Flash", "Grok 4.1 Fast"])
            
            def display_recommendations(recs_df):
                if recs_df is None or recs_df.empty:
                    st.info("No recommendations available.")
                    return
                for i in range(0, min(len(recs_df), display_limit), 3):
                    cols = st.columns(3)
                    chunk_df = recs_df.iloc[i: i+3]

                    for j, (_, row) in enumerate(chunk_df.iterrows()):
                        with cols[j]:
                            with st.container():
                                spotify_embed(row['track_id'], height=352)
                                # Use LLM reason if available
                                reason = row.get('reason') if pd.notna(
                                    row.get('reason')) else utils.generate_explanation(row, selected_song, traits)
                                st.info(f"{reason}")

            with rec_tabs[0]:
                display_recommendations(final_recs)
            with rec_tabs[1]:
                display_recommendations(final_recs_gemini)
            with rec_tabs[2]:
                display_recommendations(final_recs_grok)

            # 4. Visualization (PCA)
            st.divider()
            st.title("📊 Visualization")
            
            # Initialize viz cache
            if 'viz_cache' not in st.session_state:
                st.session_state.viz_cache = {}
            
            current_song_id = selected_song['track_id']
            
            # Function to read HTML content safely
            def get_html_content(path):
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        return f.read()
                return None

            with st.spinner("Generating Embedding Space Visualization..."):
                try:
                    viz_tabs = st.tabs(["GPT-4o", "Gemini 2.0 Flash", "Grok 4.1 Fast"])
                    
                    def render_viz(recs_df, key_suffix):
                        if recs_df is None or recs_df.empty:
                            st.warning("No data for visualization.")
                            return
                        
                        # Unique cache key for this specific plot
                        cache_key = f"{current_song_id}_{key_suffix}"
                        html_output_path = os.path.join(BASE_DIR, f"pca_plot_{key_suffix}.html")
                        
                        # Check if we have cached HTML content or need to regenerate
                        # We use memory cache to avoid unnecessary file I/O checks if logic hasn't changed,
                        # but we still need to write the file for the components.html to read it...? 
                        # Actually components.html takes a string OR safe html. 
                        # We used components.html(html_content).
                        
                        # FIX: Check if cached in session state
                        if cache_key in st.session_state.viz_cache:
                            # Use cached HTML
                            html_content = st.session_state.viz_cache[cache_key]
                        else:
                            # Generate
                            viz_recs = recs_df.head(6)
                            fig = utils.plot_pca_visualization(
                                df_songs,
                                selected_song,
                                viz_recs,
                                user_history=history,
                                step1_cands=step1_candidates,
                                step2_cands=step2_candidates,
                                df_pca=df_pca
                            )

                            # Save to file (optional mostly for debugging or if component specifically needed it, 
                            # but we can pass string directly to components.html is safer and easier)
                            # Let's keep file writing as backup but primarily use memory string.
                            pio.write_html(fig, html_output_path)
                            
                            with open(html_output_path, "r", encoding="utf-8") as f:
                                html_content = f.read()
                            
                            # Update Cache
                            st.session_state.viz_cache[cache_key] = html_content

                        # Render
                        components.html(html_content, height=750, scrolling=True)

                    with viz_tabs[0]:
                        render_viz(final_recs, "gpt")
                    with viz_tabs[1]:
                        render_viz(final_recs_gemini, "gemini")
                    with viz_tabs[2]:
                        render_viz(final_recs_grok, "grok")

                except Exception as e:
                    import traceback
                    st.error(f"Visualization Error: {e}")
                    st.code(traceback.format_exc())

            # 5. Voting System
            st.divider()
            
            with st.expander("🗳️ 我要投票"):
                with st.container(border=True):
                    st.markdown("### 📝 模型評選投票")
                    st.info("""
                    **請依照以下流程進行評選：**
                    1. 🎧 **聆聽** 左方「最近收聽紀錄」中的歌曲，及當前播放歌曲。
                    2. 👁️ **閱讀** [STEP 4] 三個模型的「推薦理由」。
                    3. 🎵 **試聽** [Recommended for You] 三個模型的「推薦歌單」。
                    4. 👇 **選擇** 下方您覺得表現最好的模型並送出。
                    """)

                    with st.form("vote_form"):
                        st.subheader("1. 最佳推薦理由 (Reasoning)")
                        reason_vote = st.radio(
                            "您覺得哪個模型給的推薦理由最符合您的口味？",
                            ["GPT-4o", "Gemini 2.0 Flash", "Grok 4.1 Fast"],
                            horizontal=True,
                            key="reason_vote_radio"
                        )
                        
                        st.divider()
                        
                        st.subheader("2. 最佳推薦歌曲 (Songs)")
                        song_vote = st.radio(
                            "您覺得哪個模型的推薦歌單最符合您的口味？",
                            ["GPT-4o", "Gemini 2.0 Flash", "Grok 4.1 Fast"],
                            horizontal=True,
                            key="song_vote_radio"
                        )

                        # Right-align the submit button
                        c1, c2 = st.columns([5, 1])
                        with c2:
                            submitted = st.form_submit_button("➤", use_container_width=True)
                        
                        if submitted:
                            import datetime
                            vote_data = {
                                "timestamp": datetime.datetime.now().isoformat(),
                                "persona": selected_persona_name,
                                # "selected_song_id": selected_song['track_id'],
                                "selected_song_name": selected_song['track_name'],
                                "vote_reason": reason_vote,
                                "vote_song": song_vote
                            }
                            vote_success = utils.save_vote(vote_data)
                            if vote_success:
                                st.success("🎉 投票成功！感謝您的回饋。")
                                if "gsheets" not in st.secrets.get("connections", {}):
                                    st.warning("⚠️ 注意：目前僅儲存於暫存區 (CSV)，重啟即遺失。請設定 Google Sheets 以永久保存。")
                                
                                # Auto-expand results
                                st.session_state.vote_expanded = True
                            else:
                                st.error("❌ 投票儲存失敗。")

                            components.html("""
                                <script>
                                    window.parent.document.querySelector('section.main').scrollTo(0, 0);
                                </script>
                            """, height=0)

            # Check if we should expand (default False)
            expand_results = st.session_state.get('vote_expanded', False)
            
            with st.expander("查看投票統計結果", expanded=expand_results):
                # Reset flag so it doesn't force open on next reload (unless voted again)
                if expand_results:
                    st.session_state.vote_expanded = False
                    
                # Debug Check
                # st.write("Secrets keys:", st.secrets.keys())
                # if "connections" in st.secrets:
                #    st.write("Connections keys:", st.secrets["connections"].keys())

                # Check connection status
                if "gsheets" in st.secrets.get("connections", {}):
                    st.caption("🟢 已連線至 Google Sheets (雲端同步中)")
                else:
                    st.caption("🔴 未連線至 Google Sheets (僅顯示暫存資料)")

                df_votes = utils.load_votes()
                if df_votes is not None and not df_votes.empty:
                    st.markdown("#### 推薦理由 (Reasoning) 得票數")
                    st.bar_chart(df_votes['vote_reason'].value_counts())
                    
                    st.markdown("#### 推薦歌單 (Songs) 得票數")
                    st.bar_chart(df_votes['vote_song'].value_counts())
                    
                    st.markdown("#### 詳細投票紀錄")
                    st.dataframe(df_votes.tail(10))
                else:
                    st.info("目前尚無投票資料。")



def main():
    # Initialize Session State
    if 'selected_song' not in st.session_state:
        st.session_state.selected_song = None
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'selected_persona' not in st.session_state:
        st.session_state.selected_persona = None
    if 'scroll_to_now_playing' not in st.session_state:
        st.session_state.scroll_to_now_playing = False

    # Load Data
    df_songs = load_data()
    personas = load_personas()
    persona_summaries = load_persona_summaries()
    df_pca = load_pca_data()

    if df_songs is None or not personas:
        st.warning("請確保資料已正確設定。")
        return

    index, metadata, model = load_faiss_resources()
    precomputed_data = load_precomputed_data()

    if st.session_state.selected_persona is None:
        render_landing_page(personas, persona_summaries)
    else:
        render_main_app(df_songs, df_pca, personas, persona_summaries,
                        precomputed_data, index, metadata, model)


if __name__ == "__main__":
    main()
