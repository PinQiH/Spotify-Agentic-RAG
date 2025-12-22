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


def load_persona_summaries():
    """Loads persona summaries from the JSON file."""
    path = "data/persona_listening_histories/persona_listening_histories/persona_summaries.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

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
        traits = utils.analyze_persona(history)

        st.title("🎵 Now Playing")

        # Inject Auto-scroll JS if analysis just started
        if st.session_state.analysis_done:
            components.html(
                f"""
                <script>
                    // {time.time()}
                    window.parent.document.getElementById('now-playing-section').scrollIntoView({{behavior: 'smooth'}});
                </script>
                """,
                height=0
            )

        col_hero_1, col_hero_2 = st.columns([3, 1])
        with col_hero_1:
            spotify_embed(selected_song['track_id'], height=152)

        # Analysis Section (Auto-triggered)
        if st.session_state.analysis_done:
            st.divider()
            st.title("🧠 Agentic Thinking")

            # Step 1: Pre-computed Candidates
            with st.status("[STEP 1: Candidates from Pre-computed Similar Items: PROCESSING...]", expanded=True) as status:
                time.sleep(0.5)
                st.markdown(
                    "<span style='font-family: Consolas, monospace;'>&gt;&gt; Accessing <span style='color:#00FF41'>[Co-occurrence_DB]</span> for high-confidence pairs...</span>", unsafe_allow_html=True)

                # Retrieve pre-computed candidates
                step1_candidates = utils.get_precomputed_candidates(
                    selected_song['track_id'], precomputed_data)

                # Filter out candidates that match the current song's Name AND Artist (ignoring case)
                # This covers cases where different IDs represent the same song
                q_name = selected_song['track_name'].strip().lower()
                q_artist = selected_song['artists'].strip().lower()

                step1_candidates = [
                    c for c in step1_candidates
                    if not (c['track_name'].strip().lower() == q_name and c['artists'].strip().lower() == q_artist)
                ]

                if step1_candidates:
                    st.markdown(
                        f"<span style='font-family: Consolas, monospace;'>&gt;&gt; Retrieved <span style='color:#00FF41'>[{len(step1_candidates)}]</span> pre-computed candidates based on audio similarity.</span>", unsafe_allow_html=True)

                    # Display all candidates
                    for i, cand in enumerate(step1_candidates):
                        score = cand.get('similarity_score', 0)
                        st.markdown(
                            f"<span style='font-family: Consolas, monospace; margin-left: 20px;'>* #{i+1} <span style='color:#00FFFF'>{cand['track_name']}</span> by {cand['artists']} (Score: {score:.4f})</span>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<span style='font-family: Consolas, monospace;'>&gt;&gt; No pre-computed candidates found for this track. Using fallback retrieval.</span>", unsafe_allow_html=True)

                status.update(
                    label="[STEP 1: Candidates from Pre-computed Similar Items: OK]", state="complete", expanded=False)

            # Step 2: Semantic Search (FAISS)
            with st.status("[STEP 2: Candidates from FAISS Semantic Search: PROCESSING...]", expanded=True) as status:
                time.sleep(0.5)
                st.markdown(
                    "<span style='font-family: Consolas, monospace;'>&gt;&gt; Converting query to vector embedding... <span style='color:#00FF41'>[All-MiniLM-L6-v2]</span></span>", unsafe_allow_html=True)
                st.markdown(
                    "<span style='font-family: Consolas, monospace;'>&gt;&gt; Querying <span style='color:#00FF41'>[FAISS]</span> for semantic similarity...</span>", unsafe_allow_html=True)

                # FAISS Retrieval Logic
                if faiss_index and faiss_metadata and embedding_model:
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

                    for score, idx in zip(scores, found_indices):
                        if idx == -1:
                            continue
                        meta = faiss_metadata[idx]
                        step2_candidates.append({
                            'track_id': meta['track_id'],
                            'track_name': meta['track_name'],
                            'artists': meta['artists'],
                            'score': float(score),
                            'rag_doc': meta['rag_doc']
                        })

                    st.markdown(
                        f"<span style='font-family: Consolas, monospace;'>&gt;&gt; Vector search complete. Retrieved <span style='color:#00FF41'>[{len(step2_candidates)}]</span> semantic candidates.</span>", unsafe_allow_html=True)

                    for i, cand in enumerate(step2_candidates):
                        score = cand['score']
                        expl_desc = cand['rag_doc'][:70] + \
                            "..." if len(cand['rag_doc']
                                         ) > 60 else cand['rag_doc']
                        st.markdown(
                            f"<span style='font-family: Consolas, monospace; margin-left: 20px;'>* #{i+1} <span style='color:#00FFFF'>{cand['track_name']}</span> by {cand['artists']} (Score: {score:.4f}) <br>  <span style='color:#AAAAAA; font-size: 0.8em; margin-left: 20px;'>Desc: {expl_desc}</span></span>", unsafe_allow_html=True)

                else:
                    st.error("FAISS resources not loaded.")
                    step2_candidates = []

                # Keep expanded so user sees the list
                status.update(
                    label="[STEP 2: Candidates from FAISS Semantic Search: OK]", state="complete", expanded=True)

            # Step 3: Re-ranking & Filtering
            with st.status("[STEP 3: Re-ranked and Filtered Recommendations: PROCESSING...]", expanded=True) as status:
                time.sleep(0.5)
                st.markdown(
                    f"<span style='font-family: Consolas, monospace;'>&gt;&gt; Loading persona profile: <span style='color:#00FF41'>['{selected_persona_name}']</span></span>", unsafe_allow_html=True)
                st.markdown(
                    f"<span style='font-family: Consolas, monospace;'>&gt;&gt; Applying filters: Name=<span style='color:#00FF41'>[{selected_song['track_name']}]</span></span>", unsafe_allow_html=True)

                # Combine and Rerank (LLM-based)
                if 'step1_candidates' not in locals():
                    step1_candidates = []

                st.markdown(f"<span style='font-family: Consolas, monospace;'>&gt;&gt; Sending candidates to <span style='color:#FF00FF'>GPT-4o Agent</span> for re-ranking...</span>", unsafe_allow_html=True)
                final_recs, llm_explanation = utils.llm_rerank_candidates(
                    df_songs, step1_candidates, step2_candidates, selected_song, traits, top_k=20)

                st.markdown(
                    f"<span style='font-family: Consolas, monospace;'>&gt;&gt; Re-ranking complete. Selected top <span style='color:#00FF41'>[{len(final_recs)}]</span> candidates based on <span style='color:#00FF41'>[User_History_Preference]</span>.</span>", unsafe_allow_html=True)

                for i, (_, row) in enumerate(final_recs.iterrows()):
                    score = row.get('ranking_score', 0)
                    # Use LLM reason if available, else fallback to template
                    reason = row.get('reason') if pd.notna(
                        row.get('reason')) else utils.generate_explanation(row, selected_song, traits)
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
                st.info(llm_explanation)
                status.update(
                    label="[STEP 4: LLM's Overall Explanation: OK]", state="complete", expanded=True)

            # Step 3: Generation
            st.divider()
            st.title("🎧 Recommended for You")

            rec_count = 0
            display_limit = 6

            # Iterate in chunks of 3 for grid layout
            for i in range(0, min(len(final_recs), display_limit), 3):
                cols = st.columns(3)
                chunk_df = final_recs.iloc[i: i+3]

                for j, (_, row) in enumerate(chunk_df.iterrows()):
                    with cols[j]:
                        with st.container():
                            spotify_embed(row['track_id'], height=352)
                            # Use LLM reason if available
                            reason = row.get('reason') if pd.notna(
                                row.get('reason')) else utils.generate_explanation(row, selected_song, traits)
                            st.info(f"{reason}")

            # 4. Visualization (PCA)
            st.divider()
            st.title("📊 Visualization")

            with st.spinner("Generating Embedding Space Visualization..."):
                try:
                    viz_recs = final_recs.head(6)
                    fig = utils.plot_pca_visualization(
                        df_songs,
                        selected_song,
                        viz_recs,
                        user_history=history,
                        step1_cands=step1_candidates,
                        step2_cands=step2_candidates,
                        df_pca=df_pca
                    )
                    # st.plotly_chart(fig, use_container_width=True)

                    # Save interactive plot to HTML (similar to test_pca_plot.py)
                    output_file = os.path.join(BASE_DIR, "pca_plot.html")
                    pio.write_html(fig, output_file)
                    # st.caption(f"PCA plot exported to {output_file}")

                    # Embed the saved HTML via iframe-like component
                    try:
                        with open(output_file, "r", encoding="utf-8") as f:
                            html_content = f.read()
                        components.html(
                            html_content, height=720, scrolling=True)
                    except Exception as embed_err:
                        st.warning(f"無法載入 PCA HTML：{embed_err}")
                except Exception as e:
                    import traceback
                    st.error(f"Visualization Error: {e}")
                    st.code(traceback.format_exc())


def main():
    # Initialize Session State
    if 'selected_song' not in st.session_state:
        st.session_state.selected_song = None
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'selected_persona' not in st.session_state:
        st.session_state.selected_persona = None

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
