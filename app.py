import streamlit as st
import pandas as pd
import time
import json
import os
import utils
import streamlit.components.v1 as components

# Set page config MUST be the first Streamlit command
st.set_page_config(
    page_title="Spotify Agentic RAG",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def spotify_embed(track_id, height=80):
    """Embeds a Spotify player for the given track_id."""
    url = f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0"
    components.iframe(url, height=height)

def main():
    # Initialize Session State
    if 'selected_song' not in st.session_state:
        st.session_state.selected_song = None
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False

    # Load Data
    df_songs = load_data()
    personas = load_personas()
    
    if df_songs is None or not personas:
        st.warning("請確保資料已正確設定。")
        return

    # Sidebar: Persona Selection
    with st.sidebar:
        # st.title("🎧 Spotify Agentic RAG")
        st.header("用戶角色 (User Persona)")
        selected_persona_name = st.selectbox("選擇角色", list(personas.keys()))
        
        # Persona Descriptions
        PERSONA_DESCRIPTIONS = {
            "Chill Vibes": "喜歡放鬆、低保真 (Lo-Fi) 和氛圍音樂的用戶。通常在休息或閱讀時聆聽。",
            "Party Animal": "熱愛高能量、舞曲和流行音樂的用戶。喜歡節奏感強烈的歌曲。",
            "Study Focus": "專注於學習和工作，偏好無歌詞或輕柔的背景音樂。",
            "Workout Motivation": "健身愛好者，喜歡高 BPM、激勵人心的音樂來提升運動表現。"
        }
        
        # Show Description
        desc = PERSONA_DESCRIPTIONS.get(selected_persona_name, "一位熱愛音樂的用戶。")
        st.info(f"📝 **角色描述:**\n{desc}")
        
        # Show mini profile
        history = personas[selected_persona_name]
        traits = utils.analyze_persona(history)
        st.caption(f"喜好風格: {', '.join(traits['top_genres'][:2])}")
        st.caption(f"最愛藝人: {traits['top_artists'][0]}")
        
        st.divider()
        
        # Listening History (Embeds)
        st.subheader("📜 最近收聽 (History)")
        for track in history[:5]: # Show top 5 recent
            spotify_embed(track['track_id'], height=80)
        
        st.divider()
        
        if st.button("⏻"):
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
        search_query = st.text_input("Search", placeholder="搜尋歌曲或藝人 (Search)", on_change=reset_page, label_visibility="collapsed")
    
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
                        if st.button("▶", key=f"btn_{row['track_id']}"): # Use track_id for unique key across pages
                            st.session_state.selected_song = row
                            st.session_state.analysis_done = True # Auto-start analysis
                            st.rerun()
                        
        # Pagination Controls
        st.write("")
        col_prev, col_info, col_next = st.columns([1, 10, 1])
        with col_prev:
            if st.button("❮", disabled=st.session_state.current_page == 1):
                st.session_state.current_page -= 1
                st.rerun()
        with col_info:
            st.markdown(f"<div style='text-align: center; padding-top: 10px;'>Page {st.session_state.current_page} of {total_pages}</div>", unsafe_allow_html=True)
        with col_next:
            if st.button("❯", disabled=st.session_state.current_page == total_pages):
                st.session_state.current_page += 1
                st.rerun()

    st.divider()

    # Section 2: Now Playing & Analysis
    if st.session_state.selected_song is not None:
        selected_song = st.session_state.selected_song
        
        st.title("🎵 Now Playing")
        
        col_hero_1, col_hero_2 = st.columns([3, 1])
        with col_hero_1:
            spotify_embed(selected_song['track_id'], height=152)
        
        # Analysis Section (Auto-triggered)
        if st.session_state.analysis_done:
            st.divider()
            st.title("🧠 Agentic Thinking")
            
            # Step 1: Candidate Retrieval
            with st.status("[STEP 1: Candidate_Retrieval: PROCESSING...]", expanded=True) as status:
                time.sleep(0.5)
                st.markdown("<span style='font-family: Consolas, monospace;'>&gt;&gt; Initiating <span style='color:#00FF41'>hybrid_retrieval</span> protocol...</span>", unsafe_allow_html=True)
                st.markdown("<span style='font-family: Consolas, monospace;'>&gt;&gt; Accessing <span style='color:#00FF41'>[Co-occurrence_DB]</span> and <span style='color:#00FF41'>[Semantic_Similarity_Model]</span>...</span>", unsafe_allow_html=True)
                
                retrieved, final_recs = utils.get_recommendations(df_songs, selected_song, traits)
                
                st.markdown(f"<span style='font-family: Consolas, monospace;'>&gt;&gt; Retrieval complete. Candidates found: <span style='color:#00FF41'>[{len(retrieved)}]</span></span>", unsafe_allow_html=True)
                status.update(label="[STEP 1: Candidate_Retrieval: OK]", state="complete", expanded=False)

            # Step 2: Filtering & Reasoning
            with st.status("[STEP 2: Filtering_Reasoning: PROCESSING...]", expanded=True) as status:
                time.sleep(0.5)
                st.markdown(f"<span style='font-family: Consolas, monospace;'>&gt;&gt; Loading persona profile: <span style='color:#00FF41'>['{selected_persona_name}']</span></span>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-family: Consolas, monospace;'>&gt;&gt; Analyzing context: Genre=<span style='color:#00FF41'>[{selected_song['track_genre']}]</span>, BPM=<span style='color:#00FF41'>[{int(selected_song['tempo'])}]</span></span>", unsafe_allow_html=True)
                st.markdown("<span style='font-family: Consolas, monospace;'>&gt;&gt; Applying filtering logic... <span style='color:#00FF41'>[FILTER_LOW_RELEVANCE]</span></span>", unsafe_allow_html=True)
                status.update(label="[STEP 2: Filtering_Reasoning: OK]", state="complete", expanded=False)

            # Step 3: Ranking & Generating
            with st.status("[STEP 3: Ranking_Generating: PROCESSING...]", expanded=True) as status:
                time.sleep(0.5)
                st.markdown("<span style='font-family: Consolas, monospace;'>&gt;&gt; Calculating final weights based on <span style='color:#00FF41'>[User_History_Preference]</span>...</span>", unsafe_allow_html=True)
                st.markdown("<span style='font-family: Consolas, monospace;'>&gt;&gt; Generating natural language explanations... <span style='color:#00FF41'>[GEN_REASONING]</span></span>", unsafe_allow_html=True)
                status.update(label="[STEP 3: Ranking_Generating: OK]", state="complete", expanded=False)

            # Step 3: Generation
            st.divider()
            st.title("🎧 Recommended for You")
            
            rec_cols = st.columns(3)
            for idx, (_, row) in enumerate(final_recs.iterrows()):
                with rec_cols[idx]:
                    with st.container():
                        spotify_embed(row['track_id'], height=352)
                        explanation = utils.generate_explanation(row, selected_song, traits)
                        st.info(f"{explanation}")

if __name__ == "__main__":
    main()
