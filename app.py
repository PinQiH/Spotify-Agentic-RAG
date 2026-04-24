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
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import streamlit.components.v1 as components
import concurrent.futures
from scripts.train_mlp import SoftPromptMLP, SoftPromptMLPV2, load_soft_prompt_mlp
from scripts.recommender_agent import get_multi_model_recommendations, calculate_cosine_similarity
from scripts.get_persona_prompt import get_persona_soft_prompt
from scripts.dynamic_rag import generate_dynamic_rag_doc

# > Spotify Agentic RAG DJ: 核心應用程序
# - 整合 3-Step 推理流程與多模型對比

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

# Custom CSS for Spotify Dark Theme
st.markdown("""
<style>
	/* 1. Global Background & Font */
	.stApp {
		background-color: #121212;
		color: #FFFFFF;
		font-family: 'Circular', 'Helvetica Neue', Helvetica, Arial, sans-serif;
	}
	section[data-testid="stSidebar"] {
		background-color: #000000;
		border-right: 1px solid #282828;
	}
	section[data-testid="stSidebar"] * { color: #B3B3B3 !important; }
	section[data-testid="stSidebar"] h1, 
	section[data-testid="stSidebar"] h2, 
	section[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
	h1, h2, h3, h4, h5, h6 { color: #FFFFFF !important; font-weight: 700; }
	.stButton > button {
		background-color: #000000;
		color: #1DB954;
		border: 2px solid #1DB954;
		border-radius: 50%;
		width: 40px;
		height: 40px;
		display: flex;
		align-items: center;
		justify-content: center;
		transition: all 0.2s ease;
	}
	.stButton > button:hover {
		background-color: #1DB954;
		color: #000000;
		transform: scale(1.1);
	}
	div[data-testid="stStatusWidget"] {
		background-color: #000000 !important;
		border: 2px solid #00FF41 !important;
		font-family: 'Consolas', monospace !important;
	}
	/* 側邊欄與矩陣按鈕置中 */
	section[data-testid="stSidebar"] .stButton,
	div[data-testid="column"] .stButton {
		display: flex;
		justify-content: center;
		width: 100%;
	}
	/* 分頁按鈕特定對齊 */
	div[data-testid="column"]:has(.prev-btn-marker) .stButton {
		justify-content: flex-start !important;
	}
	div[data-testid="column"]:has(.next-btn-marker) .stButton {
		justify-content: flex-end !important;
	}
</style>
""", unsafe_allow_html=True)

# --- Resource Loading ---

@st.cache_data
def load_data():
	if not os.path.exists("data/songs.csv"):
		return None
	df = pd.read_csv("data/songs.csv")
	# > 資料去重：防止重複的 track_id 導致 UI 崩潰 (DuplicateWidgetID)
	return df.drop_duplicates(subset='track_id').reset_index(drop=True)

@st.cache_data
def load_pca_data():
	return pd.read_csv("data/pca_songs.csv") if os.path.exists("data/pca_songs.csv") else None

@st.cache_data
def load_personas():
	personas = {}
	history_dir = "data/persona_listening_histories"
	if os.path.exists(history_dir):
		for f in os.listdir(history_dir):
			if f.endswith(".json") and f != "persona_summaries.json":
				name = f.replace("_history.json", "").replace("_", " ").title()
				with open(os.path.join(history_dir, f), "r", encoding="utf-8") as file:
					personas[name] = json.load(file)
	return personas

@st.cache_data
def load_persona_summaries():
	path = "data/persona_listening_histories/persona_summaries.json"
	if os.path.exists(path):
		with open(path, "r", encoding="utf-8") as f: return json.load(f)
	return {}

@st.cache_data
def load_soft_prompts():
	path = "data/soft_prompts_map.pkl"
	if os.path.exists(path):
		with open(path, "rb") as f: return pickle.load(f)
	return {}

@st.cache_data
def load_precomputed_data():
	path = "data/precomputed_similar_songs.json"
	if os.path.exists(path):
		with open(path, "r", encoding="utf-8") as f: return json.load(f)
	return {}

@st.cache_resource
def load_faiss_resources():
	if not os.path.exists(INDEX_PATH): return None, None, None, None
	temp_path = None
	try:
		fd, temp_path = tempfile.mkstemp(suffix=".index")
		os.close(fd)
		shutil.copy2(INDEX_PATH, temp_path)
		index = faiss.read_index(temp_path)
	except: return None, None, None, None
	finally:
		if temp_path: os.remove(temp_path)
	
	# 下載/載入語義向量模型 (384維)
	st_model = SentenceTransformer('all-MiniLM-L6-v2')
	
	# 載入微調過的 MLP 模型 (用於特徵轉向量)
	mlp_path = "data/soft_prompt_mlp_finetuned.pth"
	mlp_model, _ = load_soft_prompt_mlp(mlp_path)
	
	# 加載與 FAISS Index 嚴格對齊的元數據 (2,000筆標竿歌)
	faiss_meta_path = os.path.join(DATA_DIR, "spotify_faiss_meta.pkl")
	faiss_metadata = []
	if os.path.exists(faiss_meta_path):
		with open(faiss_meta_path, "rb") as f: faiss_metadata = pickle.load(f)
		
	return index, st_model, mlp_model, faiss_metadata

# --- Helper Functions ---
def spotify_embed(track_id, height=80):
	url = f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0"
	components.iframe(url, height=height)

# --- UI Rendering ---

def render_landing_page(personas, persona_summaries):
	"""Renders the initial persona selection landing page."""
	st.title("Choose Your Persona...")

	# 恢復原始 CSS 樣式
	st.markdown("""
	<style>
		div[data-testid="column"]:has(.persona-card-marker) {
			background-color: #181818;
			border: 1px solid #282828;
			border-radius: 12px;
			padding: 24px;
			text-align: center;
			transition: all 0.3s ease;
			height: 100%;
			min-height: 250px;
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
		/* 強制讓卡片內的按鈕置中 */
		div[data-testid="column"]:has(.persona-card-marker) .stButton {
			display: flex;
			justify-content: center;
			width: 100%;
		}
	</style>
	""", unsafe_allow_html=True)

	persona_items = list(personas.items())
	for i in range(0, len(persona_items), 2):
		cols = st.columns(2)
		batch = persona_items[i:i+2]
		for idx, (name, _) in enumerate(batch):
			with cols[idx]:
				st.markdown('<div class="persona-card-marker" style="display:none;"></div>', unsafe_allow_html=True)
				
				desc = persona_summaries.get(name, "一位熱愛音樂的用戶。")
				if len(desc) > 80: desc = desc[:77] + "..."
				
				st.markdown(f'<div class="persona-title">{name}</div>', unsafe_allow_html=True)
				st.markdown(f'<div class="persona-desc">{desc}</div>', unsafe_allow_html=True)
				
				if st.button(f"✔", key=f"select_{name}"):
					st.session_state.selected_persona = name
					st.rerun()

def render_main_app(df_songs, df_pca, personas, persona_summaries, index, st_model, mlp_model, faiss_metadata, soft_prompts_map, precomputed_data):
	with st.sidebar:
		st.header("User Persona")
		p_name = st.session_state.selected_persona
		st.subheader(f"👤 {p_name}")
		st.info(persona_summaries.get(p_name, "No summary."))

		st.divider()
		st.subheader("📜 最近收聽 (History)")
		# 確保 personas 是字典且獲取的是 list
		persona_data = personas.get(p_name, {})
		history = persona_data if isinstance(persona_data, list) else persona_data.get('listening_history', [])
		
		# 顯示前 5 首，並加入型別檢查
		if isinstance(history, list):
			for track in history[:5]:
				spotify_embed(track['track_id'], height=80)
		else:
			st.caption("無歷史紀錄或格式錯誤")
		
		st.divider()
		if st.button("⏻"):
			st.session_state.selected_persona = None
			st.session_state.analysis_done = False
			st.rerun()

	st.title("🚀 Agentic RAG DJ")
	
	# --- 歌曲選擇矩陣化與搜尋 ---
	if 'search_query' not in st.session_state: st.session_state.search_query = ""
	if 'page_num' not in st.session_state: st.session_state.page_num = 0
	
	# 搜尋與展示：直接使用 df_songs 確保全量 (20,000筆) 都能被搜到
	search_query = st.text_input("搜尋歌曲或藝人:", value=st.session_state.search_query)
	if search_query != st.session_state.search_query:
		st.session_state.search_query = search_query
		st.session_state.page_num = 0
		st.rerun()

	# 過濾全量歌曲
	mask = df_songs['track_name'].str.contains(search_query, case=False, na=False) | \
		   df_songs['artists'].str.contains(search_query, case=False, na=False)
	filtered_metadata = df_songs[mask].to_dict('records')
	
	songs_per_page = 12
	total_pages = (len(filtered_metadata) + songs_per_page - 1) // songs_per_page
	start_idx = st.session_state.page_num * songs_per_page
	end_idx = start_idx + songs_per_page
	current_page_songs = filtered_metadata[start_idx:end_idx]

	st.markdown(f"共有 `{len(filtered_metadata)}` 首歌曲滿足搜尋條件 (第 {st.session_state.page_num + 1} / {total_pages} 頁)")

	# 4x3 矩陣展示
	for i in range(0, len(current_page_songs), 3):
		cols = st.columns(3)
		batch = current_page_songs[i:i+3]
		for idx, m in enumerate(batch):
			global_idx = start_idx + i + idx # 計算全域索引作為 Key 的一部分
			with cols[idx]:
				# 每首歌一個播放器 + 一個選擇按鈕
				spotify_embed(m['track_id'], height=152)
				# @ 使用全域索引 + track_id 確保 Key 絕對唯一
				if st.button("▶", key=f"sel_grid_{m['track_id']}_{global_idx}"):
					st.session_state.selected_song_id = m['track_id']
					st.session_state.current_selected_meta = m
					st.session_state.analysis_done = True
					st.rerun()

	# 分頁控制
	col_prev, col_mid, col_next = st.columns([1, 2, 1])
	with col_prev:
		st.markdown('<div class="prev-btn-marker" style="display:none;"></div>', unsafe_allow_html=True)
		if st.button("❮", disabled=st.session_state.page_num == 0):
			st.session_state.page_num -= 1
			st.rerun()
	with col_next:
		st.markdown('<div class="next-btn-marker" style="display:none;"></div>', unsafe_allow_html=True)
		if st.button("❯", disabled=st.session_state.page_num >= total_pages - 1):
			st.session_state.page_num += 1
			st.rerun()

	st.divider()

	# --- 播放與分析區 ---
	if st.session_state.get('selected_song_id'):
		current_song_id = st.session_state.selected_song_id
		st.title("🎵 Now Playing")
		# 只保留播放器
		spotify_embed(current_song_id, height=152)
		
	if st.session_state.get('analysis_done'):
		st.divider()
		st.title("🧠 Agentic Thinking Process")

		# --- Step 1: Pre-computed Numerical Retrieval ---
		with st.status("[STEP 1: Numerical Retrieval]", expanded=False) as status:
			time.sleep(0.5)
			st.markdown("<span style='color:#00FF41'>&gt;&gt; Accessing pre-computed similarity matrix...</span>", unsafe_allow_html=True)
			
			# 從預計算數據中獲取相似歌曲 (兼容 List 與 Dict 格式)
			raw_num_cands = []
			if isinstance(precomputed_data, dict):
				raw_num_cands = precomputed_data.get(current_song_id, [])
			elif isinstance(precomputed_data, list):
				# 如果是 List，且裡面包含當前歌曲的相似清單，則尋找對應項
				# 假設 List 結構是 [{'track_id': 'xxx', 'similar': [...]}, ...]
				for item in precomputed_data:
					if item.get('track_id') == current_song_id:
						raw_num_cands = item.get('similar', [])
						break
				# 如果 List 本身就是一個候選清單，則直接使用
				if not raw_num_cands and len(precomputed_data) > 0:
					raw_num_cands = precomputed_data
			
			step1_candidates = []
			# 修正：將 metadata 替換為全量 df_songs
			df_lookup = df_songs.to_dict('records')
			meta_lookup = {m['track_id']: m for m in df_lookup}
			
			for sim_item in raw_num_cands[:20]:
				tid = sim_item.get('track_id') if isinstance(sim_item, dict) else sim_item
				if tid in meta_lookup:
					step1_candidates.append(meta_lookup[tid])
			
			st.write(f"✅ Found {len(step1_candidates)} numerical candidates.")
			status.update(label=f"Step 1: OK ({len(step1_candidates)})", state="complete")

		# --- Step 2: Semantic (MLP + FAISS) ---
		with st.status("[STEP 2: Semantic Retrieval]", expanded=False) as status:
			song_vec = soft_prompts_map.get(current_song_id)
			if song_vec is not None:
				st.markdown("<span style='color:#00FF41'>&gt;&gt; Using MLP Predicted Soft-Prompt Vector...</span>", unsafe_allow_html=True)
				query_vec = song_vec.reshape(1, -1).astype('float32')
				D, I = index.search(query_vec, 15)
				
				# 僅初步收集，稍後統一進行打分與去重
				semantic_raw_cands = []
				for idx in I[0]:
					if idx < len(faiss_metadata):
						cand = faiss_metadata[idx]
						if cand['track_id'] != current_song_id:
							semantic_raw_cands.append(cand)
				st.write(f"✅ Found {len(semantic_raw_cands)} semantic candidates.")
			else:
				semantic_raw_cands = []
				st.warning("No soft prompt vector found.")
			status.update(label="Step 2: OK", state="complete")

		# --- [NEW] Refinement: Deduplication & Scoring (跟 Notebook 一模一樣) ---
		with st.status("[INTEGRATION: Deduplication & Scoring]", expanded=False) as status:
			# 1. 建立去重清單
			all_candidates_dict = {}

			# 1.1 整合數值檢索 (Step 1)
			for c in step1_candidates:
				tid = c['track_id']
				# 確保查表資料完整 (使用全量 df_songs)
				rows = df_songs[df_songs['track_id'] == tid]
				if not rows.empty:
					all_candidates_dict[tid] = rows.iloc[0].to_dict()

			# 1.2 整合語義檢索 (Step 2)
			for c in semantic_raw_cands:
				all_candidates_dict[c['track_id']] = c

			# 2. 計算雙軸語義匹配度
			persona_vec = get_persona_soft_prompt(p_name, soft_prompts_map)
			# 取得「目前歌曲」的軟提示向量，用於計算「與當前歌曲的相似度」
			current_song_vec = soft_prompts_map.get(current_song_id)
			final_candidate_list = []

			for tid, meta in all_candidates_dict.items():
				# [補全邏輯] 如果沒有預先生成的 RAG 描述，則動態現場生成一段
				if 'rag_doc' not in meta or pd.isna(meta['rag_doc']) or meta['rag_doc'] == "" or meta['rag_doc'] == "nan":
					meta['rag_doc'] = generate_dynamic_rag_doc(meta)
					meta['is_dynamic_rag'] = True # 標註為動態生成，方便除錯

				if tid in soft_prompts_map:
					cand_vec = soft_prompts_map[tid]
					persona_score = calculate_cosine_similarity(persona_vec, cand_vec)
					# 計算候選歌與「目前選的歌」的相似度（讓推薦結果跟著選歌改變）
					song_score = calculate_cosine_similarity(current_song_vec, cand_vec) if current_song_vec is not None else 0.0
					# 混合分數：40% 個人品味 + 60% 與當前歌曲相似度
					blended_score = 0.4 * persona_score + 0.6 * song_score

					meta_with_score = meta.copy()
					meta_with_score['persona_match_score'] = round(float(persona_score), 3)
					meta_with_score['current_song_similarity'] = round(float(song_score), 3)
					meta_with_score['blended_score'] = round(float(blended_score), 3)

					# 標記來源
					is_s = any(sc['track_id'] == tid for sc in semantic_raw_cands)
					meta_with_score['retrieval_source'] = "Semantic" if is_s else "Numerical"
					final_candidate_list.append(meta_with_score)

			# 用混合分數排序，確保候選集已反映當前選歌
			final_candidate_list.sort(key=lambda x: x.get('blended_score', 0), reverse=True)

			st.write(f"✅ Refined {len(final_candidate_list)} unique candidates with taste-scores.")
			status.update(label="Sync: Completed", state="complete")

		# --- Step 3: Agentic Re-ranking ---
		with st.status("[STEP 3: Multi-Model Agent Inference]", expanded=True) as status:
			st.markdown("<span style='color:#00FF41'>&gt;&gt; Orchestrating parallel experts...</span>", unsafe_allow_html=True)
			
			current_song = st.session_state.get('current_selected_meta')
			# 使用打分後的 final_candidate_list (跟 Notebook 做法一致)
			multi_results, errors = get_multi_model_recommendations(
				p_name, current_song, persona_summaries.get(p_name, ""), final_candidate_list
			)
			
			if errors:
				st.warning("⚠️ Some models failed to respond. Check API keys or rate limits.")
				for err in errors: st.caption(err)
			
			col_gpt, col_gemini, col_grok = st.columns(3)
			model_map = {
				"GPT-4o": (col_gpt, "🟢"), 
				"Gemini-2.0-Flash": (col_gemini, "🟡"), 
				"Grok 4.1 Fast": (col_grok, "🔴")
			}
			
			for m_key, (col_obj, icon) in model_map.items():
				with col_obj:
					res = multi_results.get(m_key)
					if res:
						st.success(f"{icon} **DJ {m_key}**")
						st.caption(res.get('agent_explanation', 'No explanation'))
						for rec in res.get('recommendations', [])[:3]:
							with st.container(border=True):
								st.markdown(f"**{rec['track_name']}**")
								st.caption(rec['artists'])
								st.write(f"💡 {rec['reason']}")
								spotify_embed(rec['track_id'], height=80)
					else:
						st.warning(f"DJ {m_key} is thinking too hard or unavailable.")
			status.update(label="Step 3: OK", state="complete")

		# --- Step 4: Visualization (PCA) ---
		st.divider()
		st.title("📊 Embedding Space Visualization")
		
		# Initialize viz cache
		if 'viz_cache' not in st.session_state: st.session_state.viz_cache = {}
		
		with st.spinner("Rendering Vector Maps..."):
			try:
				# 處理可能的數據格式問題，防範 KeyError
				def safe_df(data, cols=['track_id']):
					df = pd.DataFrame(data)
					if df.empty: return pd.DataFrame(columns=cols)
					return df

				# 近期收聽紀錄
				persona_data = personas.get(p_name, {})
				history = persona_data if isinstance(persona_data, list) else persona_data.get('listening_history', [])
				history_df = safe_df(history)
				
				# 建立分頁
				viz_tabs = st.tabs(["GPT-4o", "Gemini-2.0-Flash", "Grok 4.1 Fast"])
				
				def render_viz(model_key, tab_obj):
					with tab_obj:
						res = multi_results.get(model_key)
						if not res or 'recommendations' not in res:
							st.warning(f"No {model_key} recommendations available for visualization.")
							return
						
						cache_key = f"{current_song_id}_{model_key}"
						if cache_key in st.session_state.viz_cache:
							html_content = st.session_state.viz_cache[cache_key]
						else:
							# 準備推薦結果 DataFrame
							recs_df = safe_df(res['recommendations'])
							# 準備候選集 DataFrame (從外層作用域抓取，使用最新變數名)
							s1_df = safe_df(numerical_cands if 'numerical_cands' in locals() else [])
							s2_df = safe_df(semantic_raw_cands if 'semantic_raw_cands' in locals() else [])

							try:
								fig = utils.plot_pca_visualization(
									df_songs, current_song, recs_df, 
									user_history=history_df,
									step1_cands=s1_df,
									step2_cands=s2_df,
									df_pca=df_pca
								)
								html_content = fig.to_html(include_plotlyjs='cdn', full_html=True, auto_play=False)
								# 加入 Plotly Resize Patch (防止 Tab 切換時縮成一團)
								html_content += """
								<script>
								window.addEventListener('resize', function() {
									var plotDivs = document.querySelectorAll('.plotly-graph-div');
									plotDivs.forEach(function(div) {
										Plotly.Plots.resize(div);
									});
								});
								# 觸發一次強迫渲染
								setTimeout(function(){ window.dispatchEvent(new Event('resize')); }, 100);
								</script>
								"""
								st.session_state.viz_cache[cache_key] = html_content
							except Exception as ex:
								st.error(f"Draw Error: {ex}")
								return
						
						# 使用 components.html 渲染強大的 Plotly 互動圖
						st.components.v1.html(html_content, height=720, scrolling=True)

				render_viz("GPT-4o", viz_tabs[0])
				render_viz("Gemini-2.0-Flash", viz_tabs[1])
				render_viz("Grok 4.1 Fast", viz_tabs[2])
				
			except Exception as e:
				st.error(f"Visualization Module Error: {e}")

		# --- Step 5: Voting System ---
		st.divider()
		with st.expander("🗳️ 我要投票 (Vote for your DJ)"):
			st.markdown("### 📝 模型評選投票")
			with st.form("vote_form"):
				reason_v = st.radio("哪位 DJ 的**推薦理由**最符合您的口味？", ["GPT-4o", "Gemini-2.0-Flash", "Grok 4.1 Fast"], horizontal=True)
				song_v = st.radio("哪位 DJ 的**推薦歌單**最合您心意？", ["GPT-4o", "Gemini-2.0-Flash", "Grok 4.1 Fast"], horizontal=True)
				
				if st.form_submit_button("➤"):
					import datetime
					vote_data = {
						"timestamp": datetime.datetime.now().isoformat(),
						"persona": p_name,
						"song": current_song['track_name'],
						"vote_reason": reason_v,
						"vote_song": song_v
					}
					if utils.save_vote(vote_data):
						st.success("🎉 投票成功！您的選擇將用於優化未來的 AI DJ。")
						utils.load_votes.clear() # 清除快取以刷新圖表
					else:
						st.error("投票儲存失敗。")

		# --- Step 6: 投票統計結果 ---
		st.markdown("#### 📊 DJ 評選統計")
		df_votes = utils.load_votes()
		if df_votes is not None and not df_votes.empty:
			c1, c2 = st.columns(2)
			with c1:
				st.caption("推薦理由得分")
				st.bar_chart(df_votes['vote_reason'].value_counts())
			with c2:
				st.caption("推薦歌單得分")
				st.bar_chart(df_votes['vote_song'].value_counts())
		else:
			st.info("目前尚無投票資料，快來投下第一票吧！")

def main():
	# Ensure Session State
	for key in ['selected_persona', 'analysis_done']:
		if key not in st.session_state: st.session_state[key] = None if key == 'selected_persona' else False

	# Load resources
	df_songs = load_data()
	df_pca = load_pca_data()
	personas = load_personas()
	persona_summaries = load_persona_summaries()
	soft_prompts_map = load_soft_prompts()
	# 修正接收：index, st_model, mlp_model, faiss_metadata (順序與定義對齊)
	index, st_model, mlp_model, faiss_metadata = load_faiss_resources() 
	precomputed_data = load_precomputed_data()

	if st.session_state.selected_persona is None:
		render_landing_page(personas, persona_summaries)
	else:
		render_main_app(df_songs, df_pca, personas, persona_summaries, index, 
						st_model, mlp_model, faiss_metadata, soft_prompts_map, precomputed_data)

if __name__ == "__main__":
	main()