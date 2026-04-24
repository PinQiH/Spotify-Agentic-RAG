import torch
import numpy as np
import pandas as pd
import faiss
import pickle
import os
import sys
import io

# 將當前目錄加入 path 以便 import 專案模組
sys.path.append(os.getcwd())

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from scripts.train_mlp import SoftPromptMLP, load_soft_prompt_mlp
from scripts.get_persona_prompt import get_persona_soft_prompt

# > 評估環境配置
DATA_DIR = "data"
SONGS_PATH = os.path.join(DATA_DIR, "songs.csv")
INDEX_PATH = os.path.join(DATA_DIR, "spotify.index")
FAISS_META_PATH = os.path.join(DATA_DIR, "spotify_faiss_meta.pkl")
SOFT_PROMPTS_MAP_PATH = os.path.join(DATA_DIR, "soft_prompts_map.pkl")
MODEL_ORIG_PATH = os.path.join(DATA_DIR, "soft_prompt_mlp.pth")
MODEL_FINE_PATH = os.path.join(DATA_DIR, "soft_prompt_mlp_finetuned.pth")

# --- Helper Functions ---
def calculate_cosine_similarity(vec1, vec2):
	dot_product = np.dot(vec1, vec2)
	norm_v1 = np.linalg.norm(vec1)
	norm_v2 = np.linalg.norm(vec2)
	return dot_product / (norm_v1 * norm_v2)

def load_mlp(model_path):
	if not os.path.exists(model_path):
		return None
	# 1. 載入模型
	# checkpoint = torch.load(model_path, map_location='cpu')
	# model = SoftPromptMLP(input_dim=checkpoint['input_dim'], output_dim=checkpoint['output_dim'])
	# model.load_state_dict(checkpoint['model_state_dict'])
	model, _ = load_soft_prompt_mlp(model_path)
	model.eval()
	return model

def get_recommendations(model_mlp, index, faiss_metadata, soft_prompts_map, current_song_id, persona_name):
	# 1. 取得向量
	persona_vec = get_persona_soft_prompt(persona_name, soft_prompts_map)
	current_song_vec = soft_prompts_map.get(current_song_id)
	
	if current_song_vec is None:
		return []

	# 2. FAISS 檢索 (Step 2 模擬)
	query_vec = current_song_vec.reshape(1, -1).astype('float32')
	D, I = index.search(query_vec, 100) # 取 Top 100 候選
	
	candidates = []
	for idx in I[0]:
		if idx < len(faiss_metadata):
			cand = faiss_metadata[idx]
			tid = cand['track_id']
			
			if tid == current_song_id:
				continue
				
			cand_vec = soft_prompts_map.get(tid)
			if cand_vec is not None:
				# 這裡模擬 app.py 的打分邏輯
				p_score = calculate_cosine_similarity(persona_vec, cand_vec)
				s_score = calculate_cosine_similarity(current_song_vec, cand_vec)
				
				# 40% Persona + 60% Song Context
				blended = 0.4 * p_score + 0.6 * s_score
				
				candidates.append({
					"track_id": tid,
					"track_name": cand.get("track_name", "Unknown"),
					"blended_score": blended,
					"p_score": p_score,
					"s_score": s_score
				})
	
	# 3. 排序並取 Top 20
	candidates.sort(key=lambda x: x['blended_score'], reverse=True)
	return candidates[:20]

def calculate_jaccard_distance(list1, list2):
	set1 = set([x['track_id'] for x in list1])
	set2 = set([x['track_id'] for x in list2])
	intersection = len(set1.intersection(set2))
	union = len(set1.union(set2))
	return 1.0 - (intersection / union) if union > 0 else 1.0

def run_experiment():
	print("// > 正在載入資源...")
	index = faiss.read_index(INDEX_PATH)
	with open(FAISS_META_PATH, "rb") as f: faiss_metadata = pickle.load(f)
	with open(SOFT_PROMPTS_MAP_PATH, "rb") as f: soft_prompts_map = pickle.load(f)
	
	mlp_orig = load_mlp(MODEL_ORIG_PATH)
	mlp_fine = load_mlp(MODEL_FINE_PATH)
	
	test_songs = [
		{"name": "Acoustic", "id": "5pq4v03P5PxMcnCagg4S3Z"},
		{"name": "Hardstyle", "id": "6bl8PFDgYxbDZHk0LRQhhz"},
		{"name": "Pop", "id": "0b11D9D0hMOYCIMN3OKreM"}
	]
	
	persona = "Party Animal"
	
	results = []
	
	for model_name, model in [("Original", mlp_orig), ("Fine-tuned", mlp_fine)]:
		print(f"\n--- 測試模型: {model_name} ---")
		rec_lists = {}
		
		for ts in test_songs:
			recs = get_recommendations(model, index, faiss_metadata, soft_prompts_map, ts['id'], persona)
			rec_lists[ts['name']] = recs
			print(f"   - Song: {ts['name']:<10}")
			for r in recs[:3]:
				print(f"     * {r['track_name'][:20]:<20} | Blended: {r['blended_score']:.3f} (P:{r['p_score']:.3f}, S:{r['s_score']:.3f})")
		
		# 計算多樣性 (Jaccard Distance)
		# 比較 Acoustic 與 Hardstyle
		dist_ah = calculate_jaccard_distance(rec_lists['Acoustic'], rec_lists['Hardstyle'])
		# 比較 Acoustic 與 Pop
		dist_ap = calculate_jaccard_distance(rec_lists['Acoustic'], rec_lists['Pop'])
		# 比較 Pop 與 Hardstyle
		dist_ph = calculate_jaccard_distance(rec_lists['Pop'], rec_lists['Hardstyle'])
		
		avg_dist = (dist_ah + dist_ap + dist_ph) / 3
		print(f"\n   >> 多樣性分數 (越高越好): {avg_dist:.4f}")
		print(f"      - A vs H: {dist_ah:.4f}")
		print(f"      - A vs P: {dist_ap:.4f}")
		print(f"      - P vs H: {dist_ph:.4f}")
		
		results.append({
			"model": model_name,
			"diversity": avg_dist
		})

	print("\n" + "="*30)
	print("🏆 實驗結論:")
	for r in results:
		print(f" - {r['model']:<12}: Diversity Score = {r['diversity']:.4f}")
	
	if results[1]['diversity'] < results[0]['diversity']:
		print("\n⚠️ 警訊: Fine-tuned 模型的多樣性較低，代表它可能產生了過度擬合 (Overfitting)，")
		print("或過於傾向某些特定歌曲，導致不同種子歌曲產出的結果趨同。")
	else:
		print("\n✅ Fine-tuned 模型維持或提升了多樣性，問題可能不在 MLP 模型本身，")
		print("而在於權重配置或候選集大小。")

if __name__ == "__main__":
	try:
		run_experiment()
	except Exception as e:
		print(f"// !! 執行失敗: {e}")
		import traceback
		traceback.print_exc()
