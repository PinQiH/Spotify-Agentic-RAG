import os
import json
import numpy as np
import pickle

# > Persona 語義向量提取腳本 (Persona Soft Prompt Extraction)
# - 目標：計算特定 Persona 歷史歌曲的平均向量，作為推薦系統的「偏好提示」

def get_persona_soft_prompt(persona_name, soft_prompts_map, history_dir="data/persona_listening_histories"):
	"""
	獲取特定 Persona 的代表性語義向量 (Soft Prompt)。
	"""
	# 1. 格式化檔名 (例如 "Chill Vibes" -> "chill_vibes_history.json")
	filename = persona_name.lower().replace(" ", "_") + "_history.json"
	filepath = os.path.join(history_dir, filename)
	
	if not os.path.exists(filepath):
		available = [f.replace("_history.json", "").replace("_", " ").title() for f in os.listdir(history_dir) if f.endswith(".json")]
		raise FileNotFoundError(f"找不到畫像檔案: {filepath}。可用的畫像有: {available}")

	# 2. 讀取聽歌歷史
	with open(filepath, 'r', encoding='utf-8') as f:
		data = json.load(f)
		# 修正：資料格式為字典，歌曲列表在 'listening_history' 欄位
		history = data.get('listening_history', [])
	
	# 3. 收集該 Persona 歷史歌曲的所有 MLP 預測向量
	vectors = []
	found_tracks = []
	for song in history:
		tid = song.get('track_id')
		if tid in soft_prompts_map:
			vectors.append(soft_prompts_map[tid])
			found_tracks.append(song.get('track_name', tid))
	
	if not vectors:
		raise ValueError(f"在 Soft Prompt Map 中找不到 {persona_name} 歷史歌曲的任何向量。請確認 MLP 預測已執行。")

	# 4. 計算平均向量 (Mean Pooling)
	# 這代表了該 Persona 在語義空間中的「重心」
	persona_vector = np.mean(vectors, axis=0)
	
	print(f"// @ 成功計算畫像 '{persona_name}' 的 Soft Prompt")
	print(f"   - 引用歷史歌曲數: {len(vectors)} / {len(history)}")
	print(f"   - 代表性歌曲: {', '.join(found_tracks[:3])}...")
	
	return persona_vector

def list_available_personas(history_dir="data/persona_listening_histories"):
	"""列出所有可用的畫像名稱"""
	if not os.path.exists(history_dir):
		return []
	return [f.replace("_history.json", "").replace("_", " ").title() for f in os.listdir(history_dir) if f.endswith("_history.json")]
