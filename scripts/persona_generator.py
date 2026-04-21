import pandas as pd
import os
import json
import numpy as np

# > 合成用戶畫像生成區塊
# - 定義多種音樂品味的人格，並從資料庫中分配合適的歌曲
def generate_synthetic_personas(df, save_dir="data/persona_listening_histories", num_songs=20):
	"""
	基於預設規則定義 4 個典型用戶畫像，並為其生成「虛擬收聽歷史」。
	若 save_dir 已存在 JSON 檔且內容完整，則跳過生成流程。
	"""
	# 預定義人格品味描述與篩選規則
	persona_configs = {
		"Chill Vibes": {
			"description": "偏好輕鬆、低能量、高聲學感 (Acoustic) 的音樂，如民謠或不插電歌曲。",
			"query": "energy_scaled < 0 and acousticness_scaled > 0.5" # 稍微低能量，且聲學感強
		},
		"Party Animal": {
			"description": "偏好快節奏、高能量、具有強烈舞動感的電子或流行樂。",
			"query": "energy_scaled > 0.5 and danceability_scaled > 0.5 and tempo_scaled > 0"
		},
		"Study Focus": {
			"description": "偏好中低節奏、穩定且通常為純樂器演奏 (Instrumental) 的音樂以利專注。",
			"query": "instrumentalness_binary == 1 and tempo_scaled < 0.2"
		},
		"Workout Motivation": {
			"description": "偏好極高能量、強勁節奏且能提升士氣的音樂。",
			"query": "energy_scaled > 0.8 and tempo_scaled > 0.5"
		}
	}

	# 檢查是否已存在檔案
	if os.path.exists(save_dir) and len([f for f in os.listdir(save_dir) if f.endswith('.json')]) >= len(persona_configs):
		print(f"// @ 偵測到現有的畫像歷史檔案夾: {save_dir}，直接讀取...")
		histories = {}
		for persona_name in persona_configs:
			filename = persona_name.replace(" ", "_") + "_history.json"
			filepath = os.path.join(save_dir, filename)
			if os.path.exists(filepath):
				with open(filepath, 'r', encoding='utf-8') as f:
					histories[persona_name] = json.load(f)
		return histories

	print(f"// > 正在為 {len(persona_configs)} 種人格生成合成數據...")
	os.makedirs(save_dir, exist_ok=True)
	
	generated_histories = {}
	
	# 為了確保抽樣邏輯能運作，我們需要具備必要特徵
	# 如果找不到對應特徵，會回退到隨機抽樣並顯示警告
	for persona_name, config in persona_configs.items():
		try:
			# 執行查詢
			candidates = df.query(config["query"])
			
			# 如果候選不足，則放寬條件甚至隨機
			if len(candidates) < num_songs:
				print(f"// !! 警告: {persona_name} 符合條件的歌曲不足 ({len(candidates)})，將補充隨機歌曲。")
				extra = df.sample(n=num_songs - len(candidates), random_state=42)
				candidates = pd.concat([candidates, extra])
			
			# 隨機選取指定數量
			sample_df = candidates.sample(n=num_songs, random_state=42)
			
			# 轉化為 JSON 格式 (只保留必要 Metadata)
			meta_cols = ['track_id', 'track_name', 'artists', 'album_name', 'track_genre', 'popularity']
			history_list = sample_df[meta_cols].to_dict('records')
			
			# 存檔
			filename = persona_name.replace(" ", "_") + "_history.json"
			filepath = os.path.join(save_dir, filename)
			
			# 包裝畫像資訊
			persona_data = {
				"persona_name": persona_name,
				"description": config["description"],
				"listening_history": history_list
			}
			
			with open(filepath, 'w', encoding='utf-8') as f:
				json.dump(persona_data, f, ensure_ascii=False, indent=4)
			
			generated_histories[persona_name] = persona_data
			print(f"   - ✔ {persona_name} 歷史紀錄生成成功: {filepath}")
			
		except Exception as e:
			print(f"// !! 錯誤: 生成 {persona_name} 時發生問題: {e}")

	return generated_histories
