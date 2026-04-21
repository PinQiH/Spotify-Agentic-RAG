import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from tqdm import tqdm

# > 相似度矩陣計算區塊
# - 執行 Cosine Similarity 並選取 Top-N
def calculate_similarity_matrix(df_pca, n_top=10, batch_size=1000, save_path=None):
	"""
	基於 PCA 特徵計算歌曲間的餘弦相似度，並為每首歌找出最相似的前 N 首。
	若提供 save_path 且檔案存在，則直接讀取。
	"""
	if save_path and os.path.exists(save_path):
		print(f"// @ 找到預先計算的相似歌曲檔案: {save_path}，直接載入...")
		with open(save_path, 'r', encoding='utf-8') as f:
			return json.load(f)

	# 1. 確保 track_id 唯一性
	df_unique = df_pca.drop_duplicates(subset="track_id", keep="first").reset_index(drop=True)
	num_songs = df_unique.shape[0]
	print(f"// > 開始計算 {num_songs} 首歌曲的相似度矩陣 (Top-{n_top})...")

	# 2. 提取 PCA 向量
	pca_cols = [col for col in df_unique.columns if col.startswith("PC_")]
	X = df_unique[pca_cols].values

	all_similar_songs = []

	# 3. 分批計算餘弦相似度
	for start in tqdm(range(0, num_songs, batch_size), desc="Calculating Sim"):
		end = min(start + batch_size, num_songs)
		
		# 計算目前批次對所有歌曲的相似度
		sim_batch = cosine_similarity(X[start:end], X)

		for i in range(sim_batch.shape[0]):
			idx = start + i
			scores = sim_batch[i]

			# 排除自己 (設為極小值)
			scores[idx] = -1

			# 取得排序後的索引 (由大到小)
			sorted_idx = np.argsort(-scores)

			# 選取 Top-N
			top_n_list = []
			for j in sorted_idx[:n_top]:
				row_j = df_unique.iloc[j]
				top_n_list.append({
					"track_id": str(row_j["track_id"]),
					"track_name": str(row_j["track_name"]),
					"artists": str(row_j["artists"]),
					"album_name": str(row_j["album_name"]),
					"similarity_score": round(float(scores[j]), 4)
				})

			# 組合當前歌曲與其相似歌曲
			current_song = df_unique.iloc[idx]
			all_similar_songs.append({
				"track_id": str(current_song["track_id"]),
				"track_name": str(current_song["track_name"]),
				"artists": str(current_song["artists"]),
				"album_name": str(current_song["album_name"]),
				"track_genre": str(current_song["track_genre"]),
				"popularity": int(current_song["popularity"]),
				"top_n_similar_songs": top_n_list
			})

	# 4. 存檔與完成
	if save_path:
		os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
		with open(save_path, 'w', encoding='utf-8') as f:
			json.dump(all_similar_songs, f, ensure_ascii=False, indent=4)
		print(f"// @ 相似度計算完成，結果已儲存至: {save_path}")

	return all_similar_songs
