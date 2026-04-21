import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA

# > PCA 降維處理區塊
# - 執行 PCA 運算與轉換
def apply_pca(df, save_path=None):
	"""
	選取特定數值特徵並執行 PCA 降維。
	若提供 save_path 且檔案存在，直接讀取結果。
	"""
	if save_path and os.path.exists(save_path):
		print(f"// @ 找到已處理過的 PCA 資料檔: {save_path}，直接讀取...")
		return pd.read_csv(save_path, low_memory=False)

	# 1. 定義要進行 PCA 的特徵欄位
	# 我們會抓取所有以 _scaled, key_, time_signature_, instrumentalness_binary 結尾/開頭的欄位
	feature_cols = [
		col for col in df.columns 
		if col.endswith('_scaled') or col.startswith('key_') or col.startswith('time_signature_') or col == 'instrumentalness_binary'
	]
	
	if not feature_cols:
		print("// !! 警告: 找不到任何符合 PCA 的原始特徵欄位，請檢查前處理步驟。")
		return df

	print(f"// > 正在對 {len(feature_cols)} 個特徵執行 PCA...")
	X = df[feature_cols]

	# 2. 初始化並執行 PCA (保留所有分量以供分析)
	pca = PCA(n_components=None)
	X_transformed = pca.fit_transform(X)

	# 3. 建立 PCA 結果 DataFrame
	pc_cols = [f'PC_{i+1}' for i in range(X_transformed.shape[1])]
	df_pca = pd.DataFrame(X_transformed, columns=pc_cols)

	# 4. 顯示解釋變異量 (前 5 個分量)
	explained_var = pca.explained_variance_ratio_
	print(f"// @ PCA 完成。累積解釋變異量 (前5個): {np.sum(explained_var[:5]):.4f}")
	for i in range(min(5, len(explained_var))):
		print(f"   - PC_{i+1}: {explained_var[i]:.4f}")

	# 5. 合併 Metadata
	# 定義非特徵的 Metadata 欄位
	metadata_cols = ['track_id', 'track_name', 'artists', 'album_name', 'track_genre', 'popularity', 'explicit']
	actual_metadata_cols = [col for col in metadata_cols if col in df.columns]

	# 重設 index 以確保合併對齊
	df_reset = df[actual_metadata_cols].reset_index(drop=True)
	df_final = pd.concat([df_reset, df_pca], axis=1)

	# 6. 快取存檔
	if save_path:
		os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
		df_final.to_csv(save_path, index=False)
		print(f"// @ PCA 結果已存向: {save_path}")

	return df_final
