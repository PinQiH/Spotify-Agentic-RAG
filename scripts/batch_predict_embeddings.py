import torch
import pandas as pd
import numpy as np
import os
import pickle
from scripts.train_mlp import SoftPromptMLP, load_soft_prompt_mlp

# > 批次預測語義向量腳本 (Batch Prediction)
# - 目標：將所有歌曲特徵通過 Fine-tuned 模型轉化為語義向量 (Soft Prompts)

def batch_generate_soft_prompts(df_processed, df_pca, model_path="data/soft_prompt_mlp_finetuned.pth", output_path="data/soft_prompts_map.pkl"):
	"""
	將 Dataframe 中的所有歌曲轉化為 MLP 預測向量並存入字典。
	"""
	if not os.path.exists(model_path):
		print(f"// !! 找不到微調模型: {model_path}，嘗試載入基礎預訓練模型...")
		model_path = "data/soft_prompt_mlp.pth"
		if not os.path.exists(model_path):
			raise FileNotFoundError("找不到任何 MLP 模型檔案，請先進行訓練。")

	# 1. 載入模型
	model, _ = load_soft_prompt_mlp(model_path)
	model.eval()
	print(f"// > 已載入模型: {model_path}，準備轉換 {len(df_pca)} 首歌曲...")

	# 2. 準備特徵矩陣 (與訓練/微調時邏輯完全一致)
	pca_cols = [c for c in df_pca.columns if c.startswith('PC_')]
	raw_numeric_cols = [
		'popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 
		'key', 'loudness', 'mode', 'speechiness', 'acousticness', 
		'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'
	]
	raw_numeric_cols = [c for c in raw_numeric_cols if c in df_processed.columns]
	
	print(f"// > 正在提取特徵: PCA ({len(pca_cols)}) + 原始數值 ({len(raw_numeric_cols)})")
	
	# 對齊 df_pca 與 df_processed (確保筆順一致)
	# 這裡我們以 df_pca 為基準進行拼接
	df_combined = pd.concat([
		df_pca.set_index('track_id')[pca_cols],
		df_processed.set_index('track_id')[raw_numeric_cols]
	], axis=1).reset_index()

	track_ids = df_combined['track_id'].values
	X_values = df_combined.drop(columns=['track_id']).values.astype('float32')

	# --- 維度驗證 (嚴格對齊) ---
	expected_dim = next(model.parameters()).size(1)
	if X_values.shape[1] != expected_dim:
		print(f"// !! 數據維度 ({X_values.shape[1]}) 與模型維度 ({expected_dim}) 不符，進行對齊處理...")
		if X_values.shape[1] > expected_dim:
			X_values = X_values[:, :expected_dim]
		else:
			padding = np.zeros((X_values.shape[0], expected_dim - X_values.shape[1]))
			X_values = np.hstack([X_values, padding]).astype('float32')

	# 3. 執行 Inference (Inference 不需梯度)
	print("// > 正在進行批次預測 (Inference)...")
	with torch.no_grad():
		X_tensor = torch.tensor(X_values)
		pred_embeddings = model(X_tensor).numpy()

	# 4. 建立 Dictionary
	# Format: { track_id: array([embedding_vector]) }
	soft_prompts_map = {tid: vec for tid, vec in zip(track_ids, pred_embeddings)}

	# 5. 存檔
	os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
	with open(output_path, 'wb') as f:
		pickle.dump(soft_prompts_map, f)
	
	print(f"// @ 語義向量字典建置完成！已存至: {output_path}")
	print(f"   - 總歌曲數: {len(soft_prompts_map)}")
	
	return soft_prompts_map
