import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import pickle
import sys
import io
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# 加入路徑與編碼
sys.path.append(os.getcwd())
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# > MLP V2: 加入 BatchNormalization 與 Robust Scaling
class SoftPromptMLPV2(nn.Module):
	def __init__(self, input_dim, output_dim=384):
		super(SoftPromptMLPV2, self).__init__()
		self.network = nn.Sequential(
			nn.Linear(input_dim, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			nn.Dropout(0.2),
			
			nn.Linear(256, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Dropout(0.2),
			
			nn.Linear(512, output_dim)
		)
	
	def forward(self, x):
		return self.network(x)

def train_new_model():
	print("🏗️ [TRAIN V2] 開始構建健康的 MLP 模型...")
	
	# 1. 載入資料
	df_processed = pd.read_csv("data/processed_songs.csv").drop_duplicates(subset=['track_id'])
	df_pca = pd.read_csv("data/pca_songs.csv").drop_duplicates(subset=['track_id'])
	
	with open("data/spotify_meta.pkl", "rb") as f:
		metadata = pickle.load(f)
	
	# 2. 對齊資料
	track_ids_meta = [str(m['track_id']) for m in metadata]
	df_pca_indexed = df_pca.set_index('track_id')
	df_proc_indexed = df_processed.set_index('track_id')
	
	valid_ids = [tid for tid in track_ids_meta if tid in df_pca_indexed.index and tid in df_proc_indexed.index]
	print(f"   - 有效對齊數據: {len(valid_ids)} 筆")
	
	# 3. 準備 X (特徵提取 + Scaling)
	pca_cols = [c for c in df_pca.columns if c.startswith('PC_')]
	raw_cols = [
		'popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 
		'key', 'loudness', 'mode', 'speechiness', 'acousticness', 
		'instrumentalness', 'liveness', 'valence', 'tempo'
	]
	raw_cols = [c for c in raw_cols if c in df_processed.columns]
	
	X_raw = pd.concat([
		df_pca_indexed.loc[valid_ids][pca_cols],
		df_proc_indexed.loc[valid_ids][raw_cols]
	], axis=1).values.astype('float32')
	
	# 重要：加入 StandardScaler
	scaler = StandardScaler()
	X = scaler.fit_transform(X_raw)
	
	# 儲存 Scaler 以供 Inference 使用
	with open("data/feature_scaler.pkl", "wb") as f:
		pickle.dump(scaler, f)
	
	# 4. 準備 Y (Embeddings)
	# 這裡我們需要重新計算 Y 或從 Metadata 拿
	# 如果 metadata 裡沒存，我們現場用 SentenceTransformer 算 (這裡假設 metadata 裡面的順序與 valid_ids 對應)
	from sentence_transformers import SentenceTransformer
	print("   - 正在計算訓練目標 Embeddings (Y)...")
	st_model = SentenceTransformer('all-MiniLM-L6-v2')
	id_to_meta = {str(m['track_id']): m for m in metadata}
	texts = [f"{id_to_meta[tid]['track_name']} {id_to_meta[tid]['artists']}" for tid in valid_ids]
	Y = st_model.encode(texts).astype('float32')
	
	# 5. 訓練
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = SoftPromptMLPV2(input_dim=X.shape[1], output_dim=Y.shape[1]).to(device)
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # 加入 L2 正則化
	
	dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))
	loader = DataLoader(dataset, batch_size=64, shuffle=True)
	
	print(f"🚀 開始訓練 (Epochs: 100, LR: 0.001)...")
	model.train()
	for epoch in range(100):
		total_loss = 0
		for bx, by in loader:
			bx, by = bx.to(device), by.to(device)
			optimizer.zero_grad()
			out = model(bx)
			loss = criterion(out, by)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		
		if (epoch+1) % 20 == 0:
			print(f"   * Epoch {epoch+1:3d} | Loss: {total_loss/len(loader):.6f}")

	# 6. 存檔 (覆蓋掉那個懷掉的 finetuned)
	save_path = "data/soft_prompt_mlp_finetuned.pth"
	torch.save({
		'model_state_dict': model.state_dict(),
		'input_dim': X.shape[1],
		'output_dim': Y.shape[1],
		'model_type': 'v2_robust'
	}, save_path)
	
	print(f"✅ 模型 V2 訓練完成，已存至: {save_path}")

if __name__ == "__main__":
	train_new_model()
