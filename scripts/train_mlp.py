import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import pickle
from torch.utils.data import DataLoader, TensorDataset

# > MLP 軟提示模型定義 (Soft Prompt MLP)
# - 目標：從音訊與 PCA 特徵預測語義向量 (Embedding)
class SoftPromptMLP(nn.Module):
    def __init__(self, input_dim, output_dim=384): # all-MiniLM-L6-v2 預設是 384
        super(SoftPromptMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# > MLP V2: 加入 BatchNormalization 與 Robust Scaling (全新健全架構)
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

# > 載入模型函式
def load_soft_prompt_mlp(model_path="data/soft_prompt_mlp.pth"):
    """載入現有的模型權重與訓練紀錄，自動辨識 V1 或 V2"""
    if not os.path.exists(model_path):
        return None, None
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model_type = checkpoint.get('model_type', 'v1')
    
    input_dim = checkpoint['input_dim']
    output_dim = checkpoint['output_dim']
    
    if model_type == 'v2_robust':
        model = SoftPromptMLPV2(input_dim=input_dim, output_dim=output_dim)
    else:
        model = SoftPromptMLP(input_dim=input_dim, output_dim=output_dim)
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('loss_history', [])

# > 訓練管線
def train_soft_prompt_mlp(df_processed, df_pca, meta_path="data/spotify_meta.pkl", model_save_path="data/soft_prompt_mlp.pth", epochs=50, batch_size=32, force_train=False):
	"""
	從處理過的特徵與 FAISS Metadata 中的 Embedding 進行配對並訓練 MLP。
	若 model_save_path 已存在且 force_train=False，則直接載入模型。
	"""
	# 偵測快取
	if not force_train:
		model, loss_history = load_soft_prompt_mlp(model_save_path)
		if model is not None:
			print(f"// @ 偵測到現有的 MLP 模型檔案: {model_save_path}，直接載入...")
			return model, loss_history

	# 1. 載入 Y (Embeddings) - 從此前生成的索引 Metadata 中讀取
	if not os.path.exists(meta_path):
		raise FileNotFoundError(f"找不到索引元數據: {meta_path}。請先執行向量資料庫建置。")
	
	with open(meta_path, 'rb') as f:
		metadata = pickle.load(f)
	
	# 建立 track_id 到 index 的映射，確保數據對齊
	# meta_df = pd.DataFrame(metadata) # 不需要的臨時變數
	
	from sentence_transformers import SentenceTransformer
	print("// > 重新計算訓練目標 (Y: Embeddings)...")
	model_st = SentenceTransformer('all-MiniLM-L6-v2')
	texts = [f"{row['track_name']} {row['artists']} {row['track_genre']}" for row in metadata]
	Y = model_st.encode(texts)
	Y = np.array(Y).astype('float32')

	# 2. 準備 X (Features) 與資料對齊
	# 強制對輸入數據進行去重，避免 loc 回傳多筆重複 ID 的資料
	df_pca_clean = df_pca.drop_duplicates(subset=['track_id'])
	df_proc_clean = df_processed.drop_duplicates(subset=['track_id'])

	# 我們以 metadata 裡的順序為基準
	track_ids_in_meta = [str(row['track_id']) for row in metadata]
	
	# 對齊 df_pca 與 df_processed (建立索引後快速查詢)
	df_pca_indexed = df_pca_clean.set_index('track_id')
	df_proc_indexed = df_proc_clean.set_index('track_id')
	
	# 提取對齊後的資料
	valid_ids = [tid for tid in track_ids_in_meta if tid in df_pca_indexed.index]
	
	if len(valid_ids) < len(track_ids_in_meta):
		print(f"// !! 警告: 有 {len(track_ids_in_meta) - len(valid_ids)} 筆 Metadata 找不到對應特徵，將進行過濾。")

	# @ 重新對齊並計算 Y 以確保與 X 行數相等 (2000 -> 2000)
	from sentence_transformers import SentenceTransformer
	print(f"// > 正在對齊 {len(valid_ids)} 筆訓練目標 (Y: Embeddings)...")
	model_st = SentenceTransformer('all-MiniLM-L6-v2')
	
	# 建立 ID 到 Metadata 的映射，快速取得文字描述
	id_to_meta = {str(m['track_id']): m for m in metadata}
	texts = [f"{id_to_meta[tid]['track_name']} {id_to_meta[tid]['artists']} {id_to_meta[tid]['track_genre']}" for tid in valid_ids]
	Y = model_st.encode(texts)
	Y = np.array(Y).astype('float32')

	df_pca_aligned = df_pca_indexed.loc[valid_ids].reset_index()
	df_proc_aligned = df_proc_indexed.loc[valid_ids].reset_index()
	
	# --- 特徵組裝 ---
	pca_cols = [c for c in df_pca_aligned.columns if c.startswith('PC_')]
	raw_numeric_cols = [
		'popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 
		'key', 'loudness', 'mode', 'speechiness', 'acousticness', 
		'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'
	]
	raw_numeric_cols = [c for c in raw_numeric_cols if c in df_proc_aligned.columns]
	
	print(f"// > 正在組裝特徵: PCA ({len(pca_cols)}) + 原始數值 ({len(raw_numeric_cols)})")
	
	X_df = pd.concat([
		df_pca_aligned[pca_cols],
		df_proc_aligned[raw_numeric_cols]
	], axis=1)
	
	X = X_df.values.astype('float32')

	print(f"// > 訓練數據準備完成: X shape {X.shape}, Y shape {Y.shape}")

	# 3. 轉換為 PyTorch Tensor
	X_tensor = torch.tensor(X)
	Y_tensor = torch.tensor(Y)
	dataset = TensorDataset(X_tensor, Y_tensor)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	# 4. 初始化模型
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = SoftPromptMLP(input_dim=X.shape[1], output_dim=Y.shape[1]).to(device)
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	
	# 加入縮減學習率調度器 (Scheduler)
	# 當 avg_loss 在 5 個 epoch 內沒有下降超過 1e-4 時，將學習率縮減為原來的 0.5 倍
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

	# 5. 訓練迴圈
	print(f"// > 開始訓練 (Device: {device})...")
	model.train()
	loss_history = []
	for epoch in range(epochs):
		epoch_loss = 0
		for batch_X, batch_Y in dataloader:
			batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
			
			optimizer.zero_grad()
			outputs = model(batch_X)
			loss = criterion(outputs, batch_Y)
			loss.backward()
			optimizer.step()
			
			epoch_loss += loss.item()
		
		avg_loss = epoch_loss / len(dataloader)
		loss_history.append(avg_loss)
		
		# 執行學習率調整步驟
		scheduler.step(avg_loss)
		
		if (epoch + 1) % 10 == 0:
			# 獲取當前學習率
			current_lr = optimizer.param_groups[0]['lr']
			print(f"   - Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, LR: {current_lr}")

	# 6. 存檔
	os.makedirs(os.path.dirname(model_save_path) or '.', exist_ok=True)
	torch.save({
		'model_state_dict': model.state_dict(),
		'input_dim': X.shape[1],
		'output_dim': Y.shape[1],
		'loss_history': loss_history
	}, model_save_path)
	
	print(f"// @ MLP 模型訓練完成並存至: {model_save_path}")
	return model.cpu(), loss_history
