import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import json
import pickle
from torch.utils.data import DataLoader, TensorDataset
from scripts.train_mlp import SoftPromptMLP, load_soft_prompt_mlp

# > MLP 微調區塊 (Fine-tuning)
# - 目標：讓模型學會 Persona 的具體偏好，將預估的 Soft Prompt 往用戶喜好的區間偏移

def prepare_finetune_data(df_processed, df_pca, history_dir="data/persona_listening_histories", similarity_path="data/precomputed_similar_songs.json"):
	"""
	整合 [DATA 1] 偽共現關係與 [DATA 2] 虛擬聽歌歷史，建立混合微調數據。
	"""
	from sentence_transformers import SentenceTransformer
	model_st = SentenceTransformer('all-MiniLM-L6-v2')
	
	X_list = []
	Y_list = []
	
	# 建立處理過的特徵查表
	pca_cols = [c for c in df_pca.columns if c.startswith('PC_')]
	feat_cols = pca_cols.copy()
	if 'explicit' in df_processed.columns: feat_cols.append('explicit')
	if 'instrumentalness_binary' in df_processed.columns: feat_cols.append('instrumentalness_binary')
	
	df_lookup = pd.concat([df_pca, df_processed[['explicit', 'instrumentalness_binary']]], axis=1)
	df_lookup = df_lookup.set_index('track_id')

	# --- 1. [DATA 1] 處理偽共現關係 (Pseudo Co-occurrence) ---
	if os.path.exists(similarity_path):
		print(f"// > 正在從 {similarity_path} 讀取共現關係數據...")
		with open(similarity_path, 'r', encoding='utf-8') as f:
			similarity_data = json.load(f)
		
		# 抓取每首歌最相似的 Top-1 做為對齊目標 (學會：特徵 A -> 指向歌曲 B 的語義空間)
		co_count = 0
		for item in similarity_data:
			tid = item.get('track_id')
			sim_list = item.get('top_n_similar_songs', [])
			
			if tid in df_lookup.index and len(sim_list) > 0:
				# 取 Top-1
				sim_song = sim_list[0]
				# X: 歌曲 A 的特徵
				x_feat = df_lookup.loc[tid][feat_cols].values.astype('float32')
				# Y: 歌曲 B 的 Embedding
				text_for_y = f"{sim_song['track_name']} {sim_song['artists']} {sim_song.get('track_genre', '')}"
				y_emb = model_st.encode(text_for_y)
				
				X_list.append(x_feat)
				Y_list.append(y_emb)
				co_count += 1
		print(f"// @ [DATA 1] 加入了 {co_count} 筆共現樣本。")

	# --- 2. [DATA 2] 處理客製化 Persona 收聽歷史 ---
	if os.path.exists(history_dir):
		print(f"// > 正在從 {history_dir} 收集畫像數據...")
		persona_count = 0
		for filename in os.listdir(history_dir):
			if filename.endswith(".json") and filename != "persona_summaries.json":
				with open(os.path.join(history_dir, filename), 'r', encoding='utf-8') as f:
					data = json.load(f)
					history = data.get('listening_history', [])
				
				for song in history:
					tid = song.get('track_id')
					if tid in df_lookup.index:
						x_feat = df_lookup.loc[tid][feat_cols].values.astype('float32')
						text_for_y = f"{song['track_name']} {song['artists']} {song.get('track_genre', '')}"
						y_emb = model_st.encode(text_for_y)
						X_list.append(x_feat)
						Y_list.append(y_emb)
						persona_count += 1
		print(f"// @ [DATA 2] 加入了 {persona_count} 筆畫像樣本。")

	# --- 3. 維度對齊檢查 ---
	if not X_list:
		raise ValueError("沒有收集到任何有效的訓練數據。")
	
	# 找出最常見的維度
	lengths = [len(x) for x in X_list]
	common_len = max(set(lengths), key=lengths.count)
	
	valid_X = []
	valid_Y = []
	for x, y in zip(X_list, Y_list):
		if len(x) == common_len:
			valid_X.append(x)
			valid_Y.append(y)
		else:
			print(f"// !! 跳過維度不符的樣本: 預期 {common_len}, 實際 {len(x)}")

	X = np.array(valid_X).astype('float32')
	Y = np.array(valid_Y).astype('float32')
	print(f"// @ 總計收集到 {len(X)} 筆混合微調樣本 (原始總數: {len(X_list)})。")
	return X, Y

def finetune_soft_prompt_mlp(df_processed, df_pca, base_model_path="data/soft_prompt_mlp.pth", output_path="data/soft_prompt_mlp_finetuned.pth", epochs=20):
	"""
	載入預訓練模型，並針對 Persona 數據進行微調。
	"""
	# 偵測是否已有微調模型
	if os.path.exists(output_path):
		print(f"// @ 偵測到現有的微調模型: {output_path}，直接載入...")
		checkpoint = torch.load(output_path)
		model = SoftPromptMLP(checkpoint['input_dim'], checkpoint['output_dim'])
		model.load_state_dict(checkpoint['model_state_dict'])
		return model, checkpoint.get('loss_history', [])

	# 1. 載入基礎模型
	base_model, _ = load_soft_prompt_mlp(base_model_path)
	if base_model:
		# 獲取預訓練模型所預期的輸入維度
		expected_dim = next(base_model.parameters()).size(1)
		print(f"// > 模型預期輸入維度: {expected_dim}")
	else:
		raise ValueError("找不到基礎模型 (Pre-trained)，請先執行預訓練。")
	
	# 2. 準備數據
	X, Y = prepare_finetune_data(df_processed, df_pca)
	
	# 強制維度對齊 (對齊模型的 input_dim)
	if X.shape[1] != expected_dim:
		print(f"// !! 微調數據維度 ({X.shape[1]}) 與模型維度 ({expected_dim}) 不符，進行自動對齊...")
		if X.shape[1] > expected_dim:
			X = X[:, :expected_dim] # 截斷
		else:
			padding = np.zeros((X.shape[0], expected_dim - X.shape[1]))
			X = np.hstack([X, padding]) # 補零
	
	dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))
	dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

	# 3. 設定訓練參數 (使用較小的學習率)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = base_model.to(device)
	model.train()
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=0.0001) # 微調通常使用較小 LR

	# 4. 微調迴圈
	print(f"// > 開始對 Persona 進行模型微調...")
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
		if (epoch+1) % 5 == 0:
			print(f"   - Fine-tune Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

	# 5. 存檔
	torch.save({
		'model_state_dict': model.state_dict(),
		'input_dim': X.shape[1],
		'output_dim': Y.shape[1],
		'loss_history': loss_history,
		'type': 'finetuned'
	}, output_path)

	print(f"// @ 微調完成並存至: {output_path}")
	return model.cpu(), loss_history
