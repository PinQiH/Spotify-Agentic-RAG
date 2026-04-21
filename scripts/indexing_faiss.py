import pandas as pd
import numpy as np
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# > FAISS 向量索引建置區塊
# - 使用 SentenceTransformer 進行 Embedding 並建立 FAISS 索引
def build_faiss_index(df_with_docs, index_path="data/spotify.index", meta_path="data/spotify_meta.pkl", model_name='all-MiniLM-L6-v2'):
	"""
	將帶有 rag_doc 的歌曲資料轉換為向量並建立 FAISS 索引。
	同時存儲 Metadata 以便 app.py 連結。
	若 index_path 與 meta_path 已存在，則載入現有檔案。
	"""
	if os.path.exists(index_path) and os.path.exists(meta_path):
		print(f"// @ 偵測到現有的 FAISS 索引與 Metadata，直接載入...")
		index = faiss.read_index(index_path)
		with open(meta_path, 'rb') as f:
			meta = pickle.load(f)
		return index, meta

	print(f"// > 正在初始化 Embedding 模型: {model_name}...")
	model = SentenceTransformer(model_name)

	# 1. 準備內容：結合 歌名 + 藝人 + 體裁 作為 Embedding 的主要輸入
	# 這是為了確保語義檢索能包含基本資訊
	print(f"// > 正在計算 {len(df_with_docs)} 首歌的 Embeddings...")
	texts = [
		f"{row['track_name']} {row['artists']} {row['track_genre']}" 
		for _, row in df_with_docs.iterrows()
	]
	
	embeddings = model.encode(texts, show_progress_bar=True)
	embeddings = np.array(embeddings).astype('float32')

	# 2. 正規化以使用餘弦相似度 (透過 Inner Product Index)
	faiss.normalize_L2(embeddings)

	# 3. 建立 FAISS 索引 (IndexFlatL2 或 IndexFlatIP)
	dimension = embeddings.shape[1]
	index = faiss.IndexFlatL2(dimension)
	index.add(embeddings)

	# 4. 準備 Metadata List (與 app.py 格式對齊)
	# app.py 預期是一個包含 dict 的 list，對應 index 的順序
	metadata_list = df_with_docs.to_dict('records')

	# 5. 存檔
	os.makedirs(os.path.dirname(index_path) or '.', exist_ok=True)
	faiss.write_index(index, index_path)
	with open(meta_path, 'wb') as f:
		pickle.dump(metadata_list, f)

	print(f"// @ FAISS 索引已建置完成。")
	print(f"   - 索引檔: {index_path}")
	print(f"   - Metadata: {meta_path}")

	return index, metadata_list
