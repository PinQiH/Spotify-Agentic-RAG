import faiss
import numpy as np
import pickle
import os
import pandas as pd

def build_index_from_soft_prompts(df, soft_prompts_map, index_path="data/spotify.index", meta_path="data/spotify_faiss_meta.pkl"):
    """
    將批次預測出的 Soft Prompts 轉化為 FAISS 索引與對齊元數據。
    這對於規模化 (20,000+) 檢索非常重要。
    """
    print(f"// > 正在準備 {len(soft_prompts_map)} 筆向量進行索引化...")
    
    # 1. 準備矩陣
    track_ids = list(soft_prompts_map.keys())
    vectors = np.array([soft_prompts_map[tid] for tid in track_ids]).astype('float32')

    # 2. L2 正規化 (使其 Inner Product 等於 Cosine Similarity)
    faiss.normalize_L2(vectors)

    # 3. 建立索引 (IndexFlatIP)
    d = vectors.shape[1] 
    full_index = faiss.IndexFlatIP(d)
    full_index.add(vectors)

    # 4. 準備與索引嚴格對齊的元數據
    print("// > 正在收集元數據對照表...")
    faiss_metadata = []
    # 建立一個快速查詢字典
    df_lookup = df.set_index('track_id')
    
    for tid in track_ids:
        if tid in df_lookup.index:
            row = df_lookup.loc[tid]
            # 處理可能的多行結果 (雖然 track_id 應為唯一)
            if isinstance(row, pd.DataFrame): row = row.iloc[0]
            
            faiss_metadata.append({
                'track_id': tid,
                'track_name': row['track_name'],
                'artists': row['artists'],
                'album_name': row['album_name'],
                'track_genre': row['track_genre']
            })

    # 5. 儲存結果
    if not os.path.exists("data"): os.makedirs("data")
    
    print(f"// > 正在儲存索引至 {index_path}...")
    faiss.write_index(full_index, index_path)
    
    print(f"// > 正在儲存元數據至 {meta_path}...")
    with open(meta_path, "wb") as f:
        pickle.dump(faiss_metadata, f)

    print(f"// @ 全量索引建置完成！共計 {full_index.ntotal} 筆資料已上線。")
    return full_index, faiss_metadata

if __name__ == "__main__":
    # 此腳本供模組化呼叫使用
    pass
