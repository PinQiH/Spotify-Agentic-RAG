# ... (imports handled separately if needed, but I'll replace the whole file content or large chunks)

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import time
import sys
from dotenv import load_dotenv # Changed from google.colab import userdata
import openai # Import OpenAI library

# Load environment variables from .env file
load_dotenv()

def fallback_doc(row) -> str:
    # 沒有 LLM key 時的保底版本（可跑 pipeline）
    tn = row.get("track_name", "")
    ar = row.get("artists", "")
    album = row.get("album_name", "")
    tags = row.get("semantic_tags", "")
    tempo = float(row.get("tempo", 0))
    energy = float(row.get("energy", 0))
    valence = float(row.get("valence", 0))
    danceability = float(row.get("danceability", 0))

    base = f"《{tn}》由 {ar} 演出"
    if album:
        base += f"，收錄於《{album}》"
    base += f"。這首歌的氛圍偏向 {tags}，節奏約 {tempo:.0f} BPM，能量 {energy:.2f}、正向情緒 {valence:.2f}、舞動感 {danceability:.2f}。"
    # 控制長度大概 50-100 字（粗略）
    return base[:120]

def build_prompt(row) -> str:
    tn = row.get("track_name", "")
    ar = row.get("artists", "")
    album = row.get("album_name", "")
    tags = row.get("semantic_tags", "")
    tempo = float(row.get("tempo", 0))
    energy = float(row.get("energy", 0))
    valence = float(row.get("valence", 0))
    danceability = float(row.get("danceability", 0))

    return f"""
你是一個音樂推薦文案生成器。請根據歌曲資訊生成一段「50-100字」的自然語言推薦介紹（繁體中文），語氣像串流平台推薦卡片。
必須包含：情緒/氛圍（semantic_tags）、至少兩個數值特徵（tempo/energy/valence/danceability）並自然融入，不要列點，不要出現欄位名。

歌曲資訊：
- 歌名：{tn}
- 藝人：{ar}
- 專輯：{album}
- semantic_tags：{tags}
- tempo：約 {tempo:.1f}
- energy：{energy:.2f}
- valence：{valence:.2f}
- danceability：{danceability:.2f}
""".strip()

def main():
    # Set default paths relevant to the project structure
    default_input = Path("data/spotify_semantic_mapped.csv")
    default_output = Path("data/spotify_rag_doc.csv")

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(default_input), help="input CSV path (must have semantic_tags)")
    ap.add_argument("--output", default=str(default_output), help="output CSV path (with rag_doc)")
    ap.add_argument("--id-col", default="track_id", help="track id column name (default: track_id)")
    ap.add_argument("--model", default="gpt-4o", help="OpenAI model name") 
    ap.add_argument("--limit", type=int, default=500, help="debug: only generate first N rows (0 = all)")
    ap.add_argument("--sleep", type=float, default=0.0, help="sleep seconds between calls")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    
    # Check if input file exists
    if not in_path.exists():
        print(f"❌ 錯誤：找不到輸入檔案 '{in_path}'。請確認檔案路徑。")
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {in_path}...")
    df = pd.read_csv(in_path)

    # 補值
    num_cols = df.select_dtypes(include=[np.number]).columns
    obj_cols = df.select_dtypes(include=[object]).columns
    df[num_cols] = df[num_cols].fillna(0)
    df[obj_cols] = df[obj_cols].fillna("")

    if args.id_col not in df.columns:
        df[args.id_col] = df.index.astype(str)

    if "semantic_tags" not in df.columns:
        raise ValueError("Input CSV must contain semantic_tags. Run 03_semantic_mapping.py first.")

    # Get API KEY from environment
    OPENAI_API_KEY = os.getenv('GPT_API_KEY')
    use_llm = bool(OPENAI_API_KEY)

    client = None 
    if use_llm:
        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            print(f"成功初始化 OpenAI 客戶端，將使用模型: {args.model}")
        except Exception as e:
            print(f"錯誤：無法初始化 OpenAI 客戶端。請檢查您的 GPT_API_KEY 是否有效。錯誤訊息：{e}")
            use_llm = False
    else:
        print("⚠️ 警告：未找到 GPT_API_KEY 環境變數，將使用 fallback_doc 生成簡易描述。")

    n = len(df) if args.limit <= 0 else min(args.limit, len(df))
    rag_docs = []
    
    print(f"準備處理前 {n} 筆資料 (總數: {len(df)})...")

    for i in range(n):
        row = df.iloc[i].to_dict()
        if (i+1) % 10 == 0 or i == 0:
            print(f"Processing song {i+1}/{n}: {row.get('track_name', 'Unknown Track')}")

        if not use_llm or client is None:
            rag_docs.append(fallback_doc(row))
            continue

        prompt = build_prompt(row)
        try:
            # OpenAI API call
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150, 
                temperature=0.7,
            )
            text = resp.choices[0].message.content.strip()
            if not text:
                text = fallback_doc(row)
        except openai.APIStatusError as e: 
            print(f"OpenAI API 呼叫失敗 (狀態碼: {e.status_code})：{e}")
            if e.status_code == 429:
                print("可能是請求過多，請稍後再試或增加延遲時間。")
            text = fallback_doc(row)
        except Exception as e:
            print(f"API 呼叫發生未知錯誤：{e}")
            text = fallback_doc(row)

        rag_docs.append(text)
        if args.sleep > 0:
            time.sleep(args.sleep)

    # 若只跑 limit，其他用 fallback 補齊
    if n < len(df):
        print(f"補齊剩餘 {len(df) - n} 筆資料 (使用 fallback)...")
        for i in range(n, len(df)):
            rag_docs.append(fallback_doc(df.iloc[i].to_dict()))

    df["rag_doc"] = rag_docs
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved: {out_path}")
    print(df[[args.id_col, "track_name", "artists", "semantic_tags", "rag_doc"]].head(5))

if __name__ == "__main__":
    main()