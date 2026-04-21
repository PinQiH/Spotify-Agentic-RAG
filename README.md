# 🎵 Spotify Agentic RAG: 智慧音樂推薦系統

這是一個結合了 **資料前處理**、**機器學習 (MLP Soft-Prompting)**、**向量檢索 (FAISS)** 與 **大型語言模型 (Agentic Re-ranking)** 的音樂推薦實驗專案。

## 🚀 專案特點
- **Agentic RAG 架構**：不只是簡單的相似度計算，系統會像音樂專家一樣「思考」並給出推薦理由。
- **MLP Soft-Prompting**：利用多層感知器將音訊數值特徵映射到語義嵌入空間，橋接數值與文字的鴻溝。
- **三階段推理管線**：
    1. **數值檢索**：快速定位原始聽感相似的歌曲。
    2. **語義檢索**：透過向量資料庫找尋風格、氛圍相近的候選。
    3. **Agent 推理**：呼叫 GPT-4o / Gemini / Grok 進行品味排序與理由生成。
- **多模型對比**：支援在 UI 中並排觀察不同 AI DJ 的品味差異。

## 🛠️ 環境需求
- Python 3.9+
- 建議使用虛擬環境：
  ```bash
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1  # Windows
  pip install -r requirements.txt
  ```

## 📂 快速啟動指南

### Step 1: 資料準備與模型訓練
開啟 `soft_prompt_rag.ipynb`，依序執行所有儲存格：
1. **資料前處理**：清洗與特徵工程。
2. **語義索引建構**：建立 FAISS 向量資料庫。
3. **MLP 預訓練與微調**：建立特徵與語義間的映射。
4. **畫像分析**：透過 LLM 生成 Persona 的長期偏好總結。

### Step 2: 啟動 Streamlit 應用程式
在終端機執行：
```bash
streamlit run app.py
```

## 🏗️ 核心模組說明
- `scripts/data_processing.py`: 清洗、Log 轉換、Scaling 與 Cyclic Encoding。
- `scripts/indexing_faiss.py`: 產出 ST 嵌入向量並建立向量空間索引。
- `scripts/train_mlp.py` & `finetune_mlp.py`: 負責軟提示映射與品味對齊。
- `scripts/recommender_agent.py`: 整合多模型 (GPT-4o, Gemini, Grok) 的推理大腦。

## 📊 資料來源
本專案使用 Spotify 歌曲特徵資料集，包含 11 項音訊屬性 (BPM, Energy, Valence 等) 與 Metadata。

---
*本專案為台北市立大學 資訊科學系 在職專班 資料探勘課程期末專題。*
