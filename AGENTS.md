# 🤖 Spotify Agentic RAG: Architecture & Concepts

本文檔詳細說明了本系統如何模擬「Agentic AI」的思考與決策過程，以及底層的技術實現細節。

## 🧠 核心概念: Agentic RAG

傳統的推薦系統通常是「黑盒子」，直接給出結果。本系統採用的 **Agentic RAG (Retrieval-Augmented Generation)** 架構，旨在讓 AI 像人類 DJ 或資深樂迷一樣思考：

1.  **感知 (Perception)**: 理解當前的音樂情境 (Context Song) 與用戶偏好 (Persona)。
2.  **檢索 (Retrieval)**: 從龐大的資料庫中撈取「可能適合」的候選集。
3.  **過濾 (Filtering)**: 根據語義理解 (Semantic Search) 去除不相關的噪音。
4.  **推理 (Reasoning)**: 最後再精選出最好的幾首，並給出「為什麼推薦」的理由。

## 🏗️ 系統架構流程

系統運作分為三個核心步驟 (Steps)，在 UI 上即時呈現：

### Step 1: Candidates from Pre-computed Similar Items (快速檢索)
*   **目標**: 快速縮小範圍，找出與當前歌曲「聽感相似」的基礎候選集。
*   **方法**: 使用預先計算好的 **Co-occurrence Matrix (共現矩陣)** 或 **Content-based Filtering** (基於音訊特徵如 Tempo, Energy, Valence)。
*   **技術**: `Pandas` 過濾, `Pre-computed JSON`。
*   **結果**: 產生約 20-50 首候選歌曲。

### Step 2: Candidates from ChromaDB Semantic Search (語義過濾)
*   **目標**: 引入「語義理解」，找出風格、氛圍描述相近的歌曲。
*   **方法**: 將歌曲的 Metadata (Genre, Mood, Description) 轉化為 Vector Embeddings。
*   **技術**: 
    *   **Vector DB**: `ChromaDB` (or `FAISS` in this demo implementation).
    *   **Embeddings**: `all-MiniLM-L6-v2` (Sentence Transformers).
*   **結果**: 從 Step 1 的結果中進一步篩選，或從向量資料庫補充 Step 1 沒抓到的隱藏好歌。

### Step 3: LLM Re-ranking & Reasoning (深度推理)
*   **目標**: 模擬人類專家的最終決策，並生成解釋。
*   **方法**: 將 Step 1 & 2 的候選歌曲清單，連同 User Persona 與當前歌曲資訊，打包成 Prompt 送給 LLM。
*   **技術**: `OpenAI GPT-4o-mini` / `Gemini 2.0` / `Grok 4.1`。
*   **Prompt 策略**: 要求 LLM 扮演 "Spotify Music Expert"，根據 Persona (如 "Party Animal") 的口味進行排序，並為前 3 名生成一句話的推薦理由。

## 📊 視覺化與互動 (Visualization & Interaction)

為了讓 Agent 的思考過程「可解釋 (Explainable)」，我們實作了以下功能：

*   **PCA Embedding Space**: 使用 Principal Component Analysis (PCA) 將高維度的歌曲向量降維到 2D 平面。
    *   **點的顏色**: 代表不同階段的數據 (歷史紀錄、Step 1 候選、Step 2 候選、最終推薦)。
    *   **距離意義**: 點越靠近，代表語義/風格越相似。這讓用戶能直觀看到 Agent 是如何在向量空間中「尋找」答案的。
*   **Interactive UI**: 使用 Streamlit 的 `st.status` 元件動態展示每個 Step 的處理進度，讓等待過程不再枯燥。

## ☁️ 雲端整合 (Cloud Integration)

*   **Google Sheets**: 用於持久化存儲用戶的投票結果 (Feedback Loop)。這是一個簡單但有效的 MLOps 實踐，用於收集真實用戶數據以優化未來的模型。
*   **Caching**: 大量使用 `st.cache_data` 與 `session_state` 來優化體驗，避免重複呼叫 LLM API，節省成本並降低延遲。

## 📝 未來展望

*   **真正的 RAG**: 目前主要依賴 Metadata，未來可整合歌詞 (Lyrics) 或樂評文章 (Reviews) 進入 Vector DB。
*   **Audio RAG**: 直接對音訊波形 (Waveform) 進行 Embedding，實現「聽起來像」的搜尋。
