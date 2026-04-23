# 🛠️ Spotify Agentic RAG: 技術實作與合規說明

本文件旨在詳細交代 **Spotify Agentic RAG** 系統的技術規格、研發背景、資安設計以及 AI 技術的實踐程度。

---

## 📊 1. AI 工具使用範圍與數據集

### 核心數據集

- **數據來源**: Spotify Tracks Dataset。
- **數據規模**: 原始數據包含約 11 萬筆歌曲記錄；本專案提取並標籤化約 **2 萬筆** 核心歌曲作為基礎。
- **技術特徵**: 包含分析 11 項關鍵音訊維度（Acousticness, Danceability, Energy, Valence, Tempo 等）與 Metadata（Genre, Popularity）。

### 關鍵 AI 工具

- **向量資料庫**: `FAISS (Facebook AI Similarity Search)`，用於高效的語義空間檢索。
- **LLM 推理引擎**: 整合多模型介面，包含 `OpenAI GPT-4o-mini`、`Google Gemini 2.0 Flash`、以及 `xAI Grok 4.1 Fast`。
- **Embedding 模型**: 採用 `Sentence Transformers (all-MiniLM-L6-v2)` 產生 384 維的語義密集向量。

---

## 👥 2. 團隊分工與技術來源

### 團隊組成與分工

本專案由 **台北市立大學 資訊科學系 在職專班「資料探勘」課程專題團隊** 研發，成員職責如下：

- **演算法工程**: 負責 Agentic RAG 管線建構與 MLP 權重映射學習。
- **資料科學**: 執行特徵工程、PCA 降維視覺化以及 Persona 數據生成。
- **後端與 UI 集成**: 採用 Streamlit 建構互動式儀表板，並整合全流程推薦邏輯。

### 技術起源

- **核心架構**: 基於 RAG (Retrieval-Augmented Generation) 框架的變體，並引入 **Soft-Prompt Learning** 的概念，將音訊非結構化特徵橋接至語言模型的語義空間。

---

## 🚀 3. 產品/服務之迭代與擴充設計

系統設計具備高度的擴充性與創新潛力：

### 持續優化之反饋迴圈 (Feedback Loop)

- **現狀**: 已實作 Google Sheets API 連結的投票系統，收集使用者對「DJ 推理理由」與「歌單品質」的真實評價。
- **迭代目標**: 未來將自動化處理回傳的標籤數據，進行 **Reinforcement Learning from Human Feedback (RLHF)** 的輕量化嘗試。

### 未來擴充方向

- **Audio RAG**: 引入即時音訊特徵提取，不依賴預處理的 Metadata，實現「聽音辨色」的推薦。
- **多因子混合重排**: 結合社交趨勢與地理位置數據，進一步優化 Persona 的動態感知力。

---

## 🛡️ 4. 資安考量與機制設計

系統在開發與運作時，採取了嚴格的資訊安全保護措施：

- **環境變數隔離**: 所有 API Keys (OpenRouter, Google Sheets) 均存放於 `.env` 檔案夾中，嚴格禁止進入版本控制系統 (GitHub/GitLab)，確保私鑰安全。
- **隱私去識別化**: 系統內部不儲存使用者的真實身分或裝置資訊，僅基於匿名的音樂 Persona 畫像進行推理，符合資料最小化原則。
- **API 限流與管理**: 透過 OpenRouter 中台設置配額監控與自動斷路機制，防止因異常呼叫導致的成本爆棚或 API 被鎖定。

---

## ⚡ 5. AI 程度說明與加分項（AI 輕量化）

### AI 智力等級

本系統屬於 **「輔助決策 (Augmented Decision Making)」** 程度：

- 系統自主完成海量候選集的檢索與語義分析。
- 最終推薦由 Agent 給出理由，供使用者進行播放選擇，並具備高度的人機協同特性。

### 重點加分項：MLP 達成 AI 輕量化

本專案的一大技術亮點在於使用 **多層感知器 (MLP)** 實現 **「AI 輕量化」**：

1. **低運算開銷**: 不同於傳統使用龐大的預訓練 Transformer 模型進行跨模態映射，我們使用輕量化的 MLP 模型 (僅數 MB) 完成音訊到語義的映射。
2. **毫秒級推理**: 在邊緣端或一般 CPU 環境下即可實現極速預測，大幅降低了後端伺服器的推理成本。
3. **效能對齊**: 透過特徵對齊學習，讓 MLP 的映射能力在特定任務（音樂聽感預測）中達成與大型模型相近的準確度。

---

_文件更新日期: 2026-04-23_
