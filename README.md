# Spotify Agentic RAG 音樂推薦系統 (Demo)

這是一個基於 Streamlit 的音樂推薦系統 Demo，展示了如何利用 **Agentic RAG (Retrieval-Augmented Generation)** 的概念，結合用戶畫像 (User Persona) 與當前情境 (Context)，提供個人化的音樂推薦。

本專案模擬了一個 AI 代理人的思考過程：從理解用戶偏好，到檢索候選歌曲，最後過濾並生成推薦理由。

## ✨ 特色功能

*   **🎧 視覺化音樂庫**: 透過網格狀的介面瀏覽並選擇「當前歌曲」，直接嵌入 Spotify 播放器試聽。
*   **🤖 Agentic RAG 模擬**: 視覺化展示 AI 的三階段思考流程：
    1.  **用戶理解 (User Understanding)**: 分析用戶的長期聆聽歷史與偏好。
    2.  **檢索與過濾 (Retrieval & Filtering)**: 根據當前歌曲的風格/節奏檢索候選集，並依據用戶畫像進行過濾。
    3.  **生成推薦 (Generation)**: 最終推薦 3 首歌曲，並附上 AI 生成的推薦理由。
*   **👤 多元用戶角色**: 內建 4 種不同的用戶 Persona (如 Chill Vibes, Party Animal 等)，每種角色都有獨特的聆聽歷史與描述。
*   **🎨 現代化 UI**: 採用 Glassmorphism (毛玻璃) 設計風格，搭配 Spotify 的經典深色主題與霓虹綠點綴。
*   **🎵 Spotify Embeds**: 整合 Spotify 官方播放器，提供真實的音樂播放體驗。

## 🛠️ 安裝與執行

### 1. 環境設定

建議使用 Python 3.9+。首先建立並啟動虛擬環境：

```bash
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 安裝套件

```bash
pip install -r requirements.txt
```

### 3. 準備資料

本專案使用 Kaggle 的 Spotify Tracks Dataset。請執行以下腳本自動下載並預處理資料：

```bash
python scripts/download_data.py
```
*注意：這會下載約 100MB 的資料並隨機取樣 2000 首歌曲存為 `data/songs.csv`。*

### 4. 啟動應用程式

```bash
streamlit run app.py
```

啟動後，瀏覽器應會自動開啟 `http://localhost:8501`。

## 📂 專案結構

```
root/
├── app.py                      # Streamlit 主程式 (UI 與 流程控制)
├── utils.py                    # 核心邏輯 (Persona 分析、推薦演算法模擬)
├── requirements.txt            # 專案依賴套件
├── scripts/
│   └── download_data.py        # 資料下載與預處理腳本
├── data/
│   └── songs.csv               # (執行腳本後產生) 音樂資料庫
└── persona_listening_histories/ # 用戶角色的聆聽歷史 (JSON)
    ├── chill_vibes_history.json
    ├── party_animal_history.json
    ├── study_focus_history.json
    └── workout_motivation_history.json
```

## 🚀 部署與分享

簡單來說，您可以使用 **Streamlit Community Cloud** 進行免費部署，或使用 `localtunnel` 進行臨時分享。

## 💡 技術棧

*   **Frontend**: Streamlit
*   **Data Processing**: Pandas
*   **Data Source**: Kaggle (Spotify Tracks Dataset)
*   **Visualization**: Spotify Embed Widgets, Custom CSS

---
*Created for Data Mining Final Project Demo.*
