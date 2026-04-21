import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
import os

# > 歌曲向量空間視覺化腳本 (Notebook 專用)
# - 目標：在 Jupyter Notebook 中呈現互動式的歌曲分佈圖

def run_visualization():
    # 強制設定渲染器以繞過 nbformat 報錯
    # 如果 iframe 不行，可試著改用 'notebook_connected' 或 'colab'
    pio.renderers.default = "iframe" 
    
    # 1. 檢查檔案是否存在
    pca_path = "data/pca_songs.csv"
    songs_path = "data/songs.csv"
    
    if not os.path.exists(pca_path) or not os.path.exists(songs_path):
        print("!! 錯誤: 找不到 PCA 或歌曲資料檔案。請確保已執行先前的處理步驟。")
        return

    # 2. 載入資料
    print("// > 正在讀取資料並準備繪圖...")
    df_pca_plot = pd.read_csv(pca_path)
    
    # 3. 建立視覺化圖表
    fig = px.scatter(
        df_pca_plot, 
        x='PC_1', 
        y='PC_2', 
        color='track_genre', 
        hover_data=['track_name', 'artists', 'track_genre'],
        title="Spotify Song Embedding Space (PCA 2D Projection)",
        labels={'PC_1': 'Musical Energy / Vibe', 'PC_2': 'Acoustic / Instrumental Contrast'},
        width=1000,
        height=700,
        template="plotly_dark",
        opacity=0.6
    )

    fig.update_traces(marker=dict(size=5))
    
    # 在 Notebook 中顯示
    fig.show()

    # 4. 儲存 HTML 作為備份
    output_path = "data/embedding_space_viz.html"
    fig.write_html(output_path)
    print(f"// @ 視覺化完成，互動式圖表已儲存至 {output_path}")

if __name__ == "__main__":
    run_visualization()
