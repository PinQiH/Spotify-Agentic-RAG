import os
import pandas as pd
import sys
import io

# 加入路徑
sys.path.append(os.getcwd())
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from scripts.batch_predict_embeddings import batch_generate_soft_prompts
from scripts.build_prediction_index import build_index_from_soft_prompts

def run_pipeline():
	print("🚀 [PIPELINE] 開始系統修復連鎖流程...")
	
	# 1. 載入原始特徵
	print("📂 載入原始特徵資料...")
	df_processed = pd.read_csv("data/processed_songs.csv")
	df_pca = pd.read_csv("data/pca_songs.csv")
	
	# 2. 執行批次預測 (產生新的 soft_prompts_map.pkl)
	print("\n🧠 [STEP 1/2] 正在重新生成語義向量 (MLP Inference)...")
	# 使用 fine-tuned 模型
	soft_prompts_map = batch_generate_soft_prompts(
		df_processed, 
		df_pca, 
		model_path="data/soft_prompt_mlp_finetuned.pth",
		output_path="data/soft_prompts_map.pkl"
	)
	
	# 3. 建立索引
	print("\n🔍 [STEP 2/2] 正在重建 FAISS 向量索引與元數據...")
	# 這裡我們需要原始 metadata 以對齊，但 build_index_from_soft_prompts 內部會處理
	# 我們只需要傳入 df_processed 給它作為對照表
	build_index_from_soft_prompts(df_processed, soft_prompts_map)
	
	print("\n🎉 [PIPELINE] 系統修復完成！所有向量已重新對齊且具備多樣性。")

if __name__ == "__main__":
	try:
		run_pipeline()
	except Exception as e:
		print(f"\n❌ 流程中斷: {e}")
		import traceback
		traceback.print_exc()
