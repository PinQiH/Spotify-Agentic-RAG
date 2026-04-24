import torch
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.getcwd())
from scripts.train_mlp import SoftPromptMLP, load_soft_prompt_mlp

def test_model_output(model_path):
	if not os.path.exists(model_path): return
	
	model, _ = load_soft_prompt_mlp(model_path)
	print(f"\n=== 測試模型: {model_path} ===")
	
	# 造兩個極端不同的 dummy 輸入
	# 假設 input_dim=33 (由 app.py 得知)
	checkpoint = torch.load(model_path, map_location='cpu')
	input_dim = checkpoint['input_dim']
	
	x1 = torch.zeros((1, input_dim))
	x2 = torch.ones((1, input_dim))
	
	with torch.no_grad():
		y1 = model(x1).numpy()[0]
		y2 = model(x2).numpy()[0]
	
	similarity = np.dot(y1, y2) / (np.linalg.norm(y1) * np.linalg.norm(y2))
	print(f"   - Input All Zeros vs All Ones Similarity: {similarity:.6f}")
	print(f"   - y1[:5]: {y1[:5]}")
	print(f"   - y2[:5]: {y2[:5]}")

if __name__ == "__main__":
	test_model_output("data/soft_prompt_mlp.pth")
	test_model_output("data/soft_prompt_mlp_finetuned.pth")
