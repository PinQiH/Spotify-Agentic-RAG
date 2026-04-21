import os
import json
import numpy as np
import concurrent.futures
from openai import OpenAI
from dotenv import load_dotenv

# > 推薦代理人腳本 (Multi-Model Recommender Agent)
# - 目標：支援多模型 (GPT-4o, Gemini 2.0, Grok) 並行推理，對比品味差異

load_dotenv()

def get_clients_and_models():
	"""
	獲取所有可用的 LLM 客戶端配置。
	"""
	gpt_key = os.getenv("GPT_API_KEY")
	or_key = os.getenv("OPEN_ROUTER_API_KEY")
	
	configs = []
	
	# 1. GPT-4o
	if gpt_key:
		configs.append({
			"name": "GPT-4o",
			"client": OpenAI(api_key=gpt_key),
			"model": "gpt-4o"
		})
	
	# 2. OpenRouter Models (Gemini & Grok)
	if or_key:
		or_client = OpenAI(
			base_url="https://openrouter.ai/api/v1",
			api_key=or_key,
		)
		# Gemini 2.0 Flash
		configs.append({
			"name": "Gemini-2.0-Flash",
			"client": or_client,
			"model": "google/gemini-2.0-flash-001"
		})
		# Grok 4.1 Fast (更新為用戶指定端點)
		configs.append({
			"name": "Grok 4.1 Fast",
			"client": or_client,
			"model": "x-ai/grok-4.1-fast"
		})
		
	return configs

def call_single_model(config, prompt):
	"""
	單一模型的發送與解析 Wrapper。
	"""
	try:
		print(f"// > 正在請求 {config['name']}...")
		response = config['client'].chat.completions.create(
			model=config['model'],
			messages=[
				{"role": "system", "content": "你是一位精通音樂理論與用戶心理的 Spotify 推薦 Agent。"},
				{"role": "user", "content": prompt}
			],
			temperature=0.7,
			response_format={"type": "json_object"}
		)
		content = response.choices[0].message.content
		return config['name'], json.loads(content), None
	except Exception as e:
		return config['name'], None, str(e)

def get_multi_model_recommendations(target_persona, current_song, persona_summary, candidates, n_recommendations=5):
	"""
	並行呼叫多個模型進行推薦推理。
	"""
	configs = get_clients_and_models()
	if not configs:
		return {}, "找不到任何 API Key，請檢查 .env 檔案。"

	# 1. 準備 Prompt (與舊版一致)
	prompt = f"""
你是一位專業的 Spotify 音樂 DJ。請根據以下資訊，從提供的「候選歌曲列表」中，為用戶挑選出最適合的 {n_recommendations} 首歌曲。

--- 1. 用戶長期偏好 (Persona: {target_persona}) ---
{persona_summary}

--- 2. 目前正在播放的歌曲 ---
{json.dumps(current_song, ensure_ascii=False, indent=2)}

--- 3. 候選歌曲列表 (包含音訊特徵與 RAG 描述) ---
{json.dumps(candidates, ensure_ascii=False, indent=2)}

--- 任務要求 ---
1. 重排序並挑選出適合的歌曲。
2. 每一首歌附帶一句話推薦理由，並引用某一項特徵 (如 BPM, Energy, Genre)。
3. 過濾 Explicit 內容。
4. 必須使用正體中文。

請以嚴格的 JSON 格式輸出：
{{
  "recommendations": [
    {{
      "track_id": "歌曲 ID",
      "track_name": "歌曲名稱",
      "artists": "演出者",
      "reason": "推薦理由"
    }}
  ],
  "agent_explanation": "推薦策略總結"
}}
"""

	# 2. 並行執行
	all_results = {}
	errors = []
	
	with concurrent.futures.ThreadPoolExecutor(max_workers=len(configs)) as executor:
		future_to_model = {executor.submit(call_single_model, cfg, prompt): cfg['name'] for cfg in configs}
		for future in concurrent.futures.as_completed(future_to_model):
			model_name, result, err = future.result()
			if err:
				errors.append(f"{model_name}: {err}")
			else:
				all_results[model_name] = result

	return all_results, errors if errors else None

def calculate_cosine_similarity(vec1, vec2):
	"""計算兩個向量之間的餘弦相似度"""
	dot_product = np.dot(vec1, vec2)
	norm_v1 = np.linalg.norm(vec1)
	norm_v2 = np.linalg.norm(vec2)
	return dot_product / (norm_v1 * norm_v2)
