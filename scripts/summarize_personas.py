import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# > Persona 偏好總結生成腳本 (Persona Preference Summarization)
# - 目標：讀取聽歌歷史，呼叫 LLM 總結用戶的長期音樂偏好

load_dotenv()

def get_llm_client():
	"""獲取可用之 LLM 客戶端"""
	gpt_key = os.getenv("GPT_API_KEY")
	or_key = os.getenv("OPEN_ROUTER_API_KEY")
	
	if gpt_key:
		return OpenAI(api_key=gpt_key), "gpt-4o"
	elif or_key:
		return OpenAI(
			base_url="https://openrouter.ai/api/v1",
			api_key=or_key,
		), "google/gemini-2.0-flash-001"
	return None, None

def generate_persona_summaries(history_dir="data/persona_listening_histories", save_path="data/persona_listening_histories/persona_summaries.json", force_update=False):
	"""
	讀取歷史資料夾中的所有 JSON，並使用 LLM 進行偏好總結。
	"""
	if not force_update and os.path.exists(save_path):
		print(f"// @ 找到已存在的畫像總結檔案: {save_path}，直接載入...")
		with open(save_path, 'r', encoding='utf-8') as f:
			return json.load(f)

	client, model_name = get_llm_client()
	if not client:
		print("// !! 警告：未設定 API Key，無法生成總結。")
		return {}

	summaries = {}
	print(f"// > 正在掃描歷史資料夾: {history_dir}...")
	
	for filename in os.listdir(history_dir):
		if filename.endswith("_history.json") and filename != "persona_summaries.json":
			persona_name = filename.replace("_history.json", "").replace("_", " ").title()
			filepath = os.path.join(history_dir, filename)
			
			with open(filepath, 'r', encoding='utf-8') as f:
				data = json.load(f)
				history = data.get('listening_history', [])
			
			# 準備給 LLM 的 context (簡化歌單)
			history_summary = [f"{s['track_name']} - {s['artists']} ({s.get('track_genre', 'N/A')})" for s in history[:15]]
			history_text = "\n".join(history_summary)
			
			prompt = f"""
你是一位專業的音樂推薦分析師。以下是用戶 '{persona_name}' 的部分收聽歷史歌曲清單：
{history_text}

請根據這些歌曲，總結該用戶的長期音樂偏好，並提出 3-5 個關鍵的特徵點（例如：偏好快節奏、喜歡特定的曲風融合、不喜歡露骨歌詞等）。
請以繁體中文條列式呈現，保持簡潔有力。
"""
			print(f"// > 正在為 '{persona_name}' 生成偏好總結...")
			try:
				response = client.chat.completions.create(
					model=model_name,
					messages=[
						{"role": "system", "content": "你是一位音樂分析專家。"},
						{"role": "user", "content": prompt}
					],
					temperature=0.7,
					max_tokens=300
				)
				summary = response.choices[0].message.content.strip()
				summaries[persona_name] = summary
			except Exception as e:
				print(f"// !! 處理 {persona_name} 時發生錯誤: {e}")

	# 存檔
	if summaries:
		os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
		with open(save_path, 'w', encoding='utf-8') as f:
			json.dump(summaries, f, ensure_ascii=False, indent=4)
		print(f"// @ 所有畫像總結已儲存至: {save_path}")
	
	return summaries

if __name__ == "__main__":
	generate_persona_summaries()
