import pandas as pd
import numpy as np
import os

# > RAG 文件生成區塊 (三分法 + 關鍵字映射)
# - 參考 baseline.ipynb 的邏輯，將數值特徵轉化為具備「高、中、低」層次的自然語言描述

def add_quantile_text_column(df, col, new_col, low_txt, mid_txt, high_txt):
	"""根據 0.33 與 0.66 分位數將數值切割為三個層次的文字描述"""
	s = pd.to_numeric(df[col], errors="coerce").fillna(0)
	q1, q2 = s.quantile([0.33, 0.66])
	# 處理 q1 == q2 的情況
	if q1 == q2:
		q1 = s.min() + (s.max() - s.min()) * 0.33
		q2 = s.min() + (s.max() - s.min()) * 0.66
	
	bins = [-np.inf, q1, q2, np.inf]
	labels = [low_txt, mid_txt, high_txt]
	df[new_col] = pd.cut(s, bins=bins, labels=labels, include_lowest=True).astype(str)
	return df

def generate_rag_docs(df, save_path="data/rag_docs.csv"):
	"""
	使用與 baseline.ipynb 一致的「三分法」與「調性/拍號映射」生成歌曲描述。
	"""
	if save_path and os.path.exists(save_path):
		print(f"// @ 找到已生成的 RAG 文件檔: {save_path}，直接讀取...")
		return pd.read_csv(save_path)

	print(f"// > 正在使用「三分法」為 {len(df)} 首歌曲生成專業描述...")
	df_docs = df.copy()

	# 1. 十一個維度的三分法描述
	configs = [
		("danceability", "desc_danceability", "節奏的律動較為內斂，動感元素不強，旋律走向更偏向平穩細緻。", "節奏具有適度的律動感，能帶出穩定的節拍與輕微的擺動感。", "節奏明確、律動鮮明，音樂的節拍更容易帶動身體自然律動。"),
		("energy", "desc_energy", "整體能量層次較為柔和，音樂呈現平靜、輕巧的表現。", "能量感適中，音樂保持穩定張力而不過度強烈，呈現均衡的動態。", "能量感偏高，音色、節奏與表現都具備明顯推力與張力。"),
		("valence", "desc_valence", "音樂帶有較內斂或偏沉靜的情緒色彩，呈現細膩而含蓄的氛圍。", "情緒介於中性與輕盈之間，整體感受平衡且自然。", "音樂情緒較為明亮，旋律或和聲帶有正向輕快的特質。"),
		("acousticness", "desc_acousticness", "音色以電子或合成元素為主，呈現現代化、光滑的聲響質地。", "原聲與電子音色並存，呈現兼具自然與現代感的音色平衡。", "以原聲樂器為主要呈現方式，音色溫暖自然、富有質地。"),
		("instrumentalness", "desc_instrumentalness", "以人聲演唱為主要表現，歌詞與旋律線是聆聽重點。", "人聲與樂器的比重相對平衡，旋律與編曲彼此呼應。", "器樂成分為主軸，旋律與音色更著重於非語詞性的音樂表達。"),
		("liveness", "desc_liveness", "聲音質地精緻、乾淨，多為錄音室環境的呈現方式。", "音樂帶有輕微空間感或現場氛圍，增添自然氣息。", "具有明顯的現場感，聲音中包含舞台空間的特性與觀眾互動的氛圍。"),
		("speechiness", "desc_speechiness", "以旋律與歌唱為主，口語化元素很少。", "具有一定比例的語音或說唱元素，語氣與節奏具明顯特色。", "以說話、朗讀或說唱為主要構成，語音節奏佔顯著比重。"),
		("loudness", "desc_loudness", "混音呈現較輕柔的動態範圍，聲音較為平穩細緻。", "音量與能量呈現平衡，聲音密度恰到好處，適合多數風格。", "聲音厚實飽滿，動態範圍較集中，具備較強的存在感。"),
		("tempo", "desc_tempo", "節奏步伐較為緩慢，呈現相對放鬆、柔和的節奏流動。", "速度適中，節奏線條平穩自然，具備良好的聽覺節奏感。", "速度偏快，節奏推進明顯，音樂的律動更具動力。"),
		("popularity", "desc_popularity", "屬於較少人聆聽的作品，具有獨特而個性化的風格。", "具一定熟悉度，風格親切易懂但仍保有自身特色。", "廣受聽眾喜愛，旋律與風格具有普遍吸引力。")
	]

	for cfg in configs:
		df_docs = add_quantile_text_column(df_docs, *cfg)

	# 2. Key 映射
	key_text_map = {
		0: "C 調（key=0）音色純淨、直接，展示自然且中性的聽感。",
		1: "C♯/D♭ 調（key=1）音色細緻明亮，具有豐富的穿透力。",
		2: "D 調（key=2）音色明亮清晰，呈現開放與流動的質地。",
		3: "D♯/E♭ 調（key=3）音色溫暖柔滑，具細膩而飽和的色彩。",
		4: "E 調（key=4）音色鮮明而具張力，展現強烈的現代感。",
		5: "F 調（key=5）音色沉穩圓潤，展現溫暖的包覆感。",
		6: "F♯/G♭ 調（key=6）音色透明細膩，呈現柔光般的明亮質地。",
		7: "G 調（key=7）音色開放輕快，具有很好的擴展性。",
		8: "G♯/A♭ 調（key=8）音色圓潤柔滑，呈現細膩而深邃的色彩。",
		9: "A 調（key=9）音色明朗清楚，具有流暢且動感的特質。",
		10: "A♯/B♭ 調（key=10）音色溫暖厚實，展現寬廣柔和的氛圍。",
		11: "B 調（key=11）音色明亮集中，具有清晰穿透的聲響線條。"
	}
	df_docs["desc_key"] = df_docs["key"].map(key_text_map).fillna("未知調性")

	# 3. Mode 映射
	mode_text_map = {
		1: "採用大調呈現，使音樂在調性空間中傾向開放、明亮與自然延展。",
		0: "採用小調呈現，使音樂在調性空間中帶有內斂、細膩或柔和的色彩。"
	}
	df_docs["desc_mode"] = df_docs["mode"].map(mode_text_map).fillna("")

	# 4. 整合為最終 rag_doc
	def finalize_doc(row):
		base = f"這首由 {row['artists']} 演出的歌曲《{row['track_name']}》"
		if str(row['album_name']) != 'nan' and row['album_name'] != "":
			base += f"，收錄於專輯《{row['album_name']}》"
		
		# 組合描述
		desc = f"。它屬於 {row['track_genre']}。{row['desc_key']}{row['desc_mode']}{row['desc_energy']}{row['desc_valence']}{row['desc_danceability']}{row['desc_tempo']}"
		
		# 限制長度以符合 RAG 需求
		full = base + desc
		return full[:250]

	df_docs['rag_doc'] = df_docs.apply(finalize_doc, axis=1)

	if save_path:
		os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
		df_docs.to_csv(save_path, index=False, encoding='utf-8-sig')
		print(f"// @ 三分法 RAG 文件已存至: {save_path}")

	return df_docs
