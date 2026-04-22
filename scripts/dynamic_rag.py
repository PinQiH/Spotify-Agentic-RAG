# > 動態 RAG 描述生成器
# - 參考 document_generator.py 的三分法映射
# - 目標：為 20,000 筆全量推薦提供即時語彙分析

def generate_dynamic_rag_doc(row):
	"""
	現場為缺失 RAG 描述的歌曲生成自然語言。
	"""
	# 提取特徵，並給予默認值
	try:
		energy = float(row.get('energy', 0.5))
		tempo = float(row.get('tempo', 120))
		valence = float(row.get('valence', 0.5))
		dance = float(row.get('danceability', 0.5))
		acoustic = float(row.get('acousticness', 0.5))
		popularity = float(row.get('popularity', 50))
		key = int(row.get('key', 0))
		mode = int(row.get('mode', 1))
	except:
		return "此歌曲呈現均衡的動態表現，風格適配性強。"

	def get_lvl(val, low=0.33, high=0.66):
		if val <= low: return 0
		if val <= high: return 1
		return 2

	# 1. 核心描述映射 (簡化版三分法)
	energy_txt = ["整體能量層次較為柔和，呈現平靜表現。", "能量感適中，均衡動態。", "能量感偏高，節奏具強烈張力。"][get_lvl(energy)]
	valence_txt = ["氛圍含蓄沉靜。", "情緒平衡自然。", "情緒明亮開闊。"][get_lvl(valence)]
	dance_txt = ["律動內斂平穩。", "具適度動感。", "節奏明確鮮明。"][get_lvl(dance)]
	tempo_txt = ["步伐緩慢放鬆。", "速度適中自然。", "節奏推進明顯。"][get_lvl(tempo, 90, 140)]

	# 2. 調性與藝人拼接
	key_map = {0: "C", 1: "C♯/D♭", 2: "D", 3: "D♯/E♭", 4: "E", 5: "F", 6: "F♯/G♭", 7: "G", 8: "G♯/A♭", 9: "A", 10: "A♯/B♭", 11: "B"}
	mode_txt = "大調" if mode == 1 else "小調"
	
	artists = row.get('artists', 'Unknown Artist')
	track_name = row.get('track_name', 'Unknown Song')
	genre = row.get('track_genre', '音樂')
	
	doc = f"這首由 {artists} 演出的《{track_name}》，屬於 {genre}。採用{key_map.get(key, 'C')}{mode_txt}呈現。{energy_txt}{valence_txt}{dance_txt}{tempo_txt}"
	
	return doc[:250]
