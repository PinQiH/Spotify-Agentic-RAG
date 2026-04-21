import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# > 資料清理區塊
# - 清理基礎欄位與缺失值
def clean_basic_data(df):
	"""
	執行基礎資料清理，如刪除冗餘欄位、處理缺失值、格式轉換
	"""
	df_cleaned = df.copy()
	
	# 1. 刪除 index 欄位
	if 'Unnamed: 0' in df_cleaned.columns:
		df_cleaned = df_cleaned.drop('Unnamed: 0', axis=1)

	# 2. 去除必要標籤缺失的行
	df_cleaned.dropna(subset=['artists', 'album_name', 'track_name'], inplace=True)
	
	# 3. 類別轉換
	df_cleaned['explicit'] = df_cleaned['explicit'].astype(int)
	
	return df_cleaned

# > 數值轉換區塊
# - 執行 Log 轉換、截斷 (Clip) 與標準化 (Scaling)
def transform_numerical_features(df):
	"""
	處理數值型特徵，包含 Log 轉換與 Z-score / MinMax 標準化
	"""
	# 歌曲長度處理 (排除離群值後取 Log 並縮放)
	df = df[df['duration_ms'] <= 600000].copy()
	df['duration_ms_log'] = np.log1p(df['duration_ms'])
	scaler_std = StandardScaler()
	df['duration_ms_scaled'] = scaler_std.fit_transform(df[['duration_ms_log']])

	# 響度處理 (限制範圍後縮放)
	df['loudness_clipped'] = df['loudness'].clip(lower=-30)
	df['loudness_scaled'] = scaler_std.fit_transform(df[['loudness_clipped']])

	# 速度處理 (限制範圍後縮放)
	df['tempo_clipped'] = df['tempo'].clip(lower=40, upper=220)
	df['tempo_scaled'] = scaler_std.fit_transform(df[['tempo_clipped']])

	# 一般數值特徵 Z-score 標準化
	for col in ['danceability', 'energy', 'valence']:
		df[f'{col}_scaled'] = scaler_std.fit_transform(df[[col]])

	# 說話程度 Log 轉換
	df['speechiness_log'] = np.log1p(df['speechiness'])
	df['speechiness_scaled'] = scaler_std.fit_transform(df[['speechiness_log']])

	# 聲學與現場感 MinMax 標準化
	scaler_minmax = MinMaxScaler()
	for col in ['acousticness', 'liveness']:
		df[f'{col}_scaled'] = scaler_minmax.fit_transform(df[[col]])

	# 演奏性二值化
	df['instrumentalness_binary'] = (df['instrumentalness'] > 0.5).astype(int)
	
	return df

# > 類別特徵編碼區塊
# - 處理 Key 週期性與拍號
def encode_categorical_features(df):
	"""
	處理類別型與具有週期性的特徵
	"""
	# Key 週期編碼
	df['key_sin'] = np.sin(2 * np.pi * df['key'] / 12)
	df['key_cos'] = np.cos(2 * np.pi * df['key'] / 12)

	# Mode 標準化
	scaler_std = StandardScaler()
	df['mode_scaled'] = scaler_std.fit_transform(df[['mode']])

	# 拍號處理與 One-hot
	df['time_signature_orig'] = df['time_signature'].replace(0, 4)
	df = pd.concat([df, pd.get_dummies(df['time_signature_orig'], prefix='time_signature')], axis=1)
	
	return df

# > 整合 Pipeline
# - 一次呼叫執行完整前處理
def preprocess_pipeline(df, save_path=None):
	"""
	一鍵執行完整的數值特徵前處理流程。
	若提供 save_path 且該檔案存在，會直接讀取該檔案；否則重新處理並存檔。
	"""
	if save_path and os.path.exists(save_path):
		print(f"// @ 找到已處理過的資料檔案: {save_path}，直接讀取...")
		# 讀取時可加入 low_memory=False 避免欄位型態警告
		return pd.read_csv(save_path, low_memory=False)

	df_proc = clean_basic_data(df)
	df_proc = transform_numerical_features(df_proc)
	df_proc = encode_categorical_features(df_proc)
	
	print("// @ 數值特徵處理完成")
	print(f"// @ 目前特徵維度: {df_proc.shape}")
	
	if save_path:
		os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
		df_proc.to_csv(save_path, index=False)
		print(f"// @ 處理後之資料已儲存至: {save_path}")
		
	return df_proc
