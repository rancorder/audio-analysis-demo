"""
音声解析統合システム - サンプルコード
音声ファイルから特徴抽出 + 文字起こし + 感情分析を統合

使用技術:
- Whisper: 文字起こし
- OpenSMILE: 音声特徴抽出
- BERT: 感情分析
- librosa, noisereduce: 前処理
"""

import os
import torch
import whisper
import opensmile
import librosa
import soundfile as sf
import tkinter as tk
from tkinter import filedialog
from pydub import AudioSegment
import multiprocessing
import logging
import subprocess
import sys
import noisereduce as nr
import pandas as pd
from transformers import pipeline

# ===========================
# ログ設定
# ===========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ===========================
# CPU最適化
# ===========================
torch.set_num_threads(min(2, os.cpu_count()))

# ===========================
# モデル初期化
# ===========================
WHISPER_MODEL = whisper.load_model("tiny")
OPENSMILE = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
SENTIMENT_ANALYZER = pipeline(
    "text-classification",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# ===========================
# ファイル選択
# ===========================
def select_audio_file():
    """ユーザーに音声ファイルを選択させる"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        filetypes=[("音声ファイル", "*.wav *.mp3 *.m4a")]
    )
    if not file_path:
        logging.warning("⚠️ ファイルが選択されませんでした")
        exit(1)
    logging.info(f"📂 選択: {file_path}")
    return file_path

# ===========================
# 音声前処理
# ===========================
def convert_to_wav(input_path, output_path):
    """MP3/M4AをWAVに変換（FFmpeg使用）"""
    try:
        logging.info("🔄 WAV変換中...")
        subprocess.run(
            ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", 
             "-b:a", "32k", output_path, "-y"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logging.info("✅ 変換完了")
        return output_path
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ 変換エラー: {e}")
        exit(1)

def denoise_audio(input_path, output_path):
    """ノイズ除去処理"""
    try:
        # MP3の場合は変換
        if input_path.lower().endswith(".mp3"):
            input_path = convert_to_wav(input_path, "temp_converted.wav")
        
        logging.info("🔊 ノイズ除去中...")
        audio, sr = librosa.load(input_path, sr=16000)
        reduced_noise = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8)
        sf.write(output_path, reduced_noise, sr)
        logging.info("✅ ノイズ除去完了")
        return output_path
    except Exception as e:
        logging.error(f"⚠️ エラー: {e}")
        return convert_to_wav(input_path, output_path)

def split_audio(file_path, chunk_length_ms=15000):
    """音声を15秒単位で分割"""
    logging.info("✂️ 音声分割中...")
    audio = AudioSegment.from_file(file_path)
    chunks = [
        audio[i:i+chunk_length_ms] 
        for i in range(0, len(audio), chunk_length_ms)
    ]
    
    chunk_files = []
    for i, chunk in enumerate(chunks):
        chunk_path = f"chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunk_files.append(chunk_path)
    
    logging.info(f"✅ {len(chunk_files)}個のチャンクに分割")
    return chunk_files

# ===========================
# Whisper文字起こし
# ===========================
def transcribe_audio(file_path):
    """Whisperで音声を文字起こし"""
    try:
        logging.info(f"🎙️ 文字起こし中: {file_path}")
        result = WHISPER_MODEL.transcribe(
            file_path,
            fp16=True,
            language="ja",
            temperature=0
        )
        return result["text"]
    except Exception as e:
        logging.error(f"❌ 文字起こしエラー: {e}")
        return ""

def parallel_transcription(files):
    """並列処理で複数ファイルを文字起こし"""
    num_processes = 2
    logging.info(f"⚡ 並列処理開始（プロセス数: {num_processes}）")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(transcribe_audio, files, chunksize=2)
        pool.close()
        pool.join()
    
    return results

# ===========================
# OpenSMILE音声特徴抽出
# ===========================
def extract_audio_features(audio_path):
    """OpenSMILEで音声特徴を抽出"""
    logging.info("📊 音声特徴抽出中...")
    
    # 全特徴量を抽出
    features = OPENSMILE.process_file(audio_path)
    
    # 重要な特徴量を選択
    selected_features = [
        "F0semitoneFrom27.5Hz_sma3nz_amean",      # ピッチ平均
        "F0semitoneFrom27.5Hz_sma3nz_stddevNorm", # ピッチばらつき
        "shimmerLocaldB_sma3nz_amean",            # シマー
        "jitterLocal_sma3nz_amean",               # ジッター
        "loudness_sma3_amean",                    # 音量平均
        "loudness_sma3_stddevNorm"                # 音量ばらつき
    ]
    
    # 存在チェック
    missing = [f for f in selected_features if f not in features.columns]
    if missing:
        logging.warning(f"⚠️ 未取得の特徴量: {missing}")
    
    available_features = [f for f in selected_features if f in features.columns]
    result = features[available_features]
    
    logging.info("✅ 音声特徴抽出完了")
    return result

# ===========================
# BERT感情分析
# ===========================
def analyze_sentiment(text):
    """BERTで感情分析"""
    logging.info("🧠 感情分析中...")
    
    if not text.strip():
        logging.warning("⚠️ テキストが空です")
        return 3  # 中立
    
    result = SENTIMENT_ANALYZER(text[:512])  # トークン数制限
    emotion_score = int(result[0]["label"][0])  # "4 stars" → 4
    
    logging.info(f"✅ 感情スコア: {emotion_score}/5")
    return emotion_score

# ===========================
# Late Fusion（特徴統合）
# ===========================
def late_fusion(audio_features, sentiment_score, transcription):
    """音声特徴とテキスト特徴を統合"""
    logging.info("🔗 特徴統合中...")
    
    combined = {
        "audio_features": audio_features.to_dict('records')[0],
        "sentiment_score": sentiment_score,
        "transcription": transcription,
        "feature_count": len(audio_features.columns)
    }
    
    logging.info("✅ 特徴統合完了")
    return combined

# ===========================
# 結果保存
# ===========================
def save_results(combined_features, output_file="analysis_result.txt"):
    """分析結果をファイルに保存"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== 音声解析結果 ===\n\n")
        
        f.write("【音声特徴】\n")
        for key, value in combined_features["audio_features"].items():
            f.write(f"  {key}: {value}\n")
        
        f.write(f"\n【感情スコア】\n")
        f.write(f"  {combined_features['sentiment_score']}/5\n")
        
        f.write(f"\n【文字起こし】\n")
        f.write(f"  {combined_features['transcription']}\n")
    
    logging.info(f"✅ 結果保存: {output_file}")

# ===========================
# メイン処理
# ===========================
def main():
    """統合処理のメインフロー"""
    logging.info("🔷 音声解析統合システム 起動")
    
    # 1. ファイル選択
    audio_path = select_audio_file()
    
    # 2. 前処理（ノイズ除去）
    denoised_path = "denoised_audio.wav"
    denoised_path = denoise_audio(audio_path, denoised_path)
    
    # 3. 音声分割
    chunk_files = split_audio(denoised_path)
    
    # 4. 並列文字起こし
    transcriptions = parallel_transcription(chunk_files)
    full_transcription = "\n".join(transcriptions)
    
    # 5. 音声特徴抽出
    audio_features = extract_audio_features(denoised_path)
    
    # 6. 感情分析
    sentiment_score = analyze_sentiment(full_transcription)
    
    # 7. Late Fusion
    combined = late_fusion(audio_features, sentiment_score, full_transcription)
    
    # 8. 結果保存
    save_results(combined)
    
    # 9. チャンクファイル削除
    for chunk_file in chunk_files:
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
    
    logging.info("✅ 全処理完了！")
    
    return combined

if __name__ == "__main__":
    results = main()
