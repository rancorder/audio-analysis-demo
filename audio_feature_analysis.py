"""
音声特徴統計分析 - サンプルコード
OpenSMILE特徴量の正規化・比較・可視化

使用技術:
- OpenSMILE: 音声特徴抽出
- scikit-learn: 正規化処理
- matplotlib: データ可視化
- pandas: データ処理
"""

import pandas as pd
import opensmile
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import filedialog

# ===========================
# 設定
# ===========================
# 日本語フォント設定（Windows）
plt.rcParams["font.family"] = "MS Gothic"
plt.rcParams["figure.figsize"] = (12, 6)

# OpenSMILE初期化
SMILE = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# 抽出する重要特徴量
SELECTED_FEATURES = [
    "F0semitoneFrom27.5Hz_sma3nz_amean",      # ピッチ平均
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm", # ピッチばらつき
    "shimmerLocaldB_sma3nz_amean",            # シマー
    "jitterLocal_sma3nz_amean",               # ジッター
    "loudness_sma3_amean",                    # 音量平均
    "loudness_sma3_stddevNorm"                # 音量ばらつき
]

# ===========================
# ファイル選択
# ===========================
def select_audio_file(title="音声ファイルを選択"):
    """ファイル選択ダイアログ"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[("音声ファイル", "*.wav *.mp3")]
    )
    if not file_path:
        print("⚠️ ファイルが選択されませんでした")
        return None
    print(f"✅ 選択: {file_path}")
    return file_path

# ===========================
# 音声特徴抽出
# ===========================
def extract_features(audio_path):
    """
    OpenSMILEで音声特徴を抽出
    
    Args:
        audio_path: 音声ファイルのパス
    
    Returns:
        抽出された特徴量のDataFrame
    """
    print(f"📊 音声特徴抽出中: {audio_path}")
    
    # 全特徴量を抽出
    features = SMILE.process_file(audio_path)
    
    # 指定した特徴量が存在するか確認
    missing = [f for f in SELECTED_FEATURES if f not in features.columns]
    if missing:
        print(f"⚠️ 取得できなかった特徴量: {missing}")
        # 取得可能な特徴量のみを使用
        available = [f for f in SELECTED_FEATURES if f in features.columns]
        print(f"✅ 使用可能な特徴量: {len(available)}個")
        return features[available]
    
    print(f"✅ 特徴量抽出完了: {len(SELECTED_FEATURES)}個")
    return features[SELECTED_FEATURES]

# ===========================
# データ正規化
# ===========================
def normalize_features(df):
    """
    MinMaxScalerで0-1に正規化
    
    Args:
        df: 特徴量のDataFrame
    
    Returns:
        正規化されたDataFrame
    """
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df)
    
    df_normalized = pd.DataFrame(
        normalized_data,
        columns=df.columns,
        index=df.index
    )
    
    print("✅ 正規化完了（0-1スケール）")
    return df_normalized

# ===========================
# 特徴量の統計サマリー
# ===========================
def feature_summary(features_df):
    """
    特徴量の統計情報を計算
    
    Args:
        features_df: 特徴量のDataFrame
    
    Returns:
        統計サマリーのDataFrame
    """
    summary = pd.DataFrame({
        '平均': features_df.mean(),
        '標準偏差': features_df.std(),
        '最小値': features_df.min(),
        '最大値': features_df.max(),
        '中央値': features_df.median()
    })
    
    print("\n📊 特徴量統計サマリー:")
    print(summary.round(3))
    
    return summary

# ===========================
# 比較分析
# ===========================
def compare_features(features_dict, normalize=True):
    """
    複数の音声ファイルの特徴量を比較
    
    Args:
        features_dict: {ラベル名: 特徴量DataFrame} の辞書
        normalize: 正規化するかどうか
    
    Returns:
        比較用のDataFrame
    """
    # 各特徴量の平均値を取得
    comparison_data = {}
    for label, features in features_dict.items():
        comparison_data[label] = features.iloc[0]
    
    comparison_df = pd.DataFrame(comparison_data).T
    
    # 正規化
    if normalize:
        comparison_df = normalize_features(comparison_df)
    
    print(f"✅ 比較分析完了: {len(features_dict)}ファイル")
    return comparison_df

# ===========================
# 可視化
# ===========================
def plot_single_feature(features_df, title="音声特徴量"):
    """
    単一ファイルの特徴量を棒グラフで表示
    
    Args:
        features_df: 特徴量のDataFrame
        title: グラフのタイトル
    """
    # ラベルを短縮
    shortened_labels = [col.split('_')[0] for col in features_df.columns]
    
    # 棒グラフ作成
    plt.figure(figsize=(12, 5))
    features_df.iloc[0].plot(kind='bar', color='skyblue')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("特徴量", fontsize=12)
    plt.ylabel("値", fontsize=12)
    plt.xticks(range(len(shortened_labels)), shortened_labels, rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_comparison(comparison_df, title="音声特徴比較"):
    """
    複数ファイルの特徴量を水平棒グラフで比較表示
    
    Args:
        comparison_df: 比較用DataFrame
        title: グラフのタイトル
    """
    # ラベルを短縮
    shortened_labels = [col.split('_')[0] for col in comparison_df.columns]
    
    # 水平棒グラフ作成
    comparison_df.T.plot(
        kind='barh',
        figsize=(10, 6),
        title=title
    )
    plt.xlabel("正規化値（0-1）", fontsize=12)
    plt.ylabel("特徴量", fontsize=12)
    plt.yticks(range(len(shortened_labels)), shortened_labels)
    plt.legend(loc='best')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_heatmap(comparison_df, title="特徴量ヒートマップ"):
    """
    特徴量のヒートマップを表示
    
    Args:
        comparison_df: 比較用DataFrame
        title: グラフのタイトル
    """
    plt.figure(figsize=(10, 6))
    
    # ラベルを短縮
    shortened_labels = [col.split('_')[0] for col in comparison_df.columns]
    
    # ヒートマップ作成
    plt.imshow(comparison_df.values, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='正規化値')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("特徴量", fontsize=12)
    plt.ylabel("ファイル", fontsize=12)
    plt.xticks(range(len(shortened_labels)), shortened_labels, rotation=45)
    plt.yticks(range(len(comparison_df)), comparison_df.index)
    plt.tight_layout()
    plt.show()

# ===========================
# 差分分析
# ===========================
def analyze_difference(features1, features2, label1="成功", label2="失敗"):
    """
    2つの特徴量の差分を分析
    
    Args:
        features1: 特徴量1
        features2: 特徴量2
        label1: ラベル1
        label2: ラベル2
    
    Returns:
        差分のDataFrame
    """
    diff = features1.iloc[0] - features2.iloc[0]
    diff_df = pd.DataFrame({
        label1: features1.iloc[0],
        label2: features2.iloc[0],
        '差分': diff,
        '差分率(%)': (diff / features1.iloc[0] * 100).round(2)
    })
    
    print(f"\n📊 {label1} vs {label2} の差分分析:")
    print(diff_df.round(3))
    
    return diff_df

# ===========================
# メイン処理: 単一ファイル分析
# ===========================
def analyze_single_file():
    """単一ファイルの音声特徴を分析"""
    print("🔷 単一ファイル分析モード\n")
    
    # ファイル選択
    audio_path = select_audio_file()
    if not audio_path:
        return
    
    # 特徴抽出
    features = extract_features(audio_path)
    
    # 統計サマリー
    summary = feature_summary(features)
    
    # 可視化
    plot_single_feature(features, title=f"音声特徴量: {audio_path.split('/')[-1]}")
    
    return features, summary

# ===========================
# メイン処理: 比較分析
# ===========================
def analyze_comparison():
    """複数ファイルの比較分析"""
    print("🔷 比較分析モード\n")
    
    # 成功ケースの選択
    print("📂 成功ケースの音声ファイルを選択してください")
    success_path = select_audio_file("成功ケースを選択")
    if not success_path:
        return
    
    # 失敗ケースの選択
    print("\n📂 失敗ケースの音声ファイルを選択してください")
    failure_path = select_audio_file("失敗ケースを選択")
    if not failure_path:
        return
    
    # 特徴抽出
    success_features = extract_features(success_path)
    failure_features = extract_features(failure_path)
    
    # 比較分析
    features_dict = {
        '成功': success_features,
        '失敗': failure_features
    }
    comparison_df = compare_features(features_dict, normalize=True)
    
    # 差分分析
    diff_df = analyze_difference(success_features, failure_features)
    
    # 可視化
    plot_comparison(comparison_df, title="成功 vs 失敗 の音声特徴比較")
    plot_heatmap(comparison_df, title="特徴量ヒートマップ")
    
    return comparison_df, diff_df

# ===========================
# メイン実行
# ===========================
if __name__ == "__main__":
    print("=" * 50)
    print("音声特徴統計分析システム")
    print("=" * 50)
    print("\n分析モードを選択してください:")
    print("1. 単一ファイル分析")
    print("2. 比較分析（成功 vs 失敗）")
    
    mode = input("\n選択 (1 or 2): ").strip()
    
    if mode == "1":
        results = analyze_single_file()
    elif mode == "2":
        results = analyze_comparison()
    else:
        print("⚠️ 無効な選択です")
    
    print("\n✅ 分析完了！")