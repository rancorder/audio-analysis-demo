"""
機械学習予測モデル - サンプルコード
多出力回帰問題の実装例（XGBoost + RandomForest）

特徴:
- 短期データ + 長期データの統合
- 6つの独立した予測値を同時推定
- ノイズ付加によるロバスト性向上
- 適切な特徴量エンジニアリング
"""

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from collections import Counter
import datetime
import os

# ===========================
# データ読み込み
# ===========================
def load_data():
    """CSVファイルをダイアログで選択して読み込み"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="CSVファイルを選択",
        filetypes=[("CSV files", "*.csv")]
    )
    
    if not file_path:
        raise ValueError("エラー: ファイルが選択されていません")
    
    df = pd.read_csv(file_path, encoding="utf-8")
    print(f"✅ データ読み込み完了: {len(df)}行")
    
    return df, file_path

# ===========================
# 特徴量エンジニアリング
# ===========================
def create_features(data, N=5, max_past_data=500):
    """
    短期データ + 長期データの統合特徴量を作成
    
    Args:
        data: 入力データ配列
        N: 直近N回のデータを使用
        max_past_data: 短期データの最大長
    
    Returns:
        X_recent: 直近データ特徴量
        X_past: 過去データ特徴量
        y: ターゲット値
    """
    X_recent, X_past, y = [], [], []
    full_data_length = len(data)
    
    for i in range(N, full_data_length):
        # 1. 直近N回のデータ
        recent_data = data[i-N:i].flatten()
        X_recent.append(recent_data)
        
        # 2. すべての過去データ
        all_past = data[:i].flatten()
        
        # 3. 短期データ（最新500件）
        if len(all_past) > max_past_data:
            past_500 = all_past[-max_past_data:]
        else:
            # 不足分をゼロ埋め
            pad_length = max_past_data - len(all_past)
            past_500 = np.pad(all_past, (pad_length, 0), mode='constant')
        
        # 4. 長期データの長さを制限
        all_past_limited = all_past[:full_data_length]
        
        # 5. 特徴量を結合
        combined_features = np.hstack([past_500, all_past_limited])
        
        # 6. 長さを統一（パディング）
        target_length = max_past_data + full_data_length
        if len(combined_features) < target_length:
            pad_length = target_length - len(combined_features)
            combined_features = np.pad(
                combined_features, 
                (0, pad_length), 
                mode='constant'
            )
        
        X_past.append(combined_features)
        
        # 7. ターゲット（次回の値セット）
        y.append(data[i])
    
    return np.array(X_recent), np.array(X_past), np.array(y)

# ===========================
# データ統計分析
# ===========================
def analyze_data_statistics(data, column_names):
    """データの統計情報を分析"""
    all_numbers = data.flatten()
    total_count = len(all_numbers)
    number_counts = Counter(all_numbers)
    
    print(f"\n📊 データ統計（総データ数: {total_count}）")
    
    # 頻出TOP10
    most_common = number_counts.most_common(10)
    print("\n頻出TOP10:")
    for num, count in most_common:
        percentage = (count / total_count) * 100
        print(f"  {num}: {count}回 ({percentage:.2f}%)")
    
    # 最も出ていないTOP10
    all_possible = set(range(1, 44))
    missing_counts = {num: number_counts.get(num, 0) for num in all_possible}
    least_common = sorted(missing_counts.items(), key=lambda x: x[1])[:10]
    
    print("\n出現が少ないTOP10:")
    for num, count in least_common:
        percentage = (count / total_count) * 100 if total_count > 0 else 0
        print(f"  {num}: {count}回 ({percentage:.2f}%)")
    
    return number_counts

# ===========================
# ラベルエンコーディング
# ===========================
def encode_labels(y_train, y_test, n_outputs=6):
    """
    各出力ごとにラベルエンコーディングを実行
    
    Args:
        y_train: 訓練データのターゲット
        y_test: テストデータのターゲット
        n_outputs: 出力数
    
    Returns:
        エンコード済みデータとエンコーダーのリスト
    """
    label_encoders = []
    y_train_encoded = []
    y_test_encoded = []
    
    for i in range(n_outputs):
        # 全クラスを取得
        all_classes = np.unique(
            np.concatenate([y_train[:, i], y_test[:, i]])
        )
        
        # エンコーダー作成
        encoder = LabelEncoder()
        encoder.fit(all_classes)
        label_encoders.append(encoder)
        
        # エンコード実行
        y_train_encoded.append(encoder.transform(y_train[:, i]))
        y_test_encoded.append(encoder.transform(y_test[:, i]))
    
    y_train_encoded = np.column_stack(y_train_encoded)
    y_test_encoded = np.column_stack(y_test_encoded)
    
    print(f"✅ ラベルエンコーディング完了（{n_outputs}出力）")
    
    return y_train_encoded, y_test_encoded, label_encoders

# ===========================
# XGBoostモデル訓練
# ===========================
def train_xgboost_models(X_train, y_train_encoded, n_outputs=6):
    """
    各出力ごとにXGBoostモデルを訓練
    
    Args:
        X_train: 訓練データ
        y_train_encoded: エンコード済みターゲット
        n_outputs: 出力数
    
    Returns:
        訓練済みモデルのリスト
    """
    print("\n🎯 XGBoostモデル訓練開始...")
    models = []
    
    for i in range(n_outputs):
        print(f"  モデル {i+1}/{n_outputs} 訓練中...")
        
        xgb = XGBClassifier(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.6,
            eval_metric='mlogloss',
            random_state=42
        )
        
        xgb.fit(X_train, y_train_encoded[:, i])
        models.append(xgb)
    
    print("✅ XGBoostモデル訓練完了")
    return models

# ===========================
# RandomForestモデル訓練
# ===========================
def train_randomforest_models(X_train, y_train_encoded, n_outputs=6):
    """
    各出力ごとにRandomForestモデルを訓練
    
    Args:
        X_train: 訓練データ
        y_train_encoded: エンコード済みターゲット
        n_outputs: 出力数
    
    Returns:
        訓練済みモデルのリスト
    """
    print("\n🌲 RandomForestモデル訓練開始...")
    models = []
    
    for i in range(n_outputs):
        print(f"  モデル {i+1}/{n_outputs} 訓練中...")
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        )
        
        rf.fit(X_train, y_train_encoded[:, i])
        models.append(rf)
    
    print("✅ RandomForestモデル訓練完了")
    return models

# ===========================
# 予測実行（ノイズ付加）
# ===========================
def predict_with_noise(models, X_latest, label_encoders, 
                       noise_range=(-3, 4), clip_range=(1, 43)):
    """
    ノイズを付加してロバストな予測を実行
    
    Args:
        models: 訓練済みモデルのリスト
        X_latest: 最新データ
        label_encoders: ラベルエンコーダーのリスト
        noise_range: ノイズの範囲
        clip_range: クリッピングの範囲
    
    Returns:
        予測値の配列
    """
    # ノイズ付加
    noise = np.random.randint(*noise_range, size=X_latest.shape)
    X_with_noise = np.clip(X_latest + noise, *clip_range)
    
    # 各モデルで予測
    predictions = []
    for i, model in enumerate(models):
        pred_encoded = model.predict(X_with_noise)
        pred_decoded = label_encoders[i].inverse_transform(pred_encoded)
        predictions.append(pred_decoded)
    
    result = np.column_stack(predictions)
    return np.sort(result.flatten())

# ===========================
# 結果保存
# ===========================
def save_results(xgb_pred, rf_pred, stats, output_dir):
    """予測結果とログを保存"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(output_dir, f"prediction_log_{timestamp}.txt")
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== 機械学習予測結果 ===\n\n")
        
        f.write("【XGBoost予測】\n")
        f.write(f"  {xgb_pred.tolist()}\n\n")
        
        f.write("【RandomForest予測】\n")
        f.write(f"  {rf_pred.tolist()}\n\n")
        
        f.write("【データ統計】\n")
        f.write(f"  総データ数: {sum(stats.values())}\n")
        f.write(f"  ユニーク値数: {len(stats)}\n")
    
    print(f"✅ 結果保存: {log_file}")
    return log_file

# ===========================
# メイン処理
# ===========================
def main():
    """統合処理のメインフロー"""
    print("🔷 機械学習予測システム 起動\n")
    
    # 1. データ読み込み
    df, file_path = load_data()
    output_dir = os.path.dirname(file_path)
    
    # 想定: 6列の数値データ
    number_columns = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']
    data = df[number_columns].values
    
    # データ検証
    assert data.shape[1] == 6, "エラー: データは6列必要です"
    
    # 2. データ統計分析
    stats = analyze_data_statistics(data, number_columns)
    
    # 3. 特徴量作成
    print("\n⚙️ 特徴量作成中...")
    X_recent, X_past, y = create_features(data, N=5, max_past_data=500)
    
    # 特徴量結合
    X_combined = np.hstack([X_recent, X_past])
    print(f"✅ 特徴量作成完了: {X_combined.shape}")
    
    # 4. データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42
    )
    print(f"✅ データ分割: 訓練{len(X_train)}件, テスト{len(X_test)}件")
    
    # 5. ラベルエンコーディング
    y_train_enc, y_test_enc, encoders = encode_labels(y_train, y_test)
    
    # 6. XGBoostモデル訓練
    xgb_models = train_xgboost_models(X_train, y_train_enc)
    
    # 7. RandomForestモデル訓練
    rf_models = train_randomforest_models(X_train, y_train_enc)
    
    # 8. 最新データから予測
    print("\n🎯 予測実行中...")
    num_recent = 20
    recent_samples = X_combined[-num_recent:].mean(axis=0).reshape(1, -1)
    
    xgb_pred = predict_with_noise(xgb_models, recent_samples, encoders)
    rf_pred = predict_with_noise(rf_models, recent_samples, encoders)
    
    print(f"\n🎯 XGBoost予測: {xgb_pred.tolist()}")
    print(f"🎯 RandomForest予測: {rf_pred.tolist()}")
    
    # 9. 結果保存
    save_results(xgb_pred, rf_pred, stats, output_dir)
    
    print("\n✅ 全処理完了！")
    
    return {
        'xgb_prediction': xgb_pred,
        'rf_prediction': rf_pred,
        'statistics': stats
    }

if __name__ == "__main__":
    results = main()