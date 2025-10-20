# 🎙️ 音声解析デモシステム

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Whisper + OpenSMILE + BERT** による統合音声解析Webアプリケーション

## 🌐 デモサイト

**実際に触れるデモ環境**: http://[VPSのIPアドレス]

音声ファイルをアップロードして、以下の機能を体験できます:
- 🎙️ Whisper による文字起こし
- 📊 OpenSMILE による音声特徴抽出
- 🧠 BERT による感情分析
- 🔗 Late Fusion（音声+テキスト特徴の統合）

## 📸 スクリーンショット

![Demo UI](docs/images/demo-ui.png)
*シンプルで直感的なUI*

![Analysis Results](docs/images/results.png)
*詳細な解析結果の表示*

## ✨ 特徴

### 音声処理
- **Whisper**: 高精度な文字起こし（日本語対応）
- **OpenSMILE (eGeMAPSv02)**: 感情・性格分析向け音声特徴抽出
  - ピッチ（F0）
  - シマー（声の震え）
  - ジッター（声の揺れ）
  - ラウドネス（音量特性）
- **Noise Reduction**: ノイズ除去前処理

### 自然言語処理
- **BERT Sentiment Analysis**: 多言語感情分析
- **5段階評価**: 非常にネガティブ〜非常にポジティブ

### システム設計
- **FastAPI**: 高速なWeb API
- **非同期処理**: 効率的なリクエスト処理
- **エラーハンドリング**: 堅牢なエラー処理
- **セキュリティ**: ファイルサイズ制限、バリデーション

## 🛠️ 技術スタック

### Backend
- **Python 3.10+**
- **FastAPI** - Web フレームワーク
- **Uvicorn** - ASGI サーバー

### AI/ML
- **OpenAI Whisper** - 音声認識
- **OpenSMILE** - 音声特徴抽出
- **Transformers (BERT)** - 感情分析
- **librosa** - 音声処理
- **noisereduce** - ノイズ除去

### Frontend
- **HTML5 + JavaScript**
- **Pure CSS** - フレームワーク不使用
- **Fetch API** - 非同期通信

### Infrastructure
- **VPS** - Ubuntu 22.04
- **Nginx** - リバースプロキシ
- **Systemd** - サービス管理

## 📦 インストール

### 前提条件
- Python 3.10以上
- FFmpeg
- 2GB以上のメモリ（Whisper動作のため）

### ローカル環境

```bash
# リポジトリクローン
git clone https://github.com/[your-username]/audio-analysis-demo.git
cd audio-analysis-demo

# 仮想環境作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存パッケージインストール
pip install -r requirements.txt

# サーバー起動
uvicorn main:app --reload

# ブラウザでアクセス
# http://localhost:8000/static/index.html
```

### VPS環境

詳細は [デプロイ手順書](docs/DEPLOYMENT.md) を参照

```bash
# 簡易版
cd /var/www/audio-analysis-demo
git clone https://github.com/[your-username]/audio-analysis-demo.git .
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 🚀 使い方

### API エンドポイント

#### ヘルスチェック
```bash
curl http://localhost:8000/health
```

#### 音声解析
```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@your_audio.wav" \
  -F "enable_transcription=true" \
  -F "enable_features=true" \
  -F "enable_sentiment=true"
```

### レスポンス例

```json
{
  "transcription": {
    "text": "こんにちは、今日は良い天気ですね。",
    "language": "ja",
    "duration": 3.5
  },
  "audio_features": {
    "pitch_mean": 150.5,
    "pitch_std": 25.3,
    "shimmer": 0.15,
    "jitter": 0.02,
    "loudness_mean": 65.2,
    "loudness_std": 12.1
  },
  "sentiment": {
    "score": 4,
    "label": "ポジティブ",
    "confidence": 0.85
  },
  "processing_time": 5.3,
  "timestamp": "2025-01-20T10:30:00"
}
```

## 📊 パフォーマンス

| 処理 | 平均時間 | 備考 |
|------|---------|------|
| 文字起こし | 2-5秒 | 音声の長さに依存 |
| 音声特徴抽出 | 1-2秒 | OpenSMILE |
| 感情分析 | 1秒 | BERT |
| **合計** | **4-8秒** | 10秒以内の音声の場合 |

## 🔒 セキュリティ

- ✅ ファイルサイズ制限（10MB）
- ✅ ファイル形式検証（WAV, MP3, M4Aのみ）
- ✅ アップロードファイルの自動削除
- ✅ CORS設定
- ✅ エラーハンドリング

## 🐳 Docker サポート

```bash
# イメージビルド
docker build -t audio-analysis-demo .

# コンテナ起動
docker run -d -p 8000:8000 audio-analysis-demo
```

## 📈 今後の拡張予定

- [ ] WebSocket によるリアルタイム進捗表示
- [ ] 比較分析機能（2つの音声を比較）
- [ ] 結果のPDFエクスポート
- [ ] ユーザー認証
- [ ] データベース統合
- [ ] Big Five性格推定

## 🤝 貢献

プルリクエスト歓迎です！

## 📄 ライセンス

MIT License

## 👤 作成者

**[あなたの名前]**
- GitHub: [@rancorder](https://github.com/rancorder)
- Email: xzengbu@gmail.com
- ポートフォリオ: https://github.com/rancorder/audio-analysis-portfolio

## 🙏 謝辞

このプロジェクトは以下の素晴らしいオープンソースプロジェクトを使用しています:
- [OpenAI Whisper](https://github.com/openai/whisper)
- [OpenSMILE](https://audeering.github.io/opensmile/)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [FastAPI](https://github.com/tiangolo/fastapi)

---

⭐ このプロジェクトが役に立ったら、スターをお願いします！
