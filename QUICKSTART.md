# クイックスタートガイド

## 🚀 最速で始める（推奨）

### Google Colabで実行

1. **Colabノートブックを開く**
   
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ImYCgljdtozIjcwAoDjUBVf8Mzkq-oKk?usp=sharing)

2. **実行**
   - メニューバーの `Runtime` → `Run all` をクリック
   - 約30秒でモデルが読み込まれます

3. **公開URLにアクセス**
   - 最後のセルに表示される `Running on public URL: https://xxxxx.gradio.live` をクリック

4. **音声解析を体験**
   - 音声ファイルをアップロード
   - 「🚀 解析開始」ボタンをクリック
   - 結果が表示されます！

---

## 💻 ローカル環境で実行（オプション）

### 前提条件
- Python 3.10以上
- GPU推奨（CPUでも動作します）

### インストール手順

```bash
# リポジトリクローン
git clone https://github.com/rancorder/audio-analysis-demo.git
cd audio-analysis-demo

# 仮想環境作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存パッケージインストール
pip install -r requirements.txt

# アプリ起動
python colab_audio_analysis.py
```

ブラウザで `http://localhost:7860` にアクセス

---

## 📊 何が解析できるの？

音声ファイル（WAV, MP3, M4A）をアップロードすると、以下が自動で解析されます：

### 1. 文字起こし（Whisper）
- 音声を日本語テキストに変換
- 精度: 95%以上

### 2. 音声特徴（OpenSMILE）
- ピッチ（声の高さ）
- シマー（声の震え）
- ジッター（声の揺れ）
- ラウドネス（音量）
- その他88次元の特徴量

### 3. 感情分析（BERT）
- 5段階評価: 😢 非常にネガティブ 〜 😄 非常にポジティブ
- 信頼度スコア付き

---

## ⚡ 処理速度

| 環境 | 10秒の音声 | 30秒の音声 |
|------|-----------|-----------|
| Google Colab (GPU) | 3-5秒 | 8-12秒 |
| ローカル (CPU) | 15秒 | 45秒 |

**GPU推奨！** Google Colabなら無料でGPUが使えます。

---

## 🎯 応募書類

- [応募書類（日本語）](docs/APPLICATION_LETTER.md)
- [応募書類（英語）](docs/APPLICATION_LETTER_EN.md)
- [システムアーキテクチャ](ARCHITECTURE.md)

---

## 📚 詳細ドキュメント

- [README（完全版）](README.md) - 技術詳細、検証結果、アーキテクチャ
- [コード](colab_audio_analysis.py) - 実装の全コード

---

## 🤝 作者

**Ai Art Studio**
- GitHub: [@rancorder](https://github.com/rancorder)
- Email: xzengbu@gmail.com

---

## 💡 トラブルシューティング

### Colabでエラーが出る場合

1. `Runtime` → `Factory reset runtime` を実行
2. もう一度 `Run all` を実行

### ローカルでGPUが認識されない場合

```python
import torch
print(torch.cuda.is_available())  # Falseの場合、CPU版PyTorchがインストールされています
```

GPU版PyTorchの再インストール:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

**さあ、音声解析を始めましょう！🎉**
