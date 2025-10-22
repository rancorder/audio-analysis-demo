# システムアーキテクチャ詳細図

## 全体アーキテクチャ

```mermaid
graph TB
    subgraph Client["クライアント層"]
        A[Webブラウザ<br/>HTML5 + JavaScript]
    end
    
    subgraph Proxy["プロキシ層"]
        B[Nginx<br/>Port: 8888<br/>リバースプロキシ]
    end
    
    subgraph API["アプリケーション層"]
        C[FastAPI<br/>Port: 8000<br/>非同期処理]
        C1[CORS Middleware]
        C2[File Validator]
        C3[Error Handler]
    end
    
    subgraph Processing["処理層"]
        D[前処理<br/>Noise Reduction<br/>noisereduce]
        
        E[Whisper<br/>large-v2<br/>文字起こし]
        F[OpenSMILE<br/>eGeMAPSv02<br/>音声特徴抽出]
        
        G[BERT<br/>多言語モデル<br/>感情分析]
        
        H[Late Fusion<br/>特徴統合レイヤー]
    end
    
    subgraph Storage["ストレージ"]
        I[一時ファイル<br/>/tmp<br/>自動削除]
        J[モデルキャッシュ<br/>~/.cache]
    end
    
    A -->|音声アップロード<br/>FormData| B
    B -->|Forward Request| C
    C --> C1
    C1 --> C2
    C2 --> C3
    C3 --> D
    
    D -->|音声データ| E
    D -->|音声データ| F
    
    E -->|テキスト| G
    
    F -->|音響特徴<br/>88次元| H
    G -->|テキスト特徴<br/>感情スコア| H
    
    H -->|統合結果<br/>JSON| C
    C -->|HTTP Response| B
    B -->|JSON Response| A
    
    D -.->|保存| I
    E -.->|参照| J
    G -.->|参照| J
    I -.->|削除| D
    
    style C fill:#009688,color:#fff
    style E fill:#FF6B6B,color:#fff
    style F fill:#4ECDC4,color:#fff
    style G fill:#95E1D3,color:#000
    style H fill:#F38181,color:#fff
```

## データフロー詳細

```mermaid
sequenceDiagram
    participant User as ユーザー
    participant Browser as ブラウザ
    participant Nginx as Nginx
    participant FastAPI as FastAPI
    participant Noise as Noise Reduction
    participant Whisper as Whisper
    participant OpenSMILE as OpenSMILE
    participant BERT as BERT
    participant Fusion as Late Fusion
    
    User->>Browser: 音声ファイル選択
    Browser->>Nginx: POST /api/analyze
    Nginx->>FastAPI: Forward Request
    
    FastAPI->>FastAPI: ファイル検証
    FastAPI->>Noise: 音声データ
    
    par 並列処理
        Noise->>Whisper: ノイズ除去済み音声
        Noise->>OpenSMILE: ノイズ除去済み音声
    end
    
    Whisper->>BERT: 文字起こしテキスト
    
    par 特徴抽出
        OpenSMILE->>Fusion: 音響特徴（88次元）
        BERT->>Fusion: 感情特徴（スコア+信頼度）
    end
    
    Fusion->>FastAPI: 統合結果
    FastAPI->>FastAPI: 一時ファイル削除
    FastAPI->>Nginx: JSON Response
    Nginx->>Browser: JSON Response
    Browser->>User: 結果表示
```

## 音声特徴抽出詳細

```mermaid
graph LR
    A[音声信号] --> B[OpenSMILE]
    
    B --> C[韻律特徴]
    B --> D[音響特徴]
    B --> E[時系列統計]
    
    C --> C1[Pitch F0]
    C --> C2[Jitter]
    C --> C3[Shimmer]
    
    D --> D1[Loudness]
    D --> D2[MFCCs]
    D --> D3[Spectral]
    
    E --> E1[Mean]
    E --> E2[Std Dev]
    E --> E3[Range]
    
    C1 --> F[88次元<br/>特徴ベクトル]
    C2 --> F
    C3 --> F
    D1 --> F
    D2 --> F
    D3 --> F
    E1 --> F
    E2 --> F
    E3 --> F
    
    style F fill:#4ECDC4,color:#fff
```

## Late Fusion戦略

```mermaid
graph TD
    A[音声入力] --> B[音響経路]
    A --> C[言語経路]
    
    B --> B1[OpenSMILE<br/>eGeMAPSv02]
    B1 --> B2[音響特徴<br/>88次元]
    
    C --> C1[Whisper<br/>文字起こし]
    C1 --> C2[BERT<br/>感情分析]
    C2 --> C3[テキスト特徴<br/>感情スコア+信頼度]
    
    B2 --> D[Late Fusion<br/>統合レイヤー]
    C3 --> D
    
    D --> E[統合特徴空間]
    E --> F[機械学習モデル<br/>将来的にBig Five推定]
    
    style D fill:#F38181,color:#fff
    style F fill:#95E1D3,color:#000
```

## インフラ構成

```mermaid
graph TB
    subgraph Internet["インターネット"]
        U[ユーザー]
    end
    
    subgraph VPS["VPS Ubuntu 22.04<br/>IP: 162.43.73.18"]
        N[Nginx<br/>Port: 8888]
        
        subgraph Service["Systemd Service"]
            F[FastAPI<br/>Port: 8000<br/>Uvicorn ASGI]
        end
        
        subgraph Python["Python 3.10 venv"]
            W[Whisper]
            O[OpenSMILE]
            B[BERT]
        end
        
        subgraph Storage["ストレージ"]
            T[/tmp/<br/>一時ファイル]
            C[~/.cache/<br/>モデル]
        end
    end
    
    U -->|HTTPS| N
    N -->|Proxy Pass| F
    F --> W
    F --> O
    F --> B
    
    W -.->|読込| C
    B -.->|読込| C
    F -.->|書込/削除| T
    
    style N fill:#269,color:#fff
    style F fill:#096,color:#fff
```

## 技術スタック詳細

| レイヤー | 技術 | バージョン | 役割 |
|---------|------|-----------|------|
| **Frontend** | HTML5 + JavaScript | - | UI/UX |
| **Proxy** | Nginx | 1.18+ | リバースプロキシ、静的ファイル配信 |
| **Backend** | FastAPI | 0.104+ | Web APIフレームワーク |
| **ASGI Server** | Uvicorn | 0.24+ | 非同期サーバー |
| **音声認識** | OpenAI Whisper | large-v2 | 多言語文字起こし |
| **音響特徴** | OpenSMILE | 3.0+ (eGeMAPSv02) | 88次元特徴抽出 |
| **感情分析** | BERT | 多言語 | テキスト感情分類 |
| **音声処理** | librosa | 0.10+ | 音声読込・変換 |
| **ノイズ除去** | noisereduce | 3.0+ | 前処理 |
| **OS** | Ubuntu | 22.04 | サーバーOS |
| **Service Management** | systemd | - | プロセス管理 |

## パフォーマンス最適化

```mermaid
graph LR
    A[最適化戦略]
    
    A --> B[非同期処理<br/>FastAPI async/await]
    A --> C[モデルキャッシュ<br/>初回ロードのみ]
    A --> D[一時ファイル管理<br/>自動削除]
    A --> E[Nginx圧縮<br/>gzip有効化]
    
    B --> F[複数リクエスト<br/>並列処理]
    C --> G[起動時間短縮]
    D --> H[ディスク節約]
    E --> I[転送量削減]
    
    style A fill:#F38181,color:#fff
```

## セキュリティ対策

```mermaid
graph TD
    A[セキュリティレイヤー]
    
    A --> B[入力検証]
    A --> C[ファイル管理]
    A --> D[ネットワーク]
    A --> E[エラー処理]
    
    B --> B1[ファイルサイズ制限<br/>10MB]
    B --> B2[MIME Type検証<br/>WAV/MP3/M4A]
    
    C --> C1[一時保存<br/>/tmp]
    C --> C2[自動削除<br/>処理後即座]
    
    D --> D1[CORS設定<br/>適切なOrigin]
    D --> D2[Rate Limiting<br/>将来実装予定]
    
    E --> E1[包括的エラーハンドリング]
    E --> E2[安全なエラーメッセージ]
    
    style A fill:#FF6B6B,color:#fff
```

---

## 図の使い方

これらの図は以下の用途で使用できます：

1. **README.mdに埋め込む** - Mermaidは自動レンダリングされます
2. **ドキュメント作成** - 技術仕様書や設計書に使用
3. **プレゼンテーション** - 応募時のポートフォリオ説明に活用
4. **PNG/SVGエクスポート** - Mermaid Live Editorでエクスポート可能

### Mermaid Live Editor
https://mermaid.live/

上記URLで図をコピペして、PNG/SVG形式でダウンロードできます。
