"""
音声解析デモ FastAPI メインファイル
VPS上で動作する音声解析WebアプリケーションのAPI
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
import uvicorn
import os
import shutil
from datetime import datetime
import logging

# サービスのインポート（後で作成）
from services.audio_analysis import AudioAnalysisService
from models.schemas import AnalysisResult, HealthResponse

# ===========================
# ロギング設定
# ===========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ===========================
# FastAPIアプリケーション初期化
# ===========================
app = FastAPI(
    title="音声解析デモAPI",
    description="Whisper + OpenSMILE + BERT による音声解析システム",
    version="1.0.0"
)

# ===========================
# CORS設定（フロントエンドから呼び出せるように）
# ===========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では特定のドメインに制限
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# ディレクトリ設定
# ===========================
UPLOAD_DIR = "uploads"
STATIC_DIR = "static"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# 静的ファイル（フロントエンド）を配信
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ===========================
# サービス初期化
# ===========================
analysis_service = AudioAnalysisService()

# ===========================
# ヘルスチェックエンドポイント
# ===========================
@app.get("/", response_model=HealthResponse)
async def root():
    """APIのヘルスチェック"""
    return HealthResponse(
        status="healthy",
        message="音声解析API v1.0.0 稼働中",
        timestamp=datetime.now().isoformat()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """詳細なヘルスチェック"""
    try:
        # モデルの読み込み状態を確認
        models_loaded = analysis_service.check_models()
        
        return HealthResponse(
            status="healthy" if models_loaded else "degraded",
            message="すべてのモデルが正常に読み込まれています" if models_loaded else "一部のモデルが未読み込みです",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"ヘルスチェックエラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===========================
# ファイルアップロード制限
# ===========================
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a"}

def validate_audio_file(file: UploadFile) -> None:
    """アップロードされたファイルを検証"""
    # 拡張子チェック
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"サポートされていないファイル形式です。対応形式: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # ファイルサイズチェック（簡易版）
    # 注: 実際のサイズチェックはストリーミング処理が必要
    logger.info(f"ファイルアップロード: {file.filename}, タイプ: {file.content_type}")

# ===========================
# クリーンアップ処理
# ===========================
def cleanup_file(file_path: str):
    """一定時間後にファイルを削除"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"ファイル削除: {file_path}")
    except Exception as e:
        logger.error(f"ファイル削除エラー: {e}")

# ===========================
# メインの音声解析エンドポイント
# ===========================
@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    enable_transcription: bool = True,
    enable_features: bool = True,
    enable_sentiment: bool = True
):
    """
    音声ファイルを解析
    
    Args:
        file: 音声ファイル (WAV, MP3, M4A)
        enable_transcription: 文字起こしを実行するか
        enable_features: 音声特徴抽出を実行するか
        enable_sentiment: 感情分析を実行するか
    
    Returns:
        AnalysisResult: 解析結果
    """
    try:
        # 1. ファイル検証
        validate_audio_file(file)
        
        # 2. ファイル保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_ext = os.path.splitext(file.filename)[1]
        saved_filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, saved_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"ファイル保存完了: {file_path}")
        
        # 3. 音声解析実行
        logger.info("音声解析開始...")
        result = await analysis_service.analyze(
            file_path=file_path,
            enable_transcription=enable_transcription,
            enable_features=enable_features,
            enable_sentiment=enable_sentiment
        )
        
        logger.info("音声解析完了")
        
        # 4. バックグラウンドでファイル削除（10分後）
        # background_tasks.add_task(cleanup_file, file_path)
        
        # 5. 結果を返す
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"解析エラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"音声解析中にエラーが発生しました: {str(e)}")
    finally:
        # ファイルハンドルを閉じる
        await file.close()

# ===========================
# 比較分析エンドポイント
# ===========================
@app.post("/api/compare")
async def compare_audio(
    background_tasks: BackgroundTasks,
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    """
    2つの音声ファイルを比較分析
    
    Args:
        file1: 音声ファイル1
        file2: 音声ファイル2
    
    Returns:
        比較結果
    """
    try:
        # ファイル検証
        validate_audio_file(file1)
        validate_audio_file(file2)
        
        # ファイル保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        file1_path = os.path.join(UPLOAD_DIR, f"{timestamp}_1_{file1.filename}")
        file2_path = os.path.join(UPLOAD_DIR, f"{timestamp}_2_{file2.filename}")
        
        with open(file1_path, "wb") as buffer:
            shutil.copyfileobj(file1.file, buffer)
        
        with open(file2_path, "wb") as buffer:
            shutil.copyfileobj(file2.file, buffer)
        
        logger.info(f"比較用ファイル保存: {file1_path}, {file2_path}")
        
        # 比較分析実行
        comparison = await analysis_service.compare(file1_path, file2_path)
        
        # バックグラウンドでファイル削除
        # background_tasks.add_task(cleanup_file, file1_path)
        # background_tasks.add_task(cleanup_file, file2_path)
        
        return comparison
        
    except Exception as e:
        logger.error(f"比較分析エラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"比較分析中にエラーが発生しました: {str(e)}")

# ===========================
# サンプル音声取得エンドポイント
# ===========================
@app.get("/api/samples")
async def get_samples():
    """
    デモ用のサンプル音声ファイル一覧を取得
    """
    try:
        samples_dir = os.path.join(STATIC_DIR, "samples")
        if not os.path.exists(samples_dir):
            return {"samples": []}
        
        samples = []
        for filename in os.listdir(samples_dir):
            if os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS:
                samples.append({
                    "filename": filename,
                    "url": f"/static/samples/{filename}"
                })
        
        return {"samples": samples}
        
    except Exception as e:
        logger.error(f"サンプル取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===========================
# 統計情報エンドポイント
# ===========================
@app.get("/api/stats")
async def get_stats():
    """
    API使用統計を取得（簡易版）
    """
    try:
        upload_count = len([f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))])
        
        return {
            "total_analyses": upload_count,
            "uptime": "稼働中",
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"統計取得エラー: {e}")
        return {"error": str(e)}

# ===========================
# エラーハンドラー
# ===========================
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """グローバルエラーハンドラー"""
    logger.error(f"未処理エラー: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "サーバー内部エラーが発生しました"}
    )

# ===========================
# 起動設定
# ===========================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 開発時のみ
        log_level="info"
    )