import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_SECRET = "dev-only-insecure-secret-key-change-me-in-env-file"


class Settings:
    APP_NAME = "Fake Detection in Newspaper and Images"
    SECRET_KEY = os.getenv("SECRET_KEY", DEFAULT_SECRET)
    JWT_ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "720"))

    DATA_DIR = BASE_DIR / "data"
    UPLOAD_DIR = BASE_DIR / "uploads"
    FRONTEND_DIR = BASE_DIR / "frontend"
    WEIGHTS_DIR = BASE_DIR / "models" / "weights"
    TRAINING_DATA_DIR = BASE_DIR / "training" / "data"

    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(BASE_DIR / 'data' / 'app.db').as_posix()}")

    MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "10"))
    MAX_VIDEO_MB = int(os.getenv("MAX_VIDEO_MB", "100"))
    MAX_TEXT_CHARS = 20000
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    ENABLE_TRANSFORMER = _bool("ENABLE_TRANSFORMER", True)
    TEXT_TRANSFORMER_MODEL = os.getenv("TEXT_TRANSFORMER_MODEL", "hamzab/roberta-fake-news-classification")
    ENABLE_PRETRAINED_DOWNLOAD = _bool("ENABLE_PRETRAINED_DOWNLOAD", True)
    VIDEO_NUM_FRAMES = int(os.getenv("VIDEO_NUM_FRAMES", "16"))
    TORCH_THREADS = int(os.getenv("TORCH_THREADS", "4"))


settings = Settings()

for _d in (settings.DATA_DIR, settings.UPLOAD_DIR, settings.WEIGHTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
