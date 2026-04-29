import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# 数据库配置
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/database/images.db")

# 上传配置
UPLOAD_DIR = BASE_DIR / "uploads"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 相似度配置
SIMILARITY_THRESHOLD = 0.5  # 50% 相似度阈值

# YOLO配置
YOLO_MODEL = "yolov8n.pt"  # 使用nano版本，速度快
YOLO_FEATURE_SIZE = 256  # 特征向量大小

# 服务器配置
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# 确保必要的目录存在
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "database").mkdir(parents=True, exist_ok=True)
