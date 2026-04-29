from sqlalchemy import Column, Integer, String, DateTime, Float, Text
from sqlalchemy.sql import func
from app.database import Base

class ImageRecord(Base):
    __tablename__ = "image_records"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), index=True)
    original_filename = Column(String(255))
    file_path = Column(String(500))
    file_size = Column(Integer)
    content_type = Column(String(100))
    feature_vector = Column(Text)  # 存储特征向量的JSON字符串
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class SimilarityResult(Base):
    __tablename__ = "similarity_results"

    id = Column(Integer, primary_key=True, index=True)
    image_id_1 = Column(Integer, index=True)
    image_id_2 = Column(Integer, index=True)
    similarity_score = Column(Float)
    is_duplicate = Column(Integer, default=0)  # 0: 不重复, 1: 重复嫌疑
    created_at = Column(DateTime(timezone=True), server_default=func.now())
