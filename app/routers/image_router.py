from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
import os
from pathlib import Path
from datetime import datetime

from app.database import get_db
from app.models import ImageRecord, SimilarityResult
from app.schemas import (
    ImageResponse,
    UploadResponse,
    SimilarityCheckResponse,
    SimilarityResultResponse
)
from app.services.yolo_service import yolo_service
from app.services.similarity_service import similarity_service
from config import (
    UPLOAD_DIR,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
    SIMILARITY_THRESHOLD
)
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates")

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def generate_unique_filename(original_filename: str) -> str:
    ext = Path(original_filename).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    return unique_name

def save_upload_file(upload_file: UploadFile, filename: str) -> str:
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as buffer:
        content = upload_file.file.read()
        buffer.write(content)
    return str(file_path)

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/upload", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式。支持的格式: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件大小超过限制。最大允许: {MAX_FILE_SIZE / 1024 / 1024} MB"
        )
    
    unique_filename = generate_unique_filename(file.filename)
    file_path = UPLOAD_DIR / unique_filename
    
    with open(file_path, "wb") as buffer:
        buffer.write(content)
    
    features = yolo_service.extract_features(str(file_path))
    if features is None:
        return UploadResponse(
            success=False,
            message="图片特征提取失败，请尝试其他图片"
        )
    
    existing_images = db.query(ImageRecord).all()
    database_features = []
    for img in existing_images:
        db_features = similarity_service.json_to_features(img.feature_vector)
        if db_features is not None:
            database_features.append((img.id, db_features))
    
    similar_images = []
    if database_features:
        similar_results = similarity_service.find_top_similar(
            features, database_features, top_n=10
        )
        
        for result in similar_results:
            db_image = db.query(ImageRecord).filter(
                ImageRecord.id == result["image_id"]
            ).first()
            
            if db_image:
                similarity_record = SimilarityResult(
                    image_id_1=None,
                    image_id_2=db_image.id,
                    similarity_score=result["similarity_score"],
                    is_duplicate=1 if result["is_duplicate"] else 0
                )
                db.add(similarity_record)
                
                similar_images.append({
                    "image_id": db_image.id,
                    "original_filename": db_image.original_filename,
                    "similarity_score": result["similarity_score"],
                    "similarity_percentage": result["similarity_percentage"],
                    "is_duplicate": result["is_duplicate"]
                })
    
    image_record = ImageRecord(
        filename=unique_filename,
        original_filename=file.filename,
        file_path=str(file_path),
        file_size=len(content),
        content_type=file.content_type or "application/octet-stream",
        feature_vector=similarity_service.features_to_json(features)
    )
    db.add(image_record)
    db.commit()
    db.refresh(image_record)
    
    for sim in similar_images:
        if sim.get("similarity_record"):
            sim["similarity_record"].image_id_1 = image_record.id
    db.commit()
    
    duplicate_count = sum(1 for sim in similar_images if sim["is_duplicate"])
    
    if duplicate_count > 0:
        message = f"检测到 {duplicate_count} 张相似图片，相似度超过 {SIMILARITY_THRESHOLD * 100}%！"
    else:
        message = "图片上传成功，未检测到高度相似的图片"
    
    return UploadResponse(
        success=True,
        message=message,
        image=ImageResponse(
            id=image_record.id,
            filename=image_record.filename,
            original_filename=image_record.original_filename,
            file_path=image_record.file_path,
            file_size=image_record.file_size,
            content_type=image_record.content_type,
            created_at=image_record.created_at
        ),
        similar_images=[
            SimilarityResultResponse(
                id=0,
                image_id_1=image_record.id,
                image_id_2=sim["image_id"],
                similarity_score=sim["similarity_score"],
                is_duplicate=1 if sim["is_duplicate"] else 0,
                created_at=datetime.now()
            ) for sim in similar_images
        ]
    )

@router.get("/all", response_model=List[ImageResponse])
async def get_all_images(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    images = db.query(ImageRecord).offset(skip).limit(limit).all()
    return [
        ImageResponse(
            id=img.id,
            filename=img.filename,
            original_filename=img.original_filename,
            file_path=img.file_path,
            file_size=img.file_size,
            content_type=img.content_type,
            created_at=img.created_at
        ) for img in images
    ]

@router.get("/{image_id}/info", response_model=ImageResponse)
async def get_image_info(
    image_id: int,
    db: Session = Depends(get_db)
):
    image = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="图片不存在")
    
    return ImageResponse(
        id=image.id,
        filename=image.filename,
        original_filename=image.original_filename,
        file_path=image.file_path,
        file_size=image.file_size,
        content_type=image.content_type,
        created_at=image.created_at
    )

@router.get("/{image_id}/similar", response_model=SimilarityCheckResponse)
async def check_similarity(
    image_id: int,
    threshold: float = SIMILARITY_THRESHOLD,
    db: Session = Depends(get_db)
):
    target_image = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    if not target_image:
        raise HTTPException(status_code=404, detail="图片不存在")
    
    target_features = similarity_service.json_to_features(target_image.feature_vector)
    if target_features is None:
        raise HTTPException(status_code=500, detail="无法提取图片特征")
    
    other_images = db.query(ImageRecord).filter(
        ImageRecord.id != image_id
    ).all()
    
    if not other_images:
        return SimilarityCheckResponse(
            success=True,
            message="没有其他图片进行对比",
            is_duplicate=False,
            similar_images=[]
        )
    
    database_features = []
    for img in other_images:
        db_features = similarity_service.json_to_features(img.feature_vector)
        if db_features is not None:
            database_features.append((img.id, db_features))
    
    similar_results = similarity_service.find_top_similar(
        target_features, database_features, top_n=10
    )
    
    detailed_results = []
    for result in similar_results:
        db_image = db.query(ImageRecord).filter(
            ImageRecord.id == result["image_id"]
        ).first()
        
        if db_image:
            detailed_results.append({
                "image_id": db_image.id,
                "original_filename": db_image.original_filename,
                "similarity_score": result["similarity_score"],
                "similarity_percentage": result["similarity_percentage"],
                "is_duplicate": result["is_duplicate"],
                "created_at": db_image.created_at.isoformat() if db_image.created_at else None
            })
    
    has_duplicate = any(result["is_duplicate"] for result in detailed_results)
    
    if has_duplicate:
        message = f"检测到相似图片，相似度超过 {threshold * 100}%！"
    else:
        message = "未检测到高度相似的图片"
    
    return SimilarityCheckResponse(
        success=True,
        message=message,
        is_duplicate=has_duplicate,
        similar_images=detailed_results
    )

@router.delete("/{image_id}")
async def delete_image(
    image_id: int,
    db: Session = Depends(get_db)
):
    image = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="图片不存在")
    
    try:
        file_path = Path(image.file_path)
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        print(f"Warning: Could not delete file {image.file_path}: {e}")
    
    db.query(SimilarityResult).filter(
        (SimilarityResult.image_id_1 == image_id) |
        (SimilarityResult.image_id_2 == image_id)
    ).delete()
    
    db.delete(image)
    db.commit()
    
    return {"success": True, "message": "图片已删除"}

@router.get("/file/{image_id}")
async def get_image_file(
    image_id: int,
    db: Session = Depends(get_db)
):
    image = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="图片不存在")
    
    file_path = Path(image.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="图片文件不存在")
    
    return FileResponse(
        path=str(file_path),
        media_type=image.content_type or "application/octet-stream"
    )
