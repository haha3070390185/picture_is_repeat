from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ImageBase(BaseModel):
    original_filename: str
    content_type: str
    file_size: int

class ImageCreate(ImageBase):
    pass

class ImageResponse(ImageBase):
    id: int
    filename: str
    file_path: str
    created_at: datetime
    
    class Config:
        orm_mode = True

class SimilarityResultBase(BaseModel):
    image_id_1: int
    image_id_2: int
    similarity_score: float
    is_duplicate: int

class SimilarityResultResponse(SimilarityResultBase):
    id: int
    created_at: datetime
    
    class Config:
        orm_mode = True

class UploadResponse(BaseModel):
    success: bool
    message: str
    image: Optional[ImageResponse] = None
    similar_images: Optional[List[SimilarityResultResponse]] = None

class SimilarityCheckResponse(BaseModel):
    success: bool
    message: str
    is_duplicate: bool
    similar_images: Optional[List[dict]] = None
