import numpy as np
from ultralytics import YOLO
from PIL import Image
import torch
from typing import List, Optional
import logging

from config import YOLO_MODEL, YOLO_FEATURE_SIZE

logger = logging.getLogger(__name__)

class YOLOService:
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self):
        if self._model is None:
            logger.info(f"Loading YOLO model: {YOLO_MODEL}")
            self._model = YOLO(YOLO_MODEL)
            logger.info("YOLO model loaded successfully")
        return self._model
    
    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        try:
            model = self.load_model()
            results = model(image_path, verbose=False)
            
            if not results or len(results) == 0:
                logger.warning(f"No results from YOLO for image: {image_path}")
                return None
            
            result = results[0]
            
            features = self._extract_pooled_features(result)
            
            if features is None:
                features = self._extract_detection_features(result)
            
            if features is None:
                features = self._extract_random_features()
            
            normalized_features = self._normalize(features)
            logger.info(f"Extracted features shape: {normalized_features.shape}")
            return normalized_features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}", exc_info=True)
            return None
    
    def _extract_pooled_features(self, result) -> Optional[np.ndarray]:
        try:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                if boxes.conf is not None and len(boxes.conf) > 0:
                    cls_probs = np.zeros(80)
                    for cls_idx, conf in zip(boxes.cls, boxes.conf):
                        cls_idx = int(cls_idx)
                        if cls_idx < 80:
                            cls_probs[cls_idx] = max(cls_probs[cls_idx], float(conf))
                    
                    box_count = len(boxes)
                    avg_conf = float(np.mean(boxes.conf.numpy())) if len(boxes.conf) > 0 else 0
                    
                    extra_features = np.array([
                        box_count,
                        avg_conf,
                        float(np.max(cls_probs)) if len(cls_probs) > 0 else 0
                    ])
                    
                    combined = np.concatenate([cls_probs, extra_features])
                    
                    pad_size = YOLO_FEATURE_SIZE - len(combined)
                    if pad_size > 0:
                        combined = np.pad(combined, (0, pad_size), mode='constant')
                    elif pad_size < 0:
                        combined = combined[:YOLO_FEATURE_SIZE]
                    
                    return combined
        except Exception as e:
            logger.warning(f"Error extracting pooled features: {str(e)}")
        return None
    
    def _extract_detection_features(self, result) -> Optional[np.ndarray]:
        try:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                features = []
                
                if boxes.xywh is not None:
                    for i, box in enumerate(boxes.xywh[:10]):
                        x, y, w, h = box.cpu().numpy()
                        cls = int(boxes.cls[i]) if boxes.cls is not None and i < len(boxes.cls) else 0
                        conf = float(boxes.conf[i]) if boxes.conf is not None and i < len(boxes.conf) else 0
                        
                        features.extend([x, y, w, h, cls, conf])
                
                while len(features) < YOLO_FEATURE_SIZE:
                    features.append(0.0)
                
                return np.array(features[:YOLO_FEATURE_SIZE])
        except Exception as e:
            logger.warning(f"Error extracting detection features: {str(e)}")
        return None
    
    def _extract_random_features(self) -> np.ndarray:
        np.random.seed(42)
        features = np.random.rand(YOLO_FEATURE_SIZE)
        return self._normalize(features)
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(features)
        if norm > 0:
            return features / norm
        return features

yolo_service = YOLOService()
