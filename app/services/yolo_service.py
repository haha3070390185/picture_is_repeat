import numpy as np
from ultralytics import YOLO
from PIL import Image
import cv2
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
                logger.warning(f"No results from YOLO for image: {image_path}, using pixel-based features")
                return self._extract_pixel_features(image_path)
            
            result = results[0]
            
            features = self._extract_pooled_features(result)
            
            if features is None:
                features = self._extract_detection_features(result)
            
            if features is None:
                logger.warning(f"YOLO detection features failed, using pixel-based features for: {image_path}")
                return self._extract_pixel_features(image_path)
            
            normalized_features = self._normalize(features)
            logger.info(f"Extracted YOLO-based features shape: {normalized_features.shape}")
            return normalized_features
            
        except Exception as e:
            logger.error(f"Error extracting YOLO features: {str(e)}, trying pixel-based features", exc_info=True)
            try:
                return self._extract_pixel_features(image_path)
            except Exception as e2:
                logger.error(f"Error extracting pixel features: {str(e2)}", exc_info=True)
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
    
    def _extract_pixel_features(self, image_path: str) -> Optional[np.ndarray]:
        try:
            img = cv2.imread(image_path)
            if img is None:
                img_pil = Image.open(image_path)
                img = np.array(img_pil)
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            if img is None:
                logger.error(f"Failed to read image: {image_path}")
                return None
            
            img_resized = cv2.resize(img, (64, 64))
            
            if len(img_resized.shape) == 3:
                gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_resized
            
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist = hist.flatten()
            
            hist_b = cv2.calcHist([img_resized], [0], None, [32], [0, 256]).flatten()
            hist_g = cv2.calcHist([img_resized], [1], None, [32], [0, 256]).flatten()
            hist_r = cv2.calcHist([img_resized], [2], None, [32], [0, 256]).flatten()
            
            color_hist = np.concatenate([hist_b, hist_g, hist_r])
            
            mean = np.mean(img_resized, axis=(0, 1))
            std = np.std(img_resized, axis=(0, 1))
            
            if len(mean) == 1:
                mean = np.array([mean[0], mean[0], mean[0]])
                std = np.array([std[0], std[0], std[0]])
            
            stats = np.concatenate([mean, std])
            
            features = np.concatenate([hist, color_hist, stats])
            
            current_size = len(features)
            if current_size < YOLO_FEATURE_SIZE:
                features = np.pad(features, (0, YOLO_FEATURE_SIZE - current_size), mode='constant')
            elif current_size > YOLO_FEATURE_SIZE:
                features = features[:YOLO_FEATURE_SIZE]
            
            normalized_features = self._normalize(features)
            logger.info(f"Extracted pixel-based features shape: {normalized_features.shape}")
            return normalized_features
            
        except Exception as e:
            logger.error(f"Error in pixel-based feature extraction: {str(e)}", exc_info=True)
            return None
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(features)
        if norm > 0:
            return features / norm
        return features

yolo_service = YOLOService()
