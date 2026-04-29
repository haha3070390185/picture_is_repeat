import numpy as np
from typing import List, Tuple, Optional
import logging
import json

from config import SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

class SimilarityService:
    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        try:
            if v1 is None or v2 is None:
                return 0.0
            
            v1 = np.array(v1, dtype=np.float64)
            v2 = np.array(v2, dtype=np.float64)
            
            if v1.ndim > 1:
                v1 = v1.flatten()
            if v2.ndim > 1:
                v2 = v2.flatten()
            
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
    
    @staticmethod
    def euclidean_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        try:
            if v1 is None or v2 is None:
                return 0.0
            
            v1 = np.array(v1, dtype=np.float64)
            v2 = np.array(v2, dtype=np.float64)
            
            distance = np.linalg.norm(v1 - v2)
            similarity = 1 / (1 + distance)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating euclidean similarity: {str(e)}")
            return 0.0
    
    @staticmethod
    def combined_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        cosine_sim = SimilarityService.cosine_similarity(v1, v2)
        euclidean_sim = SimilarityService.euclidean_similarity(v1, v2)
        
        combined = (cosine_sim * 0.7) + (euclidean_sim * 0.3)
        return float(combined)
    
    @staticmethod
    def is_duplicate(similarity_score: float, threshold: float = None) -> Tuple[bool, float]:
        if threshold is None:
            threshold = SIMILARITY_THRESHOLD
        
        is_dup = similarity_score >= threshold
        confidence = similarity_score
        
        return is_dup, confidence
    
    @staticmethod
    def compare_with_database(
        new_features: np.ndarray,
        database_features: List[Tuple[int, np.ndarray]]
    ) -> List[Tuple[int, float, bool]]:
        results = []
        
        for image_id, features in database_features:
            try:
                similarity = SimilarityService.combined_similarity(new_features, features)
                is_dup, _ = SimilarityService.is_duplicate(similarity)
                results.append((image_id, similarity, is_dup))
            except Exception as e:
                logger.error(f"Error comparing with image {image_id}: {str(e)}")
                continue
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    @staticmethod
    def find_top_similar(
        new_features: np.ndarray,
        database_features: List[Tuple[int, np.ndarray]],
        top_n: int = 10
    ) -> List[dict]:
        results = SimilarityService.compare_with_database(new_features, database_features)
        
        top_results = []
        for image_id, similarity, is_dup in results[:top_n]:
            top_results.append({
                "image_id": image_id,
                "similarity_score": similarity,
                "is_duplicate": is_dup,
                "similarity_percentage": round(similarity * 100, 2)
            })
        
        return top_results
    
    @staticmethod
    def features_to_json(features: np.ndarray) -> str:
        if features is None:
            return "[]"
        try:
            return json.dumps(features.tolist())
        except Exception as e:
            logger.error(f"Error converting features to JSON: {str(e)}")
            return "[]"
    
    @staticmethod
    def json_to_features(json_str: str) -> Optional[np.ndarray]:
        try:
            if not json_str:
                return None
            features_list = json.loads(json_str)
            return np.array(features_list, dtype=np.float64)
        except Exception as e:
            logger.error(f"Error converting JSON to features: {str(e)}")
            return None

similarity_service = SimilarityService()
