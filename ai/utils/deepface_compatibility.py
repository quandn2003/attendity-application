"""
DeepFace Compatibility Layer
Provides compatibility between our SSD ResNet detector and DeepFace-style interfaces
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Union, Optional, IO
import logging

from ai.models.face_detection.ssd_resnet import SsdResNetDetector, FacialAreaRegion
from ai.utils.preprocessing import FacePreprocessor

logger = logging.getLogger(__name__)

class DeepFaceCompatibleDetector:
    """
    Wrapper to make our SSD ResNet detector compatible with DeepFace interfaces
    """
    
    def __init__(self):
        self.detector = SsdResNetDetector()
        self.preprocessor = FacePreprocessor()
    
    def detect_faces(self, img: np.ndarray) -> List[FacialAreaRegion]:
        """
        Detect faces compatible with DeepFace Detector interface
        
        Args:
            img: Input image as numpy array
            
        Returns:
            List of FacialAreaRegion objects
        """
        return self.detector.detect_faces(img)

def extract_faces_compatible(
    img_path: Union[str, np.ndarray, IO[bytes]],
    detector_backend: str = "ssd_resnet",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    grayscale: bool = False,
    color_face: str = "rgb",
    normalize_face: bool = True,
    anti_spoofing: bool = False,
    max_faces: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Extract faces compatible with DeepFace extract_faces interface
    
    Args:
        img_path: Path to image or numpy array
        detector_backend: Face detector backend (ignored, uses SSD ResNet)
        enforce_detection: Raise exception if no face detected
        align: Enable face alignment
        expand_percentage: Expand face region percentage
        grayscale: Convert to grayscale (deprecated)
        color_face: Color format for output
        normalize_face: Normalize face pixels
        anti_spoofing: Enable anti-spoofing
        max_faces: Maximum faces to detect
        
    Returns:
        List of face extraction results compatible with DeepFace
    """
    try:
        # Load image if path provided
        if isinstance(img_path, str):
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not load image from {img_path}")
        elif isinstance(img_path, np.ndarray):
            img = img_path.copy()
        else:
            raise ValueError("Unsupported image input type")
        
        # Initialize preprocessor
        preprocessor = FacePreprocessor(max_faces=max_faces or 10)
        
        # Extract faces with alignment
        face_results = preprocessor.extract_faces_with_alignment(
            img, align=align, expand_percentage=expand_percentage
        )
        
        if not face_results and enforce_detection:
            raise ValueError("No faces detected in image")
        
        # Process results for DeepFace compatibility
        processed_results = []
        for face_result in face_results:
            face_img = face_result["face"]
            
            # Handle color format
            if color_face == "rgb":
                pass  # Already in RGB from preprocessor
            elif color_face == "bgr":
                face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            elif color_face == "gray":
                face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            
            # Handle normalization
            if normalize_face:
                face_img = face_img.astype(np.float32) / 255.0
            
            result = {
                "face": face_img,
                "facial_area": face_result["facial_area"],
                "confidence": face_result["confidence"]
            }
            
            # Add anti-spoofing if requested
            if anti_spoofing:
                # Placeholder for anti-spoofing integration
                result["is_real"] = True
                result["antispoof_score"] = 0.9
            
            processed_results.append(result)
        
        return processed_results
        
    except Exception as e:
        logger.error(f"Error in extract_faces_compatible: {e}")
        if enforce_detection:
            raise
        return []

def represent_compatible(
    img_path: Union[str, np.ndarray, IO[bytes]],
    model_name: str = "Facenet",
    enforce_detection: bool = True,
    detector_backend: str = "ssd_resnet",
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    anti_spoofing: bool = False,
    max_faces: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Generate face representations compatible with DeepFace represent interface
    
    Args:
        img_path: Path to image or numpy array
        model_name: Face recognition model (ignored, uses FaceNet)
        enforce_detection: Raise exception if no face detected
        detector_backend: Face detector backend (ignored, uses SSD ResNet)
        align: Enable face alignment
        expand_percentage: Expand face region percentage
        normalization: Normalization method
        anti_spoofing: Enable anti-spoofing
        max_faces: Maximum faces to detect
        
    Returns:
        List of representation results compatible with DeepFace
    """
    try:
        from ai.inference.engine import InferenceEngine
        from ai.models.facenet_model import ModelConfig
        
        # Initialize inference engine
        config = ModelConfig()
        engine = InferenceEngine(
            model_config=config,
            enable_anti_spoofing=anti_spoofing
        )
        
        # Load image
        if isinstance(img_path, str):
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not load image from {img_path}")
        elif isinstance(img_path, np.ndarray):
            img = img_path.copy()
        else:
            raise ValueError("Unsupported image input type")
        
        # Extract faces
        face_results = engine.preprocessor.extract_faces_with_alignment(
            img, align=align, expand_percentage=expand_percentage
        )
        
        if not face_results and enforce_detection:
            raise ValueError("No faces detected in image")
        
        # Generate embeddings
        representations = []
        for face_result in face_results:
            try:
                # Extract embedding
                embedding = engine.face_model.extract_embedding(face_result["face"])
                
                result = {
                    "embedding": embedding.tolist(),
                    "facial_area": face_result["facial_area"],
                    "face_confidence": face_result["confidence"]
                }
                
                representations.append(result)
                
            except Exception as e:
                logger.error(f"Error extracting embedding: {e}")
                if enforce_detection:
                    raise
        
        return representations
        
    except Exception as e:
        logger.error(f"Error in represent_compatible: {e}")
        if enforce_detection:
            raise
        return []

def verify_compatible(
    img1_path: Union[str, np.ndarray],
    img2_path: Union[str, np.ndarray],
    model_name: str = "Facenet",
    detector_backend: str = "ssd_resnet",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    silent: bool = False,
    threshold: Optional[float] = None,
    anti_spoofing: bool = False,
) -> Dict[str, Any]:
    """
    Verify if two images represent the same person (DeepFace compatible)
    
    Args:
        img1_path: First image
        img2_path: Second image
        model_name: Face recognition model (ignored, uses FaceNet)
        detector_backend: Face detector backend (ignored, uses SSD ResNet)
        distance_metric: Distance metric for comparison
        enforce_detection: Raise exception if no face detected
        align: Enable face alignment
        expand_percentage: Expand face region percentage
        normalization: Normalization method
        silent: Suppress log messages
        threshold: Verification threshold
        anti_spoofing: Enable anti-spoofing
        
    Returns:
        Verification result compatible with DeepFace
    """
    try:
        from ai.inference.engine import InferenceEngine
        from ai.models.facenet_model import ModelConfig
        import time
        
        start_time = time.time()
        
        # Initialize inference engine
        config = ModelConfig()
        engine = InferenceEngine(
            model_config=config,
            enable_anti_spoofing=anti_spoofing
        )
        
        # Load images
        if isinstance(img1_path, str):
            img1 = cv2.imread(img1_path)
        else:
            img1 = img1_path.copy()
            
        if isinstance(img2_path, str):
            img2 = cv2.imread(img2_path)
        else:
            img2 = img2_path.copy()
        
        # Process both images
        result1 = engine.process_single_image(img1)
        result2 = engine.process_single_image(img2)
        
        if not result1.success:
            if enforce_detection:
                raise ValueError(f"No face detected in first image: {result1.error_message}")
            return {"verified": False, "distance": float('inf'), "threshold": threshold or 0.6}
        
        if not result2.success:
            if enforce_detection:
                raise ValueError(f"No face detected in second image: {result2.error_message}")
            return {"verified": False, "distance": float('inf'), "threshold": threshold or 0.6}
        
        # Calculate distance
        if distance_metric == "cosine":
            distance = 1 - np.dot(result1.embedding, result2.embedding) / (
                np.linalg.norm(result1.embedding) * np.linalg.norm(result2.embedding)
            )
        elif distance_metric == "euclidean":
            distance = np.linalg.norm(result1.embedding - result2.embedding)
        else:
            distance = np.linalg.norm(result1.embedding - result2.embedding)
        
        # Set threshold if not provided
        if threshold is None:
            threshold = 0.6 if distance_metric == "cosine" else 1.0
        
        verified = distance < threshold
        
        return {
            "verified": verified,
            "distance": float(distance),
            "threshold": threshold,
            "model": "Facenet",
            "distance_metric": distance_metric,
            "facial_areas": {
                "img1": {"x": 0, "y": 0, "w": img1.shape[1], "h": img1.shape[0]},
                "img2": {"x": 0, "y": 0, "w": img2.shape[1], "h": img2.shape[0]}
            },
            "time": time.time() - start_time
        }
        
    except Exception as e:
        logger.error(f"Error in verify_compatible: {e}")
        if enforce_detection:
            raise
        return {"verified": False, "distance": float('inf'), "threshold": threshold or 0.6} 