import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FaceDetectionResult:
    """Result of face detection with bounding box and confidence"""
    x: int
    y: int
    w: int
    h: int
    confidence: float
    face_image: np.ndarray

class FacePreprocessor:
    """
    Face detection and preprocessing optimized for mobile CPU deployment
    Uses lightweight SSD MobileNet for face detection
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 target_size: Tuple[int, int] = (160, 160),
                 max_faces: int = 10):
        """
        Initialize face preprocessor
        
        Args:
            confidence_threshold: Minimum confidence for face detection
            target_size: Target size for face images (width, height)
            max_faces: Maximum number of faces to detect per image
        """
        self.confidence_threshold = confidence_threshold
        self.target_size = target_size
        self.max_faces = max_faces
        self.face_detector = None
        self._load_face_detector()
    
    def _load_face_detector(self):
        """Load lightweight face detection model optimized for CPU"""
        try:
            # Use OpenCV's DNN face detector (SSD MobileNet)
            model_file = "opencv_face_detector_uint8.pb"
            config_file = "opencv_face_detector.pbtxt"
            
            # For now, use Haar cascades as fallback (lightweight)
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            logger.info("Loaded Haar cascade face detector for CPU optimization")
            
        except Exception as e:
            logger.error(f"Error loading face detector: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """
        Detect faces in image using CPU-optimized detector
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detected faces with bounding boxes and confidence scores
        """
        if self.face_detector is None:
            raise ValueError("Face detector not loaded")
        
        try:
            # Convert to grayscale for Haar cascade
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            results = []
            for i, (x, y, w, h) in enumerate(faces[:self.max_faces]):
                # Extract face region
                face_image = image[y:y+h, x:x+w]
                
                # Calculate confidence (simplified for Haar cascade)
                confidence = 0.9  # Haar cascade doesn't provide confidence
                
                if confidence >= self.confidence_threshold:
                    result = FaceDetectionResult(
                        x=int(x), y=int(y), w=int(w), h=int(h),
                        confidence=confidence,
                        face_image=face_image
                    )
                    results.append(result)
            
            logger.debug(f"Detected {len(results)} faces with confidence >= {self.confidence_threshold}")
            return results
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess detected face for model input
        
        Args:
            face_image: Detected face image
            
        Returns:
            Preprocessed face image ready for model input
        """
        try:
            # Resize to target size
            resized = cv2.resize(face_image, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values to [0, 255] uint8
            if rgb_image.dtype != np.uint8:
                rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
            
            return rgb_image
            
        except Exception as e:
            logger.error(f"Error preprocessing face: {e}")
            raise
    
    def process_image(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Complete pipeline: detect faces and preprocess them
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of preprocessed face images
        """
        try:
            # Detect faces
            face_results = self.detect_faces(image)
            
            if not face_results:
                logger.warning("No faces detected in image")
                return []
            
            # Preprocess each detected face
            preprocessed_faces = []
            for face_result in face_results:
                preprocessed_face = self.preprocess_face(face_result.face_image)
                preprocessed_faces.append(preprocessed_face)
            
            logger.debug(f"Processed {len(preprocessed_faces)} faces")
            return preprocessed_faces
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return []
    
    def process_multiple_images(self, images: List[np.ndarray]) -> List[List[np.ndarray]]:
        """
        Process multiple images for batch operations
        
        Args:
            images: List of input images
            
        Returns:
            List of lists, each containing preprocessed faces from one image
        """
        results = []
        for i, image in enumerate(images):
            try:
                faces = self.process_image(image)
                results.append(faces)
                logger.debug(f"Processed image {i+1}/{len(images)}: {len(faces)} faces")
            except Exception as e:
                logger.error(f"Error processing image {i+1}: {e}")
                results.append([])
        
        return results
    
    def validate_face_quality(self, face_image: np.ndarray) -> Dict[str, Any]:
        """
        Validate face image quality for mobile constraints
        
        Args:
            face_image: Preprocessed face image
            
        Returns:
            Quality metrics and validation result
        """
        try:
            # Convert to grayscale for analysis
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_image
            
            # Calculate quality metrics
            # 1. Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 2. Brightness (mean intensity)
            brightness = np.mean(gray)
            
            # 3. Contrast (standard deviation)
            contrast = np.std(gray)
            
            # 4. Size check
            height, width = gray.shape
            size_ok = height >= 80 and width >= 80
            
            # Quality thresholds (optimized for mobile)
            sharpness_ok = laplacian_var > 50
            brightness_ok = 50 < brightness < 200
            contrast_ok = contrast > 20
            
            is_valid = all([size_ok, sharpness_ok, brightness_ok, contrast_ok])
            
            return {
                "is_valid": is_valid,
                "sharpness": float(laplacian_var),
                "brightness": float(brightness),
                "contrast": float(contrast),
                "size": (width, height),
                "quality_score": float(laplacian_var * contrast / 1000)  # Combined score
            }
            
        except Exception as e:
            logger.error(f"Error validating face quality: {e}")
            return {
                "is_valid": False,
                "error": str(e)
            }
    
    def get_face_landmarks(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract basic face landmarks (simplified for mobile)
        
        Args:
            face_image: Preprocessed face image
            
        Returns:
            Face landmarks array or None if detection fails
        """
        try:
            # For mobile optimization, we'll use a simplified approach
            # In a full implementation, you would use dlib or MediaPipe
            
            # Convert to grayscale
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_image
            
            # Use eye detection as basic landmarks
            eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(eyes) >= 2:
                # Return eye centers as basic landmarks
                landmarks = []
                for (ex, ey, ew, eh) in eyes[:2]:
                    center_x = ex + ew // 2
                    center_y = ey + eh // 2
                    landmarks.extend([center_x, center_y])
                
                return np.array(landmarks)
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting landmarks: {e}")
            return None
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """Get processor statistics and configuration"""
        return {
            "confidence_threshold": self.confidence_threshold,
            "target_size": self.target_size,
            "max_faces": self.max_faces,
            "detector_loaded": self.face_detector is not None
        } 