import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass

from ai.models.face_detection.ssd_resnet import SsdResNetDetector, FacialAreaRegion

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
    Uses SSD ResNet for accurate face detection
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
        """Load SSD ResNet face detection model optimized for CPU"""
        try:
            self.face_detector = SsdResNetDetector(
                confidence_threshold=self.confidence_threshold,
                nms_threshold=0.4,
                input_size=(300, 300)
            )
            logger.info("SSD ResNet face detector loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading face detector: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """
        Detect faces in image using SSD ResNet detector
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detected faces with bounding boxes and confidence scores
        """
        if self.face_detector is None:
            raise ValueError("Face detector not loaded")
        
        try:
            # Preprocess image for detection
            processed_img = self.face_detector.preprocess_image(image)
            
            # Detect faces using SSD ResNet
            facial_areas = self.face_detector.detect_faces(processed_img)
            
            results = []
            for i, facial_area in enumerate(facial_areas[:self.max_faces]):
                # Extract face region
                face_image = self.face_detector.extract_face(
                    image, facial_area, target_size=None
                )
                
                if face_image.size > 0:
                    result = FaceDetectionResult(
                        x=facial_area.x,
                        y=facial_area.y,
                        w=facial_area.w,
                        h=facial_area.h,
                        confidence=facial_area.confidence or 0.9,
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
                "quality_score": float(laplacian_var * contrast / 1000)
            }
            
        except Exception as e:
            logger.error(f"Error validating face quality: {e}")
            return {
                "is_valid": False,
                "error": str(e)
            }
    
    def extract_faces_with_alignment(self, image: np.ndarray, 
                                   align: bool = True,
                                   expand_percentage: int = 0) -> List[Dict[str, Any]]:
        """
        Extract faces with optional alignment (DeepFace compatible)
        
        Args:
            image: Input image
            align: Whether to align faces
            expand_percentage: Expand face region by percentage
            
        Returns:
            List of face extraction results
        """
        try:
            face_results = self.detect_faces(image)
            
            extracted_faces = []
            for face_result in face_results:
                # Get facial area
                facial_area = FacialAreaRegion(
                    x=face_result.x,
                    y=face_result.y,
                    w=face_result.w,
                    h=face_result.h,
                    confidence=face_result.confidence
                )
                
                # Extract face with optional expansion
                if expand_percentage > 0:
                    expanded_w = int(face_result.w * (1 + expand_percentage / 100))
                    expanded_h = int(face_result.h * (1 + expand_percentage / 100))
                    
                    new_x = max(0, face_result.x - (expanded_w - face_result.w) // 2)
                    new_y = max(0, face_result.y - (expanded_h - face_result.h) // 2)
                    
                    facial_area.x = new_x
                    facial_area.y = new_y
                    facial_area.w = min(image.shape[1] - new_x, expanded_w)
                    facial_area.h = min(image.shape[0] - new_y, expanded_h)
                
                face_img = self.face_detector.extract_face(image, facial_area)
                
                # Preprocess face
                preprocessed_face = self.preprocess_face(face_img)
                
                result = {
                    "face": preprocessed_face,
                    "facial_area": {
                        "x": facial_area.x,
                        "y": facial_area.y,
                        "w": facial_area.w,
                        "h": facial_area.h
                    },
                    "confidence": facial_area.confidence or 0.9
                }
                
                extracted_faces.append(result)
            
            return extracted_faces
            
        except Exception as e:
            logger.error(f"Error extracting faces with alignment: {e}")
            return []
    
    def get_face_landmarks(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract basic face landmarks (simplified for mobile)
        
        Args:
            face_image: Preprocessed face image
            
        Returns:
            Face landmarks array or None if detection fails
        """
        try:
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
        detector_info = {}
        if self.face_detector:
            detector_info = self.face_detector.get_detector_info()
        
        return {
            "confidence_threshold": self.confidence_threshold,
            "target_size": self.target_size,
            "max_faces": self.max_faces,
            "detector_loaded": self.face_detector is not None,
            "detector_info": detector_info
        }

def normalize_input(img: np.ndarray, normalization: str = "base") -> np.ndarray:
    """Normalize input image.

    Args:
        img (numpy array): the input image.
        normalization (str, optional): the normalization technique. Defaults to "base",
        for no normalization.

    Returns:
        numpy array: the normalized image.
    """

    if normalization == "base":
        return img

    img *= 255

    if normalization == "raw":
        pass

    elif normalization == "Facenet":
        mean, std = img.mean(), img.std()
        img = (img - mean) / std

    elif normalization == "Facenet2018":
        img /= 127.5
        img -= 1

    elif normalization == "VGGFace":
        img[..., 0] -= 93.5940
        img[..., 1] -= 104.7624
        img[..., 2] -= 129.1863

    elif normalization == "VGGFace2":
        img[..., 0] -= 91.4953
        img[..., 1] -= 103.8827
        img[..., 2] -= 131.0912

    elif normalization == "ArcFace":
        img -= 127.5
        img /= 128
    else:
        raise ValueError(f"unimplemented normalization type - {normalization}")

    return img


def resize_image(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize an image to expected size of a ml model with adding black pixels.
    Args:
        img (np.ndarray): pre-loaded image as numpy array
        target_size (tuple): input shape of ml model
    Returns:
        img (np.ndarray): resized input image
    """
    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)

    dsize = (
        int(img.shape[1] * factor),
        int(img.shape[0] * factor),
    )
    img = cv2.resize(img, dsize)

    diff_0 = target_size[0] - img.shape[0]
    diff_1 = target_size[1] - img.shape[1]

    img = np.pad(
        img,
        (
            (diff_0 // 2, diff_0 - diff_0 // 2),
            (diff_1 // 2, diff_1 - diff_1 // 2),
            (0, 0),
        ),
        "constant",
    )

    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    img = np.expand_dims(img, axis=0)

    if img.max() > 1:
        img = (img.astype(np.float32) / 255.0).astype(np.float32)

    return img 