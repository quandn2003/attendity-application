import cv2
import numpy as np
from typing import List, Optional, Tuple
import logging
import os
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FacialAreaRegion:
    """Face detection result compatible with DeepFace structure"""
    x: int
    y: int
    w: int
    h: int
    left_eye: Optional[Tuple[int, int]] = None
    right_eye: Optional[Tuple[int, int]] = None
    confidence: Optional[float] = None
    nose: Optional[Tuple[int, int]] = None
    mouth_right: Optional[Tuple[int, int]] = None
    mouth_left: Optional[Tuple[int, int]] = None

class SsdResNetDetector:
    """
    SSD ResNet face detector optimized for mobile CPU deployment
    Compatible with DeepFace Detector interface
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 nms_threshold: float = 0.4,
                 input_size: Tuple[int, int] = (300, 300)):
        """
        Initialize SSD ResNet face detector
        
        Args:
            confidence_threshold: Minimum confidence for face detection
            nms_threshold: Non-maximum suppression threshold
            input_size: Input size for the detector network
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.net = None
        self.model_loaded = False
        
        self._load_model()
    
    def _load_model(self):
        """Load SSD ResNet model for face detection"""
        try:
            model_dir = os.path.join(os.path.dirname(__file__), "weights")
            os.makedirs(model_dir, exist_ok=True)
            
            prototxt_path = os.path.join(model_dir, "deploy.prototxt")
            model_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
            
            prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            
            if not os.path.exists(prototxt_path):
                logger.info("Downloading SSD ResNet prototxt...")
                urllib.request.urlretrieve(prototxt_url, prototxt_path)
            
            if not os.path.exists(model_path):
                logger.info("Downloading SSD ResNet model weights...")
                urllib.request.urlretrieve(model_url, model_path)
            
            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            self.model_loaded = True
            logger.info("SSD ResNet face detector loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading SSD ResNet model: {e}")
            self._load_fallback_detector()
    
    def _load_fallback_detector(self):
        """Load Haar cascade as fallback if SSD ResNet fails"""
        try:
            self.net = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.model_loaded = True
            logger.warning("Using Haar cascade as fallback face detector")
        except Exception as e:
            logger.error(f"Failed to load fallback detector: {e}")
            raise
    
    def detect_faces(self, img: np.ndarray) -> List[FacialAreaRegion]:
        """
        Detect faces in image using SSD ResNet
        Compatible with DeepFace Detector interface
        
        Args:
            img: Input image as numpy array (BGR format)
            
        Returns:
            List of FacialAreaRegion objects with detected faces
        """
        if not self.model_loaded:
            raise ValueError("Face detector not loaded")
        
        try:
            if isinstance(self.net, cv2.dnn.Net):
                return self._detect_with_ssd(img)
            else:
                return self._detect_with_haar(img)
                
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []
    
    def _detect_with_ssd(self, img: np.ndarray) -> List[FacialAreaRegion]:
        """Detect faces using SSD ResNet"""
        h, w = img.shape[:2]
        
        blob = cv2.dnn.blobFromImage(
            img, 1.0, self.input_size, 
            (104.0, 177.0, 123.0), 
            swapRB=False, crop=False
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x, y, x1, y1 = box.astype(int)
                
                face_w = x1 - x
                face_h = y1 - y
                
                if face_w > 0 and face_h > 0:
                    facial_area = FacialAreaRegion(
                        x=max(0, x),
                        y=max(0, y),
                        w=min(w - x, face_w),
                        h=min(h - y, face_h),
                        confidence=float(confidence)
                    )
                    faces.append(facial_area)
        
        faces = self._apply_nms(faces)
        return faces
    
    def _detect_with_haar(self, img: np.ndarray) -> List[FacialAreaRegion]:
        """Detect faces using Haar cascade fallback"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        detections = self.net.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        faces = []
        for (x, y, w, h) in detections:
            facial_area = FacialAreaRegion(
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                confidence=0.9
            )
            faces.append(facial_area)
        
        return faces
    
    def _apply_nms(self, faces: List[FacialAreaRegion]) -> List[FacialAreaRegion]:
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        if len(faces) <= 1:
            return faces
        
        boxes = []
        confidences = []
        
        for face in faces:
            boxes.append([face.x, face.y, face.w, face.h])
            confidences.append(face.confidence or 0.9)
        
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            confidences.tolist(), 
            self.confidence_threshold, 
            self.nms_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [faces[i] for i in indices]
        
        return []
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for face detection
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            Preprocessed image
        """
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        return img
    
    def extract_face(self, img: np.ndarray, facial_area: FacialAreaRegion, 
                    target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Extract and preprocess face region
        
        Args:
            img: Source image
            facial_area: Detected face region
            target_size: Target size for extracted face
            
        Returns:
            Extracted face image
        """
        x, y, w, h = facial_area.x, facial_area.y, facial_area.w, facial_area.h
        
        x = max(0, x)
        y = max(0, y)
        w = min(img.shape[1] - x, w)
        h = min(img.shape[0] - y, h)
        
        face_img = img[y:y+h, x:x+w]
        
        if target_size:
            face_img = cv2.resize(face_img, target_size, interpolation=cv2.INTER_AREA)
        
        return face_img
    
    def get_detector_info(self) -> dict:
        """Get detector information"""
        return {
            "name": "SSD ResNet",
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "input_size": self.input_size,
            "model_loaded": self.model_loaded,
            "backend": "SSD ResNet" if isinstance(self.net, cv2.dnn.Net) else "Haar Cascade"
        } 