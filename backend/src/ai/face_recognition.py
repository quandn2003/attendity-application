import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import base64
import io
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class FaceRecognitionService:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        
        try:
            # Initialize MTCNN for face detection
            self.mtcnn = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=True,
                device=self.device
            )
            
            # Initialize FaceNet for face recognition
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            logger.info(f"Face recognition service initialized on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize face recognition service: {e}")
            raise
    
    def base64_to_image(self, base64_string: str) -> Image.Image:
        """Convert base64 string to PIL Image"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
        except Exception as e:
            logger.error(f"Error converting base64 to image: {str(e)}")
            raise ValueError(f"Invalid base64 image: {str(e)}")
    
    def generate_embedding(self, image_base64: str) -> Optional[List[float]]:
        """Generate face embedding from base64 image"""
        try:
            # Convert base64 to PIL image
            pil_image = self.base64_to_image(image_base64)
            
            # Extract face using MTCNN
            face_tensor = self.mtcnn(pil_image)
            
            if face_tensor is None:
                logger.warning("No face detected in image")
                return None
            
            # Generate embedding
            face_tensor = face_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.resnet(face_tensor)
                embedding = embedding.cpu().numpy().flatten()
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
    
    def compare_embeddings(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Error comparing embeddings: {str(e)}")
            return 0.0

# Global instance - will be initialized when needed
face_recognition_service = None

def get_face_recognition_service():
    global face_recognition_service
    if face_recognition_service is None:
        from ..core.config import settings
        face_recognition_service = FaceRecognitionService(device=settings.MODEL_DEVICE)
    return face_recognition_service 