from sqlalchemy.orm import Session
from typing import Optional
from ..models.student import Student
from ..models.schemas import RecognitionResponse, StudentResponse
from ..ai.face_recognition import get_face_recognition_service
from .vector_service import vector_service
from ..core.config import settings
import logging

logger = logging.getLogger(__name__)

class RecognitionService:
    def __init__(self):
        self.similarity_threshold = settings.FACE_SIMILARITY_THRESHOLD
    
    async def assign_face(self, db: Session, student_id: str, face_image: str) -> dict:
        """Assign face embedding to student"""
        try:
            # Check if student exists
            student = db.query(Student).filter(Student.student_id == student_id).first()
            if not student:
                return {"success": False, "message": "Student not found"}
            
            # Generate embedding
            face_service = get_face_recognition_service()
            embedding = face_service.generate_embedding(face_image)
            if not embedding:
                return {"success": False, "message": "No face detected in image"}
            
            # Store in vector database
            if not vector_service.insert_embedding(student_id, embedding):
                return {"success": False, "message": "Failed to store face embedding"}
            
            # Update student record
            student.face_embedding = embedding
            db.commit()
            
            logger.info(f"Face assigned to student: {student_id}")
            return {"success": True, "message": "Face assigned successfully"}
            
        except Exception as e:
            logger.error(f"Error assigning face: {str(e)}")
            db.rollback()
            return {"success": False, "message": "Internal server error"}
    
    async def recognize_face(self, db: Session, image: str) -> RecognitionResponse:
        """Recognize face in image"""
        try:
            # Generate embedding
            face_service = get_face_recognition_service()
            embedding = face_service.generate_embedding(image)
            if not embedding:
                return RecognitionResponse(
                    success=False,
                    reason="No face detected in image"
                )
            
            # Search for similar faces
            matches = vector_service.search_similar(embedding, top_k=1)
            
            if not matches:
                return RecognitionResponse(
                    success=False,
                    reason="No matching face found in database"
                )
            
            best_match = matches[0]
            
            if best_match["similarity"] < self.similarity_threshold:
                return RecognitionResponse(
                    success=False,
                    reason=f"Face similarity too low ({best_match['similarity']:.1%})"
                )
            
            # Get student details
            student = db.query(Student).filter(
                Student.student_id == best_match["student_id"]
            ).first()
            
            if not student:
                return RecognitionResponse(
                    success=False,
                    reason="Student record not found"
                )
            
            if not student.is_active:
                return RecognitionResponse(
                    success=False,
                    reason="Student account is inactive"
                )
            
            return RecognitionResponse(
                success=True,
                student=StudentResponse.model_validate(student),
                confidence=best_match["similarity"]
            )
            
        except Exception as e:
            logger.error(f"Error recognizing face: {str(e)}")
            return RecognitionResponse(
                success=False,
                reason="Recognition service error"
            )

recognition_service = RecognitionService() 