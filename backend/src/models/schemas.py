from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime
import uuid

class StudentBase(BaseModel):
    student_id: str
    name: str
    email: EmailStr

class StudentCreate(StudentBase):
    profile_image_url: Optional[str] = None
    student_card_image_url: Optional[str] = None

class StudentResponse(StudentBase):
    id: str  # Changed from uuid.UUID to str since we're using String in SQLAlchemy
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class AssignFaceRequest(BaseModel):
    student_id: str
    face_image: str  # base64 encoded

class AssignFaceResponse(BaseModel):
    success: bool
    message: Optional[str] = None

class RecognitionRequest(BaseModel):
    image: str  # base64 encoded

class RecognitionResponse(BaseModel):
    success: bool
    student: Optional[StudentResponse] = None
    confidence: Optional[float] = None
    reason: Optional[str] = None 