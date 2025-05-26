from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ...core.database import get_db
from ...models.schemas import AssignFaceRequest, AssignFaceResponse, RecognitionRequest, RecognitionResponse
from ...services.recognition_service import recognition_service

router = APIRouter()

@router.post("/assign", response_model=AssignFaceResponse)
async def assign_face(
    request: AssignFaceRequest,
    db: Session = Depends(get_db)
):
    """Assign face to student"""
    result = await recognition_service.assign_face(
        db=db,
        student_id=request.student_id,
        face_image=request.face_image
    )
    return AssignFaceResponse(**result)

@router.post("/recognize", response_model=RecognitionResponse)
async def recognize_face(
    request: RecognitionRequest,
    db: Session = Depends(get_db)
):
    """Recognize face in image"""
    result = await recognition_service.recognize_face(
        db=db,
        image=request.image
    )
    return result 