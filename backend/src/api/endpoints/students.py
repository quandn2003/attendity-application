from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from ...core.database import get_db
from ...models.student import Student
from ...models.schemas import StudentCreate, StudentResponse

router = APIRouter()

@router.post("/", response_model=StudentResponse)
async def create_student(
    student: StudentCreate,
    db: Session = Depends(get_db)
):
    """Create a new student"""
    # Check if student ID already exists
    existing = db.query(Student).filter(Student.student_id == student.student_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Student ID already exists")
    
    # Check if email already exists
    existing_email = db.query(Student).filter(Student.email == student.email).first()
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already exists")
    
    db_student = Student(**student.model_dump())
    db.add(db_student)
    db.commit()
    db.refresh(db_student)
    
    return StudentResponse.model_validate(db_student)

@router.get("/", response_model=List[StudentResponse])
async def get_students(db: Session = Depends(get_db)):
    """Get all students"""
    students = db.query(Student).all()
    return [StudentResponse.model_validate(student) for student in students]

@router.get("/{student_id}", response_model=StudentResponse)
async def get_student(student_id: str, db: Session = Depends(get_db)):
    """Get student by ID"""
    student = db.query(Student).filter(Student.student_id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    return StudentResponse.model_validate(student) 