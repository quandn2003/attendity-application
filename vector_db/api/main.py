"""
Vector Database FastAPI Application
Provides endpoints for student management and attendance verification with voting system.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import logging
import time
import numpy as np
from datetime import datetime

from vector_db.database.chroma_client import ChromaClient
from vector_db.database.student_manager import StudentManager
from vector_db.database.attendance_manager import AttendanceManager
from vector_db.voting.similarity_voting import SimilarityVoting
from vector_db.config.database_config import DatabaseConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vector Database API",
    description="Mobile-optimized vector database for attendance system with voting",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = DatabaseConfig()
chroma_client = ChromaClient(config)
student_manager = StudentManager(chroma_client, config)
attendance_manager = AttendanceManager(chroma_client, student_manager, config)
voting_system = SimilarityVoting(
    similarity_threshold=config.similarity_threshold,
    voting_threshold=config.voting_threshold
)

class CreateClassRequest(BaseModel):
    class_code: str = Field(..., min_length=1, max_length=50, description="Unique class identifier")

class DeleteClassRequest(BaseModel):
    class_code: str = Field(..., min_length=1, max_length=50, description="Class identifier to delete")

class StudentData(BaseModel):
    student_id: str = Field(..., min_length=1, max_length=50, description="Unique student identifier")
    class_code: str = Field(..., min_length=1, max_length=50, description="Class identifier")
    embedding: List[float] = Field(..., min_items=512, max_items=512, description="512-dimensional face embedding")
    
    @validator('embedding')
    def validate_embedding_dimension(cls, v):
        if len(v) != 512:
            raise ValueError('Embedding must be exactly 512 dimensions')
        return v

class InsertStudentRequest(BaseModel):
    students: List[StudentData] = Field(..., min_items=1, description="List of students to insert")

class DeleteStudentData(BaseModel):
    student_id: str = Field(..., min_length=1, max_length=50, description="Student identifier to delete")
    class_code: str = Field(..., min_length=1, max_length=50, description="Class identifier")

class DeleteStudentRequest(BaseModel):
    students: List[DeleteStudentData] = Field(..., min_items=1, description="List of students to delete")

class SearchWithVotingRequest(BaseModel):
    embedding: List[float] = Field(..., min_items=512, max_items=512, description="512-dimensional face embedding")
    class_code: str = Field(..., min_length=1, max_length=50, description="Class identifier to search in")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold for matching")
    
    @validator('embedding')
    def validate_embedding_dimension(cls, v):
        if len(v) != 512:
            raise ValueError('Embedding must be exactly 512 dimensions')
        return v

class MatchResult(BaseModel):
    student_id: str
    similarity: float
    confidence: float

class SearchResponse(BaseModel):
    status: str
    student_id: Optional[str] = None
    confidence: Optional[float] = None
    top_matches: List[MatchResult]
    reason: Optional[str] = None
    voting_details: Optional[Dict[str, Any]] = None

class StandardResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None

class BatchResponse(BaseModel):
    status: str
    processed_count: int
    details: Optional[Dict[str, Any]] = None

@app.on_event("startup")
async def startup_event():
    """Initialize database connections and perform health checks."""
    try:
        logger.info("Starting Vector Database API...")
        chroma_client.initialize()
        logger.info("Vector Database API started successfully")
    except Exception as e:
        logger.error(f"Failed to start Vector Database API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up database connections."""
    try:
        logger.info("Shutting down Vector Database API...")
        chroma_client.cleanup()
        logger.info("Vector Database API shut down successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Check system health and database connectivity."""
    try:
        start_time = time.time()
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "response_time_ms": 0,
            "collections": {},
            "memory_usage": chroma_client.get_memory_usage()
        }
        
        collections = chroma_client.list_collections()
        for collection in collections:
            try:
                count = chroma_client.get_collection_count(collection['class_code'])
                health_status["collections"][collection['class_code']] = {
                    "student_count": count,
                    "status": "active"
                }
            except Exception as e:
                health_status["collections"][collection['class_code']] = {
                    "status": "error",
                    "error": str(e)
                }
        
        health_status["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )

@app.post("/create_class", response_model=StandardResponse)
async def create_class(request: CreateClassRequest):
    """Create a new class collection for storing student embeddings."""
    try:
        logger.info(f"Creating class: {request.class_code}")
        
        if chroma_client.collection_exists(request.class_code):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Class '{request.class_code}' already exists"
            )
        
        chroma_client.create_collection(request.class_code)
        
        return StandardResponse(
            status="success",
            message=f"Class '{request.class_code}' created successfully",
            details={"class_code": request.class_code, "created_at": datetime.now().isoformat()}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create class {request.class_code}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create class: {str(e)}"
        )

@app.delete("/delete_class", response_model=StandardResponse)
async def delete_class(request: DeleteClassRequest):
    """Delete a class collection and all associated student data."""
    try:
        logger.info(f"Deleting class: {request.class_code}")
        
        if not chroma_client.collection_exists(request.class_code):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Class '{request.class_code}' not found"
            )
        
        student_count = chroma_client.get_collection_count(request.class_code)
        chroma_client.delete_collection(request.class_code)
        
        return StandardResponse(
            status="success",
            message=f"Class '{request.class_code}' deleted successfully",
            details={
                "class_code": request.class_code,
                "deleted_students": student_count,
                "deleted_at": datetime.now().isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete class {request.class_code}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete class: {str(e)}"
        )

@app.post("/insert_student", response_model=BatchResponse)
async def insert_student(request: InsertStudentRequest):
    """Insert students with consensus embeddings into their respective classes."""
    try:
        logger.info(f"Inserting {len(request.students)} students")
        
        inserted_count = 0
        failed_insertions = []
        
        for student_data in request.students:
            try:
                if not chroma_client.collection_exists(student_data.class_code):
                    chroma_client.create_collection(student_data.class_code)
                
                if student_manager.student_exists(student_data.student_id, student_data.class_code):
                    logger.warning(f"Student {student_data.student_id} already exists in class {student_data.class_code}")
                    failed_insertions.append({
                        "student_id": student_data.student_id,
                        "class_code": student_data.class_code,
                        "reason": "Student already exists"
                    })
                    continue
                
                success = student_manager.insert_consensus_embedding(
                    student_id=student_data.student_id,
                    class_code=student_data.class_code,
                    consensus_embedding=student_data.embedding,
                    metadata={
                        "insertion_method": "consensus",
                        "created_at": datetime.now().isoformat()
                    }
                )
                
                if success:
                    inserted_count += 1
                    logger.info(f"Successfully inserted student {student_data.student_id} into class {student_data.class_code}")
                else:
                    failed_insertions.append({
                        "student_id": student_data.student_id,
                        "class_code": student_data.class_code,
                        "reason": "Failed to insert into database"
                    })
                
            except Exception as e:
                logger.error(f"Failed to insert student {student_data.student_id}: {e}")
                failed_insertions.append({
                    "student_id": student_data.student_id,
                    "class_code": student_data.class_code,
                    "reason": str(e)
                })
        
        response_details = {
            "total_requested": len(request.students),
            "successfully_inserted": inserted_count,
            "failed_insertions": failed_insertions
        }
        
        if failed_insertions:
            logger.warning(f"Some insertions failed: {len(failed_insertions)} out of {len(request.students)}")
        
        return BatchResponse(
            status="success" if inserted_count > 0 else "failed",
            processed_count=inserted_count,
            details=response_details
        )
        
    except Exception as e:
        logger.error(f"Failed to insert students: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to insert students: {str(e)}"
        )

@app.delete("/delete_student", response_model=BatchResponse)
async def delete_student(request: DeleteStudentRequest):
    """Delete students from their respective classes."""
    try:
        logger.info(f"Deleting {len(request.students)} students")
        
        deleted_count = 0
        failed_deletions = []
        
        for student_data in request.students:
            try:
                if not chroma_client.collection_exists(student_data.class_code):
                    failed_deletions.append({
                        "student_id": student_data.student_id,
                        "class_code": student_data.class_code,
                        "reason": "Class not found"
                    })
                    continue
                
                if not student_manager.student_exists(student_data.student_id, student_data.class_code):
                    failed_deletions.append({
                        "student_id": student_data.student_id,
                        "class_code": student_data.class_code,
                        "reason": "Student not found"
                    })
                    continue
                
                success = student_manager.delete_student(student_data.student_id, student_data.class_code)
                if success:
                    deleted_count += 1
                    logger.info(f"Successfully deleted student {student_data.student_id} from class {student_data.class_code}")
                else:
                    failed_deletions.append({
                        "student_id": student_data.student_id,
                        "class_code": student_data.class_code,
                        "reason": "Failed to delete from database"
                    })
                
            except Exception as e:
                logger.error(f"Failed to delete student {student_data.student_id}: {e}")
                failed_deletions.append({
                    "student_id": student_data.student_id,
                    "class_code": student_data.class_code,
                    "reason": str(e)
                })
        
        response_details = {
            "total_requested": len(request.students),
            "successfully_deleted": deleted_count,
            "failed_deletions": failed_deletions
        }
        
        return BatchResponse(
            status="success" if deleted_count > 0 else "failed",
            processed_count=deleted_count,
            details=response_details
        )
        
    except Exception as e:
        logger.error(f"Failed to delete students: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete students: {str(e)}"
        )

@app.post("/search_with_voting", response_model=SearchResponse)
async def search_with_voting(request: SearchWithVotingRequest):
    """Search for student matches using top-3 voting system for attendance verification."""
    try:
        logger.info(f"Searching with voting in class: {request.class_code}")
        
        if not chroma_client.collection_exists(request.class_code):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Class '{request.class_code}' not found"
            )
        
        search_results = chroma_client.similarity_search(
            collection_name=request.class_code,
            query_embedding=request.embedding,
            n_results=3
        )
        
        if not search_results or not search_results.get('ids') or len(search_results['ids'][0]) == 0:
            return SearchResponse(
                status="no_students_found",
                top_matches=[],
                reason="No students found in the class",
                voting_details={"class_size": 0}
            )
        
        top_matches = []
        for i, (full_student_id, distance) in enumerate(zip(search_results['ids'][0], search_results['distances'][0])):
            # Extract actual student_id from ChromaDB format: {class_code}_{student_id}
            student_id = full_student_id.replace(f"{request.class_code}_", "")
            similarity = 1.0 - distance
            confidence = max(0.0, min(1.0, similarity))
            
            top_matches.append(MatchResult(
                student_id=student_id,
                similarity=similarity,
                confidence=confidence
            ))
        
        # Convert search results to the format expected by voting system
        voting_search_results = []
        for i, (full_student_id, distance) in enumerate(zip(search_results['ids'][0], search_results['distances'][0])):
            # Extract actual student_id from ChromaDB format: {class_code}_{student_id}
            student_id = full_student_id.replace(f"{request.class_code}_", "")
            similarity = 1.0 - distance
            voting_search_results.append({
                "student_id": student_id,
                "embedding": np.zeros(512),  # Placeholder, not used in voting
                "similarity": similarity,
                "distance": distance,
                "metadata": search_results['metadatas'][0][i] if search_results.get('metadatas') else {}
            })
        
        voting_result = voting_system.vote_for_attendance(
            query_embedding=np.array(request.embedding),
            search_results=voting_search_results
        )
        
        if voting_result.decision in ["clear_match", "consensus_match"]:
            attendance_manager.record_attendance(
                student_id=voting_result.student_id,
                class_code=request.class_code,
                confidence=voting_result.confidence,
                voting_details=voting_result.voting_details
            )
            
            return SearchResponse(
                status="match_found",
                student_id=voting_result.student_id,
                confidence=voting_result.confidence,
                top_matches=top_matches,
                voting_details=voting_result.voting_details
            )
        else:
            return SearchResponse(
                status="no_clear_match",
                top_matches=top_matches,
                reason=voting_result.reason,
                voting_details=voting_result.voting_details
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search with voting: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@app.get("/class_stats/{class_code}", response_model=Dict[str, Any])
async def get_class_statistics(class_code: str):
    """Get detailed statistics for a specific class."""
    try:
        if not chroma_client.collection_exists(class_code):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Class '{class_code}' not found"
            )
        
        student_count = chroma_client.get_collection_count(class_code)
        
        # Get all students using StudentManager
        try:
            all_students = student_manager.list_students_in_class(class_code)
            students_data = []
            
            for student in all_students:
                # Get real attendance records for this student
                attendance_records = attendance_manager.get_student_attendance_history(
                    student.student_id, class_code, limit=30
                )
                
                # Calculate real attendance statistics
                total_sessions = len(attendance_records) if attendance_records else 0
                present_sessions = len([r for r in attendance_records if r.get('status') == 'present'])
                attendance_rate = (present_sessions / total_sessions * 100) if total_sessions > 0 else 0
                
                # Check if student is present today
                today = datetime.now().date()
                is_present_today = any(
                    r.get('status') == 'present' and 
                    datetime.fromisoformat(r.get('timestamp', '')).date() == today
                    for r in attendance_records
                )
                
                students_data.append({
                    "student_id": student.student_id,
                    "name": student.name or student.student_id,
                    "email": student.email or '',
                    "is_present": is_present_today,
                    "attendance_rate": round(attendance_rate, 2),
                    "total_sessions": total_sessions,
                    "present_sessions": present_sessions,
                    "created_at": student.created_at or ''
                })
            
        except Exception as e:
            logger.warning(f"Could not fetch detailed student data for {class_code}: {e}")
            # Fallback to basic data
            students_data = []
        
        # Get recent attendance records for the class
        recent_attendance = attendance_manager.get_recent_attendance(class_code, limit=10)
        
        # Calculate overall class attendance statistics
        if students_data:
            total_possible_sessions = len(students_data) * 20 if students_data else 0  # Assuming 20 sessions per semester
            total_attended = sum(student['present_sessions'] for student in students_data)
            overall_attendance_rate = (total_attended / total_possible_sessions * 100) if total_possible_sessions > 0 else 0
            
            # Count students present today
            present_today = len([s for s in students_data if s['is_present']])
        else:
            # Fallback calculation using recent attendance
            total_records = len(recent_attendance)
            present_records = len([r for r in recent_attendance if r.get('status') == 'present'])
            overall_attendance_rate = (present_records / total_records * 100) if total_records > 0 else 0
            
            # Calculate present today from recent attendance
            today = datetime.now().date()
            present_today = len(set([
                r['student_id'] for r in recent_attendance 
                if r.get('status') == 'present' and 
                datetime.fromisoformat(r.get('timestamp', '')).date() == today
            ]))
        
        return {
            "status": "success",
            "class_code": class_code,
            "student_count": student_count,
            "present_today": present_today,
            "attendance_rate": round(overall_attendance_rate, 2),
            "students": students_data,
            "recent_attendance": recent_attendance,
            "last_updated": datetime.now().isoformat(),
            "class_name": f"{class_code} Course"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get class stats for {class_code}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve class statistics: {str(e)}"
        )

@app.get("/student_attendance/{class_code}/{student_id}", response_model=Dict[str, Any])
async def get_student_attendance(class_code: str, student_id: str):
    """Get attendance history for a specific student."""
    try:
        if not chroma_client.collection_exists(class_code):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Class '{class_code}' not found"
            )
        
        # Check if student exists
        student = student_manager.get_student(student_id, class_code)
        if not student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Student '{student_id}' not found in class '{class_code}'"
            )
        
        # Get real attendance records
        attendance_records = attendance_manager.get_student_attendance_history(
            student_id, class_code, limit=50
        )
        
        # Calculate real statistics
        total_sessions = len(attendance_records) if attendance_records else 0
        attended_sessions = len([r for r in attendance_records if r.get('status') == 'present'])
        attendance_rate = (attended_sessions / total_sessions * 100) if total_sessions > 0 else 0
        
        # Get student metadata
        student_metadata = student.get('metadata', {})
        
        return {
            "status": "success",
            "student_id": student_id,
            "class_code": class_code,
            "total_sessions": total_sessions,
            "attended_sessions": attended_sessions,
            "attendance_rate": round(attendance_rate, 2),
            "attendance_records": attendance_records,
            "student_info": {
                "name": student_metadata.get('name', student_id),
                "email": student_metadata.get('email', ''),
                "created_at": student_metadata.get('created_at', '')
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get attendance for {student_id} in {class_code}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve student attendance: {str(e)}"
        )

@app.get("/classes", response_model=Dict[str, Any])
async def get_all_classes():
    """Get all classes with real statistics."""
    try:
        # Get collections using the correct method
        collections_info = chroma_client.list_collections()
        classes_data = []
        
        for collection_info in collections_info:
            try:
                class_code = collection_info['class_code']
                student_count = chroma_client.get_collection_count(class_code)
                
                # Get real attendance data for this class
                recent_attendance = attendance_manager.get_recent_attendance(class_code, limit=100)
                
                # Calculate present today
                today = datetime.now().date()
                present_today = len(set([
                    r['student_id'] for r in recent_attendance 
                    if r.get('status') == 'present' and 
                    datetime.fromisoformat(r.get('timestamp', '')).date() == today
                ]))
                
                # Calculate attendance rate
                total_records = len(recent_attendance)
                present_records = len([r for r in recent_attendance if r.get('status') == 'present'])
                attendance_rate = (present_records / total_records * 100) if total_records > 0 else 0
                
                # Get last session date
                last_session = None
                if recent_attendance:
                    last_session = max([
                        datetime.fromisoformat(r.get('timestamp', ''))
                        for r in recent_attendance
                    ]).date().isoformat()
                
                classes_data.append({
                    "class_code": class_code,
                    "class_name": f"{class_code} Course",
                    "student_count": student_count,
                    "present_today": present_today,
                    "attendance_rate": round(attendance_rate, 2),
                    "last_session": last_session or collection_info.get('created_at', 'No sessions yet'),
                    "created_at": collection_info.get('created_at', 'Unknown'),
                    "status": "active"
                })
            except Exception as e:
                logger.error(f"Error processing class {collection_info.get('class_code', 'unknown')}: {e}")
                # Add basic info even if detailed stats fail
                classes_data.append({
                    "class_code": collection_info.get('class_code', 'unknown'),
                    "class_name": f"{collection_info.get('class_code', 'unknown')} Course",
                    "student_count": 0,
                    "present_today": 0,
                    "attendance_rate": 0,
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "status": "success",
            "classes": classes_data,
            "total_classes": len(classes_data)
        }
        
    except Exception as e:
        logger.error(f"Failed to get classes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve classes: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    ) 