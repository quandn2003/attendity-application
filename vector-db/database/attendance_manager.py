import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime, date
from dataclasses import dataclass

from .chroma_client import ChromaClient
from .student_manager import StudentManager
from ..config.database_config import DatabaseConfig

logger = logging.getLogger(__name__)

@dataclass
class AttendanceRecord:
    """Attendance record data model"""
    student_id: str
    class_code: str
    timestamp: str
    confidence: float
    similarity_score: float
    verification_method: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AttendanceResult:
    """Result of attendance verification"""
    success: bool
    student_id: Optional[str]
    class_code: str
    confidence: float
    similarity_score: float
    decision: str
    reason: str
    top_matches: List[Dict[str, Any]]
    timestamp: str
    processing_time: float

class AttendanceManager:
    """
    Manager for attendance verification with voting system
    """
    
    def __init__(self, 
                 chroma_client: ChromaClient,
                 student_manager: StudentManager,
                 config: DatabaseConfig):
        self.chroma_client = chroma_client
        self.student_manager = student_manager
        self.config = config
        self.attendance_records = []  # In-memory storage for demo
    
    def verify_attendance(self, 
                         class_code: str,
                         query_embedding: np.ndarray,
                         processing_time: float = 0.0) -> AttendanceResult:
        """
        Verify student attendance using embedding similarity and voting
        
        Args:
            class_code: Class identifier
            query_embedding: Face embedding from attendance photo
            processing_time: Time taken for face processing
            
        Returns:
            Attendance verification result
        """
        start_time = datetime.now()
        
        try:
            # Search for similar embeddings
            similar_results = self.chroma_client.search_similar(
                class_code=class_code,
                query_embedding=query_embedding,
                top_k=self.config.top_k_results
            )
            
            if not similar_results:
                return AttendanceResult(
                    success=False,
                    student_id=None,
                    class_code=class_code,
                    confidence=0.0,
                    similarity_score=0.0,
                    decision="no_students_found",
                    reason="No students registered in this class",
                    top_matches=[],
                    timestamp=start_time.isoformat(),
                    processing_time=processing_time
                )
            
            # Apply voting logic for attendance verification
            voting_result = self._apply_attendance_voting(similar_results)
            
            # Create attendance record if successful
            if voting_result["decision"] == "clear_match":
                student_id = voting_result["student_id"]
                attendance_record = AttendanceRecord(
                    student_id=student_id,
                    class_code=class_code,
                    timestamp=start_time.isoformat(),
                    confidence=voting_result["confidence"],
                    similarity_score=voting_result["similarity_score"],
                    verification_method="embedding_similarity_voting",
                    metadata={
                        "top_matches": voting_result["top_matches"],
                        "processing_time": processing_time
                    }
                )
                self.attendance_records.append(attendance_record)
                
                logger.info(f"Attendance verified for student {student_id} in class {class_code}")
            
            return AttendanceResult(
                success=voting_result["decision"] == "clear_match",
                student_id=voting_result.get("student_id"),
                class_code=class_code,
                confidence=voting_result["confidence"],
                similarity_score=voting_result.get("similarity_score", 0.0),
                decision=voting_result["decision"],
                reason=voting_result["reason"],
                top_matches=voting_result["top_matches"],
                timestamp=start_time.isoformat(),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in attendance verification: {e}")
            return AttendanceResult(
                success=False,
                student_id=None,
                class_code=class_code,
                confidence=0.0,
                similarity_score=0.0,
                decision="error",
                reason=f"Verification error: {str(e)}",
                top_matches=[],
                timestamp=start_time.isoformat(),
                processing_time=processing_time
            )
    
    def _apply_attendance_voting(self, similar_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply voting logic for attendance verification
        
        Args:
            similar_results: List of similar embeddings from ChromaDB
            
        Returns:
            Voting decision result
        """
        try:
            if not similar_results:
                return {
                    "decision": "no_match",
                    "reason": "no_candidates_found",
                    "confidence": 0.0,
                    "top_matches": []
                }
            
            # Get the best match
            best_match = similar_results[0]
            best_similarity = best_match["similarity"]
            
            # Format top matches for response
            top_matches = []
            for result in similar_results:
                top_matches.append({
                    "student_id": result["student_id"],
                    "similarity": result["similarity"],
                    "distance": result["distance"]
                })
            
            # Check if the best match meets the voting threshold
            if best_similarity >= self.config.voting_threshold:
                # Check if there's a clear winner (significant margin)
                if len(similar_results) > 1:
                    second_best_similarity = similar_results[1]["similarity"]
                    margin = best_similarity - second_best_similarity
                    
                    if margin >= 0.1:  # Clear margin threshold
                        return {
                            "decision": "clear_match",
                            "student_id": best_match["student_id"],
                            "confidence": best_similarity,
                            "similarity_score": best_similarity,
                            "margin": margin,
                            "top_matches": top_matches,
                            "reason": "clear_winner_with_sufficient_margin"
                        }
                    else:
                        return {
                            "decision": "ambiguous_match",
                            "confidence": best_similarity,
                            "similarity_score": best_similarity,
                            "margin": margin,
                            "top_matches": top_matches,
                            "reason": f"insufficient_margin_{margin:.3f}_between_top_matches"
                        }
                else:
                    # Only one candidate, but strong match
                    return {
                        "decision": "clear_match",
                        "student_id": best_match["student_id"],
                        "confidence": best_similarity,
                        "similarity_score": best_similarity,
                        "top_matches": top_matches,
                        "reason": "single_strong_match"
                    }
            else:
                return {
                    "decision": "no_clear_match",
                    "confidence": best_similarity,
                    "similarity_score": best_similarity,
                    "top_matches": top_matches,
                    "reason": f"best_similarity_{best_similarity:.3f}_below_threshold_{self.config.voting_threshold}"
                }
                
        except Exception as e:
            logger.error(f"Error in attendance voting: {e}")
            return {
                "decision": "error",
                "reason": f"voting_error: {str(e)}",
                "confidence": 0.0,
                "top_matches": []
            }
    
    def get_attendance_records(self, 
                              class_code: str,
                              date_filter: Optional[str] = None) -> List[AttendanceRecord]:
        """
        Get attendance records for a class
        
        Args:
            class_code: Class identifier
            date_filter: Optional date filter (YYYY-MM-DD format)
            
        Returns:
            List of attendance records
        """
        try:
            filtered_records = []
            
            for record in self.attendance_records:
                if record.class_code == class_code:
                    if date_filter:
                        record_date = record.timestamp.split('T')[0]  # Extract date part
                        if record_date == date_filter:
                            filtered_records.append(record)
                    else:
                        filtered_records.append(record)
            
            # Sort by timestamp (most recent first)
            filtered_records.sort(key=lambda x: x.timestamp, reverse=True)
            
            return filtered_records
            
        except Exception as e:
            logger.error(f"Error getting attendance records: {e}")
            return []
    
    def get_attendance_statistics(self, 
                                 class_code: str,
                                 date_range: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
        """
        Get attendance statistics for a class
        
        Args:
            class_code: Class identifier
            date_range: Optional tuple of (start_date, end_date) in YYYY-MM-DD format
            
        Returns:
            Attendance statistics
        """
        try:
            # Get all students in the class
            students = self.student_manager.list_students_in_class(class_code)
            total_students = len(students)
            
            # Filter attendance records
            records = self.get_attendance_records(class_code)
            
            if date_range:
                start_date, end_date = date_range
                filtered_records = []
                for record in records:
                    record_date = record.timestamp.split('T')[0]
                    if start_date <= record_date <= end_date:
                        filtered_records.append(record)
                records = filtered_records
            
            # Calculate statistics
            total_attendances = len(records)
            unique_students_attended = len(set(record.student_id for record in records))
            
            # Attendance by date
            attendance_by_date = {}
            for record in records:
                record_date = record.timestamp.split('T')[0]
                if record_date not in attendance_by_date:
                    attendance_by_date[record_date] = []
                attendance_by_date[record_date].append(record.student_id)
            
            # Calculate daily attendance rates
            daily_rates = {}
            for date_str, student_ids in attendance_by_date.items():
                unique_daily_attendance = len(set(student_ids))
                daily_rates[date_str] = {
                    "attended": unique_daily_attendance,
                    "total_students": total_students,
                    "rate": unique_daily_attendance / total_students if total_students > 0 else 0.0
                }
            
            # Student attendance frequency
            student_frequency = {}
            for record in records:
                if record.student_id not in student_frequency:
                    student_frequency[record.student_id] = 0
                student_frequency[record.student_id] += 1
            
            # Average confidence and similarity scores
            avg_confidence = np.mean([record.confidence for record in records]) if records else 0.0
            avg_similarity = np.mean([record.similarity_score for record in records]) if records else 0.0
            
            return {
                "class_code": class_code,
                "total_students": total_students,
                "total_attendances": total_attendances,
                "unique_students_attended": unique_students_attended,
                "overall_attendance_rate": unique_students_attended / total_students if total_students > 0 else 0.0,
                "daily_attendance": daily_rates,
                "student_frequency": student_frequency,
                "average_confidence": float(avg_confidence),
                "average_similarity": float(avg_similarity),
                "date_range": date_range,
                "analysis_period": {
                    "start": min(record.timestamp for record in records) if records else None,
                    "end": max(record.timestamp for record in records) if records else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating attendance statistics: {e}")
            return {"error": str(e)}
    
    def get_student_attendance_history(self, 
                                     student_id: str,
                                     class_code: str) -> Dict[str, Any]:
        """
        Get attendance history for a specific student
        
        Args:
            student_id: Student identifier
            class_code: Class identifier
            
        Returns:
            Student attendance history
        """
        try:
            # Get student's attendance records
            student_records = [
                record for record in self.attendance_records
                if record.student_id == student_id and record.class_code == class_code
            ]
            
            # Sort by timestamp
            student_records.sort(key=lambda x: x.timestamp)
            
            # Calculate statistics
            total_attendances = len(student_records)
            
            # Attendance by date
            attendance_dates = [record.timestamp.split('T')[0] for record in student_records]
            unique_dates = list(set(attendance_dates))
            unique_dates.sort()
            
            # Confidence and similarity trends
            confidence_scores = [record.confidence for record in student_records]
            similarity_scores = [record.similarity_score for record in student_records]
            
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
            
            # Recent attendance (last 7 days)
            recent_records = student_records[-7:] if len(student_records) >= 7 else student_records
            
            return {
                "student_id": student_id,
                "class_code": class_code,
                "total_attendances": total_attendances,
                "unique_attendance_dates": len(unique_dates),
                "attendance_dates": unique_dates,
                "average_confidence": float(avg_confidence),
                "average_similarity": float(avg_similarity),
                "confidence_trend": confidence_scores,
                "similarity_trend": similarity_scores,
                "recent_attendances": [
                    {
                        "timestamp": record.timestamp,
                        "confidence": record.confidence,
                        "similarity": record.similarity_score
                    }
                    for record in recent_records
                ],
                "first_attendance": student_records[0].timestamp if student_records else None,
                "last_attendance": student_records[-1].timestamp if student_records else None
            }
            
        except Exception as e:
            logger.error(f"Error getting student attendance history: {e}")
            return {"error": str(e)}
    
    def export_attendance_report(self, 
                                class_code: str,
                                date_range: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
        """
        Export comprehensive attendance report
        
        Args:
            class_code: Class identifier
            date_range: Optional date range filter
            
        Returns:
            Comprehensive attendance report
        """
        try:
            # Get basic statistics
            stats = self.get_attendance_statistics(class_code, date_range)
            
            # Get all students in class
            students = self.student_manager.list_students_in_class(class_code)
            
            # Get detailed student reports
            student_reports = []
            for student in students:
                student_history = self.get_student_attendance_history(
                    student.student_id, class_code
                )
                student_reports.append({
                    "student_id": student.student_id,
                    "name": student.name,
                    "email": student.email,
                    "attendance_summary": student_history
                })
            
            # Get all attendance records for the period
            records = self.get_attendance_records(class_code)
            if date_range:
                start_date, end_date = date_range
                records = [
                    record for record in records
                    if start_date <= record.timestamp.split('T')[0] <= end_date
                ]
            
            return {
                "report_metadata": {
                    "class_code": class_code,
                    "generated_at": datetime.now().isoformat(),
                    "date_range": date_range,
                    "total_records": len(records)
                },
                "class_statistics": stats,
                "student_reports": student_reports,
                "raw_attendance_records": [
                    {
                        "student_id": record.student_id,
                        "timestamp": record.timestamp,
                        "confidence": record.confidence,
                        "similarity_score": record.similarity_score,
                        "verification_method": record.verification_method
                    }
                    for record in records
                ]
            }
            
        except Exception as e:
            logger.error(f"Error exporting attendance report: {e}")
            return {"error": str(e)}
    
    def record_attendance(self, 
                         student_id: str,
                         class_code: str,
                         confidence: float,
                         voting_details: Dict[str, Any]) -> bool:
        """
        Record attendance for a student
        
        Args:
            student_id: Student identifier
            class_code: Class identifier
            confidence: Confidence score
            voting_details: Details from voting process
            
        Returns:
            Success status
        """
        try:
            attendance_record = AttendanceRecord(
                student_id=student_id,
                class_code=class_code,
                timestamp=datetime.now().isoformat(),
                confidence=confidence,
                similarity_score=voting_details.get("similarity_score", confidence),
                verification_method="voting_system",
                metadata=voting_details
            )
            
            self.attendance_records.append(attendance_record)
            logger.info(f"Recorded attendance for student {student_id} in class {class_code}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording attendance: {e}")
            return False
    
    def get_class_attendance_stats(self, class_code: str) -> Dict[str, Any]:
        """
        Get attendance statistics for a class
        
        Args:
            class_code: Class identifier
            
        Returns:
            Attendance statistics
        """
        try:
            records = self.get_attendance_records(class_code)
            
            if not records:
                return {
                    "total_attendances": 0,
                    "unique_students": 0,
                    "average_confidence": 0.0,
                    "last_attendance": None
                }
            
            unique_students = len(set(record.student_id for record in records))
            avg_confidence = np.mean([record.confidence for record in records])
            last_attendance = max(record.timestamp for record in records)
            
            return {
                "total_attendances": len(records),
                "unique_students": unique_students,
                "average_confidence": float(avg_confidence),
                "last_attendance": last_attendance
            }
            
        except Exception as e:
            logger.error(f"Error getting class attendance stats: {e}")
            return {"error": str(e)}

    def clear_attendance_records(self, 
                                class_code: str,
                                before_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Clear attendance records for cleanup
        
        Args:
            class_code: Class identifier
            before_date: Optional date filter (clear records before this date)
            
        Returns:
            Cleanup statistics
        """
        try:
            initial_count = len(self.attendance_records)
            
            if before_date:
                # Remove records before specified date
                self.attendance_records = [
                    record for record in self.attendance_records
                    if not (record.class_code == class_code and 
                           record.timestamp.split('T')[0] < before_date)
                ]
            else:
                # Remove all records for the class
                self.attendance_records = [
                    record for record in self.attendance_records
                    if record.class_code != class_code
                ]
            
            final_count = len(self.attendance_records)
            removed_count = initial_count - final_count
            
            logger.info(f"Cleared {removed_count} attendance records for class {class_code}")
            
            return {
                "class_code": class_code,
                "initial_count": initial_count,
                "final_count": final_count,
                "removed_count": removed_count,
                "before_date": before_date
            }
            
        except Exception as e:
            logger.error(f"Error clearing attendance records: {e}")
            return {"error": str(e)} 