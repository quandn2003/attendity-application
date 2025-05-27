import numpy as np
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from dataclasses import dataclass

from vector_db.database.chroma_client import ChromaClient
from vector_db.config.database_config import DatabaseConfig

logger = logging.getLogger(__name__)

@dataclass
class Student:
    """Student data model"""
    student_id: str
    class_code: str
    embedding: np.ndarray
    name: Optional[str] = None
    email: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None

@dataclass
class StudentInsertResult:
    """Result of student insertion operation"""
    success: bool
    student_id: str
    class_code: str
    message: str
    inserted_count: int = 0

class StudentManager:
    """
    Manager for student operations with consensus embeddings
    """
    
    def __init__(self, chroma_client: ChromaClient, config: DatabaseConfig):
        self.chroma_client = chroma_client
        self.config = config
    
    def insert_student(self, 
                      student_id: str,
                      class_code: str,
                      consensus_embedding: np.ndarray,
                      name: Optional[str] = None,
                      email: Optional[str] = None,
                      additional_metadata: Optional[Dict[str, Any]] = None) -> StudentInsertResult:
        """
        Insert a single student with consensus embedding
        
        Args:
            student_id: Unique student identifier
            class_code: Class identifier
            consensus_embedding: Consensus embedding from 3-image voting
            name: Student name (optional)
            email: Student email (optional)
            additional_metadata: Additional metadata
            
        Returns:
            Insert operation result
        """
        try:
            # Validate embedding
            if consensus_embedding.shape[0] != self.config.embedding_dimension:
                return StudentInsertResult(
                    success=False,
                    student_id=student_id,
                    class_code=class_code,
                    message=f"Invalid embedding dimension: {consensus_embedding.shape[0]}, expected {self.config.embedding_dimension}"
                )
            
            # Check if student already exists
            existing = self.get_student(student_id, class_code)
            if existing:
                return StudentInsertResult(
                    success=False,
                    student_id=student_id,
                    class_code=class_code,
                    message=f"Student {student_id} already exists in class {class_code}"
                )
            
            # Prepare metadata
            metadata = {
                "name": name,
                "email": email,
                "embedding_type": "consensus",
                "created_at": datetime.now().isoformat()
            }
            
            if additional_metadata:
                metadata.update(additional_metadata)
            
            # Insert into ChromaDB
            success = self.chroma_client.insert_embeddings(
                class_code=class_code,
                student_ids=[student_id],
                embeddings=[consensus_embedding],
                metadata=[metadata]
            )
            
            if success:
                logger.info(f"Successfully inserted student {student_id} into class {class_code}")
                return StudentInsertResult(
                    success=True,
                    student_id=student_id,
                    class_code=class_code,
                    message="Student inserted successfully",
                    inserted_count=1
                )
            else:
                return StudentInsertResult(
                    success=False,
                    student_id=student_id,
                    class_code=class_code,
                    message="Failed to insert student into database"
                )
                
        except Exception as e:
            logger.error(f"Error inserting student {student_id}: {e}")
            return StudentInsertResult(
                success=False,
                student_id=student_id,
                class_code=class_code,
                message=f"Error: {str(e)}"
            )
    
    def insert_multiple_students(self, students: List[Student]) -> List[StudentInsertResult]:
        """
        Insert multiple students in batch
        
        Args:
            students: List of Student objects
            
        Returns:
            List of insert results
        """
        results = []
        
        # Group students by class for batch insertion
        class_groups = {}
        for student in students:
            if student.class_code not in class_groups:
                class_groups[student.class_code] = []
            class_groups[student.class_code].append(student)
        
        # Insert each class group
        for class_code, class_students in class_groups.items():
            try:
                # Prepare batch data
                student_ids = [s.student_id for s in class_students]
                embeddings = [s.embedding for s in class_students]
                metadata_list = []
                
                for student in class_students:
                    metadata = {
                        "name": student.name,
                        "email": student.email,
                        "embedding_type": "consensus",
                        "created_at": student.created_at or datetime.now().isoformat()
                    }
                    if student.metadata:
                        metadata.update(student.metadata)
                    metadata_list.append(metadata)
                
                # Batch insert
                success = self.chroma_client.insert_embeddings(
                    class_code=class_code,
                    student_ids=student_ids,
                    embeddings=embeddings,
                    metadata=metadata_list
                )
                
                # Create results
                for student in class_students:
                    if success:
                        results.append(StudentInsertResult(
                            success=True,
                            student_id=student.student_id,
                            class_code=class_code,
                            message="Student inserted successfully",
                            inserted_count=1
                        ))
                    else:
                        results.append(StudentInsertResult(
                            success=False,
                            student_id=student.student_id,
                            class_code=class_code,
                            message="Batch insertion failed"
                        ))
                        
            except Exception as e:
                logger.error(f"Error in batch insertion for class {class_code}: {e}")
                for student in class_students:
                    results.append(StudentInsertResult(
                        success=False,
                        student_id=student.student_id,
                        class_code=class_code,
                        message=f"Batch error: {str(e)}"
                    ))
        
        return results
    
    def delete_student(self, student_id: str, class_code: str) -> bool:
        """
        Delete a student from the database
        
        Args:
            student_id: Student identifier
            class_code: Class identifier
            
        Returns:
            Success status
        """
        try:
            success = self.chroma_client.delete_embeddings(
                class_code=class_code,
                student_ids=[student_id]
            )
            
            if success:
                logger.info(f"Successfully deleted student {student_id} from class {class_code}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting student {student_id}: {e}")
            return False
    
    def delete_multiple_students(self, student_data: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Delete multiple students
        
        Args:
            student_data: List of dicts with student_id and class_code
            
        Returns:
            Deletion statistics
        """
        try:
            # Group by class
            class_groups = {}
            for data in student_data:
                class_code = data["class_code"]
                student_id = data["student_id"]
                
                if class_code not in class_groups:
                    class_groups[class_code] = []
                class_groups[class_code].append(student_id)
            
            total_deleted = 0
            failed_deletions = []
            
            # Delete by class groups
            for class_code, student_ids in class_groups.items():
                try:
                    success = self.chroma_client.delete_embeddings(
                        class_code=class_code,
                        student_ids=student_ids
                    )
                    
                    if success:
                        total_deleted += len(student_ids)
                    else:
                        failed_deletions.extend([{"student_id": sid, "class_code": class_code} 
                                               for sid in student_ids])
                        
                except Exception as e:
                    logger.error(f"Error deleting students from class {class_code}: {e}")
                    failed_deletions.extend([{"student_id": sid, "class_code": class_code, "error": str(e)} 
                                           for sid in student_ids])
            
            return {
                "total_requested": len(student_data),
                "total_deleted": total_deleted,
                "failed_deletions": failed_deletions,
                "success_rate": total_deleted / len(student_data) if student_data else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in batch deletion: {e}")
            return {
                "total_requested": len(student_data),
                "total_deleted": 0,
                "failed_deletions": student_data,
                "error": str(e)
            }
    
    def get_student(self, student_id: str, class_code: str) -> Optional[Student]:
        """
        Get student information
        
        Args:
            student_id: Student identifier
            class_code: Class identifier
            
        Returns:
            Student object or None if not found
        """
        try:
            # Search for the specific student
            collection = self.chroma_client.get_collection(class_code)
            if collection is None:
                return None
            
            # Get student by ID
            results = collection.get(
                ids=[f"{class_code}_{student_id}"],
                include=["metadatas", "embeddings"]
            )
            
            if results["ids"] and len(results["ids"]) > 0:
                metadata = results["metadatas"][0]
                embedding = np.array(results["embeddings"][0])
                
                return Student(
                    student_id=student_id,
                    class_code=class_code,
                    embedding=embedding,
                    name=metadata.get("name"),
                    email=metadata.get("email"),
                    metadata=metadata,
                    created_at=metadata.get("created_at")
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting student {student_id}: {e}")
            return None
    
    def list_students_in_class(self, class_code: str) -> List[Student]:
        """
        List all students in a class
        
        Args:
            class_code: Class identifier
            
        Returns:
            List of Student objects
        """
        try:
            collection = self.chroma_client.get_collection(class_code)
            if collection is None:
                return []
            
            # Get all students in the class
            results = collection.get(include=["metadatas", "embeddings"])
            
            students = []
            for i, student_id_full in enumerate(results["ids"]):
                # Extract student_id from full ID (format: class_code_student_id)
                student_id = student_id_full.replace(f"{class_code}_", "")
                metadata = results["metadatas"][i]
                embedding = np.array(results["embeddings"][i])
                
                student = Student(
                    student_id=student_id,
                    class_code=class_code,
                    embedding=embedding,
                    name=metadata.get("name"),
                    email=metadata.get("email"),
                    metadata=metadata,
                    created_at=metadata.get("created_at")
                )
                students.append(student)
            
            return students
            
        except Exception as e:
            logger.error(f"Error listing students in class {class_code}: {e}")
            return []
    
    def update_student_metadata(self, 
                               student_id: str,
                               class_code: str,
                               new_metadata: Dict[str, Any]) -> bool:
        """
        Update student metadata (name, email, etc.)
        
        Args:
            student_id: Student identifier
            class_code: Class identifier
            new_metadata: New metadata to update
            
        Returns:
            Success status
        """
        try:
            # Get current student data
            student = self.get_student(student_id, class_code)
            if student is None:
                logger.warning(f"Student {student_id} not found in class {class_code}")
                return False
            
            # Update metadata
            updated_metadata = student.metadata.copy() if student.metadata else {}
            updated_metadata.update(new_metadata)
            updated_metadata["updated_at"] = datetime.now().isoformat()
            
            # Delete and re-insert with updated metadata
            delete_success = self.delete_student(student_id, class_code)
            if not delete_success:
                return False
            
            insert_result = self.insert_student(
                student_id=student_id,
                class_code=class_code,
                consensus_embedding=student.embedding,
                name=updated_metadata.get("name"),
                email=updated_metadata.get("email"),
                additional_metadata=updated_metadata
            )
            
            return insert_result.success
            
        except Exception as e:
            logger.error(f"Error updating student {student_id} metadata: {e}")
            return False
    
    def get_class_statistics(self, class_code: str) -> Dict[str, Any]:
        """
        Get statistics for a class
        
        Args:
            class_code: Class identifier
            
        Returns:
            Class statistics
        """
        try:
            students = self.list_students_in_class(class_code)
            
            # Calculate statistics
            total_students = len(students)
            students_with_names = sum(1 for s in students if s.name)
            students_with_emails = sum(1 for s in students if s.email)
            
            # Creation date analysis
            creation_dates = [s.created_at for s in students if s.created_at]
            latest_addition = max(creation_dates) if creation_dates else None
            earliest_addition = min(creation_dates) if creation_dates else None
            
            return {
                "class_code": class_code,
                "total_students": total_students,
                "students_with_names": students_with_names,
                "students_with_emails": students_with_emails,
                "completion_rate": {
                    "names": students_with_names / total_students if total_students > 0 else 0,
                    "emails": students_with_emails / total_students if total_students > 0 else 0
                },
                "latest_addition": latest_addition,
                "earliest_addition": earliest_addition,
                "embedding_dimension": self.config.embedding_dimension
            }
            
        except Exception as e:
            logger.error(f"Error getting class statistics for {class_code}: {e}")
            return {"error": str(e)}
    
    def student_exists(self, student_id: str, class_code: str) -> bool:
        """
        Check if a student exists in the database
        
        Args:
            student_id: Student identifier
            class_code: Class identifier
            
        Returns:
            True if student exists, False otherwise
        """
        try:
            student = self.get_student(student_id, class_code)
            return student is not None
        except Exception as e:
            logger.error(f"Error checking student existence: {e}")
            return False
    
    def insert_consensus_embedding(self, 
                                  student_id: str,
                                  class_code: str,
                                  consensus_embedding: List[float],
                                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Insert a student with consensus embedding
        
        Args:
            student_id: Student identifier
            class_code: Class identifier
            consensus_embedding: Consensus embedding from voting
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        try:
            # Convert list to numpy array
            embedding_array = np.array(consensus_embedding)
            
            # Use the existing insert_student method
            result = self.insert_student(
                student_id=student_id,
                class_code=class_code,
                consensus_embedding=embedding_array,
                additional_metadata=metadata
            )
            
            return result.success
            
        except Exception as e:
            logger.error(f"Error inserting consensus embedding: {e}")
            return False

    def search_students_by_name(self, class_code: str, name_query: str) -> List[Student]:
        """
        Search students by name in a class
        
        Args:
            class_code: Class identifier
            name_query: Name search query
            
        Returns:
            List of matching students
        """
        try:
            students = self.list_students_in_class(class_code)
            
            # Simple name matching (case-insensitive)
            matching_students = []
            name_query_lower = name_query.lower()
            
            for student in students:
                if student.name and name_query_lower in student.name.lower():
                    matching_students.append(student)
            
            return matching_students
            
        except Exception as e:
            logger.error(f"Error searching students by name in class {class_code}: {e}")
            return [] 