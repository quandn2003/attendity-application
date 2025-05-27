from vector_db.database.chroma_client import ChromaClient
from vector_db.database.student_manager import StudentManager
from vector_db.database.attendance_manager import AttendanceManager
from vector_db.voting.similarity_voting import SimilarityVoting
from vector_db.config.database_config import DatabaseConfig

__version__ = "1.0.0"
__all__ = [
    "ChromaClient",
    "StudentManager", 
    "AttendanceManager",
    "SimilarityVoting",
    "DatabaseConfig"
] 