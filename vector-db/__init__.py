from .database.chroma_client import ChromaClient
from .database.student_manager import StudentManager
from .database.attendance_manager import AttendanceManager
from .voting.similarity_voting import SimilarityVoting
from .config.database_config import DatabaseConfig

__version__ = "1.0.0"
__all__ = [
    "ChromaClient",
    "StudentManager", 
    "AttendanceManager",
    "SimilarityVoting",
    "DatabaseConfig"
] 