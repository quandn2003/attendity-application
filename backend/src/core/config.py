from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Face Recognition API"
    
    # Database - Using SQLite for development
    DATABASE_URL: str = "sqlite:///./face_recognition.db"
    
    # Redis (optional for development)
    REDIS_URL: str = "redis://localhost:6379"
    
    # Vector Database
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # AI Settings
    FACE_SIMILARITY_THRESHOLD: float = 0.8
    MODEL_DEVICE: str = "cpu"
    
    class Config:
        env_file = ".env"

settings = Settings() 