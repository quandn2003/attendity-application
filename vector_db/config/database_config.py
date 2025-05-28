from dataclasses import dataclass
from typing import Optional, Dict, Any
import os

@dataclass
class DatabaseConfig:
    """Configuration for ChromaDB vector database optimized for mobile deployment"""
    
    persist_directory: str = "./chroma_db"
    collection_prefix: str = "attendity"
    embedding_dimension: int = 512
    distance_metric: str = "cosine"
    max_batch_size: int = 100
    similarity_threshold: float = 0.6
    top_k_results: int = 3
    voting_threshold: float = 0.6
    
    # Mobile optimization settings
    enable_compression: bool = True
    max_collection_size: int = 10000
    cleanup_interval_hours: int = 24
    backup_enabled: bool = True
    backup_directory: str = "./chroma_backups"
    
    # Performance settings
    query_timeout_seconds: int = 30
    batch_timeout_seconds: int = 60
    max_memory_usage_mb: int = 512
    
    def __post_init__(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        if self.backup_enabled:
            os.makedirs(self.backup_directory, exist_ok=True)
    
    def get_collection_name(self, class_code: str) -> str:
        """Generate collection name for a class"""
        return class_code
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "persist_directory": self.persist_directory,
            "collection_prefix": self.collection_prefix,
            "embedding_dimension": self.embedding_dimension,
            "distance_metric": self.distance_metric,
            "max_batch_size": self.max_batch_size,
            "similarity_threshold": self.similarity_threshold,
            "top_k_results": self.top_k_results,
            "voting_threshold": self.voting_threshold,
            "enable_compression": self.enable_compression,
            "max_collection_size": self.max_collection_size,
            "cleanup_interval_hours": self.cleanup_interval_hours,
            "backup_enabled": self.backup_enabled,
            "backup_directory": self.backup_directory,
            "query_timeout_seconds": self.query_timeout_seconds,
            "batch_timeout_seconds": self.batch_timeout_seconds,
            "max_memory_usage_mb": self.max_memory_usage_mb
        } 