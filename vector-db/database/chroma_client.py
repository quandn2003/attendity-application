import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
import json
from datetime import datetime, timedelta

from ..config.database_config import DatabaseConfig

logger = logging.getLogger(__name__)

class ChromaClient:
    """
    ChromaDB client optimized for mobile deployment with vector operations
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.client = None
        self.collections = {}
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client with mobile optimizations"""
        try:
            settings = Settings(
                persist_directory=self.config.persist_directory,
                anonymized_telemetry=False,
                allow_reset=True
            )
            
            self.client = chromadb.PersistentClient(settings=settings)
            logger.info(f"ChromaDB client initialized with persist directory: {self.config.persist_directory}")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {e}")
            raise
    
    def create_collection(self, class_code: str) -> bool:
        """
        Create a new collection for a class
        
        Args:
            class_code: Unique class identifier
            
        Returns:
            Success status
        """
        try:
            collection_name = self.config.get_collection_name(class_code)
            
            if collection_name in self.collections:
                logger.warning(f"Collection {collection_name} already exists")
                return True
            
            collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "class_code": class_code,
                    "created_at": datetime.now().isoformat(),
                    "embedding_dimension": self.config.embedding_dimension,
                    "distance_metric": self.config.distance_metric
                }
            )
            
            self.collections[collection_name] = collection
            logger.info(f"Created collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection for class {class_code}: {e}")
            return False
    
    def delete_collection(self, class_code: str) -> bool:
        """
        Delete a collection and all its data
        
        Args:
            class_code: Class identifier
            
        Returns:
            Success status
        """
        try:
            collection_name = self.config.get_collection_name(class_code)
            
            if collection_name in self.collections:
                del self.collections[collection_name]
            
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting collection for class {class_code}: {e}")
            return False
    
    def get_collection(self, class_code: str):
        """Get or load collection for a class"""
        try:
            collection_name = self.config.get_collection_name(class_code)
            
            if collection_name not in self.collections:
                collection = self.client.get_collection(name=collection_name)
                self.collections[collection_name] = collection
            
            return self.collections[collection_name]
            
        except Exception as e:
            logger.error(f"Error getting collection for class {class_code}: {e}")
            return None
    
    def insert_embeddings(self, 
                         class_code: str,
                         student_ids: List[str],
                         embeddings: List[np.ndarray],
                         metadata: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Insert student embeddings into collection
        
        Args:
            class_code: Class identifier
            student_ids: List of student IDs
            embeddings: List of face embeddings
            metadata: Optional metadata for each student
            
        Returns:
            Success status
        """
        try:
            collection = self.get_collection(class_code)
            if collection is None:
                if not self.create_collection(class_code):
                    return False
                collection = self.get_collection(class_code)
            
            # Prepare data for insertion
            ids = [f"{class_code}_{student_id}" for student_id in student_ids]
            embeddings_list = [embedding.tolist() for embedding in embeddings]
            
            # Prepare metadata
            if metadata is None:
                metadata = [{"student_id": sid, "class_code": class_code, 
                           "inserted_at": datetime.now().isoformat()} 
                          for sid in student_ids]
            else:
                for i, meta in enumerate(metadata):
                    meta.update({
                        "student_id": student_ids[i],
                        "class_code": class_code,
                        "inserted_at": datetime.now().isoformat()
                    })
            
            # Insert in batches for mobile optimization
            batch_size = min(self.config.max_batch_size, len(ids))
            
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i+batch_size]
                batch_embeddings = embeddings_list[i:i+batch_size]
                batch_metadata = metadata[i:i+batch_size]
                
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadata
                )
            
            logger.info(f"Inserted {len(ids)} embeddings into collection {class_code}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting embeddings for class {class_code}: {e}")
            return False
    
    def delete_embeddings(self, class_code: str, student_ids: List[str]) -> bool:
        """
        Delete student embeddings from collection
        
        Args:
            class_code: Class identifier
            student_ids: List of student IDs to delete
            
        Returns:
            Success status
        """
        try:
            collection = self.get_collection(class_code)
            if collection is None:
                logger.warning(f"Collection for class {class_code} not found")
                return False
            
            ids_to_delete = [f"{class_code}_{student_id}" for student_id in student_ids]
            
            collection.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} embeddings from collection {class_code}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting embeddings for class {class_code}: {e}")
            return False
    
    def search_similar(self, 
                      class_code: str,
                      query_embedding: np.ndarray,
                      top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings in collection
        
        Args:
            class_code: Class identifier
            query_embedding: Query face embedding
            top_k: Number of top results to return
            
        Returns:
            List of similar embeddings with metadata and distances
        """
        try:
            collection = self.get_collection(class_code)
            if collection is None:
                logger.warning(f"Collection for class {class_code} not found")
                return []
            
            top_k = top_k or self.config.top_k_results
            
            # Perform similarity search
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["metadatas", "distances", "embeddings"]
            )
            
            # Format results
            formatted_results = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    # Convert distance to similarity (cosine distance -> cosine similarity)
                    distance = results["distances"][0][i]
                    similarity = 1.0 - distance  # For cosine distance
                    
                    result = {
                        "id": results["ids"][0][i],
                        "student_id": results["metadatas"][0][i]["student_id"],
                        "similarity": similarity,
                        "distance": distance,
                        "metadata": results["metadatas"][0][i],
                        "embedding": np.array(results["embeddings"][0][i])
                    }
                    formatted_results.append(result)
            
            logger.debug(f"Found {len(formatted_results)} similar embeddings for class {class_code}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching similar embeddings for class {class_code}: {e}")
            return []
    
    def get_collection_stats(self, class_code: str) -> Dict[str, Any]:
        """
        Get statistics for a collection
        
        Args:
            class_code: Class identifier
            
        Returns:
            Collection statistics
        """
        try:
            collection = self.get_collection(class_code)
            if collection is None:
                return {"error": f"Collection for class {class_code} not found"}
            
            count = collection.count()
            metadata = collection.metadata
            
            return {
                "class_code": class_code,
                "student_count": count,
                "collection_metadata": metadata,
                "created_at": metadata.get("created_at") if metadata else None,
                "embedding_dimension": metadata.get("embedding_dimension") if metadata else None
            }
            
        except Exception as e:
            logger.error(f"Error getting stats for class {class_code}: {e}")
            return {"error": str(e)}
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections with their statistics"""
        try:
            collections = self.client.list_collections()
            collection_info = []
            
            for collection in collections:
                if collection.name.startswith(self.config.collection_prefix):
                    class_code = collection.name.replace(f"{self.config.collection_prefix}_", "")
                    stats = self.get_collection_stats(class_code)
                    collection_info.append(stats)
            
            return collection_info
            
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def cleanup_old_data(self, days_old: int = 30) -> Dict[str, Any]:
        """
        Cleanup old data for mobile storage optimization
        
        Args:
            days_old: Remove data older than this many days
            
        Returns:
            Cleanup statistics
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cleanup_stats = {"collections_processed": 0, "items_removed": 0}
            
            collections = self.client.list_collections()
            
            for collection in collections:
                if collection.name.startswith(self.config.collection_prefix):
                    # Get all items with metadata
                    results = collection.get(include=["metadatas"])
                    
                    old_ids = []
                    for i, metadata in enumerate(results["metadatas"]):
                        if "inserted_at" in metadata:
                            inserted_date = datetime.fromisoformat(metadata["inserted_at"])
                            if inserted_date < cutoff_date:
                                old_ids.append(results["ids"][i])
                    
                    if old_ids:
                        collection.delete(ids=old_ids)
                        cleanup_stats["items_removed"] += len(old_ids)
                    
                    cleanup_stats["collections_processed"] += 1
            
            logger.info(f"Cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {"error": str(e)}
    
    def backup_collection(self, class_code: str) -> bool:
        """
        Backup collection data for mobile deployment
        
        Args:
            class_code: Class identifier
            
        Returns:
            Success status
        """
        try:
            if not self.config.backup_enabled:
                return True
            
            collection = self.get_collection(class_code)
            if collection is None:
                return False
            
            # Get all data
            results = collection.get(include=["metadatas", "embeddings"])
            
            backup_data = {
                "class_code": class_code,
                "backup_timestamp": datetime.now().isoformat(),
                "data": {
                    "ids": results["ids"],
                    "embeddings": results["embeddings"],
                    "metadatas": results["metadatas"]
                }
            }
            
            # Save backup
            backup_file = f"{self.config.backup_directory}/{class_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            logger.info(f"Backup created for class {class_code}: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error backing up collection for class {class_code}: {e}")
            return False
    
    def initialize(self):
        """Initialize the client (already done in __init__, but provided for API compatibility)"""
        if self.client is None:
            self._initialize_client()
    
    def cleanup(self):
        """Cleanup client resources"""
        try:
            if self.client:
                # Clear collections cache
                self.collections.clear()
                logger.info("ChromaDB client cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def collection_exists(self, class_code: str) -> bool:
        """Check if a collection exists for the given class"""
        try:
            collection_name = self.config.get_collection_name(class_code)
            collections = self.client.list_collections()
            return any(col.name == collection_name for col in collections)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False
    
    def get_collection_count(self, class_code: str) -> int:
        """Get the number of students in a collection"""
        try:
            collection = self.get_collection(class_code)
            if collection is None:
                return 0
            return collection.count()
        except Exception as e:
            logger.error(f"Error getting collection count: {e}")
            return 0
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
                "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
                "percent": round(process.memory_percent(), 2)
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {"error": str(e)}
    
    def similarity_search(self, collection_name: str, query_embedding: List[float], n_results: int = 3) -> Dict[str, Any]:
        """Perform similarity search in a collection"""
        try:
            # Extract class code from collection name
            class_code = collection_name.replace(f"{self.config.collection_prefix}_", "")
            collection = self.get_collection(class_code)
            
            if collection is None:
                return {"ids": [[]], "distances": [[]], "metadatas": [[]]}
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["metadatas", "distances"]
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}

    def get_client_info(self) -> Dict[str, Any]:
        """Get client information and status"""
        try:
            collections = self.list_collections()
            total_students = sum(col.get("student_count", 0) for col in collections)
            
            return {
                "config": self.config.to_dict(),
                "total_collections": len(collections),
                "total_students": total_students,
                "collections": collections,
                "client_status": "connected" if self.client else "disconnected"
            }
            
        except Exception as e:
            logger.error(f"Error getting client info: {e}")
            return {"error": str(e)} 