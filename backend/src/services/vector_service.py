import json
import os
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SimpleVectorService:
    """Simple file-based vector storage for development"""
    
    def __init__(self):
        self.storage_file = "face_embeddings.json"
        self.embeddings = self.load_embeddings()
    
    def load_embeddings(self) -> Dict:
        """Load embeddings from file"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading embeddings: {e}")
        return {}
    
    def save_embeddings(self):
        """Save embeddings to file"""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.embeddings, f)
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
    
    def insert_embedding(self, student_id: str, embedding: List[float]) -> bool:
        """Insert face embedding"""
        try:
            self.embeddings[student_id] = embedding
            self.save_embeddings()
            logger.info(f"Inserted embedding for student: {student_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to insert embedding: {str(e)}")
            return False
    
    def search_similar(self, embedding: List[float], top_k: int = 1) -> List[Dict]:
        """Search for similar embeddings using cosine similarity"""
        try:
            if not self.embeddings:
                return []
            
            similarities = []
            query_embedding = np.array(embedding)
            
            for student_id, stored_embedding in self.embeddings.items():
                stored_emb = np.array(stored_embedding)
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, stored_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_emb)
                )
                
                similarities.append({
                    "student_id": student_id,
                    "similarity": float(similarity)
                })
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            return similarities[:top_k]
        except Exception as e:
            logger.error(f"Failed to search embeddings: {str(e)}")
            return []
    
    def delete_embedding(self, student_id: str) -> bool:
        """Delete embedding for student"""
        try:
            if student_id in self.embeddings:
                del self.embeddings[student_id]
                self.save_embeddings()
            return True
        except Exception as e:
            logger.error(f"Failed to delete embedding: {str(e)}")
            return False

vector_service = SimpleVectorService() 