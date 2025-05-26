import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

@dataclass
class VotingResult:
    """Result of voting system for multi-image processing"""
    is_consistent: bool
    consensus_embedding: Optional[np.ndarray]
    similarity_matrix: np.ndarray
    individual_scores: List[float]
    reason: str
    confidence: float

class VotingSystem:
    """
    Voting system for 3-image processing and consensus building
    Validates consistency between multiple face images of the same person
    """
    
    def __init__(self,
                 similarity_threshold: float = 0.6,
                 consistency_threshold: float = 0.7,
                 min_valid_pairs: int = 2):
        """
        Initialize voting system
        
        Args:
            similarity_threshold: Minimum similarity between face embeddings
            consistency_threshold: Threshold for overall consistency
            min_valid_pairs: Minimum number of valid pairs required
        """
        self.similarity_threshold = similarity_threshold
        self.consistency_threshold = consistency_threshold
        self.min_valid_pairs = min_valid_pairs
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            # Convert to [0, 1] range
            similarity = (similarity + 1) / 2
            
            return float(np.clip(similarity, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def build_similarity_matrix(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Build pairwise similarity matrix for all embeddings
        
        Args:
            embeddings: List of face embeddings
            
        Returns:
            Similarity matrix (n x n)
        """
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity_matrix[i, j] = self.calculate_similarity(
                        embeddings[i], embeddings[j]
                    )
        
        return similarity_matrix
    
    def validate_consistency(self, embeddings: List[np.ndarray]) -> VotingResult:
        """
        Validate consistency between multiple face embeddings
        
        Args:
            embeddings: List of face embeddings (typically 3 for student insertion)
            
        Returns:
            Voting result with consistency validation
        """
        try:
            if len(embeddings) < 2:
                return VotingResult(
                    is_consistent=False,
                    consensus_embedding=None,
                    similarity_matrix=np.array([]),
                    individual_scores=[],
                    reason="insufficient_embeddings",
                    confidence=0.0
                )
            
            # Build similarity matrix
            similarity_matrix = self.build_similarity_matrix(embeddings)
            
            # Extract pairwise similarities (excluding diagonal)
            n = len(embeddings)
            pairwise_similarities = []
            for i in range(n):
                for j in range(i + 1, n):
                    pairwise_similarities.append(similarity_matrix[i, j])
            
            # Calculate individual scores (average similarity with others)
            individual_scores = []
            for i in range(n):
                scores = [similarity_matrix[i, j] for j in range(n) if i != j]
                individual_scores.append(np.mean(scores))
            
            # Check consistency criteria
            valid_pairs = sum(1 for sim in pairwise_similarities if sim >= self.similarity_threshold)
            overall_consistency = np.mean(pairwise_similarities)
            
            is_consistent = (
                valid_pairs >= self.min_valid_pairs and
                overall_consistency >= self.consistency_threshold
            )
            
            # Generate consensus embedding if consistent
            consensus_embedding = None
            if is_consistent:
                consensus_embedding = self._generate_consensus_embedding(embeddings, individual_scores)
            
            # Determine reason
            reason = self._determine_reason(
                is_consistent, valid_pairs, overall_consistency, individual_scores
            )
            
            return VotingResult(
                is_consistent=is_consistent,
                consensus_embedding=consensus_embedding,
                similarity_matrix=similarity_matrix,
                individual_scores=individual_scores,
                reason=reason,
                confidence=overall_consistency
            )
            
        except Exception as e:
            logger.error(f"Error in consistency validation: {e}")
            return VotingResult(
                is_consistent=False,
                consensus_embedding=None,
                similarity_matrix=np.array([]),
                individual_scores=[],
                reason=f"validation_error: {str(e)}",
                confidence=0.0
            )
    
    def _generate_consensus_embedding(self, 
                                    embeddings: List[np.ndarray], 
                                    individual_scores: List[float]) -> np.ndarray:
        """
        Generate consensus embedding using weighted average
        
        Args:
            embeddings: List of face embeddings
            individual_scores: Individual quality scores for weighting
            
        Returns:
            Consensus embedding
        """
        try:
            # Use individual scores as weights
            weights = np.array(individual_scores)
            weights = weights / np.sum(weights)  # Normalize weights
            
            # Calculate weighted average
            consensus = np.zeros_like(embeddings[0])
            for i, embedding in enumerate(embeddings):
                consensus += weights[i] * embedding
            
            # Normalize the consensus embedding
            norm = np.linalg.norm(consensus)
            if norm > 0:
                consensus = consensus / norm
            
            return consensus
            
        except Exception as e:
            logger.error(f"Error generating consensus embedding: {e}")
            # Fallback to simple average
            return np.mean(embeddings, axis=0)
    
    def _determine_reason(self, 
                         is_consistent: bool,
                         valid_pairs: int,
                         overall_consistency: float,
                         individual_scores: List[float]) -> str:
        """Determine the reason for consistency result"""
        if is_consistent:
            return "consistent_faces_detected"
        
        if valid_pairs < self.min_valid_pairs:
            return f"insufficient_valid_pairs_{valid_pairs}_of_{self.min_valid_pairs}_required"
        
        if overall_consistency < self.consistency_threshold:
            return f"low_overall_consistency_{overall_consistency:.3f}_below_{self.consistency_threshold}"
        
        # Check for outliers
        if len(individual_scores) > 0:
            min_score = min(individual_scores)
            if min_score < 0.3:
                return f"outlier_detected_min_score_{min_score:.3f}"
        
        return "unknown_inconsistency"
    
    def process_three_images(self, embeddings: List[np.ndarray]) -> VotingResult:
        """
        Process exactly 3 images for student insertion
        
        Args:
            embeddings: List of 3 face embeddings
            
        Returns:
            Voting result for 3-image validation
        """
        if len(embeddings) != 3:
            return VotingResult(
                is_consistent=False,
                consensus_embedding=None,
                similarity_matrix=np.array([]),
                individual_scores=[],
                reason=f"expected_3_images_got_{len(embeddings)}",
                confidence=0.0
            )
        
        return self.validate_consistency(embeddings)
    
    def find_best_matches(self, 
                         query_embedding: np.ndarray,
                         candidate_embeddings: List[np.ndarray],
                         top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Find top-k best matches for attendance verification
        
        Args:
            query_embedding: Query face embedding
            candidate_embeddings: List of candidate embeddings from database
            top_k: Number of top matches to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        try:
            similarities = []
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.calculate_similarity(query_embedding, candidate)
                similarities.append((i, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding best matches: {e}")
            return []
    
    def apply_voting_for_attendance(self,
                                  query_embedding: np.ndarray,
                                  top_matches: List[Tuple[int, float]],
                                  voting_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Apply voting logic for attendance verification
        
        Args:
            query_embedding: Query face embedding
            top_matches: Top matches from database search
            voting_threshold: Threshold for clear match decision
            
        Returns:
            Voting decision for attendance
        """
        try:
            if not top_matches:
                return {
                    "decision": "no_match",
                    "reason": "no_candidates_found",
                    "confidence": 0.0,
                    "top_matches": []
                }
            
            # Get the best match
            best_match_idx, best_similarity = top_matches[0]
            
            # Check if there's a clear winner
            if best_similarity >= voting_threshold:
                # Check if the best match is significantly better than others
                if len(top_matches) > 1:
                    second_best_similarity = top_matches[1][1]
                    margin = best_similarity - second_best_similarity
                    
                    if margin >= 0.1:  # Clear margin
                        return {
                            "decision": "clear_match",
                            "student_index": best_match_idx,
                            "confidence": best_similarity,
                            "margin": margin,
                            "top_matches": top_matches,
                            "reason": "clear_winner_with_margin"
                        }
                    else:
                        return {
                            "decision": "ambiguous_match",
                            "confidence": best_similarity,
                            "margin": margin,
                            "top_matches": top_matches,
                            "reason": "insufficient_margin_between_top_matches"
                        }
                else:
                    return {
                        "decision": "clear_match",
                        "student_index": best_match_idx,
                        "confidence": best_similarity,
                        "top_matches": top_matches,
                        "reason": "single_strong_match"
                    }
            else:
                return {
                    "decision": "no_clear_match",
                    "confidence": best_similarity,
                    "top_matches": top_matches,
                    "reason": f"best_similarity_{best_similarity:.3f}_below_threshold_{voting_threshold}"
                }
                
        except Exception as e:
            logger.error(f"Error in attendance voting: {e}")
            return {
                "decision": "error",
                "reason": f"voting_error: {str(e)}",
                "confidence": 0.0,
                "top_matches": []
            }
    
    def get_voting_stats(self) -> Dict[str, Any]:
        """Get voting system configuration and statistics"""
        return {
            "similarity_threshold": self.similarity_threshold,
            "consistency_threshold": self.consistency_threshold,
            "min_valid_pairs": self.min_valid_pairs
        } 