import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VotingCandidate:
    """Candidate for similarity voting"""
    student_id: str
    embedding: np.ndarray
    similarity: float
    distance: float
    metadata: Dict[str, Any]

@dataclass
class VotingResult:
    """Result of similarity voting process"""
    decision: str
    confidence: float
    student_id: Optional[str]
    top_candidates: List[VotingCandidate]
    voting_details: Dict[str, Any]
    reason: str

class SimilarityVoting:
    """
    Enhanced similarity voting system for attendance verification
    Implements top-3 voting with consensus building and margin analysis
    """
    
    def __init__(self,
                 similarity_threshold: float = 0.6,
                 voting_threshold: float = 0.8,
                 margin_threshold: float = 0.1,
                 consensus_threshold: float = 0.7):
        """
        Initialize similarity voting system
        
        Args:
            similarity_threshold: Minimum similarity to consider a candidate
            voting_threshold: Threshold for clear match decision
            margin_threshold: Minimum margin between top candidates
            consensus_threshold: Threshold for consensus building
        """
        self.similarity_threshold = similarity_threshold
        self.voting_threshold = voting_threshold
        self.margin_threshold = margin_threshold
        self.consensus_threshold = consensus_threshold
    
    def calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
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
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def prepare_candidates(self, search_results: List[Dict[str, Any]]) -> List[VotingCandidate]:
        """
        Prepare voting candidates from search results
        
        Args:
            search_results: Results from ChromaDB similarity search
            
        Returns:
            List of voting candidates
        """
        candidates = []
        
        for result in search_results:
            try:
                candidate = VotingCandidate(
                    student_id=result["student_id"],
                    embedding=result["embedding"],
                    similarity=result["similarity"],
                    distance=result["distance"],
                    metadata=result["metadata"]
                )
                candidates.append(candidate)
                
            except Exception as e:
                logger.error(f"Error preparing candidate: {e}")
                continue
        
        # Sort by similarity (descending)
        candidates.sort(key=lambda x: x.similarity, reverse=True)
        
        return candidates
    
    def apply_top3_voting(self, 
                         query_embedding: np.ndarray,
                         candidates: List[VotingCandidate]) -> VotingResult:
        """
        Apply top-3 voting logic for attendance verification
        
        Args:
            query_embedding: Query face embedding
            candidates: List of voting candidates
            
        Returns:
            Voting result with decision
        """
        try:
            if not candidates:
                return VotingResult(
                    decision="no_candidates",
                    confidence=0.0,
                    student_id=None,
                    top_candidates=[],
                    voting_details={"reason": "no_candidates_found"},
                    reason="No candidates found for voting"
                )
            
            # Filter candidates by similarity threshold
            valid_candidates = [
                c for c in candidates if c.similarity >= self.similarity_threshold
            ]
            
            if not valid_candidates:
                return VotingResult(
                    decision="below_threshold",
                    confidence=0.0,
                    student_id=None,
                    top_candidates=candidates[:3],
                    voting_details={
                        "best_similarity": candidates[0].similarity,
                        "threshold": self.similarity_threshold
                    },
                    reason=f"Best similarity {candidates[0].similarity:.3f} below threshold {self.similarity_threshold}"
                )
            
            # Take top 3 valid candidates
            top3_candidates = valid_candidates[:3]
            
            # Analyze voting patterns
            voting_analysis = self._analyze_voting_patterns(query_embedding, top3_candidates)
            
            # Make decision based on voting analysis
            decision_result = self._make_voting_decision(top3_candidates, voting_analysis)
            
            return VotingResult(
                decision=decision_result["decision"],
                confidence=decision_result["confidence"],
                student_id=decision_result.get("student_id"),
                top_candidates=top3_candidates,
                voting_details=voting_analysis,
                reason=decision_result["reason"]
            )
            
        except Exception as e:
            logger.error(f"Error in top-3 voting: {e}")
            return VotingResult(
                decision="error",
                confidence=0.0,
                student_id=None,
                top_candidates=[],
                voting_details={"error": str(e)},
                reason=f"Voting error: {str(e)}"
            )
    
    def _analyze_voting_patterns(self, 
                                query_embedding: np.ndarray,
                                candidates: List[VotingCandidate]) -> Dict[str, Any]:
        """
        Analyze voting patterns among top candidates
        
        Args:
            query_embedding: Query embedding
            candidates: Top candidates for analysis
            
        Returns:
            Voting pattern analysis
        """
        try:
            analysis = {
                "candidate_count": len(candidates),
                "similarities": [c.similarity for c in candidates],
                "student_ids": [c.student_id for c in candidates]
            }
            
            if len(candidates) >= 1:
                best_candidate = candidates[0]
                analysis["best_match"] = {
                    "student_id": best_candidate.student_id,
                    "similarity": best_candidate.similarity,
                    "distance": best_candidate.distance
                }
            
            if len(candidates) >= 2:
                # Calculate margin between top 2
                margin = candidates[0].similarity - candidates[1].similarity
                analysis["top2_margin"] = margin
                analysis["margin_significant"] = margin >= self.margin_threshold
                
                # Check for duplicate students in top results
                unique_students = len(set(c.student_id for c in candidates))
                analysis["unique_students"] = unique_students
                analysis["has_duplicates"] = unique_students < len(candidates)
            
            if len(candidates) >= 3:
                # Analyze top-3 consensus
                top3_similarities = [c.similarity for c in candidates[:3]]
                analysis["top3_mean"] = np.mean(top3_similarities)
                analysis["top3_std"] = np.std(top3_similarities)
                analysis["top3_consensus"] = analysis["top3_std"] < 0.1  # Low variance indicates consensus
                
                # Calculate pairwise similarities between top 3 embeddings
                pairwise_sims = []
                for i in range(3):
                    for j in range(i + 1, 3):
                        sim = self.calculate_cosine_similarity(
                            candidates[i].embedding, candidates[j].embedding
                        )
                        pairwise_sims.append(sim)
                
                analysis["pairwise_similarities"] = pairwise_sims
                analysis["avg_pairwise_similarity"] = np.mean(pairwise_sims)
            
            # Overall confidence calculation
            if candidates:
                base_confidence = candidates[0].similarity
                
                # Boost confidence if there's a clear margin
                if len(candidates) >= 2 and analysis.get("margin_significant", False):
                    base_confidence += 0.1
                
                # Reduce confidence if there are duplicates or low consensus
                if analysis.get("has_duplicates", False):
                    base_confidence -= 0.05
                
                if len(candidates) >= 3 and not analysis.get("top3_consensus", True):
                    base_confidence -= 0.05
                
                analysis["adjusted_confidence"] = min(base_confidence, 1.0)
            else:
                analysis["adjusted_confidence"] = 0.0
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in voting pattern analysis: {e}")
            return {"error": str(e), "adjusted_confidence": 0.0}
    
    def _make_voting_decision(self, 
                             candidates: List[VotingCandidate],
                             analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make final voting decision based on analysis
        
        Args:
            candidates: Top candidates
            analysis: Voting pattern analysis
            
        Returns:
            Decision result
        """
        try:
            if not candidates:
                return {
                    "decision": "no_candidates",
                    "confidence": 0.0,
                    "reason": "No valid candidates"
                }
            
            best_candidate = candidates[0]
            best_similarity = best_candidate.similarity
            adjusted_confidence = analysis.get("adjusted_confidence", best_similarity)
            
            # Decision logic based on voting threshold
            if best_similarity >= self.voting_threshold:
                # High confidence match
                if len(candidates) == 1:
                    return {
                        "decision": "clear_match",
                        "student_id": best_candidate.student_id,
                        "confidence": adjusted_confidence,
                        "reason": "Single strong match above voting threshold"
                    }
                
                # Multiple candidates - check margin
                if analysis.get("margin_significant", False):
                    return {
                        "decision": "clear_match",
                        "student_id": best_candidate.student_id,
                        "confidence": adjusted_confidence,
                        "reason": f"Clear winner with margin {analysis.get('top2_margin', 0):.3f}"
                    }
                else:
                    return {
                        "decision": "ambiguous_match",
                        "confidence": adjusted_confidence,
                        "reason": f"Insufficient margin {analysis.get('top2_margin', 0):.3f} between top matches"
                    }
            
            elif best_similarity >= self.similarity_threshold:
                # Medium confidence - additional checks
                if len(candidates) >= 3 and analysis.get("top3_consensus", False):
                    # Good consensus among top 3
                    return {
                        "decision": "consensus_match",
                        "student_id": best_candidate.student_id,
                        "confidence": adjusted_confidence,
                        "reason": "Consensus match based on top-3 agreement"
                    }
                else:
                    return {
                        "decision": "weak_match",
                        "confidence": adjusted_confidence,
                        "reason": f"Similarity {best_similarity:.3f} below voting threshold {self.voting_threshold}"
                    }
            
            else:
                # Below similarity threshold
                return {
                    "decision": "no_match",
                    "confidence": adjusted_confidence,
                    "reason": f"Best similarity {best_similarity:.3f} below threshold {self.similarity_threshold}"
                }
                
        except Exception as e:
            logger.error(f"Error making voting decision: {e}")
            return {
                "decision": "error",
                "confidence": 0.0,
                "reason": f"Decision error: {str(e)}"
            }
    
    def vote_for_attendance(self, 
                           query_embedding: np.ndarray,
                           search_results: List[Dict[str, Any]]) -> VotingResult:
        """
        Main voting function for attendance verification
        
        Args:
            query_embedding: Query face embedding
            search_results: Results from similarity search
            
        Returns:
            Voting result with attendance decision
        """
        try:
            # Prepare candidates
            candidates = self.prepare_candidates(search_results)
            
            # Apply top-3 voting
            voting_result = self.apply_top3_voting(query_embedding, candidates)
            
            # Log voting decision
            logger.info(f"Voting decision: {voting_result.decision} for student {voting_result.student_id}")
            
            return voting_result
            
        except Exception as e:
            logger.error(f"Error in attendance voting: {e}")
            return VotingResult(
                decision="error",
                confidence=0.0,
                student_id=None,
                top_candidates=[],
                voting_details={"error": str(e)},
                reason=f"Voting error: {str(e)}"
            )
    
    def get_voting_statistics(self, voting_results: List[VotingResult]) -> Dict[str, Any]:
        """
        Calculate statistics from multiple voting results
        
        Args:
            voting_results: List of voting results
            
        Returns:
            Voting statistics
        """
        try:
            if not voting_results:
                return {"error": "No voting results provided"}
            
            # Decision distribution
            decisions = [result.decision for result in voting_results]
            decision_counts = {}
            for decision in decisions:
                decision_counts[decision] = decision_counts.get(decision, 0) + 1
            
            # Confidence statistics
            confidences = [result.confidence for result in voting_results]
            
            # Success rate (clear matches)
            successful_matches = sum(1 for result in voting_results 
                                   if result.decision in ["clear_match", "consensus_match"])
            success_rate = successful_matches / len(voting_results)
            
            return {
                "total_votes": len(voting_results),
                "decision_distribution": decision_counts,
                "success_rate": success_rate,
                "confidence_stats": {
                    "mean": np.mean(confidences),
                    "std": np.std(confidences),
                    "min": np.min(confidences),
                    "max": np.max(confidences)
                },
                "successful_matches": successful_matches,
                "ambiguous_matches": decision_counts.get("ambiguous_match", 0),
                "no_matches": decision_counts.get("no_match", 0) + decision_counts.get("no_candidates", 0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating voting statistics: {e}")
            return {"error": str(e)}
    
    def tune_thresholds(self, 
                       validation_data: List[Dict[str, Any]],
                       target_accuracy: float = 0.95) -> Dict[str, float]:
        """
        Tune voting thresholds based on validation data
        
        Args:
            validation_data: List of validation samples with ground truth
            target_accuracy: Target accuracy for threshold tuning
            
        Returns:
            Optimized thresholds
        """
        try:
            # This is a simplified threshold tuning approach
            # In practice, you would use more sophisticated optimization
            
            best_thresholds = {
                "similarity_threshold": self.similarity_threshold,
                "voting_threshold": self.voting_threshold,
                "margin_threshold": self.margin_threshold
            }
            best_accuracy = 0.0
            
            # Grid search over threshold values
            similarity_values = np.arange(0.5, 0.9, 0.05)
            voting_values = np.arange(0.7, 0.95, 0.05)
            margin_values = np.arange(0.05, 0.2, 0.025)
            
            for sim_thresh in similarity_values:
                for vote_thresh in voting_values:
                    for margin_thresh in margin_values:
                        # Temporarily set thresholds
                        original_thresholds = (
                            self.similarity_threshold,
                            self.voting_threshold,
                            self.margin_threshold
                        )
                        
                        self.similarity_threshold = sim_thresh
                        self.voting_threshold = vote_thresh
                        self.margin_threshold = margin_thresh
                        
                        # Evaluate on validation data
                        correct_predictions = 0
                        for sample in validation_data:
                            # Simulate voting (simplified)
                            result = self.vote_for_attendance(
                                sample["query_embedding"],
                                sample["search_results"]
                            )
                            
                            predicted_id = result.student_id
                            true_id = sample["ground_truth_id"]
                            
                            if predicted_id == true_id:
                                correct_predictions += 1
                        
                        accuracy = correct_predictions / len(validation_data)
                        
                        if accuracy > best_accuracy and accuracy >= target_accuracy:
                            best_accuracy = accuracy
                            best_thresholds = {
                                "similarity_threshold": sim_thresh,
                                "voting_threshold": vote_thresh,
                                "margin_threshold": margin_thresh
                            }
                        
                        # Restore original thresholds
                        (self.similarity_threshold,
                         self.voting_threshold,
                         self.margin_threshold) = original_thresholds
            
            # Set best thresholds
            self.similarity_threshold = best_thresholds["similarity_threshold"]
            self.voting_threshold = best_thresholds["voting_threshold"]
            self.margin_threshold = best_thresholds["margin_threshold"]
            
            logger.info(f"Tuned thresholds: {best_thresholds}, accuracy: {best_accuracy:.3f}")
            
            return {
                **best_thresholds,
                "achieved_accuracy": best_accuracy,
                "target_accuracy": target_accuracy
            }
            
        except Exception as e:
            logger.error(f"Error tuning thresholds: {e}")
            return {"error": str(e)}
    
    def get_voting_config(self) -> Dict[str, Any]:
        """Get current voting configuration"""
        return {
            "similarity_threshold": self.similarity_threshold,
            "voting_threshold": self.voting_threshold,
            "margin_threshold": self.margin_threshold,
            "consensus_threshold": self.consensus_threshold
        } 