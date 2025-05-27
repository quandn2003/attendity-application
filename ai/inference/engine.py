import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
from dataclasses import dataclass

from ai.models.facenet_model import FaceNetModel, ModelConfig
from ai.utils.preprocessing import FacePreprocessor
from ai.utils.anti_spoofing import AntiSpoofingDetector, AntiSpoofingResult
from ai.utils.voting import VotingSystem, VotingResult

logger = logging.getLogger(__name__)

@dataclass
class InferenceResult:
    """Result of face recognition inference"""
    success: bool
    embedding: Optional[np.ndarray]
    confidence: float
    is_real: bool
    anti_spoofing_result: Optional[AntiSpoofingResult]
    processing_time: float
    error_message: Optional[str]

@dataclass
class MultiImageResult:
    """Result of multi-image processing for student insertion"""
    success: bool
    consensus_embedding: Optional[np.ndarray]
    voting_result: Optional[VotingResult]
    individual_results: List[InferenceResult]
    total_processing_time: float
    error_message: Optional[str]

class InferenceEngine:
    """
    Main inference engine optimized for mobile CPU deployment
    Combines face detection, anti-spoofing, and embedding extraction
    """
    
    def __init__(self, 
                 model_config: Optional[ModelConfig] = None,
                 enable_anti_spoofing: bool = True,
                 enable_voting: bool = True):
        """
        Initialize inference engine
        
        Args:
            model_config: Configuration for FaceNet model
            enable_anti_spoofing: Whether to enable anti-spoofing detection
            enable_voting: Whether to enable voting system for multi-image processing
        """
        self.model_config = model_config or ModelConfig()
        self.enable_anti_spoofing = enable_anti_spoofing
        self.enable_voting = enable_voting
        
        # Initialize components
        self.face_model = None
        self.preprocessor = None
        self.anti_spoofing_detector = None
        self.voting_system = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all inference components"""
        try:
            # Initialize face model
            self.face_model = FaceNetModel(self.model_config)
            logger.info("FaceNet model initialized")
            
            # Initialize preprocessor
            self.preprocessor = FacePreprocessor(
                confidence_threshold=0.7,
                target_size=self.model_config.input_size
            )
            logger.info("Face preprocessor initialized")
            
            # Initialize anti-spoofing detector
            if self.enable_anti_spoofing:
                self.anti_spoofing_detector = AntiSpoofingDetector()
                logger.info("Anti-spoofing detector initialized")
            
            # Initialize voting system
            if self.enable_voting:
                self.voting_system = VotingSystem()
                logger.info("Voting system initialized")
                
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def load_model(self, model_path: Optional[str] = None):
        """Load the FaceNet model"""
        if self.face_model is None:
            raise ValueError("Face model not initialized")
        
        self.face_model.load_model(model_path)
        logger.info("FaceNet model loaded successfully")
    
    def process_single_image(self, image: np.ndarray) -> InferenceResult:
        """
        Process a single image for face recognition
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Inference result with embedding and anti-spoofing information
        """
        start_time = time.time()
        
        try:
            # Detect and preprocess faces
            preprocessed_faces = self.preprocessor.process_image(image)
            
            if not preprocessed_faces:
                return InferenceResult(
                    success=False,
                    embedding=None,
                    confidence=0.0,
                    is_real=False,
                    anti_spoofing_result=None,
                    processing_time=time.time() - start_time,
                    error_message="No faces detected in image"
                )
            
            # Use the first detected face
            face_image = preprocessed_faces[0]
            
            # Perform anti-spoofing detection
            anti_spoofing_result = None
            is_real = True
            if self.enable_anti_spoofing and self.anti_spoofing_detector:
                anti_spoofing_result = self.anti_spoofing_detector.detect_spoofing(face_image)
                is_real = anti_spoofing_result.is_real
                
                if not is_real:
                    return InferenceResult(
                        success=False,
                        embedding=None,
                        confidence=anti_spoofing_result.confidence,
                        is_real=False,
                        anti_spoofing_result=anti_spoofing_result,
                        processing_time=time.time() - start_time,
                        error_message=f"Spoofing detected: {anti_spoofing_result.reason}"
                    )
            
            # Extract face embedding
            embedding = self.face_model.extract_embedding(face_image)
            
            # Calculate confidence (based on anti-spoofing if available)
            confidence = anti_spoofing_result.confidence if anti_spoofing_result else 0.9
            
            return InferenceResult(
                success=True,
                embedding=embedding,
                confidence=confidence,
                is_real=is_real,
                anti_spoofing_result=anti_spoofing_result,
                processing_time=time.time() - start_time,
                error_message=None
            )
            
        except Exception as e:
            logger.error(f"Error in single image processing: {e}")
            return InferenceResult(
                success=False,
                embedding=None,
                confidence=0.0,
                is_real=False,
                anti_spoofing_result=None,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def process_multiple_images(self, images: List[np.ndarray]) -> MultiImageResult:
        """
        Process multiple images for student insertion with voting
        
        Args:
            images: List of input images (typically 3 for student insertion)
            
        Returns:
            Multi-image processing result with consensus embedding
        """
        start_time = time.time()
        
        try:
            if len(images) != 3:
                return MultiImageResult(
                    success=False,
                    consensus_embedding=None,
                    voting_result=None,
                    individual_results=[],
                    total_processing_time=time.time() - start_time,
                    error_message=f"Expected 3 images, got {len(images)}"
                )
            
            # Process each image individually
            individual_results = []
            valid_embeddings = []
            
            for i, image in enumerate(images):
                result = self.process_single_image(image)
                individual_results.append(result)
                
                if result.success and result.embedding is not None:
                    valid_embeddings.append(result.embedding)
                else:
                    logger.warning(f"Image {i+1} processing failed: {result.error_message}")
            
            # Check if we have enough valid embeddings
            if len(valid_embeddings) < 2:
                return MultiImageResult(
                    success=False,
                    consensus_embedding=None,
                    voting_result=None,
                    individual_results=individual_results,
                    total_processing_time=time.time() - start_time,
                    error_message=f"Insufficient valid embeddings: {len(valid_embeddings)}/3"
                )
            
            # Apply voting system if enabled
            voting_result = None
            consensus_embedding = None
            
            if self.enable_voting and self.voting_system:
                if len(valid_embeddings) == 3:
                    voting_result = self.voting_system.process_three_images(valid_embeddings)
                else:
                    voting_result = self.voting_system.validate_consistency(valid_embeddings)
                
                if voting_result.is_consistent:
                    consensus_embedding = voting_result.consensus_embedding
                else:
                    return MultiImageResult(
                        success=False,
                        consensus_embedding=None,
                        voting_result=voting_result,
                        individual_results=individual_results,
                        total_processing_time=time.time() - start_time,
                        error_message=f"Inconsistent faces detected: {voting_result.reason}"
                    )
            else:
                # Simple average if voting is disabled
                consensus_embedding = np.mean(valid_embeddings, axis=0)
                # Normalize
                norm = np.linalg.norm(consensus_embedding)
                if norm > 0:
                    consensus_embedding = consensus_embedding / norm
            
            return MultiImageResult(
                success=True,
                consensus_embedding=consensus_embedding,
                voting_result=voting_result,
                individual_results=individual_results,
                total_processing_time=time.time() - start_time,
                error_message=None
            )
            
        except Exception as e:
            logger.error(f"Error in multi-image processing: {e}")
            return MultiImageResult(
                success=False,
                consensus_embedding=None,
                voting_result=None,
                individual_results=[],
                total_processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def extract_embedding_only(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract embedding without anti-spoofing (for performance)
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Face embedding or None if extraction fails
        """
        try:
            # Detect and preprocess faces
            preprocessed_faces = self.preprocessor.process_image(image)
            
            if not preprocessed_faces:
                return None
            
            # Extract embedding from first face
            embedding = self.face_model.extract_embedding(preprocessed_faces[0])
            return embedding
            
        except Exception as e:
            logger.error(f"Error in embedding extraction: {e}")
            return None
    
    def validate_face_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Validate face image quality
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Quality validation result
        """
        try:
            preprocessed_faces = self.preprocessor.process_image(image)
            
            if not preprocessed_faces:
                return {
                    "is_valid": False,
                    "error": "No faces detected"
                }
            
            return self.preprocessor.validate_face_quality(preprocessed_faces[0])
            
        except Exception as e:
            logger.error(f"Error in quality validation: {e}")
            return {
                "is_valid": False,
                "error": str(e)
            }
    
    def benchmark_performance(self, test_image: np.ndarray, num_iterations: int = 10) -> Dict[str, Any]:
        """
        Benchmark inference performance on mobile CPU
        
        Args:
            test_image: Test image for benchmarking
            num_iterations: Number of iterations for averaging
            
        Returns:
            Performance benchmark results
        """
        try:
            times = {
                "face_detection": [],
                "anti_spoofing": [],
                "embedding_extraction": [],
                "total": []
            }
            
            for _ in range(num_iterations):
                # Face detection timing
                start = time.time()
                preprocessed_faces = self.preprocessor.process_image(test_image)
                times["face_detection"].append(time.time() - start)
                
                if not preprocessed_faces:
                    continue
                
                face_image = preprocessed_faces[0]
                
                # Anti-spoofing timing
                if self.enable_anti_spoofing and self.anti_spoofing_detector:
                    start = time.time()
                    self.anti_spoofing_detector.detect_spoofing(face_image)
                    times["anti_spoofing"].append(time.time() - start)
                
                # Embedding extraction timing
                start = time.time()
                self.face_model.extract_embedding(face_image)
                times["embedding_extraction"].append(time.time() - start)
                
                # Total timing
                start = time.time()
                self.process_single_image(test_image)
                times["total"].append(time.time() - start)
            
            # Calculate statistics
            stats = {}
            for component, time_list in times.items():
                if time_list:
                    stats[component] = {
                        "mean": np.mean(time_list),
                        "std": np.std(time_list),
                        "min": np.min(time_list),
                        "max": np.max(time_list)
                    }
            
            return {
                "num_iterations": num_iterations,
                "timing_stats": stats,
                "fps_estimate": 1.0 / stats["total"]["mean"] if "total" in stats else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in performance benchmark: {e}")
            return {"error": str(e)}
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get inference engine information and status"""
        return {
            "model_config": {
                "embedding_dim": self.model_config.embedding_dim,
                "input_size": self.model_config.input_size,
                "cpu_threads": self.model_config.cpu_threads,
                "quantization": self.model_config.quantization
            },
            "components": {
                "face_model_loaded": self.face_model is not None and self.face_model.model is not None,
                "preprocessor_loaded": self.preprocessor is not None,
                "anti_spoofing_enabled": self.enable_anti_spoofing,
                "voting_enabled": self.enable_voting
            },
            "model_info": self.face_model.get_model_info() if self.face_model else {}
        } 