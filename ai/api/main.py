from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import cv2
import logging
import time
from io import BytesIO

from ..inference.engine import InferenceEngine
from ..models.facenet_model import ModelConfig

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Attendity AI Module",
    description="Face recognition and anti-spoofing API optimized for mobile CPU",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine
inference_engine: Optional[InferenceEngine] = None

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    confidence: float
    is_real: bool
    processing_time: float
    anti_spoofing_details: Optional[Dict[str, Any]] = None

class MultiImageResponse(BaseModel):
    consensus_embedding: List[float]
    individual_results: List[Dict[str, Any]]
    voting_result: Dict[str, Any]
    total_processing_time: float
    status: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    cpu_optimized: bool
    components_status: Dict[str, bool]
    system_info: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Initialize the inference engine on startup"""
    global inference_engine
    try:
        config = ModelConfig(
            embedding_dim=512,
            cpu_threads=4,
            quantization=True
        )
        
        inference_engine = InferenceEngine(
            model_config=config,
            enable_anti_spoofing=True,
            enable_voting=True
        )
        
        logger.info("AI inference engine initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        raise

def load_image_from_upload(file: UploadFile) -> np.ndarray:
    """Load image from uploaded file"""
    try:
        # Read file content
        content = file.file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(content, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image")
        
        return image
        
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

@app.post("/inference", response_model=EmbeddingResponse)
async def extract_embedding(file: UploadFile = File(...)):
    """
    Extract face embedding from single image with anti-spoofing
    
    - **file**: Image file (JPEG, PNG)
    - Returns face embedding, confidence, and anti-spoofing result
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        # Load image
        image = load_image_from_upload(file)
        
        # Process image
        result = inference_engine.process_single_image(image)
        
        if not result.success:
            raise HTTPException(
                status_code=400, 
                detail=f"Processing failed: {result.error_message}"
            )
        
        # Prepare anti-spoofing details
        anti_spoofing_details = None
        if result.anti_spoofing_result:
            anti_spoofing_details = {
                "is_real": result.anti_spoofing_result.is_real,
                "confidence": result.anti_spoofing_result.confidence,
                "reason": result.anti_spoofing_result.reason,
                "scores": result.anti_spoofing_result.scores
            }
        
        return EmbeddingResponse(
            embedding=result.embedding.tolist(),
            confidence=result.confidence,
            is_real=result.is_real,
            processing_time=result.processing_time,
            anti_spoofing_details=anti_spoofing_details
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in inference endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/insert_student", response_model=MultiImageResponse)
async def insert_student(
    class_code: str = Form(...),
    student_id: str = Form(...),
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    image3: UploadFile = File(...)
):
    """
    Process 3 images for student insertion with voting mechanism
    
    - **class_code**: Class identifier
    - **student_id**: Student identifier  
    - **image1, image2, image3**: Three face images for consensus
    - Returns consensus embedding if faces are consistent
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        # Load all three images
        images = []
        for i, file in enumerate([image1, image2, image3], 1):
            try:
                image = load_image_from_upload(file)
                images.append(image)
            except Exception as e:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Error loading image {i}: {str(e)}"
                )
        
        # Process multiple images
        result = inference_engine.process_multiple_images(images)
        
        # Prepare individual results
        individual_results = []
        for i, individual_result in enumerate(result.individual_results):
            individual_data = {
                "image_index": i + 1,
                "success": individual_result.success,
                "confidence": individual_result.confidence,
                "is_real": individual_result.is_real,
                "processing_time": individual_result.processing_time
            }
            
            if individual_result.error_message:
                individual_data["error"] = individual_result.error_message
            
            if individual_result.anti_spoofing_result:
                individual_data["anti_spoofing"] = {
                    "is_real": individual_result.anti_spoofing_result.is_real,
                    "confidence": individual_result.anti_spoofing_result.confidence,
                    "reason": individual_result.anti_spoofing_result.reason
                }
            
            individual_results.append(individual_data)
        
        # Prepare voting result
        voting_data = {}
        if result.voting_result:
            voting_data = {
                "is_consistent": result.voting_result.is_consistent,
                "confidence": result.voting_result.confidence,
                "reason": result.voting_result.reason,
                "individual_scores": result.voting_result.individual_scores,
                "similarity_matrix": result.voting_result.similarity_matrix.tolist() if result.voting_result.similarity_matrix.size > 0 else []
            }
        
        if result.success:
            return MultiImageResponse(
                consensus_embedding=result.consensus_embedding.tolist(),
                individual_results=individual_results,
                voting_result=voting_data,
                total_processing_time=result.total_processing_time,
                status="success"
            )
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "failed",
                    "reason": result.error_message,
                    "individual_results": individual_results,
                    "voting_result": voting_data
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in insert_student endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/extract_embedding_fast")
async def extract_embedding_fast(file: UploadFile = File(...)):
    """
    Fast embedding extraction without anti-spoofing (for performance)
    
    - **file**: Image file
    - Returns only the face embedding
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        # Load image
        image = load_image_from_upload(file)
        
        # Extract embedding only
        embedding = inference_engine.extract_embedding_only(image)
        
        if embedding is None:
            raise HTTPException(status_code=400, detail="Could not extract embedding")
        
        return {
            "embedding": embedding.tolist(),
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in fast embedding endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/validate_quality")
async def validate_face_quality(file: UploadFile = File(...)):
    """
    Validate face image quality for mobile constraints
    
    - **file**: Image file
    - Returns quality metrics and validation result
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        # Load image
        image = load_image_from_upload(file)
        
        # Validate quality
        quality_result = inference_engine.validate_face_quality(image)
        
        return quality_result
        
    except Exception as e:
        logger.error(f"Error in quality validation endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/benchmark")
async def benchmark_performance(file: UploadFile = File(...), iterations: int = 10):
    """
    Benchmark inference performance on mobile CPU
    
    - **file**: Test image file
    - **iterations**: Number of iterations for averaging
    - Returns performance metrics
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        # Load image
        image = load_image_from_upload(file)
        
        # Run benchmark
        benchmark_result = inference_engine.benchmark_performance(image, iterations)
        
        return benchmark_result
        
    except Exception as e:
        logger.error(f"Error in benchmark endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check system health and model loading status
    """
    try:
        if inference_engine is None:
            return HealthResponse(
                status="unhealthy",
                model_loaded=False,
                cpu_optimized=False,
                components_status={},
                system_info={}
            )
        
        # Get engine info
        engine_info = inference_engine.get_engine_info()
        
        return HealthResponse(
            status="healthy",
            model_loaded=engine_info["components"]["face_model_loaded"],
            cpu_optimized=engine_info["model_config"]["quantization"],
            components_status=engine_info["components"],
            system_info={
                "cpu_threads": engine_info["model_config"]["cpu_threads"],
                "embedding_dim": engine_info["model_config"]["embedding_dim"],
                "input_size": engine_info["model_config"]["input_size"]
            }
        )
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return HealthResponse(
            status="error",
            model_loaded=False,
            cpu_optimized=False,
            components_status={},
            system_info={"error": str(e)}
        )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Attendity AI Module API",
        "version": "1.0.0",
        "description": "Face recognition and anti-spoofing API optimized for mobile CPU",
        "endpoints": {
            "/inference": "Extract face embedding with anti-spoofing",
            "/insert_student": "Process 3 images for student insertion",
            "/extract_embedding_fast": "Fast embedding extraction",
            "/validate_quality": "Validate face image quality",
            "/benchmark": "Performance benchmarking",
            "/health": "Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 