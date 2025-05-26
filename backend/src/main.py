from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.endpoints import recognition, students
from .core.config import settings
from .core.database import engine, Base
from .models import student  # Import to register the model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    recognition.router,
    prefix=f"{settings.API_V1_STR}/recognition",
    tags=["recognition"]
)

app.include_router(
    students.router,
    prefix=f"{settings.API_V1_STR}/students",
    tags=["students"]
)

@app.get("/")
async def root():
    return {"message": "Face Recognition API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 