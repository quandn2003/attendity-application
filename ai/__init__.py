"""
AI Module for Mobile Attendance System
Provides face recognition, anti-spoofing, and inference capabilities optimized for mobile CPU deployment.
"""

__version__ = "1.0.0"
__author__ = "Attendity Team"

from .inference.engine import InferenceEngine
from .models.facenet_model import FaceNetModel
from .utils.preprocessing import FacePreprocessor
from .utils.anti_spoofing import AntiSpoofingDetector

__all__ = [
    "InferenceEngine",
    "FaceNetModel", 
    "FacePreprocessor",
    "AntiSpoofingDetector"
] 