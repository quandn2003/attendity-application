"""
AI Module for Mobile Attendance System
Provides face recognition, anti-spoofing, and inference capabilities optimized for mobile CPU deployment.
"""

__version__ = "1.0.0"
__author__ = "Attendity Team"

from ai.inference.engine import InferenceEngine
from ai.models.facenet_model import FaceNetModel
from ai.utils.preprocessing import FacePreprocessor
from ai.utils.anti_spoofing import AntiSpoofingDetector

__all__ = [
    "InferenceEngine",
    "FaceNetModel", 
    "FacePreprocessor",
    "AntiSpoofingDetector"
] 