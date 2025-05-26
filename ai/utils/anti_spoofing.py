import cv2
import numpy as np
from typing import Dict, Any, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AntiSpoofingResult:
    """Result of anti-spoofing detection"""
    is_real: bool
    confidence: float
    scores: Dict[str, float]
    reason: str

class AntiSpoofingDetector:
    """
    Lightweight rule-based anti-spoofing detection optimized for mobile CPU
    Uses texture analysis, color space analysis, and basic liveness detection
    """
    
    def __init__(self,
                 texture_threshold: float = 0.3,
                 color_threshold: float = 0.4,
                 quality_threshold: float = 0.5,
                 combined_threshold: float = 0.6):
        """
        Initialize anti-spoofing detector
        
        Args:
            texture_threshold: Threshold for texture analysis
            color_threshold: Threshold for color analysis
            quality_threshold: Threshold for quality analysis
            combined_threshold: Final threshold for classification
        """
        self.texture_threshold = texture_threshold
        self.color_threshold = color_threshold
        self.quality_threshold = quality_threshold
        self.combined_threshold = combined_threshold
        
    def analyze_texture(self, face_image: np.ndarray) -> Dict[str, float]:
        """
        Analyze texture patterns using Local Binary Patterns (LBP)
        
        Args:
            face_image: Preprocessed face image (RGB)
            
        Returns:
            Texture analysis scores
        """
        try:
            # Convert to grayscale
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_image
            
            # Calculate LBP (simplified version for mobile)
            lbp = self._calculate_simple_lbp(gray)
            
            # Calculate texture uniformity
            hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()
            
            # Texture metrics
            uniformity = np.sum(hist ** 2)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            avg_gradient = np.mean(gradient_magnitude)
            
            return {
                "uniformity": float(uniformity),
                "entropy": float(entropy),
                "edge_density": float(edge_density),
                "avg_gradient": float(avg_gradient)
            }
            
        except Exception as e:
            logger.error(f"Error in texture analysis: {e}")
            return {"uniformity": 0.0, "entropy": 0.0, "edge_density": 0.0, "avg_gradient": 0.0}
    
    def _calculate_simple_lbp(self, gray_image: np.ndarray) -> np.ndarray:
        """Calculate simplified Local Binary Pattern for mobile optimization"""
        height, width = gray_image.shape
        lbp = np.zeros((height-2, width-2), dtype=np.uint8)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                center = gray_image[i, j]
                code = 0
                
                # 8-neighbor LBP
                neighbors = [
                    gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                    gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                    gray_image[i+1, j-1], gray_image[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp[i-1, j-1] = code
        
        return lbp
    
    def analyze_color_distribution(self, face_image: np.ndarray) -> Dict[str, float]:
        """
        Analyze color distribution in different color spaces
        
        Args:
            face_image: Preprocessed face image (RGB)
            
        Returns:
            Color analysis scores
        """
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(face_image, cv2.COLOR_RGB2HSV)
            yuv = cv2.cvtColor(face_image, cv2.COLOR_RGB2YUV)
            
            # RGB channel analysis
            r_mean, g_mean, b_mean = np.mean(face_image, axis=(0, 1))
            r_std, g_std, b_std = np.std(face_image, axis=(0, 1))
            
            # HSV analysis
            h_mean, s_mean, v_mean = np.mean(hsv, axis=(0, 1))
            h_std, s_std, v_std = np.std(hsv, axis=(0, 1))
            
            # YUV analysis
            y_mean, u_mean, v_yuv_mean = np.mean(yuv, axis=(0, 1))
            
            # Color distribution metrics
            rgb_balance = min(r_mean, g_mean, b_mean) / max(r_mean, g_mean, b_mean)
            saturation_variance = s_std / (s_mean + 1e-10)
            
            # Skin color likelihood (simplified)
            skin_score = self._calculate_skin_likelihood(r_mean, g_mean, b_mean)
            
            return {
                "rgb_balance": float(rgb_balance),
                "saturation_variance": float(saturation_variance),
                "skin_score": float(skin_score),
                "hue_mean": float(h_mean),
                "saturation_mean": float(s_mean)
            }
            
        except Exception as e:
            logger.error(f"Error in color analysis: {e}")
            return {
                "rgb_balance": 0.0, "saturation_variance": 0.0, 
                "skin_score": 0.0, "hue_mean": 0.0, "saturation_mean": 0.0
            }
    
    def _calculate_skin_likelihood(self, r: float, g: float, b: float) -> float:
        """Calculate likelihood of skin color (simplified model)"""
        # Simplified skin color model
        if r > 95 and g > 40 and b > 20:
            if max(r, g, b) - min(r, g, b) > 15:
                if abs(r - g) > 15 and r > g and r > b:
                    return 1.0
        return 0.0
    
    def analyze_image_quality(self, face_image: np.ndarray) -> Dict[str, float]:
        """
        Analyze image quality metrics for liveness detection
        
        Args:
            face_image: Preprocessed face image (RGB)
            
        Returns:
            Quality analysis scores
        """
        try:
            # Convert to grayscale
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_image
            
            # Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Brightness analysis
            brightness = np.mean(gray)
            brightness_std = np.std(gray)
            
            # Contrast analysis
            contrast = np.std(gray)
            
            # Noise estimation (using high-frequency components)
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            noise_response = cv2.filter2D(gray, cv2.CV_64F, kernel)
            noise_level = np.std(noise_response)
            
            # Pixel variance in local regions
            local_variance = self._calculate_local_variance(gray)
            
            # Motion blur detection (simplified)
            motion_blur_score = self._detect_motion_blur(gray)
            
            return {
                "sharpness": float(laplacian_var),
                "brightness": float(brightness),
                "contrast": float(contrast),
                "noise_level": float(noise_level),
                "local_variance": float(local_variance),
                "motion_blur": float(motion_blur_score)
            }
            
        except Exception as e:
            logger.error(f"Error in quality analysis: {e}")
            return {
                "sharpness": 0.0, "brightness": 0.0, "contrast": 0.0,
                "noise_level": 0.0, "local_variance": 0.0, "motion_blur": 0.0
            }
    
    def _calculate_local_variance(self, gray_image: np.ndarray, window_size: int = 8) -> float:
        """Calculate average local variance"""
        height, width = gray_image.shape
        variances = []
        
        for i in range(0, height - window_size, window_size):
            for j in range(0, width - window_size, window_size):
                window = gray_image[i:i+window_size, j:j+window_size]
                variances.append(np.var(window))
        
        return np.mean(variances) if variances else 0.0
    
    def _detect_motion_blur(self, gray_image: np.ndarray) -> float:
        """Detect motion blur using FFT analysis (simplified)"""
        try:
            # Apply FFT
            f_transform = np.fft.fft2(gray_image)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Calculate high frequency content
            height, width = magnitude_spectrum.shape
            center_y, center_x = height // 2, width // 2
            
            # Create high-pass filter
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 > (min(height, width) // 4)**2
            
            high_freq_content = np.mean(magnitude_spectrum[mask])
            
            # Normalize to [0, 1] range
            return min(high_freq_content / 10.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error in motion blur detection: {e}")
            return 0.0
    
    def detect_spoofing(self, face_image: np.ndarray) -> AntiSpoofingResult:
        """
        Main anti-spoofing detection function
        
        Args:
            face_image: Preprocessed face image (RGB)
            
        Returns:
            Anti-spoofing detection result
        """
        try:
            # Perform all analyses
            texture_scores = self.analyze_texture(face_image)
            color_scores = self.analyze_color_distribution(face_image)
            quality_scores = self.analyze_image_quality(face_image)
            
            # Calculate individual component scores
            texture_score = self._calculate_texture_score(texture_scores)
            color_score = self._calculate_color_score(color_scores)
            quality_score = self._calculate_quality_score(quality_scores)
            
            # Combine scores with weights
            weights = {"texture": 0.4, "color": 0.3, "quality": 0.3}
            combined_score = (
                weights["texture"] * texture_score +
                weights["color"] * color_score +
                weights["quality"] * quality_score
            )
            
            # Make final decision
            is_real = combined_score >= self.combined_threshold
            
            # Determine reason if classified as fake
            reason = "real_face"
            if not is_real:
                if texture_score < self.texture_threshold:
                    reason = "poor_texture_quality"
                elif color_score < self.color_threshold:
                    reason = "unnatural_color_distribution"
                elif quality_score < self.quality_threshold:
                    reason = "poor_image_quality"
                else:
                    reason = "combined_score_too_low"
            
            all_scores = {
                "texture_score": texture_score,
                "color_score": color_score,
                "quality_score": quality_score,
                "combined_score": combined_score,
                **texture_scores,
                **color_scores,
                **quality_scores
            }
            
            return AntiSpoofingResult(
                is_real=is_real,
                confidence=combined_score,
                scores=all_scores,
                reason=reason
            )
            
        except Exception as e:
            logger.error(f"Error in spoofing detection: {e}")
            return AntiSpoofingResult(
                is_real=False,
                confidence=0.0,
                scores={},
                reason=f"detection_error: {str(e)}"
            )
    
    def _calculate_texture_score(self, texture_scores: Dict[str, float]) -> float:
        """Calculate normalized texture score"""
        try:
            # Normalize individual metrics
            uniformity = min(texture_scores["uniformity"] * 2, 1.0)
            entropy = min(texture_scores["entropy"] / 8.0, 1.0)
            edge_density = min(texture_scores["edge_density"] * 10, 1.0)
            gradient = min(texture_scores["avg_gradient"] / 50.0, 1.0)
            
            # Combine with weights
            score = 0.3 * uniformity + 0.3 * entropy + 0.2 * edge_density + 0.2 * gradient
            return float(score)
            
        except Exception:
            return 0.0
    
    def _calculate_color_score(self, color_scores: Dict[str, float]) -> float:
        """Calculate normalized color score"""
        try:
            rgb_balance = color_scores["rgb_balance"]
            saturation_var = min(color_scores["saturation_variance"], 1.0)
            skin_score = color_scores["skin_score"]
            
            # Combine with weights
            score = 0.4 * rgb_balance + 0.3 * skin_score + 0.3 * (1.0 - saturation_var)
            return float(score)
            
        except Exception:
            return 0.0
    
    def _calculate_quality_score(self, quality_scores: Dict[str, float]) -> float:
        """Calculate normalized quality score"""
        try:
            sharpness = min(quality_scores["sharpness"] / 100.0, 1.0)
            brightness = 1.0 - abs(quality_scores["brightness"] - 128) / 128.0
            contrast = min(quality_scores["contrast"] / 50.0, 1.0)
            noise = max(0.0, 1.0 - quality_scores["noise_level"] / 10.0)
            local_var = min(quality_scores["local_variance"] / 100.0, 1.0)
            motion = quality_scores["motion_blur"]
            
            # Combine with weights
            score = (0.25 * sharpness + 0.15 * brightness + 0.2 * contrast + 
                    0.15 * noise + 0.15 * local_var + 0.1 * motion)
            return float(score)
            
        except Exception:
            return 0.0
    
    def get_detector_config(self) -> Dict[str, Any]:
        """Get detector configuration"""
        return {
            "texture_threshold": self.texture_threshold,
            "color_threshold": self.color_threshold,
            "quality_threshold": self.quality_threshold,
            "combined_threshold": self.combined_threshold
        } 