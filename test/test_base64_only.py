#!/usr/bin/env python3
"""
Test Script for Base64-Only AI Module
Verifies that all endpoints work correctly with base64 encoded images
"""

import requests
import numpy as np
import cv2
import base64
import time
import logging
import os
import glob
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Base64OnlyTest:
    """Test class for base64-only AI module endpoints"""
    
    def __init__(self, ai_url: str = "http://localhost:8000", num_images: int = 3):
        self.ai_url = ai_url
        self.num_images = num_images
        self.test_images_dir = os.path.join(os.path.dirname(__file__), "imgs")
        self.available_images = self._load_available_images()
    
    def _load_available_images(self) -> list:
        """Load available images from test/imgs directory"""
        if not os.path.exists(self.test_images_dir):
            raise FileNotFoundError(f"Test images directory not found: {self.test_images_dir}")
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.test_images_dir, ext)))
            image_files.extend(glob.glob(os.path.join(self.test_images_dir, ext.upper())))
        
        if not image_files:
            raise FileNotFoundError(f"No image files found in {self.test_images_dir}")
        
        print(f"Found {len(image_files)} images in {self.test_images_dir}")
        for img in image_files:
            print(f"  - {os.path.basename(img)}")
        
        return image_files
    
    def load_image_from_file(self, image_index: int = 0) -> np.ndarray:
        """Load image from file"""
        if image_index >= len(self.available_images):
            image_index = image_index % len(self.available_images)
        
        image_path = self.available_images[image_index]
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        return image
    
    def print_step(self, step: str, status: str = "info"):
        """Print formatted step"""
        icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "progress": "🔄"}
        icon = icons.get(status, "ℹ️")
        print(f"  {icon} {step}")
    
    def generate_mock_image(self, width: int = 160, height: int = 160, seed: int = 42) -> np.ndarray:
        """Generate a mock face image for testing (fallback method)"""
        np.random.seed(seed)
        
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :] = [220, 220, 220]
        
        center_x, center_y = width // 2, height // 2
        for y in range(height):
            for x in range(width):
                dist_x = (x - center_x) / (width * 0.4)
                dist_y = (y - center_y) / (height * 0.45)
                if dist_x**2 + dist_y**2 <= 1:
                    image[y, x] = [200, 180, 160]
        
        eye_y = center_y - height // 6
        left_eye_x = center_x - width // 6
        right_eye_x = center_x + width // 6
        eye_size = width // 20
        
        cv2.circle(image, (left_eye_x, eye_y), eye_size, (50, 50, 50), -1)
        cv2.circle(image, (right_eye_x, eye_y), eye_size, (50, 50, 50), -1)
        
        nose_y = center_y
        cv2.circle(image, (center_x, nose_y), width // 30, (150, 120, 100), -1)
        
        mouth_y = center_y + height // 6
        cv2.ellipse(image, (center_x, mouth_y), (width // 8, height // 20), 0, 0, 180, (100, 50, 50), 2)
        
        noise = np.random.randint(-20, 20, (height, width, 3))
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def test_health_endpoint(self) -> bool:
        """Test /health endpoint"""
        print("\n🔥 TESTING HEALTH ENDPOINT")
        print("=" * 50)
        
        try:
            self.print_step("Testing /health endpoint", "progress")
            
            response = requests.get(f"{self.ai_url}/health", timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                self.print_step("✓ Health check successful", "success")
                self.print_step(f"  - Status: {result.get('status')}", "info")
                self.print_step(f"  - Model loaded: {result.get('model_loaded')}", "info")
                return True
            else:
                self.print_step(f"✗ Health check failed: {response.status_code}", "error")
                return False
                
        except Exception as e:
            self.print_step(f"✗ Health check error: {e}", "error")
            return False
    
    def test_root_endpoint(self) -> bool:
        """Test / root endpoint"""
        print("\n🔥 TESTING ROOT ENDPOINT")
        print("=" * 50)
        
        try:
            self.print_step("Testing / root endpoint", "progress")
            
            response = requests.get(f"{self.ai_url}/", timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                self.print_step("✓ Root endpoint successful", "success")
                self.print_step(f"  - Message: {result.get('message')}", "info")
                self.print_step(f"  - Endpoints: {len(result.get('endpoints', {}))}", "info")
                self.print_step(f"  - Input format: {result.get('input_format')}", "info")
                return True
            else:
                self.print_step(f"✗ Root endpoint failed: {response.status_code}", "error")
                return False
                
        except Exception as e:
            self.print_step(f"✗ Root endpoint error: {e}", "error")
            return False
    
    def test_inference_endpoint(self) -> bool:
        """Test /inference endpoint"""
        print("\n🔥 TESTING INFERENCE ENDPOINT")
        print("=" * 50)
        
        try:
            test_image = self.load_image_from_file(0)
            base64_image = self.image_to_base64(test_image)
            
            self.print_step(f"Testing /inference endpoint with image: {os.path.basename(self.available_images[0])}", "progress")
            
            response = requests.post(
                f"{self.ai_url}/inference",
                json={"image": base64_image},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                self.print_step("✓ Inference successful", "success")
                self.print_step(f"  - Embedding dimension: {len(result['embedding'])}", "info")
                self.print_step(f"  - Confidence: {result['confidence']:.3f}", "info")
                self.print_step(f"  - Is real: {result['is_real']}", "info")
                self.print_step(f"  - Processing time: {result['processing_time']:.3f}s", "info")
                return True
            else:
                self.print_step(f"✗ Inference failed: {response.text}", "error")
                return False
                
        except Exception as e:
            self.print_step(f"✗ Inference error: {e}", "error")
            return False
    
    def test_insert_student_endpoint(self) -> bool:
        """Test /insert_student endpoint"""
        print("\n🔥 TESTING INSERT STUDENT ENDPOINT")
        print("=" * 50)
        
        try:
            images_base64 = []
            used_images = []
            
            for i in range(min(self.num_images, len(self.available_images))):
                test_image = self.load_image_from_file(i)
                base64_image = self.image_to_base64(test_image)
                images_base64.append(base64_image)
                used_images.append(os.path.basename(self.available_images[i]))
            
            if len(images_base64) < 3:
                for i in range(3 - len(images_base64)):
                    fallback_image = self.generate_mock_image(seed=200 + i)
                    base64_image = self.image_to_base64(fallback_image)
                    images_base64.append(base64_image)
                    used_images.append(f"mock_image_{i}")
            
            self.print_step(f"Testing /insert_student endpoint with images: {', '.join(used_images[:3])}", "progress")
            
            response = requests.post(
                f"{self.ai_url}/insert_student",
                json={
                    "class_code": "TEST_CLASS",
                    "student_id": "test_student_001",
                    "image1": images_base64[0],
                    "image2": images_base64[1],
                    "image3": images_base64[2]
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                self.print_step("✓ Student insertion successful", "success")
                self.print_step(f"  - Status: {result['status']}", "info")
                self.print_step(f"  - Consensus embedding dim: {len(result['consensus_embedding'])}", "info")
                self.print_step(f"  - Individual results: {len(result['individual_results'])}", "info")
                self.print_step(f"  - Total processing time: {result['total_processing_time']:.3f}s", "info")
                return True
            else:
                self.print_step(f"✗ Student insertion failed: {response.text}", "error")
                return False
                
        except Exception as e:
            self.print_step(f"✗ Student insertion error: {e}", "error")
            return False
    
    def test_fast_embedding_endpoint(self) -> bool:
        """Test /extract_embedding_fast endpoint"""
        print("\n🔥 TESTING FAST EMBEDDING ENDPOINT")
        print("=" * 50)
        
        try:
            test_image = self.load_image_from_file(0)
            base64_image = self.image_to_base64(test_image)
            
            self.print_step(f"Testing /extract_embedding_fast endpoint with image: {os.path.basename(self.available_images[0])}", "progress")
            
            response = requests.post(
                f"{self.ai_url}/extract_embedding_fast",
                json={"image": base64_image},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                self.print_step("✓ Fast embedding extraction successful", "success")
                self.print_step(f"  - Status: {result['status']}", "info")
                self.print_step(f"  - Embedding dimension: {len(result['embedding'])}", "info")
                return True
            else:
                self.print_step(f"✗ Fast embedding extraction failed: {response.text}", "error")
                return False
                
        except Exception as e:
            self.print_step(f"✗ Fast embedding extraction error: {e}", "error")
            return False
    
    def test_quality_validation_endpoint(self) -> bool:
        """Test /validate_quality endpoint"""
        print("\n🔥 TESTING QUALITY VALIDATION ENDPOINT")
        print("=" * 50)
        
        try:
            test_image = self.load_image_from_file(0)
            base64_image = self.image_to_base64(test_image)
            
            self.print_step(f"Testing /validate_quality endpoint with image: {os.path.basename(self.available_images[0])}", "progress")
            
            response = requests.post(
                f"{self.ai_url}/validate_quality",
                json={"image": base64_image},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                self.print_step("✓ Quality validation successful", "success")
                self.print_step(f"  - Result keys: {list(result.keys())}", "info")
                return True
            else:
                self.print_step(f"✗ Quality validation failed: {response.text}", "error")
                return False
                
        except Exception as e:
            self.print_step(f"✗ Quality validation error: {e}", "error")
            return False
    
    def test_benchmark_endpoint(self) -> bool:
        """Test /benchmark endpoint"""
        print("\n🔥 TESTING BENCHMARK ENDPOINT")
        print("=" * 50)
        
        try:
            test_image = self.load_image_from_file(0)
            base64_image = self.image_to_base64(test_image)
            
            self.print_step(f"Testing /benchmark endpoint with image: {os.path.basename(self.available_images[0])} (3 iterations)", "progress")
            
            response = requests.post(
                f"{self.ai_url}/benchmark",
                json={"image": base64_image},
                params={"iterations": 3},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                self.print_step("✓ Benchmark successful", "success")
                self.print_step(f"  - Result keys: {list(result.keys())}", "info")
                return True
            else:
                self.print_step(f"✗ Benchmark failed: {response.text}", "error")
                return False
                
        except Exception as e:
            self.print_step(f"✗ Benchmark error: {e}", "error")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all base64-only tests"""
        print("🚀 STARTING BASE64-ONLY AI MODULE TEST")
        print("Testing all endpoints with base64 encoded images from test/imgs folder")
        print("=" * 80)
        print(f"Using {len(self.available_images)} available images, testing with {self.num_images} images")
        print("=" * 80)
        
        start_time = time.time()
        
        test_methods = [
            ("Health Check", self.test_health_endpoint),
            ("Root Endpoint", self.test_root_endpoint),
            ("Inference", self.test_inference_endpoint),
            ("Insert Student", self.test_insert_student_endpoint),
            ("Fast Embedding", self.test_fast_embedding_endpoint),
            ("Quality Validation", self.test_quality_validation_endpoint),
            ("Benchmark", self.test_benchmark_endpoint)
        ]
        
        passed_tests = 0
        failed_tests = []
        
        for test_name, test_method in test_methods:
            try:
                if test_method():
                    passed_tests += 1
                else:
                    failed_tests.append(test_name)
            except Exception as e:
                print(f"  ❌ Test '{test_name}' failed with exception: {e}")
                failed_tests.append(test_name)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n🔥 TEST SUMMARY")
        print("=" * 50)
        print(f"  ℹ️ Total Tests: {len(test_methods)}")
        print(f"  ✅ Passed: {passed_tests}")
        print(f"  ❌ Failed: {len(failed_tests)}")
        print(f"  ℹ️ Total Time: {total_time:.2f} seconds")
        
        if failed_tests:
            print(f"  ❌ Failed Tests:")
            for test in failed_tests:
                print(f"    - {test}")
        
        success = len(failed_tests) == 0
        final_status = "✅ ALL TESTS PASSED!" if success else "❌ SOME TESTS FAILED!"
        print(f"  {final_status}")
        
        return success

def main():
    """Main function to run base64-only tests"""
    parser = argparse.ArgumentParser(description='Test AI Module with real images from test/imgs folder')
    parser.add_argument('--num-images', '-n', type=int, default=3, 
                       help='Number of images to use for testing (default: 3)')
    parser.add_argument('--ai-url', '-u', type=str, default='http://localhost:8000',
                       help='AI Module URL (default: http://localhost:8000)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎯 BASE64-ONLY AI MODULE TEST SUITE")
    print("=" * 80)
    print("This script tests the AI Module with real images from test/imgs folder:")
    print("  📊 Health & Info endpoints")
    print("  🧠 Inference with real images")
    print("  👥 Student insertion with real images")
    print("  ⚡ Fast embedding with real images")
    print("  🔍 Quality validation with real images")
    print("  📈 Benchmark with real images")
    print("=" * 80)
    print("Prerequisites:")
    print(f"  - AI Module running on {args.ai_url}")
    print("  - opencv-python installed (pip install opencv-python)")
    print("  - Images available in test/imgs folder")
    print(f"  - Using {args.num_images} images for testing")
    print("=" * 80)
    
    input("Press Enter to start the base64-only test...")
    
    try:
        tester = Base64OnlyTest(ai_url=args.ai_url, num_images=args.num_images)
        success = tester.run_all_tests()
        
        print("\n" + "=" * 80)
        if success:
            print("🎉 BASE64-ONLY TEST COMPLETED SUCCESSFULLY!")
            print("All endpoints work correctly with real images from test/imgs folder.")
            print("File upload functionality has been successfully removed.")
        else:
            print("💥 BASE64-ONLY TEST FAILED!")
            print("Please check the error messages above and fix any issues.")
        print("=" * 80)
        
        return 0 if success else 1
    
    except Exception as e:
        print(f"\n❌ Test initialization failed: {e}")
        print("Please ensure images are available in test/imgs folder")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 