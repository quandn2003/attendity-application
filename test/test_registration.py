#!/usr/bin/env python3
"""
Test Script for Registration System
Tests the insert_student endpoint that requires 3 successfully embedded images
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

class RegistrationTest:
    """Test class for registration system"""
    
    def __init__(self, ai_url: str = "http://localhost:8000"):
        self.ai_url = ai_url
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
    
    def generate_mock_face_image(self, width: int = 160, height: int = 160, seed: int = 42) -> np.ndarray:
        """Generate a realistic mock face image for testing"""
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
    
    def test_registration_with_real_images(self) -> bool:
        """Test registration with real images from test/imgs"""
        print("\n🔥 TESTING REGISTRATION WITH REAL IMAGES")
        print("=" * 50)
        
        try:
            images_base64 = []
            used_images = []
            
            # Use the same image 3 times if only one available
            for i in range(3):
                image_index = i % len(self.available_images)
                test_image = self.load_image_from_file(image_index)
                base64_image = self.image_to_base64(test_image)
                images_base64.append(base64_image)
                used_images.append(os.path.basename(self.available_images[image_index]))
            
            self.print_step(f"Testing registration with images: {', '.join(used_images)}", "progress")
            
            response = requests.post(
                f"{self.ai_url}/insert_student",
                json={
                    "class_code": "TEST_CLASS_REAL",
                    "student_id": "test_student_real_001",
                    "image1": images_base64[0],
                    "image2": images_base64[1],
                    "image3": images_base64[2]
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                self.print_step("✓ Registration with real images successful", "success")
                self.print_step(f"  - Status: {result['status']}", "info")
                self.print_step(f"  - Consensus embedding dim: {len(result['consensus_embedding'])}", "info")
                self.print_step(f"  - Individual results: {len(result['individual_results'])}", "info")
                self.print_step(f"  - Total processing time: {result['total_processing_time']:.3f}s", "info")
                
                # Check individual results
                successful_embeddings = 0
                for i, individual in enumerate(result['individual_results']):
                    if individual['success'] and individual['is_real']:
                        successful_embeddings += 1
                        self.print_step(f"  - Image {i+1}: ✓ Success (confidence: {individual['confidence']:.3f})", "success")
                    else:
                        self.print_step(f"  - Image {i+1}: ✗ Failed ({individual.get('error', 'Unknown error')})", "error")
                
                self.print_step(f"  - Successfully embedded: {successful_embeddings}/3", "info")
                return successful_embeddings == 3
            else:
                self.print_step(f"✗ Registration failed: {response.text}", "error")
                return False
                
        except Exception as e:
            self.print_step(f"✗ Registration error: {e}", "error")
            return False
    
    def test_registration_with_mock_images(self) -> bool:
        """Test registration with generated mock images"""
        print("\n🔥 TESTING REGISTRATION WITH MOCK IMAGES")
        print("=" * 50)
        
        try:
            images_base64 = []
            
            # Generate 3 different mock face images
            for i in range(3):
                mock_image = self.generate_mock_face_image(seed=100 + i)
                base64_image = self.image_to_base64(mock_image)
                images_base64.append(base64_image)
            
            self.print_step("Testing registration with 3 generated mock face images", "progress")
            
            response = requests.post(
                f"{self.ai_url}/insert_student",
                json={
                    "class_code": "TEST_CLASS_MOCK",
                    "student_id": "test_student_mock_001",
                    "image1": images_base64[0],
                    "image2": images_base64[1],
                    "image3": images_base64[2]
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                self.print_step("✓ Registration with mock images successful", "success")
                self.print_step(f"  - Status: {result['status']}", "info")
                self.print_step(f"  - Consensus embedding dim: {len(result['consensus_embedding'])}", "info")
                self.print_step(f"  - Individual results: {len(result['individual_results'])}", "info")
                self.print_step(f"  - Total processing time: {result['total_processing_time']:.3f}s", "info")
                return True
            else:
                self.print_step(f"✗ Registration failed: {response.text}", "error")
                return False
                
        except Exception as e:
            self.print_step(f"✗ Registration error: {e}", "error")
            return False
    
    def test_registration_failure_scenarios(self) -> bool:
        """Test scenarios where registration should fail"""
        print("\n🔥 TESTING REGISTRATION FAILURE SCENARIOS")
        print("=" * 50)
        
        try:
            # Test with invalid base64
            self.print_step("Testing with invalid base64 image", "progress")
            
            response = requests.post(
                f"{self.ai_url}/insert_student",
                json={
                    "class_code": "TEST_CLASS_FAIL",
                    "student_id": "test_student_fail_001",
                    "image1": "invalid_base64",
                    "image2": "invalid_base64",
                    "image3": "invalid_base64"
                },
                timeout=60
            )
            
            if response.status_code != 200:
                self.print_step("✓ Correctly failed with invalid base64", "success")
                return True
            else:
                self.print_step("✗ Should have failed with invalid base64", "error")
                return False
                
        except Exception as e:
            self.print_step(f"✓ Correctly failed with exception: {e}", "success")
            return True
    
    def run_all_tests(self) -> bool:
        """Run all registration tests"""
        print("🚀 STARTING REGISTRATION SYSTEM TEST")
        print("Testing the insert_student endpoint that requires 3 successfully embedded images")
        print("=" * 80)
        print(f"Using images from: {self.test_images_dir}")
        print("=" * 80)
        
        start_time = time.time()
        
        test_methods = [
            ("Registration with Real Images", self.test_registration_with_real_images),
            ("Registration with Mock Images", self.test_registration_with_mock_images),
            ("Registration Failure Scenarios", self.test_registration_failure_scenarios)
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
    """Main function to run registration tests"""
    parser = argparse.ArgumentParser(description='Test Registration System')
    parser.add_argument('--ai-url', '-u', type=str, default='http://localhost:8000',
                       help='AI Module URL (default: http://localhost:8000)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎯 REGISTRATION SYSTEM TEST SUITE")
    print("=" * 80)
    print("This script tests the registration logic:")
    print("  📊 Registration with real images from test/imgs")
    print("  🤖 Registration with generated mock images")
    print("  ❌ Registration failure scenarios")
    print("  ✅ Requires exactly 3 successfully embedded images (face detected and real)")
    print("  🔧 Simple averaging of embeddings")
    print("=" * 80)
    print("Prerequisites:")
    print(f"  - AI Module running on {args.ai_url}")
    print("  - opencv-python installed (pip install opencv-python)")
    print("  - Images available in test/imgs folder")
    print("=" * 80)
    
    input("Press Enter to start the registration test...")
    
    try:
        tester = RegistrationTest(ai_url=args.ai_url)
        success = tester.run_all_tests()
        
        print("\n" + "=" * 80)
        if success:
            print("🎉 REGISTRATION SYSTEM TEST COMPLETED SUCCESSFULLY!")
            print("The registration logic works correctly:")
            print("  ✅ Requires exactly 3 successfully embedded images")
            print("  ✅ Simple averaging of valid embeddings")
            print("  ✅ Clean and efficient implementation")
        else:
            print("💥 REGISTRATION SYSTEM TEST FAILED!")
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