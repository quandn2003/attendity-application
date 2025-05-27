import asyncio
import httpx
import base64
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any

class APITester:
    def __init__(self):
        self.ai_api_url = "http://localhost:8000"
        self.vector_db_url = "http://localhost:8001"
        self.test_class = "TEST_CLASS_001"
        self.student_ids = ["232323", "2114547"]
        self.test_images_dir = Path("test/imgs")
        self.results = []
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string"""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    
    def log_result(self, test_name: str, status: str, details: str = ""):
        """Log test result"""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.results.append(result)
        status_symbol = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_symbol} {test_name}: {status}")
        if details:
            print(f"   Details: {details}")
    
    async def test_ai_api_health(self):
        """Test AI API health endpoint"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.ai_api_url}/health", timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "healthy":
                        self.log_result("AI API Health Check", "PASS", "AI API is healthy")
                    else:
                        self.log_result("AI API Health Check", "FAIL", f"AI API unhealthy: {data}")
                else:
                    self.log_result("AI API Health Check", "FAIL", f"Status code: {response.status_code}")
        except Exception as e:
            self.log_result("AI API Health Check", "FAIL", str(e))
    
    async def test_vector_db_health(self):
        """Test Vector DB API health endpoint"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.vector_db_url}/health", timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "healthy":
                        self.log_result("Vector DB Health Check", "PASS", "Vector DB is healthy")
                    else:
                        self.log_result("Vector DB Health Check", "FAIL", f"Vector DB unhealthy: {data}")
                else:
                    self.log_result("Vector DB Health Check", "FAIL", f"Status code: {response.status_code}")
        except Exception as e:
            self.log_result("Vector DB Health Check", "FAIL", str(e))
    
    async def test_ai_single_inference(self):
        """Test AI API single image inference"""
        try:
            image_path = self.test_images_dir / self.student_ids[0] / os.listdir(self.test_images_dir / self.student_ids[0])[0]
            base64_image = self.encode_image_to_base64(str(image_path))
            
            async with httpx.AsyncClient() as client:
                payload = {"image": base64_image}
                response = await client.post(f"{self.ai_api_url}/inference", json=payload, timeout=30.0)
                
                if response.status_code == 200:
                    data = response.json()
                    if "embedding" in data and len(data["embedding"]) == 512:
                        self.log_result("AI Single Inference", "PASS", f"Embedding extracted, confidence: {data.get('confidence', 'N/A')}")
                    else:
                        self.log_result("AI Single Inference", "FAIL", "Invalid embedding response")
                else:
                    self.log_result("AI Single Inference", "FAIL", f"Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            self.log_result("AI Single Inference", "FAIL", str(e))
    
    async def test_ai_fast_embedding(self):
        """Test AI API fast embedding extraction"""
        try:
            image_path = self.test_images_dir / self.student_ids[1] / os.listdir(self.test_images_dir / self.student_ids[1])[0]
            base64_image = self.encode_image_to_base64(str(image_path))
            
            async with httpx.AsyncClient() as client:
                payload = {"image": base64_image}
                response = await client.post(f"{self.ai_api_url}/extract_embedding_fast", json=payload, timeout=30.0)
                
                if response.status_code == 200:
                    data = response.json()
                    if "embedding" in data and len(data["embedding"]) == 512:
                        self.log_result("AI Fast Embedding", "PASS", "Fast embedding extracted successfully")
                    else:
                        self.log_result("AI Fast Embedding", "FAIL", "Invalid embedding response")
                else:
                    self.log_result("AI Fast Embedding", "FAIL", f"Status: {response.status_code}")
        except Exception as e:
            self.log_result("AI Fast Embedding", "FAIL", str(e))
    
    async def test_ai_quality_validation(self):
        """Test AI API face quality validation"""
        try:
            image_path = self.test_images_dir / self.student_ids[0] / os.listdir(self.test_images_dir / self.student_ids[0])[0]
            base64_image = self.encode_image_to_base64(str(image_path))
            
            async with httpx.AsyncClient() as client:
                payload = {"image": base64_image}
                response = await client.post(f"{self.ai_api_url}/validate_quality", json=payload, timeout=30.0)
                
                if response.status_code == 200:
                    self.log_result("AI Quality Validation", "PASS", "Quality validation completed")
                else:
                    self.log_result("AI Quality Validation", "FAIL", f"Status: {response.status_code}")
        except Exception as e:
            self.log_result("AI Quality Validation", "FAIL", str(e))
    
    async def test_vector_db_create_class_success(self):
        """Test Vector DB create class - success case"""
        try:
            async with httpx.AsyncClient() as client:
                payload = {"class_code": self.test_class}
                response = await client.post(f"{self.vector_db_url}/create_class", json=payload, timeout=10.0)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success":
                        self.log_result("Vector DB Create Class (Success)", "PASS", f"Class {self.test_class} created")
                    else:
                        self.log_result("Vector DB Create Class (Success)", "FAIL", f"Unexpected response: {data}")
                else:
                    self.log_result("Vector DB Create Class (Success)", "FAIL", f"Status: {response.status_code}")
        except Exception as e:
            self.log_result("Vector DB Create Class (Success)", "FAIL", str(e))
    
    async def test_vector_db_create_class_duplicate(self):
        """Test Vector DB create class - duplicate class (should fail)"""
        try:
            async with httpx.AsyncClient() as client:
                payload = {"class_code": self.test_class}
                response = await client.post(f"{self.vector_db_url}/create_class", json=payload, timeout=10.0)
                
                if response.status_code == 409:
                    self.log_result("Vector DB Create Class (Duplicate)", "PASS", "Correctly rejected duplicate class")
                else:
                    self.log_result("Vector DB Create Class (Duplicate)", "FAIL", f"Expected 409, got {response.status_code}")
        except Exception as e:
            self.log_result("Vector DB Create Class (Duplicate)", "FAIL", str(e))
    
    async def test_ai_insert_student_success(self):
        """Test AI API insert student - success case"""
        try:
            student_id = self.student_ids[0]
            student_dir = self.test_images_dir / student_id
            image_files = os.listdir(student_dir)[:3]
            
            base64_images = []
            for img_file in image_files:
                img_path = student_dir / img_file
                base64_images.append(self.encode_image_to_base64(str(img_path)))
            
            async with httpx.AsyncClient() as client:
                payload = {
                    "class_code": self.test_class,
                    "student_id": student_id,
                    "image1": base64_images[0],
                    "image2": base64_images[1],
                    "image3": base64_images[2]
                }
                response = await client.post(f"{self.ai_api_url}/insert_student", json=payload, timeout=60.0)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success" and "consensus_embedding" in data:
                        embedding_len = len(data.get("consensus_embedding", []))
                        self.log_result("AI Insert Student (Success)", "PASS", f"Student {student_id} inserted successfully (embedding dim: {embedding_len})")
                    else:
                        self.log_result("AI Insert Student (Success)", "FAIL", f"Unexpected response: {data}")
                else:
                    self.log_result("AI Insert Student (Success)", "FAIL", f"Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            self.log_result("AI Insert Student (Success)", "FAIL", str(e))
    
    async def test_ai_insert_student_second(self):
        """Test AI API insert second student"""
        try:
            student_id = self.student_ids[1]
            student_dir = self.test_images_dir / student_id
            image_files = os.listdir(student_dir)[:3]
            
            base64_images = []
            for img_file in image_files:
                img_path = student_dir / img_file
                base64_images.append(self.encode_image_to_base64(str(img_path)))
            
            async with httpx.AsyncClient() as client:
                payload = {
                    "class_code": self.test_class,
                    "student_id": student_id,
                    "image1": base64_images[0],
                    "image2": base64_images[1],
                    "image3": base64_images[2]
                }
                response = await client.post(f"{self.ai_api_url}/insert_student", json=payload, timeout=60.0)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success":
                        embedding_len = len(data.get("consensus_embedding", []))
                        self.log_result("AI Insert Second Student", "PASS", f"Student {student_id} inserted successfully (embedding dim: {embedding_len})")
                    else:
                        self.log_result("AI Insert Second Student", "FAIL", f"Unexpected response: {data}")
                else:
                    self.log_result("AI Insert Second Student", "FAIL", f"Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            self.log_result("AI Insert Second Student", "FAIL", str(e))
    
    async def test_vector_db_search_with_voting(self):
        """Test Vector DB search with voting"""
        try:
            image_path = self.test_images_dir / self.student_ids[0] / os.listdir(self.test_images_dir / self.student_ids[0])[0]
            base64_image = self.encode_image_to_base64(str(image_path))
            
            async with httpx.AsyncClient() as client:
                embedding_payload = {"image": base64_image}
                embedding_response = await client.post(f"{self.ai_api_url}/extract_embedding_fast", json=embedding_payload, timeout=30.0)
                
                if embedding_response.status_code == 200:
                    embedding_data = embedding_response.json()
                    embedding = embedding_data["embedding"]
                    
                    # Ensure embedding has exactly 512 dimensions as required by Pydantic model
                    if len(embedding) != 512:
                        self.log_result("Vector DB Search with Voting", "FAIL", f"Invalid embedding dimension: {len(embedding)}, expected 512")
                        return
                    
                    # Match SearchWithVotingRequest Pydantic model
                    search_payload = {
                        "embedding": embedding,
                        "class_code": self.test_class,
                        "threshold": 0.7
                    }
                    search_response = await client.post(f"{self.vector_db_url}/search_with_voting", json=search_payload, timeout=30.0)
                    
                    if search_response.status_code == 200:
                        search_data = search_response.json()
                        if search_data.get("status") == "match_found":
                            self.log_result("Vector DB Search with Voting", "PASS", f"Found match: {search_data.get('student_id')}")
                        else:
                            self.log_result("Vector DB Search with Voting", "WARN", f"No match found: {search_data.get('reason')}")
                    else:
                        self.log_result("Vector DB Search with Voting", "FAIL", f"Search failed: {search_response.status_code}, Response: {search_response.text}")
                else:
                    self.log_result("Vector DB Search with Voting", "FAIL", "Failed to extract embedding for search")
        except Exception as e:
            self.log_result("Vector DB Search with Voting", "FAIL", str(e))
    
    async def test_vector_db_class_stats(self):
        """Test Vector DB class statistics"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.vector_db_url}/class_stats/{self.test_class}", timeout=10.0)
                
                if response.status_code == 200:
                    data = response.json()
                    student_count = data.get("student_count", 0)
                    self.log_result("Vector DB Class Stats", "PASS", f"Class has {student_count} students. Full response: {data}")
                else:
                    self.log_result("Vector DB Class Stats", "FAIL", f"Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            self.log_result("Vector DB Class Stats", "FAIL", str(e))
    
    async def test_vector_db_student_attendance(self):
        """Test Vector DB student attendance history"""
        try:
            student_id = self.student_ids[0]
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.vector_db_url}/student_attendance/{self.test_class}/{student_id}", timeout=10.0)
                
                if response.status_code == 200:
                    data = response.json()
                    attendance_count = data.get("total_attendances", 0)
                    self.log_result("Vector DB Student Attendance", "PASS", f"Student has {attendance_count} attendance records")
                else:
                    self.log_result("Vector DB Student Attendance", "FAIL", f"Status: {response.status_code}")
        except Exception as e:
            self.log_result("Vector DB Student Attendance", "FAIL", str(e))
    
    async def test_vector_db_delete_student_success(self):
        """Test Vector DB delete student - success case"""
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "students": [{
                        "student_id": self.student_ids[0],
                        "class_code": self.test_class
                    }]
                }
                response = await client.request("DELETE", f"{self.vector_db_url}/delete_student", json=payload, timeout=10.0)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success" and data.get("processed_count") > 0:
                        self.log_result("Vector DB Delete Student (Success)", "PASS", f"Student {self.student_ids[0]} deleted")
                    else:
                        self.log_result("Vector DB Delete Student (Success)", "FAIL", f"Unexpected response: {data}")
                else:
                    self.log_result("Vector DB Delete Student (Success)", "FAIL", f"Status: {response.status_code}")
        except Exception as e:
            self.log_result("Vector DB Delete Student (Success)", "FAIL", str(e))
    
    async def test_vector_db_delete_student_not_exist(self):
        """Test Vector DB delete student - student doesn't exist (should fail gracefully)"""
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "students": [{
                        "student_id": "NONEXISTENT_STUDENT",
                        "class_code": self.test_class
                    }]
                }
                response = await client.request("DELETE", f"{self.vector_db_url}/delete_student", json=payload, timeout=10.0)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("processed_count") == 0 and len(data.get("details", {}).get("failed_deletions", [])) > 0:
                        self.log_result("Vector DB Delete Student (Not Exist)", "PASS", "Correctly handled non-existent student")
                    else:
                        self.log_result("Vector DB Delete Student (Not Exist)", "FAIL", f"Unexpected response: {data}")
                else:
                    self.log_result("Vector DB Delete Student (Not Exist)", "FAIL", f"Status: {response.status_code}")
        except Exception as e:
            self.log_result("Vector DB Delete Student (Not Exist)", "FAIL", str(e))
    
    async def test_vector_db_delete_class_success(self):
        """Test Vector DB delete class - success case"""
        try:
            async with httpx.AsyncClient() as client:
                payload = {"class_code": self.test_class}
                response = await client.request("DELETE", f"{self.vector_db_url}/delete_class", json=payload, timeout=10.0)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success":
                        self.log_result("Vector DB Delete Class (Success)", "PASS", f"Class {self.test_class} deleted")
                    else:
                        self.log_result("Vector DB Delete Class (Success)", "FAIL", f"Unexpected response: {data}")
                else:
                    self.log_result("Vector DB Delete Class (Success)", "FAIL", f"Status: {response.status_code}")
        except Exception as e:
            self.log_result("Vector DB Delete Class (Success)", "FAIL", str(e))
    
    async def test_vector_db_delete_class_not_exist(self):
        """Test Vector DB delete class - class doesn't exist (should fail)"""
        try:
            async with httpx.AsyncClient() as client:
                payload = {"class_code": "NONEXISTENT_CLASS"}
                response = await client.request("DELETE", f"{self.vector_db_url}/delete_class", json=payload, timeout=10.0)
                
                if response.status_code == 404:
                    self.log_result("Vector DB Delete Class (Not Exist)", "PASS", "Correctly rejected non-existent class")
                else:
                    self.log_result("Vector DB Delete Class (Not Exist)", "FAIL", f"Expected 404, got {response.status_code}")
        except Exception as e:
            self.log_result("Vector DB Delete Class (Not Exist)", "FAIL", str(e))
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        total_tests = len(self.results)
        passed = len([r for r in self.results if r["status"] == "PASS"])
        failed = len([r for r in self.results if r["status"] == "FAIL"])
        warnings = len([r for r in self.results if r["status"] == "WARN"])
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Warnings: {warnings}")
        print(f"Success Rate: {(passed/total_tests)*100:.1f}%")
        
        if failed > 0:
            print("\nFAILED TESTS:")
            for result in self.results:
                if result["status"] == "FAIL":
                    print(f"❌ {result['test']}: {result['details']}")
        
        print("\nDetailed Results:")
        for result in self.results:
            status_symbol = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⚠️"
            print(f"{status_symbol} {result['test']}")
    
    async def run_all_tests(self):
        """Run all tests in sequence"""
        print("Starting comprehensive API testing...")
        print("="*80)
        
        # 1. Health checks first
        await self.test_ai_api_health()
        await self.test_vector_db_health()
        
        # 2. AI API standalone tests (no database interaction)
        await self.test_ai_single_inference()
        await self.test_ai_fast_embedding()
        await self.test_ai_quality_validation()
        
        # 3. Vector DB setup - create class
        await self.test_vector_db_create_class_success()
        await self.test_vector_db_create_class_duplicate()
        
        # 4. Student insertion tests (MUST be after class creation, before search/delete)
        await self.test_ai_insert_student_success()
        await self.test_ai_insert_student_second()
        
        # 5. Search and stats tests (MUST be after student insertion)
        await self.test_vector_db_search_with_voting()
        await self.test_vector_db_class_stats()
        await self.test_vector_db_student_attendance()
        
        # 6. Deletion tests (MUST be after insertion and search)
        await self.test_vector_db_delete_student_success()
        await self.test_vector_db_delete_student_not_exist()
        await self.test_vector_db_delete_class_success()
        await self.test_vector_db_delete_class_not_exist()
        
        self.print_summary()

async def main():
    """Main test runner"""
    tester = APITester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main()) 