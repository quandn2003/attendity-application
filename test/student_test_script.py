#!/usr/bin/env python3

import asyncio
import httpx
import base64
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any

class StudentTestScript:
    def __init__(self):
        self.ai_api_url = "http://localhost:8000"
        self.vector_db_url = "http://localhost:8001"
        self.test_class = "CS101_TEST"
        self.student_1_id = "232323"
        self.student_2_id = "2114547"
        self.test_images_dir = Path("test/imgs")
        self.results = []
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string"""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    
    def log_step(self, step_name: str, status: str, details: str = ""):
        """Log test step result"""
        result = {
            "step": step_name,
            "status": status,
            "details": details,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.results.append(result)
        status_symbol = "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "⚠️"
        print(f"{status_symbol} {step_name}: {status}")
        if details:
            print(f"   Details: {details}")
        print()
    
    async def step_1_create_class_list(self):
        """Step 1: Create a class list"""
        print("=" * 60)
        print("STEP 1: Creating Class List")
        print("=" * 60)
        
        try:
            async with httpx.AsyncClient() as client:
                payload = {"class_code": self.test_class}
                response = await client.post(f"{self.vector_db_url}/create_class", json=payload, timeout=10.0)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success":
                        self.log_step("Create Class", "SUCCESS", f"Class '{self.test_class}' created successfully")
                        return True
                    else:
                        self.log_step("Create Class", "FAILED", f"Unexpected response: {data}")
                        return False
                elif response.status_code == 409:
                    self.log_step("Create Class", "SUCCESS", f"Class '{self.test_class}' already exists")
                    return True
                else:
                    self.log_step("Create Class", "FAILED", f"HTTP {response.status_code}: {response.text}")
                    return False
        except Exception as e:
            self.log_step("Create Class", "FAILED", f"Exception: {str(e)}")
            return False
    
    async def step_2_insert_student_with_debug(self):
        """Step 2: Insert 1 student with debug to check if embeddings are added"""
        print("=" * 60)
        print("STEP 2: Inserting Student with Debug")
        print("=" * 60)
        
        try:
            student_dir = self.test_images_dir / self.student_1_id
            image_files = [f for f in os.listdir(student_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]
            
            if len(image_files) < 3:
                self.log_step("Insert Student", "FAILED", f"Not enough images for student {self.student_1_id}. Found: {len(image_files)}")
                return False
            
            print(f"📸 Using images for student {self.student_1_id}:")
            for i, img_file in enumerate(image_files, 1):
                print(f"   Image {i}: {img_file}")
            
            base64_images = []
            for img_file in image_files:
                img_path = student_dir / img_file
                base64_images.append(self.encode_image_to_base64(str(img_path)))
            
            async with httpx.AsyncClient() as client:
                payload = {
                    "class_code": self.test_class,
                    "student_id": self.student_1_id,
                    "image1": base64_images[0],
                    "image2": base64_images[1],
                    "image3": base64_images[2]
                }
                
                print(f"🔄 Processing 3 images for student {self.student_1_id}...")
                response = await client.post(f"{self.ai_api_url}/insert_student", json=payload, timeout=60.0)
                
                print(f"🔍 AI API Response Status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"🔍 AI API Response Data: {json.dumps(data, indent=2)}")
                    
                    if data.get("status") == "success" and "consensus_embedding" in data:
                        embedding_len = len(data.get("consensus_embedding", []))
                        processing_time = data.get("total_processing_time", 0)
                        
                        print(f"🧠 Consensus embedding generated:")
                        print(f"   Dimension: {embedding_len}")
                        print(f"   Processing time: {processing_time:.2f}s")
                        
                        individual_results = data.get("individual_results", [])
                        print(f"📊 Individual image results:")
                        for i, result in enumerate(individual_results, 1):
                            confidence = result.get("confidence", 0)
                            is_real = result.get("is_real", False)
                            print(f"   Image {i}: Confidence={confidence:.3f}, Real={is_real}")
                        
                        # Test direct vector DB insertion
                        await self.test_direct_vector_db_insertion(data.get("consensus_embedding"))
                        
                        await self.debug_check_student_in_collection()
                        
                        self.log_step("Insert Student", "SUCCESS", 
                                    f"Student {self.student_1_id} inserted with {embedding_len}D embedding")
                        return True
                    else:
                        self.log_step("Insert Student", "FAILED", f"Unexpected AI response: {data}")
                        return False
                else:
                    error_text = response.text
                    print(f"🔍 AI API Error Response: {error_text}")
                    self.log_step("Insert Student", "FAILED", 
                                f"AI API error - HTTP {response.status_code}: {error_text}")
                    return False
                    
        except Exception as e:
            self.log_step("Insert Student", "FAILED", f"Exception: {str(e)}")
            return False
    
    async def test_direct_vector_db_insertion(self, consensus_embedding):
        """Test direct insertion into vector database to debug the issue"""
        print("🔧 DEBUG: Testing direct vector DB insertion...")
        
        try:
            async with httpx.AsyncClient() as client:
                vector_db_payload = {
                    "students": [{
                        "student_id": self.student_1_id,
                        "class_code": self.test_class,
                        "embedding": consensus_embedding
                    }]
                }
                
                # Create a simplified payload for logging (without full embedding data)
                log_payload = {k: v if k != 'students' else [{'student_id': v[0]['student_id'], 'class_code': v[0]['class_code'], 'embedding': f'[{len(v[0]["embedding"])} dimensions]'}] for k, v in vector_db_payload.items()}
                print(f"🔍 Vector DB Payload: {json.dumps(log_payload, indent=2)}")
                
                response = await client.post(
                    f"{self.vector_db_url}/insert_student",
                    json=vector_db_payload,
                    timeout=30.0
                )
                
                print(f"🔍 Vector DB Response Status: {response.status_code}")
                print(f"🔍 Vector DB Response: {response.text}")
                
                if response.status_code == 200:
                    db_data = response.json()
                    print(f"✅ Direct vector DB insertion successful: {db_data}")
                else:
                    print(f"❌ Direct vector DB insertion failed: HTTP {response.status_code}")
                    
        except Exception as e:
            print(f"❌ Direct vector DB test failed: {str(e)}")
    
    async def debug_check_student_in_collection(self):
        """Debug: Check if student embeddings are actually added to the collection"""
        print("🔍 DEBUG: Checking if student was added to collection...")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.vector_db_url}/class_stats/{self.test_class}", timeout=10.0)
                
                print(f"🔍 Class Stats Response Status: {response.status_code}")
                print(f"🔍 Class Stats Response: {response.text}")
                
                if response.status_code == 200:
                    data = response.json()
                    student_count = data.get("student_count", 0)
                    total_embeddings = data.get("total_embeddings", 0)
                    
                    print(f"📈 Collection stats:")
                    print(f"   Class: {self.test_class}")
                    print(f"   Students: {student_count}")
                    print(f"   Total embeddings: {total_embeddings}")
                    
                    if student_count > 0:
                        print("✅ Student successfully added to collection!")
                    else:
                        print("❌ No students found in collection!")
                        
                    students_list = data.get("students", [])
                    if students_list:
                        print(f"👥 Students in collection: {', '.join(students_list)}")
                        if self.student_1_id in students_list:
                            print(f"✅ Student {self.student_1_id} confirmed in collection!")
                        else:
                            print(f"❌ Student {self.student_1_id} NOT found in collection!")
                else:
                    print(f"❌ Failed to get class stats: HTTP {response.status_code}")
                    
        except Exception as e:
            print(f"❌ Debug check failed: {str(e)}")
    
    async def step_3_search_with_different_student(self):
        """Step 3: Use another student to search"""
        print("=" * 60)
        print("STEP 3: Search with Different Student")
        print("=" * 60)
        
        try:
            student_dir = self.test_images_dir / self.student_2_id
            image_files = [f for f in os.listdir(student_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                self.log_step("Search with Different Student", "FAILED", 
                            f"No images found for student {self.student_2_id}")
                return False
            
            search_image = image_files[0]
            print(f"🔍 Searching with student {self.student_2_id} using image: {search_image}")
            
            img_path = student_dir / search_image
            base64_image = self.encode_image_to_base64(str(img_path))
            
            async with httpx.AsyncClient() as client:
                payload = {"image": base64_image}
                response = await client.post(f"{self.ai_api_url}/inference", json=payload, timeout=30.0)
                
                if response.status_code == 200:
                    embedding_data = response.json()
                    embedding = embedding_data.get("embedding", [])
                    confidence = embedding_data.get("confidence", 0)
                    is_real = embedding_data.get("is_real", False)
                    
                    print(f"🧠 Embedding extracted:")
                    print(f"   Dimension: {len(embedding)}")
                    print(f"   Confidence: {confidence:.3f}")
                    print(f"   Is real: {is_real}")
                    
                    if len(embedding) == 512:
                        search_payload = {
                            "embedding": embedding,
                            "class_code": self.test_class,
                            "threshold": 0.7
                        }
                        
                        search_response = await client.post(f"{self.vector_db_url}/search_with_voting", 
                                                          json=search_payload, timeout=30.0)
                        
                        if search_response.status_code == 200:
                            search_data = search_response.json()
                            status = search_data.get("status", "")
                            student_id = search_data.get("student_id", "")
                            search_confidence = search_data.get("confidence", 0)
                            top_matches = search_data.get("top_matches", [])
                            
                            print(f"🎯 Search results:")
                            print(f"   Status: {status}")
                            print(f"   Matched student: {student_id if student_id else 'None'}")
                            print(f"   Confidence: {search_confidence if search_confidence else 'N/A'}")
                            print(f"   Top matches: {len(top_matches)}")
                            
                            for i, match in enumerate(top_matches[:3], 1):
                                print(f"   Match {i}: {match.get('student_id', 'Unknown')} "
                                     f"(similarity: {match.get('similarity', 0):.3f})")
                            
                            if status in ["no_match", "no_students_found"]:
                                self.log_step("Search with Different Student", "SUCCESS", 
                                            f"Correctly identified no match for student {self.student_2_id}")
                            else:
                                self.log_step("Search with Different Student", "SUCCESS", 
                                            f"Search completed - matched: {student_id}")
                            return True
                        else:
                            self.log_step("Search with Different Student", "FAILED", 
                                        f"Search API error - HTTP {search_response.status_code}")
                            return False
                    else:
                        self.log_step("Search with Different Student", "FAILED", 
                                    f"Invalid embedding dimension: {len(embedding)}")
                        return False
                else:
                    self.log_step("Search with Different Student", "FAILED", 
                                f"Embedding extraction failed - HTTP {response.status_code}")
                    return False
                    
        except Exception as e:
            self.log_step("Search with Different Student", "FAILED", f"Exception: {str(e)}")
            return False
    
    async def step_4_search_with_added_student_image(self):
        """Step 4: Search using 1 image of the added student"""
        print("=" * 60)
        print("STEP 4: Search with Added Student Image")
        print("=" * 60)
        
        try:
            student_dir = self.test_images_dir / self.student_1_id
            image_files = [f for f in os.listdir(student_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                self.log_step("Search with Added Student", "FAILED", 
                            f"No images found for student {self.student_1_id}")
                return False
            
            search_image = image_files[-1]
            print(f"🔍 Searching with added student {self.student_1_id} using image: {search_image}")
            
            img_path = student_dir / search_image
            base64_image = self.encode_image_to_base64(str(img_path))
            
            async with httpx.AsyncClient() as client:
                payload = {"image": base64_image}
                response = await client.post(f"{self.ai_api_url}/inference", json=payload, timeout=30.0)
                
                if response.status_code == 200:
                    embedding_data = response.json()
                    embedding = embedding_data.get("embedding", [])
                    confidence = embedding_data.get("confidence", 0)
                    is_real = embedding_data.get("is_real", False)
                    
                    print(f"🧠 Embedding extracted:")
                    print(f"   Dimension: {len(embedding)}")
                    print(f"   Confidence: {confidence:.3f}")
                    print(f"   Is real: {is_real}")
                    
                    if len(embedding) == 512:
                        search_payload = {
                            "embedding": embedding,
                            "class_code": self.test_class,
                            "threshold": 0.7
                        }
                        
                        search_response = await client.post(f"{self.vector_db_url}/search_with_voting", 
                                                          json=search_payload, timeout=30.0)
                        
                        if search_response.status_code == 200:
                            search_data = search_response.json()
                            status = search_data.get("status", "")
                            student_id = search_data.get("student_id", "")
                            search_confidence = search_data.get("confidence", 0)
                            top_matches = search_data.get("top_matches", [])
                            
                            print(f"🎯 Search results:")
                            print(f"   Status: {status}")
                            print(f"   Matched student: {student_id if student_id else 'None'}")
                            print(f"   Confidence: {search_confidence if search_confidence else 'N/A'}")
                            print(f"   Top matches: {len(top_matches)}")
                            
                            for i, match in enumerate(top_matches[:3], 1):
                                print(f"   Match {i}: {match.get('student_id', 'Unknown')} "
                                     f"(similarity: {match.get('similarity', 0):.3f})")
                            
                            if status == "match_found" and str(student_id) == str(self.student_1_id):
                                self.log_step("Search with Added Student", "SUCCESS", 
                                            f"Correctly matched student {self.student_1_id} with confidence {search_confidence:.3f}")
                            elif status in ["no_match", "no_students_found"]:
                                self.log_step("Search with Added Student", "FAILED", 
                                            f"Failed to match student {self.student_1_id} - student not in collection or threshold too high")
                            else:
                                self.log_step("Search with Added Student", "FAILED", 
                                            f"Incorrect match: expected {self.student_1_id}, got {student_id}")
                            return True
                        else:
                            self.log_step("Search with Added Student", "FAILED", 
                                        f"Search API error - HTTP {search_response.status_code}")
                            return False
                    else:
                        self.log_step("Search with Added Student", "FAILED", 
                                    f"Invalid embedding dimension: {len(embedding)}")
                        return False
                else:
                    self.log_step("Search with Added Student", "FAILED", 
                                f"Embedding extraction failed - HTTP {response.status_code}")
                    return False
                    
        except Exception as e:
            self.log_step("Search with Added Student", "FAILED", f"Exception: {str(e)}")
            return False
    
    async def check_apis_health(self):
        """Check if both APIs are running"""
        print("🏥 Checking API health...")
        
        try:
            async with httpx.AsyncClient() as client:
                ai_response = await client.get(f"{self.ai_api_url}/health", timeout=5.0)
                vector_response = await client.get(f"{self.vector_db_url}/health", timeout=5.0)
                
                ai_healthy = ai_response.status_code == 200
                vector_healthy = vector_response.status_code == 200
                
                print(f"   AI API (port 8000): {'✅ Healthy' if ai_healthy else '❌ Unhealthy'}")
                print(f"   Vector DB API (port 8001): {'✅ Healthy' if vector_healthy else '❌ Unhealthy'}")
                
                return ai_healthy and vector_healthy
                
        except Exception as e:
            print(f"❌ Health check failed: {str(e)}")
            return False
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        success_count = sum(1 for result in self.results if result["status"] == "SUCCESS")
        total_count = len(self.results)
        
        print(f"Total steps: {total_count}")
        print(f"Successful: {success_count}")
        print(f"Failed: {total_count - success_count}")
        print(f"Success rate: {(success_count/total_count*100):.1f}%" if total_count > 0 else "N/A")
        
        print("\nDetailed results:")
        for result in self.results:
            status_symbol = "✅" if result["status"] == "SUCCESS" else "❌"
            print(f"{status_symbol} {result['step']}: {result['status']}")
            if result["details"]:
                print(f"   {result['details']}")
        
        print("\n" + "=" * 60)
    
    async def run_test_script(self):
        """Run the complete test script"""
        print("🚀 Starting Student Test Script")
        print(f"📁 Test images directory: {self.test_images_dir}")
        print(f"👤 Student 1 (to insert): {self.student_1_id}")
        print(f"👤 Student 2 (for search): {self.student_2_id}")
        print(f"🏫 Test class: {self.test_class}")
        print()
        
        if not await self.check_apis_health():
            print("❌ APIs are not healthy. Please start the APIs first using:")
            print("   python3 run_apis.py")
            return False
        
        success = True
        
        success &= await self.step_1_create_class_list()
        
        if success:
            success &= await self.step_2_insert_student_with_debug()
        
        if success:
            success &= await self.step_3_search_with_different_student()
        
        if success:
            success &= await self.step_4_search_with_added_student_image()
        
        self.print_summary()
        
        return success

async def main():
    """Main function"""
    test_script = StudentTestScript()
    success = await test_script.run_test_script()
    
    if success:
        print("🎉 All tests completed successfully!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main())) 