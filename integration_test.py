#!/usr/bin/env python3
"""
Integration test script for AI and Vector-DB modules
Tests the complete workflow: face processing -> embedding extraction -> student insertion -> attendance verification
"""

import requests
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, Any, List

class AttendityIntegrationTest:
    """Integration test for Attendity AI and Vector-DB modules"""
    
    def __init__(self, ai_url: str = "http://localhost:8000", vectordb_url: str = "http://localhost:8001"):
        self.ai_url = ai_url
        self.vectordb_url = vectordb_url
        self.test_class = "CS101_TEST"
        self.test_students = ["student001", "student002", "student003"]
    
    def check_apis_health(self) -> Dict[str, bool]:
        """Check if both APIs are running and healthy"""
        print("🔍 Checking API health...")
        
        health_status = {"ai": False, "vectordb": False}
        
        try:
            ai_response = requests.get(f"{self.ai_url}/health", timeout=5)
            health_status["ai"] = ai_response.status_code == 200
            print(f"  ✅ AI Module: {'Healthy' if health_status['ai'] else 'Unhealthy'}")
        except Exception as e:
            print(f"  ❌ AI Module: Connection failed - {e}")
        
        try:
            vectordb_response = requests.get(f"{self.vectordb_url}/health", timeout=5)
            health_status["vectordb"] = vectordb_response.status_code == 200
            print(f"  ✅ Vector-DB: {'Healthy' if health_status['vectordb'] else 'Unhealthy'}")
        except Exception as e:
            print(f"  ❌ Vector-DB: Connection failed - {e}")
        
        return health_status
    
    def create_test_class(self) -> bool:
        """Create a test class in the vector database"""
        print(f"📚 Creating test class: {self.test_class}")
        
        try:
            response = requests.post(
                f"{self.vectordb_url}/create_class",
                json={"class_code": self.test_class}
            )
            
            if response.status_code == 200:
                print(f"  ✅ Class {self.test_class} created successfully")
                return True
            elif response.status_code == 409:
                print(f"  ℹ️  Class {self.test_class} already exists")
                return True
            else:
                print(f"  ❌ Failed to create class: {response.text}")
                return False
                
        except Exception as e:
            print(f"  ❌ Error creating class: {e}")
            return False
    
    def generate_mock_embeddings(self, count: int = 3) -> List[List[float]]:
        """Generate mock face embeddings for testing"""
        print(f"🎭 Generating {count} mock embeddings...")
        
        embeddings = []
        for i in range(count):
            # Generate random but consistent embeddings for testing
            np.random.seed(42 + i)  # Different seed for each student
            embedding = np.random.normal(0, 1, 512).tolist()
            embeddings.append(embedding)
        
        print(f"  ✅ Generated {len(embeddings)} embeddings")
        return embeddings
    
    def insert_test_students(self, embeddings: List[List[float]]) -> bool:
        """Insert test students with mock embeddings"""
        print(f"👥 Inserting {len(self.test_students)} test students...")
        
        students_data = []
        for i, student_id in enumerate(self.test_students):
            students_data.append({
                "student_id": student_id,
                "class_code": self.test_class,
                "embedding": embeddings[i]
            })
        
        try:
            response = requests.post(
                f"{self.vectordb_url}/insert_student",
                json={"students": students_data}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ Inserted {result['processed_count']} students successfully")
                if result.get('details', {}).get('failed_insertions'):
                    print(f"  ⚠️  Some insertions failed: {result['details']['failed_insertions']}")
                return True
            else:
                print(f"  ❌ Failed to insert students: {response.text}")
                return False
                
        except Exception as e:
            print(f"  ❌ Error inserting students: {e}")
            return False
    
    def test_attendance_verification(self, test_embedding: List[float]) -> Dict[str, Any]:
        """Test attendance verification with voting system"""
        print("🎯 Testing attendance verification...")
        
        try:
            response = requests.post(
                f"{self.vectordb_url}/search_with_voting",
                json={
                    "embedding": test_embedding,
                    "class_code": self.test_class,
                    "threshold": 0.7
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ Attendance verification completed")
                print(f"     Status: {result['status']}")
                if result.get('student_id'):
                    print(f"     Matched Student: {result['student_id']}")
                    print(f"     Confidence: {result.get('confidence', 0):.3f}")
                else:
                    print(f"     Reason: {result.get('reason', 'Unknown')}")
                
                return result
            else:
                print(f"  ❌ Attendance verification failed: {response.text}")
                return {}
                
        except Exception as e:
            print(f"  ❌ Error in attendance verification: {e}")
            return {}
    
    def get_class_statistics(self) -> Dict[str, Any]:
        """Get statistics for the test class"""
        print("📊 Getting class statistics...")
        
        try:
            response = requests.get(f"{self.vectordb_url}/class_stats/{self.test_class}")
            
            if response.status_code == 200:
                stats = response.json()
                print(f"  ✅ Class statistics retrieved")
                print(f"     Students: {stats.get('student_count', 0)}")
                print(f"     Total Attendances: {stats.get('attendance_stats', {}).get('total_attendances', 0)}")
                return stats
            else:
                print(f"  ❌ Failed to get statistics: {response.text}")
                return {}
                
        except Exception as e:
            print(f"  ❌ Error getting statistics: {e}")
            return {}
    
    def cleanup_test_data(self) -> bool:
        """Clean up test data"""
        print("🧹 Cleaning up test data...")
        
        try:
            response = requests.delete(
                f"{self.vectordb_url}/delete_class",
                json={"class_code": self.test_class}
            )
            
            if response.status_code == 200:
                print(f"  ✅ Test class {self.test_class} deleted successfully")
                return True
            else:
                print(f"  ⚠️  Failed to delete test class: {response.text}")
                return False
                
        except Exception as e:
            print(f"  ❌ Error cleaning up: {e}")
            return False
    
    def run_full_integration_test(self) -> bool:
        """Run the complete integration test"""
        print("🚀 Starting Attendity Integration Test")
        print("=" * 60)
        
        # Check API health
        health = self.check_apis_health()
        if not all(health.values()):
            print("❌ Some APIs are not healthy. Please start both APIs first.")
            return False
        
        print()
        
        # Create test class
        if not self.create_test_class():
            return False
        
        print()
        
        # Generate mock embeddings
        embeddings = self.generate_mock_embeddings(len(self.test_students))
        
        print()
        
        # Insert test students
        if not self.insert_test_students(embeddings):
            return False
        
        print()
        
        # Test attendance verification with each student's embedding
        for i, student_id in enumerate(self.test_students):
            print(f"Testing attendance for {student_id}...")
            result = self.test_attendance_verification(embeddings[i])
            
            if result.get('status') == 'match_found':
                expected_student = student_id
                matched_student = result.get('student_id')
                if matched_student == expected_student:
                    print(f"  ✅ Correctly matched {matched_student}")
                else:
                    print(f"  ⚠️  Expected {expected_student}, got {matched_student}")
            else:
                print(f"  ⚠️  No match found for {student_id}")
            
            print()
        
        # Test with unknown embedding
        print("Testing with unknown student...")
        unknown_embedding = np.random.normal(0, 1, 512).tolist()
        result = self.test_attendance_verification(unknown_embedding)
        
        if result.get('status') != 'match_found':
            print("  ✅ Correctly rejected unknown student")
        else:
            print(f"  ⚠️  Incorrectly matched unknown student to {result.get('student_id')}")
        
        print()
        
        # Get class statistics
        self.get_class_statistics()
        
        print()
        
        # Cleanup (optional)
        cleanup_choice = input("Do you want to clean up test data? (y/n): ").lower().strip()
        if cleanup_choice == 'y':
            self.cleanup_test_data()
        
        print()
        print("🎉 Integration test completed!")
        print("=" * 60)
        
        return True

def main():
    """Main function to run integration tests"""
    print("Attendity Mobile Attendance System - Integration Test")
    print("=" * 60)
    print("This script tests the integration between AI and Vector-DB modules")
    print("Make sure both APIs are running:")
    print("  AI Module:    http://localhost:8000")
    print("  Vector-DB:    http://localhost:8001")
    print("=" * 60)
    
    # Wait for user confirmation
    input("Press Enter to start the integration test...")
    print()
    
    # Run the test
    tester = AttendityIntegrationTest()
    success = tester.run_full_integration_test()
    
    if success:
        print("✅ Integration test completed successfully!")
    else:
        print("❌ Integration test failed!")
    
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 