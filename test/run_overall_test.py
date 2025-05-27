#!/usr/bin/env python3

import os
import sys
import subprocess
import time

def check_api_running(url: str, name: str) -> bool:
    """Check if API is running"""
    try:
        import requests
        response = requests.get(f"{url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("🚀 Attendity API Overall Test Runner")
    print("="*50)
    
    # Check if both APIs are running
    ai_running = check_api_running("http://localhost:8000", "AI API")
    db_running = check_api_running("http://localhost:8001", "Vector DB API")
    
    if not ai_running:
        print("❌ AI API (port 8000) is not running!")
        print("   Please start it with: python3 ai/api/main.py")
        
    if not db_running:
        print("❌ Vector DB API (port 8001) is not running!")
        print("   Please start it with: python3 vector_db/api/main.py")
    
    if not (ai_running and db_running):
        print("\n⚠️  Please start both APIs before running tests")
        print("   You can use: python3 run_apis.py")
        return 1
    
    print("✅ Both APIs are running")
    print("🧪 Starting comprehensive tests...\n")
    
    # Change to project root directory
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Run the overall test
    try:
        result = subprocess.run([sys.executable, "test/overall_test.py"], 
                              capture_output=False, text=True)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 