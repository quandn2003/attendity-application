#!/usr/bin/env python3

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the student test script with proper setup"""
    
    print("Hi Boss Quan! 🚀")
    print("=" * 60)
    print("Student Test Script Runner")
    print("=" * 60)
    
    workspace_root = Path(__file__).parent.parent
    os.chdir(workspace_root)
    
    test_script_path = "test/student_test_script.py"
    
    if not Path(test_script_path).exists():
        print(f"❌ Test script not found: {test_script_path}")
        return 1
    
    print("📋 Prerequisites:")
    print("   1. Both APIs should be running (AI API on port 8000, Vector DB on port 8001)")
    print("   2. Start APIs with: python3 run_apis.py")
    print("   3. Test images should be in test/imgs/232323/ and test/imgs/2114547/")
    print()
    
    try:
        print("🏃 Running student test script...")
        print("=" * 60)
        
        result = subprocess.run([
            sys.executable, test_script_path
        ], check=False, cwd=workspace_root)
        
        print("=" * 60)
        if result.returncode == 0:
            print("✅ Test script completed successfully!")
        else:
            print("❌ Test script failed!")
            
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running test script: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 