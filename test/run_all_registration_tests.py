#!/usr/bin/env python3
"""
Test Runner for Registration Tests
Runs registration test scripts in sequence
"""

import subprocess
import sys
import os
import argparse
import time

def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"🎯 {title}")
    print("=" * 80)

def print_step(step: str, status: str = "info"):
    """Print formatted step"""
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "progress": "🔄"}
    icon = icons.get(status, "ℹ️")
    print(f"  {icon} {step}")

def run_test_script(script_path: str, ai_url: str, additional_args: list = None) -> bool:
    """Run a test script and return success status"""
    try:
        cmd = [sys.executable, script_path, "--ai-url", ai_url]
        if additional_args:
            cmd.extend(additional_args)
        
        print_step(f"Running: {os.path.basename(script_path)}", "progress")
        
        # Run the script
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(script_path),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print_step(f"✓ {os.path.basename(script_path)} completed successfully", "success")
            return True
        else:
            print_step(f"✗ {os.path.basename(script_path)} failed", "error")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print_step(f"✗ {os.path.basename(script_path)} timed out", "error")
        return False
    except Exception as e:
        print_step(f"✗ {os.path.basename(script_path)} error: {e}", "error")
        return False

def main():
    """Main function to run all registration tests"""
    parser = argparse.ArgumentParser(description='Run Registration Tests')
    parser.add_argument('--ai-url', '-u', type=str, default='http://localhost:8000',
                       help='AI Module URL (default: http://localhost:8000)')
    parser.add_argument('--skip-interactive', '-s', action='store_true',
                       help='Skip interactive prompts')
    
    args = parser.parse_args()
    
    print_header("REGISTRATION TEST SUITE RUNNER")
    print("This script runs registration test scripts in sequence:")
    print("  📊 Base64 Only Test (existing functionality)")
    print("  🔧 Registration System Test (clean implementation)")
    print("=" * 80)
    print("Prerequisites:")
    print(f"  - AI Module running on {args.ai_url}")
    print("  - opencv-python installed (pip install opencv-python)")
    print("  - Images available in test/imgs folder")
    print("=" * 80)
    
    if not args.skip_interactive:
        input("Press Enter to start all tests...")
    
    # Get the test directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define test scripts to run
    test_scripts = [
        {
            "name": "Base64 Only Test",
            "script": os.path.join(test_dir, "test_base64_only.py"),
            "description": "Tests existing base64-only functionality",
            "args": []
        },
        {
            "name": "Registration System Test",
            "script": os.path.join(test_dir, "test_registration.py"),
            "description": "Tests registration system logic",
            "args": []
        }
    ]
    
    start_time = time.time()
    passed_tests = 0
    failed_tests = []
    
    for test_info in test_scripts:
        print_header(test_info["name"])
        print(f"Description: {test_info['description']}")
        print(f"Script: {os.path.basename(test_info['script'])}")
        
        if not os.path.exists(test_info["script"]):
            print_step(f"✗ Script not found: {test_info['script']}", "error")
            failed_tests.append(test_info["name"])
            continue
        
        if run_test_script(test_info["script"], args.ai_url, test_info["args"]):
            passed_tests += 1
        else:
            failed_tests.append(test_info["name"])
        
        print_step(f"Completed: {test_info['name']}", "info")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Final summary
    print_header("FINAL TEST SUMMARY")
    print_step(f"Total Tests: {len(test_scripts)}", "info")
    print_step(f"Passed: {passed_tests}", "success" if passed_tests == len(test_scripts) else "info")
    print_step(f"Failed: {len(failed_tests)}", "error" if failed_tests else "info")
    print_step(f"Total Time: {total_time:.2f} seconds", "info")
    
    if failed_tests:
        print_step("Failed Tests:", "error")
        for test in failed_tests:
            print(f"    - {test}")
    
    print("\n🎯 REGISTRATION SYSTEM SUMMARY:")
    print("  ✅ Clean registration implementation")
    print("  ✅ Requires exactly 3 successfully embedded images")
    print("  ✅ Simple averaging of valid embeddings")
    print("  ✅ Efficient processing and better user experience")
    print("  ✅ No complex voting or similarity calculations")
    
    success = len(failed_tests) == 0
    final_status = "✅ ALL TESTS PASSED!" if success else "❌ SOME TESTS FAILED!"
    print_step(final_status, "success" if success else "error")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 