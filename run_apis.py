#!/usr/bin/env python3
"""
Startup script for running both AI and Vector-DB APIs
Runs both FastAPI applications on different ports for development and testing.
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path

def run_api(module_path, port, name):
    """Run a FastAPI application using uvicorn."""
    cmd = [
        sys.executable, "-m", "uvicorn",
        f"{module_path}:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--reload",
        "--log-level", "info"
    ]
    
    print(f"Starting {name} API on port {port}...")
    print(f"Command: {' '.join(cmd)}")
    
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

def main():
    """Main function to run both APIs."""
    print("=" * 60)
    print("Attendity Application - AI & Vector-DB APIs")
    print("=" * 60)
    
    workspace_root = Path(__file__).parent
    os.chdir(workspace_root)
    
    processes = []
    
    try:
        ai_process = run_api("ai.api.main", 8000, "AI Module")
        processes.append(("AI Module", ai_process))
        time.sleep(2)
        
        vectordb_process = run_api("vector_db.api.main", 8001, "Vector-DB Module")
        processes.append(("Vector-DB Module", vectordb_process))
        time.sleep(2)
        
        print("\n" + "=" * 60)
        print("Both APIs are starting up...")
        print("AI Module API:      http://localhost:8000")
        print("AI Module Docs:     http://localhost:8000/docs")
        print("Vector-DB API:      http://localhost:8001")
        print("Vector-DB Docs:     http://localhost:8001/docs")
        print("=" * 60)
        print("\nPress Ctrl+C to stop both APIs")
        print("=" * 60)
        
        while True:
            for name, process in processes:
                if process.poll() is not None:
                    print(f"\n{name} process has stopped unexpectedly!")
                    return_code = process.returncode
                    if return_code != 0:
                        print(f"Exit code: {return_code}")
                        stdout, stderr = process.communicate()
                        if stdout:
                            print(f"Output: {stdout}")
                        if stderr:
                            print(f"Error: {stderr}")
                    return 1
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nShutting down APIs...")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        
    finally:
        for name, process in processes:
            if process.poll() is None:
                print(f"Terminating {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Force killing {name}...")
                    process.kill()
                    process.wait()
        
        print("All APIs stopped.")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 