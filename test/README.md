# Attendity API Test Suite

This directory contains comprehensive tests for both AI API and Vector Database API endpoints.

## Test Structure

### Test Images
- `imgs/232323/` - Test images for student ID 232323 (3 images)
- `imgs/2114547/` - Test images for student ID 2114547 (3 images)

### Test Scripts
- `overall_test.py` - Main comprehensive test script
- `run_overall_test.py` - Test runner with API health checks
- Other legacy test files for specific scenarios

## Running Tests

### Prerequisites
1. Both APIs must be running:
   ```bash
   # Terminal 1: Start AI API
   python3 ai/api/main.py
   
   # Terminal 2: Start Vector DB API  
   python3 vector_db/api/main.py
   
   # Or use the combined runner:
   python3 run_apis.py
   ```

2. Install required dependencies:
   ```bash
   pip install httpx requests
   ```

### Execute Tests

#### Option 1: Using the test runner (Recommended)
```bash
python3 test/run_overall_test.py
```

#### Option 2: Direct execution
```bash
python3 test/overall_test.py
```

## Test Coverage

### AI API Tests
- ✅ Health check (`/health`)
- ✅ Single image inference (`/inference`)
- ✅ Fast embedding extraction (`/extract_embedding_fast`)
- ✅ Face quality validation (`/validate_quality`)
- ✅ Student insertion with 3 images (`/insert_student`)

### Vector Database API Tests
- ✅ Health check (`/health`)
- ✅ Create class - success case (`/create_class`)
- ✅ Create class - duplicate failure (`/create_class`)
- ✅ Search with voting system (`/search_with_voting`)
- ✅ Class statistics (`/class_stats/{class_code}`)
- ✅ Student attendance history (`/student_attendance/{class_code}/{student_id}`)
- ✅ Delete student - success case (`/delete_student`)
- ✅ Delete student - non-existent student (`/delete_student`)
- ✅ Delete class - success case (`/delete_class`)
- ✅ Delete class - non-existent class (`/delete_class`)

## Test Scenarios

### Success Cases
- Create new class
- Insert students with valid face images
- Search for existing students
- Delete existing students and classes

### Failure Cases
- Create duplicate class (409 Conflict)
- Delete non-existent student (graceful failure)
- Delete non-existent class (404 Not Found)

## Test Output

The test suite provides:
- ✅ Real-time test status with emojis
- 📊 Comprehensive summary with pass/fail counts
- 🔍 Detailed error messages for failed tests
- ⏱️ Performance timing information

## Example Output

```
🚀 Attendity API Overall Test Runner
==================================================
✅ Both APIs are running
🧪 Starting comprehensive tests...

Starting comprehensive API testing...
================================================================================
✅ AI API Health Check: PASS
   Details: AI API is healthy
✅ Vector DB Health Check: PASS
   Details: Vector DB is healthy
✅ AI Single Inference: PASS
   Details: Embedding extracted, confidence: 0.95
...

================================================================================
TEST SUMMARY
================================================================================
Total Tests: 15
✅ Passed: 14
❌ Failed: 0
⚠️  Warnings: 1
Success Rate: 93.3%
```

## Notes

- Tests use student IDs from the `imgs` folder structure
- Each test is independent and can be run separately
- The test suite automatically cleans up test data
- All tests use base64 encoded images as per API requirements
- Tests include proper error handling and timeout management 