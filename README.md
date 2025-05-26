# Attendity - Mobile Attendance System

A comprehensive mobile attendance system using AI face recognition and vector database technology, optimized for mobile CPU deployment.

## 🏗️ Project Architecture

```
attendity-application/
├── ai/                          # AI Module - Face Recognition & Anti-Spoofing
│   ├── models/                  # FaceNet model implementation
│   ├── utils/                   # Preprocessing, anti-spoofing, voting
│   ├── inference/               # Inference engine
│   ├── api/                     # FastAPI endpoints
│   └── requirements.txt         # AI module dependencies
├── vector-db/                   # Vector Database Module
│   ├── database/                # ChromaDB client and managers
│   ├── voting/                  # Similarity voting system
│   ├── config/                  # Database configuration
│   ├── api/                     # FastAPI endpoints
│   └── requirements.txt         # Vector-DB dependencies
├── run_apis.py                  # Startup script for both APIs
├── integration_test.py          # Integration testing script
└── README.md                    # This file
```

## 🚀 Key Features

### AI Module
- **Mobile CPU Optimization**: Quantized models, CPU threading, memory management
- **3-Image Voting System**: Consensus building from multiple face images for student insertion
- **Anti-Spoofing Detection**: Texture analysis, color distribution, quality metrics
- **FaceNet Implementation**: 512-dimensional embeddings with Inception ResNet v1
- **Quality Validation**: Face detection confidence, image quality checks

### Vector Database Module
- **ChromaDB Integration**: Persistent vector storage optimized for mobile
- **Top-3 Voting System**: Advanced similarity voting for attendance verification
- **Student Management**: Batch operations, consensus embeddings
- **Attendance Tracking**: Real-time attendance recording with confidence scores
- **Mobile Optimization**: Compression, cleanup, memory management

## 📋 Requirements

### System Requirements
- Python 3.8+
- CPU: Multi-core processor (4+ cores recommended)
- RAM: 4GB+ (8GB recommended)
- Storage: 2GB+ free space

### Dependencies
All dependencies are managed through requirements.txt files in each module.

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd attendity-application
```

### 2. Install AI Module Dependencies
```bash
cd ai
pip install -r requirements.txt
cd ..
```

### 3. Install Vector-DB Module Dependencies
```bash
cd vector-db
pip install -r requirements.txt
cd ..
```

### 4. Install Integration Test Dependencies
```bash
pip install requests numpy
```

## 🚀 Running the System

### Option 1: Run Both APIs Together (Recommended)
```bash
python run_apis.py
```

This will start:
- AI Module API on `http://localhost:8000`
- Vector-DB API on `http://localhost:8001`

### Option 2: Run APIs Separately

**Terminal 1 - AI Module:**
```bash
cd ai
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Vector-DB Module:**
```bash
cd vector-db
python -m uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
```

## 📚 API Documentation

### AI Module API (`http://localhost:8000`)
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

#### Key Endpoints:
- `POST /inference` - Extract face embedding with anti-spoofing
- `POST /insert_student` - Process 3 images for student insertion
- `POST /extract_embedding_fast` - Fast embedding extraction
- `GET /health` - Health check

### Vector-DB API (`http://localhost:8001`)
- **Swagger UI**: `http://localhost:8001/docs`
- **ReDoc**: `http://localhost:8001/redoc`

#### Key Endpoints:
- `POST /create_class` - Create new class collection
- `POST /insert_student` - Insert students with consensus embeddings
- `POST /search_with_voting` - Attendance verification with voting
- `GET /class_stats/{class_code}` - Class statistics
- `GET /health` - Health check

## 🧪 Testing

### Integration Test
Run the comprehensive integration test to verify both modules work together:

```bash
python integration_test.py
```

This test will:
1. Check API health
2. Create a test class
3. Insert test students with mock embeddings
4. Test attendance verification
5. Verify voting system functionality
6. Display statistics

### Manual Testing

#### 1. Create a Class
```bash
curl -X POST "http://localhost:8001/create_class" \
     -H "Content-Type: application/json" \
     -d '{"class_code": "CS101"}'
```

#### 2. Insert Students (with mock embeddings)
```bash
curl -X POST "http://localhost:8001/insert_student" \
     -H "Content-Type: application/json" \
     -d '{
       "students": [
         {
           "student_id": "student001",
           "class_code": "CS101",
           "embedding": [0.1, 0.2, ..., 0.512]
         }
       ]
     }'
```

#### 3. Verify Attendance
```bash
curl -X POST "http://localhost:8001/search_with_voting" \
     -H "Content-Type: application/json" \
     -d '{
       "embedding": [0.1, 0.2, ..., 0.512],
       "class_code": "CS101",
       "threshold": 0.7
     }'
```

## 🔧 Configuration

### AI Module Configuration
Located in `ai/models/facenet_model.py`:
```python
@dataclass
class ModelConfig:
    embedding_dim: int = 512
    input_size: Tuple[int, int] = (160, 160)
    cpu_threads: int = 4
    quantization: bool = True
```

### Vector-DB Configuration
Located in `vector-db/config/database_config.py`:
```python
@dataclass
class DatabaseConfig:
    similarity_threshold: float = 0.6
    voting_threshold: float = 0.8
    top_k_results: int = 3
    max_batch_size: int = 100
```

## 📱 Mobile Deployment Considerations

### CPU Optimization
- Model quantization (INT8/FP16)
- CPU threading optimization
- Memory management
- Efficient batch processing

### Storage Optimization
- Database compression
- Automatic cleanup
- Backup management
- Size limits

### Performance Monitoring
- Processing time tracking
- Memory usage monitoring
- CPU utilization
- Battery impact assessment

## 🔄 Workflow

### Student Registration (3-Image Voting)
1. **Capture 3 Images**: Take 3 face photos of the student
2. **AI Processing**: Extract embeddings from each image
3. **Voting System**: Verify consistency between the 3 images
4. **Consensus Building**: Generate consensus embedding if consistent
5. **Database Storage**: Store student with consensus embedding

### Attendance Verification
1. **Capture Image**: Take attendance photo
2. **AI Processing**: Extract face embedding with anti-spoofing
3. **Database Search**: Find top-3 similar students in class
4. **Voting Decision**: Apply voting logic for final decision
5. **Record Attendance**: Log attendance if match found

## 🛡️ Security Features

- **Anti-Spoofing Detection**: Prevents photo/video attacks
- **Quality Validation**: Ensures good face image quality
- **Confidence Thresholds**: Configurable security levels
- **Voting System**: Reduces false positives
- **Audit Trail**: Complete attendance logging

## 📊 Performance Metrics

### AI Module
- Face detection: ~50-100ms (CPU)
- Embedding extraction: ~100-200ms (CPU)
- Anti-spoofing: ~20-50ms (CPU)
- Total processing: ~200-400ms per image

### Vector-DB Module
- Similarity search: ~10-50ms
- Voting decision: ~5-20ms
- Database operations: ~5-30ms

## 🐛 Troubleshooting

### Common Issues

1. **APIs not starting**
   - Check port availability (8000, 8001)
   - Verify dependencies installation
   - Check Python version (3.8+)

2. **Memory issues**
   - Reduce batch sizes in configuration
   - Enable model quantization
   - Monitor memory usage

3. **Slow performance**
   - Increase CPU threads
   - Enable quantization
   - Optimize database settings

### Logs
- AI Module logs: Check console output
- Vector-DB logs: Check console output
- Integration test logs: Detailed test results

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- FaceNet paper and implementation
- ChromaDB for vector database
- FastAPI for API framework
- PyTorch for deep learning

---

**Note**: This system is optimized for mobile CPU deployment. For production use, consider additional security measures, scalability requirements, and compliance with privacy regulations. 