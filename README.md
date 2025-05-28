# Attendity - Mobile Face Recognition Attendance System

A mobile-optimized attendance system using face recognition technology with anti-spoofing capabilities, built with Python, PyTorch, and ChromaDB.

## 🏗️ Architecture

```
attendity-application/
├── ai/                          # AI Module
│   ├── models/                  # FaceNet model implementation
│   ├── inference/               # Inference engine
│   ├── utils/                   # Preprocessing & anti-spoofing
│   └── api/                     # FastAPI AI service
└── vector_db/                   # Vector Database Module
    ├── database/                # ChromaDB client & managers
    ├── voting/                  # Similarity voting system
    ├── config/                  # Database configuration
    └── api/                     # FastAPI database service
```

## 🚀 Features

### AI Module
- **Face Recognition**: FaceNet (Inception-ResNet-V1) with 512-dimensional embeddings
- **Anti-Spoofing**: Multiple detection methods for presentation attack detection
- **Mobile Optimization**: CPU-optimized inference with model quantization
- **Multi-Image Processing**: Student registration with 3-image consensus

### Vector Database Module
- **ChromaDB Integration**: Persistent vector storage with similarity search
- **Voting System**: Top-3 candidate voting for attendance verification
- **Class Management**: Separate collections per class with batch operations
- **Attendance Tracking**: Real-time verification with history logging

## 📋 Requirements

- Python 3.8+
- Git

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/quandn2003/attendity-application.git
cd attendity-application
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Quick Start

### 1. Start AI Service
```bash
uvicorn ai.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start Vector Database Service
```bash
uvicorn vector_db.api.main:app --host 0.0.0.0 --port 8001 --reload
```

### 3. Health Check
```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
```

## 📚 API Documentation

### AI Module Endpoints (Port 8000)

#### Extract Face Embedding
```bash
POST /inference
{
  "image": "base64_encoded_image"
}
```

#### Student Registration (3 Images)
```bash
POST /insert_student
{
  "class_code": "CS101",
  "student_id": "12345",
  "image1": "base64_image_1",
  "image2": "base64_image_2", 
  "image3": "base64_image_3"
}
```

### Vector Database Endpoints (Port 8001)

#### Create Class
```bash
POST /create_class
{
  "class_code": "CS101"
}
```

#### Register Student
```bash
POST /insert_student
{
  "students": [{
    "student_id": "12345",
    "class_code": "CS101",
    "embedding": [0.1, 0.2, ...]
  }]
}
```

#### Verify Attendance
```bash
POST /search_with_voting
{
  "embedding": [0.1, 0.2, ...],
  "class_code": "CS101",
  "threshold": 0.7
}
```

## 🔧 Configuration

### AI Module Configuration
```python
ModelConfig(
    embedding_dim=512,
    input_size=(160, 160),
    pretrained='vggface2',
    cpu_threads=4,
    quantization=True,
    device='cpu'
)
```

### Vector Database Configuration
```python
DatabaseConfig(
    persist_directory="./chroma_db",
    embedding_dimension=512,
    similarity_threshold=0.6,
    voting_threshold=0.8,
    max_batch_size=100
)
```

## 🔄 Key Workflows

### Student Registration
1. Capture 3 face images
2. AI Module processes images → extracts embeddings
3. Vector DB stores embeddings with metadata
4. Return success/failure status

### Attendance Verification
1. Capture single face image
2. AI Module extracts embedding + anti-spoofing check
3. Vector DB searches similar embeddings
4. Voting system applies top-3 voting logic
5. Return attendance decision

## 🛡️ Security Features

### Anti-Spoofing Protection
- Multiple detection algorithms
- Liveness detection
- Quality assessment
- Confidence thresholding

### Data Protection
- Local data storage (no cloud dependencies)
- Encrypted embeddings support
- Access control via API authentication

## ⚡ Performance Optimizations

### Mobile CPU Optimizations
- Model quantization for faster inference
- Optimized batch sizes for memory constraints
- Configurable CPU thread usage
- Efficient memory management

### Database Optimizations
- Collection partitioning per class
- Batch insert/delete operations
- Optimized vector search algorithms
- Automated data cleanup

## 🔍 Monitoring

### Health Checks
- Model loading status
- Database connectivity
- Memory usage monitoring
- Performance metrics

### Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 🧪 Testing

### Test Face Recognition
```bash
curl -X POST "http://localhost:8000/inference" \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_test_image"}'
```

### Benchmark Performance
```bash
curl -X POST "http://localhost:8000/benchmark" \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image", "iterations": 10}'
```


## 🔮 Future Enhancements

- Model fine-tuning capabilities
- Advanced anti-spoofing methods
- Real-time performance monitoring
- Automated threshold tuning
- Multi-modal biometric fusion
- Distributed database support

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For technical support or questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review API documentation at `/docs` endpoints

---

**Attendity Team** - Mobile Face Recognition Attendance System 