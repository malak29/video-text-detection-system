# Video Text Detection System

Production-ready API for detecting and recognizing text in video files using deep learning models. Supports real-time processing, batch operations, and horizontal scaling.

## Architecture Overview

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Streamlit Web App<br/>Port: 8501]
    end
    
    subgraph "API Layer"
        API[FastAPI REST API<br/>Port: 8000]
        MW[Middleware<br/>Auth, CORS, Rate Limiting]
        API --> MW
    end
    
    subgraph "Authentication"
        AUTH[JWT Authentication<br/>User Management]
    end
    
    subgraph "Database Layer"
        DB[(PostgreSQL<br/>Port: 5432)]
        MODELS[SQLAlchemy Models<br/>Users, Videos, Jobs, Detections]
        DB --> MODELS
    end
    
    subgraph "Message Queue"
        REDIS[(Redis<br/>Port: 6379)]
        BROKER[Celery Broker<br/>Task Queue]
        REDIS --> BROKER
    end
    
    subgraph "Background Processing"
        WORKER1[Celery Worker 1<br/>GPU Enabled]
        WORKER2[Celery Worker 2<br/>GPU Enabled]
        BEAT[Celery Beat<br/>Scheduler]
        WORKER1 --> PIPELINE
        WORKER2 --> PIPELINE
    end
    
    subgraph "ML Pipeline"
        PIPELINE[Video Text Pipeline]
        DETECTOR[Text Detector<br/>DBNet Model]
        RECOGNIZER[Text Recognizer<br/>CRNN + Transformers]
        PIPELINE --> DETECTOR
        PIPELINE --> RECOGNIZER
    end
    
    subgraph "Storage Layer"
        LOCAL[Local Storage<br/>./uploads]
        S3[AWS S3<br/>Optional]
        TEMP[Temp Files<br/>./temp]
        MODELS_STORAGE[Model Files<br/>./models]
    end
    
    subgraph "Monitoring"
        PROMETHEUS[Prometheus<br/>Port: 9090]
        HEALTH[Health Checks<br/>System Status]
        METRICS[Custom Metrics<br/>Processing Stats]
        PROMETHEUS --> HEALTH
        PROMETHEUS --> METRICS
    end
    
    subgraph "Container Orchestration"
        K8S[Kubernetes Cluster]
        HPA[Horizontal Pod Autoscaler]
        PVC[Persistent Volume Claims]
        K8S --> HPA
        K8S --> PVC
    end
    
    UI --> API
    API --> AUTH
    API --> DB
    API --> REDIS
    REDIS --> WORKER1
    REDIS --> WORKER2
    REDIS --> BEAT
    
    WORKER1 --> LOCAL
    WORKER1 --> S3
    WORKER2 --> LOCAL
    WORKER2 --> S3
    
    PIPELINE --> TEMP
    PIPELINE --> MODELS_STORAGE
    
    API --> PROMETHEUS
    WORKER1 --> PROMETHEUS
    WORKER2 --> PROMETHEUS
    
    K8S -.-> API
    K8S -.-> WORKER1
    K8S -.-> WORKER2
    K8S -.-> DB
    K8S -.-> REDIS
```

## System Components

### Frontend Layer
- **Streamlit Web App**: User interface for video upload, processing monitoring, and results visualization
- **Features**: File upload, real-time progress tracking, results export, analytics dashboard

### API Layer
- **FastAPI REST API**: High-performance async API with automatic OpenAPI documentation
- **Authentication**: JWT-based user authentication and authorization
- **Middleware**: Rate limiting, CORS handling, request logging, security headers

### Database Layer
- **PostgreSQL**: Primary database for persistent data storage
- **Models**: Users, Videos, Frames, TextDetections, ProcessingJobs, ModelVersions
- **Migrations**: Alembic-based database schema versioning

### Background Processing
- **Celery Workers**: Async task processing with GPU support for ML inference
- **Task Queue**: Redis-based message broker for distributed task management
- **Scheduler**: Celery Beat for periodic maintenance tasks

### ML Pipeline
- **Text Detection**: DBNet model for locating text regions in video frames
- **Text Recognition**: CRNN + Transformer models for reading detected text
- **Preprocessing**: Frame extraction, image enhancement, batch processing

### Storage Layer
- **Local Storage**: File system storage for uploaded videos and results
- **AWS S3**: Optional cloud storage for scalable file management
- **Temporary Storage**: Processing workspace with automatic cleanup

### Monitoring
- **Prometheus**: Metrics collection and monitoring
- **Health Checks**: System component status monitoring
- **Custom Metrics**: Processing statistics and performance tracking

## Technology Stack

**Backend**: Python 3.11, FastAPI, SQLAlchemy, Alembic
**ML Framework**: PyTorch, PyTorch Lightning, Transformers, OpenCV
**Database**: PostgreSQL 15
**Message Queue**: Redis 7, Celery
**Frontend**: Streamlit
**Containerization**: Docker, Docker Compose
**Orchestration**: Kubernetes
**Monitoring**: Prometheus, Custom health checks
**CI/CD**: GitHub Actions
**Storage**: Local filesystem, AWS S3 (optional)

## Key Features

- **Real-time Processing**: Async video text detection with progress tracking
- **Scalable Architecture**: Horizontal scaling with Kubernetes and auto-scaling
- **GPU Acceleration**: CUDA support for ML model inference
- **Multiple Storage Options**: Local storage and AWS S3 integration
- **Comprehensive Monitoring**: Prometheus metrics and health checks
- **Production Security**: JWT authentication, rate limiting, input validation
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation
- **Background Tasks**: Celery-based async processing with retry mechanisms
- **Database Migrations**: Alembic for schema versioning
- **Testing Suite**: Unit, integration, and API tests with CI/CD

## Quick Start

### Local Development

1. **Clone Repository**
   ```bash
   git clone https://github.com/your-org/video-text-detection.git
   cd video-text-detection
   ```

2. **Environment Setup**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start Services**
   ```bash
   docker-compose up -d
   ```

4. **Access Applications**
   - API: http://localhost:8000
   - Frontend: http://localhost:8501
   - API Docs: http://localhost:8000/docs

### Production Deployment

1. **Kubernetes Deployment**
   ```bash
   kubectl apply -f k8s/deployment.yaml
   ```

2. **Configure Ingress**
   ```bash
   # Update k8s/deployment.yaml with your domain
   kubectl apply -f k8s/deployment.yaml
   ```

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `GET /api/v1/auth/me` - Current user info

### Videos
- `POST /api/v1/videos/upload` - Upload video file
- `GET /api/v1/videos/` - List user videos
- `GET /api/v1/videos/{id}` - Get video details
- `DELETE /api/v1/videos/{id}` - Delete video

### Processing
- `POST /api/v1/processing/videos/{id}/detect` - Start text detection
- `GET /api/v1/processing/jobs/{id}/status` - Check processing status
- `GET /api/v1/processing/videos/{id}/results` - Get detection results
- `GET /api/v1/processing/videos/{id}/annotated` - Download annotated video

### System
- `GET /health` - System health check
- `GET /metrics` - Prometheus metrics

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/videotext
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-super-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Storage
MAX_FILE_SIZE=524288000  # 500MB
MAX_VIDEO_DURATION=300   # 5 minutes
TEMP_DIR=./temp
OUTPUT_DIR=./output

# ML Configuration
GPU_ENABLED=true
BATCH_SIZE=32
CONFIDENCE_THRESHOLD=0.5

# AWS (Optional)
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
S3_BUCKET_NAME=your-bucket

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_LEVEL=INFO
```

## Development

### Setup Development Environment

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Database Setup**
   ```bash
   alembic upgrade head
   ```

3. **Run Tests**
   ```bash
   pytest tests/ -v
   ```

4. **Code Formatting**
   ```bash
   black app tests
   isort app tests
   flake8 app tests
   ```

### Model Training

1. **Prepare Dataset**
   ```python
   python scripts/prepare_training_data.py
   ```

2. **Train Detection Model**
   ```python
   python -m app.ml.training.train_detector
   ```

3. **Train Recognition Model**
   ```python
   python -m app.ml.training.train_recognizer
   ```

## Monitoring and Alerting

### Prometheus Metrics
- Request rates and response times
- Processing job statistics
- System resource usage
- Model inference performance
- Error rates and types

### Health Checks
- Database connectivity
- Redis availability
- Model file integrity
- Disk space monitoring
- Memory usage tracking


## License

MIT License - see LICENSE file for details
