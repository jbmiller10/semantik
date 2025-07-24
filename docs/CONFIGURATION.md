# Configuration Guide

This guide covers all configuration options for the Document Embedding System.

## Docker Configuration (Recommended)

When using Docker Compose, configuration is simplified through the `.env` file and docker-compose.yml.

### Quick Setup

1. Copy the Docker environment template:
   ```bash
   cp .env.docker.example .env
   ```

2. Edit `.env` with your configuration (see sections below)

3. Start services:
   ```bash
   docker compose up -d
   ```

### Docker-Specific Settings

When running with Docker Compose, the following settings are handled automatically:

- **Service URLs**: Services communicate using Docker service names (e.g., `qdrant`, `vecpipe`)
- **Volumes**: Data persistence is managed through Docker volumes
- **Networking**: All services are on the same Docker network
- **Health Checks**: Built-in health monitoring for all services

## Environment Variables

The system uses environment variables for configuration. When using Docker, copy `.env.docker.example` to `.env`. For manual installation, use `.env.example`.

### Core Settings

#### Qdrant Configuration
```bash
# Qdrant vector database connection
QDRANT_HOST=localhost          # Qdrant server hostname/IP
QDRANT_PORT=6333              # Qdrant server port
DEFAULT_COLLECTION=work_docs   # Default collection name
```

#### Embedding Model Configuration
```bash
# Embedding service settings
USE_MOCK_EMBEDDINGS=false     # Use mock embeddings for testing (true/false)
DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B  # Default model
DEFAULT_QUANTIZATION=float16  # Quantization mode: float32, float16, int8
MODEL_UNLOAD_AFTER_SECONDS=300  # Unload models after N seconds of inactivity (default: 300)
```

#### Authentication
```bash
# JWT authentication for Web UI
JWT_SECRET_KEY=your-secret-key-here  # Generate with: openssl rand -hex 32
JWT_ALGORITHM=HS256                   # JWT signing algorithm (default: HS256)
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30    # Access token expiration (default: 30)
JWT_REFRESH_TOKEN_EXPIRE_DAYS=30      # Refresh token expiration (default: 30)
DISABLE_AUTH=false                    # Disable authentication (development only!)
```

### Complete Environment Variables Reference

Here's a comprehensive list of all environment variables with their defaults:

#### Application Settings
```bash
# General
ENVIRONMENT=development               # Environment: development, staging, production
LOG_LEVEL=INFO                       # Logging level: DEBUG, INFO, WARNING, ERROR
DEBUG=false                          # Enable debug mode (default: false)

# Service Ports
SEARCH_API_PORT=8000                 # Search API port (default: 8000)
WEBUI_PORT=8080                      # WebUI port (default: 8080)

# Service URLs (for Docker/production)
SEARCH_API_URL=http://localhost:8000 # Search API URL for WebUI proxy
```

#### Database Configuration
```bash
# PostgreSQL
DATABASE_URL=postgresql://user:password@localhost:5432/semantik  # PostgreSQL connection string
# Alternative PostgreSQL configuration (if not using DATABASE_URL):
POSTGRES_HOST=localhost             # PostgreSQL host (default: localhost)
POSTGRES_PORT=5432                  # PostgreSQL port (default: 5432)
POSTGRES_DB=semantik                # Database name (default: semantik)
POSTGRES_USER=postgres              # Database user
POSTGRES_PASSWORD=password          # Database password

# Qdrant
QDRANT_HOST=localhost               # Qdrant host (default: localhost)
QDRANT_PORT=6333                    # Qdrant port (default: 6333)
QDRANT_API_KEY=                     # Qdrant API key (optional, for secured instances)
QDRANT_USE_TLS=false                # Use TLS for Qdrant connection (default: false)
```

#### Model and Processing Configuration
```bash
# Embedding Models
DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B  # Default: Qwen/Qwen3-Embedding-0.6B
DEFAULT_QUANTIZATION=float16                       # Default: float16
USE_MOCK_EMBEDDINGS=false                         # Default: false
MODEL_UNLOAD_AFTER_SECONDS=300                    # Default: 300 (5 minutes)
FORCE_CPU=false                                   # Force CPU mode (default: false)

# Processing Settings
MAX_WORKERS=4                       # Parallel processing workers (default: 4)
BATCH_SIZE=32                       # Embedding batch size (default: 32)
CHUNK_SIZE=512                      # Document chunk size in tokens (default: 512)
CHUNK_OVERLAP=128                   # Chunk overlap in tokens (default: 128)
MAX_FILE_SIZE_MB=100                # Maximum file size to process (default: 100)

# Reranking Configuration
USE_RERANKER=false                  # Enable reranking by default (default: false)
RERANK_CANDIDATE_MULTIPLIER=5       # Retrieve k*5 candidates (default: 5)
RERANK_MIN_CANDIDATES=20            # Minimum candidates (default: 20)
RERANK_MAX_CANDIDATES=200           # Maximum candidates (default: 200)
```

#### Storage and Paths
```bash
# Data Directories
DATA_DIR=/app/data                  # Main data directory
MODELS_DIR=/app/models              # Model cache directory
LOGS_DIR=/app/logs                  # Log files directory
UPLOAD_DIR=/app/uploads             # Temporary upload directory
DOCUMENTS_DIR=/documents            # Source documents directory

# Job Processing Paths
EXTRACT_DIR=/app/jobs/extract       # Extraction working directory
INGEST_DIR=/app/jobs/ingest         # Ingestion working directory
OUTPUT_DIR=/app/output              # Output directory for processed files
```

#### Performance and Resource Management
```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0              # GPU device IDs (comma-separated)
GPU_MEMORY_FRACTION=0.9             # Fraction of GPU memory to use (default: 0.9)
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # PyTorch memory config

# Memory Management
MIN_BATCH_SIZE=4                    # Minimum batch size on OOM (default: 4)
BATCH_RESTORE_THRESHOLD=5           # Batches before size increase (default: 5)
ADAPTIVE_BATCH_SIZING=true          # Enable adaptive batching (default: true)
```

#### Security Settings
```bash
# Authentication
JWT_SECRET_KEY=                     # Required: Generate with openssl rand -hex 32
JWT_ALGORITHM=HS256                 # JWT algorithm (default: HS256)
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30  # Access token lifetime (default: 30)
JWT_REFRESH_TOKEN_EXPIRE_DAYS=30    # Refresh token lifetime (default: 30)
DISABLE_AUTH=false                  # Disable auth - NEVER in production!

# CORS
CORS_ALLOWED_ORIGINS=*              # Allowed origins (default: *)
CORS_ALLOW_CREDENTIALS=true         # Allow credentials (default: true)

# Security Headers
SECURE_HEADERS_ENABLED=true         # Enable security headers (default: true)
HSTS_MAX_AGE=31536000              # HSTS max age in seconds (default: 1 year)
```

#### Monitoring and Metrics
```bash
# Prometheus Metrics
METRICS_ENABLED=true                # Enable metrics collection (default: true)
METRICS_PORT=9091                   # Metrics port (default: 9091)
METRICS_PATH=/metrics               # Metrics endpoint path (default: /metrics)

# Health Checks
HEALTH_CHECK_ENABLED=true           # Enable health checks (default: true)
HEALTH_CHECK_INTERVAL=30            # Check interval in seconds (default: 30)
```

### Optional Path Configuration

These paths have sensible defaults but can be overridden:

```bash
# Data directories
FILE_TRACKING_DB=/var/embeddings/file_tracking.json
EXTRACT_DIR=/opt/semantik/extract
INGEST_DIR=/var/embeddings/ingest
LOADED_DIR=/var/embeddings/loaded
REJECT_DIR=/var/embeddings/rejects

# File lists and logs
MANIFEST_FILE=/var/embeddings/filelist.null
ERROR_LOG=/var/embeddings/error_extract.log
CLEANUP_LOG=/var/embeddings/cleanup.log
```

## Model Configuration

### Available Models

The system supports various embedding models:

| Model | Environment Variable | Dimensions | Notes |
|-------|---------------------|------------|-------|
| BAAI/bge-large-en-v1.5 | `DEFAULT_EMBEDDING_MODEL` | 1024 | High quality, general purpose |
| BAAI/bge-base-en-v1.5 | `DEFAULT_EMBEDDING_MODEL` | 768 | Balanced performance |
| Qwen/Qwen3-Embedding-0.6B | `DEFAULT_EMBEDDING_MODEL` | 1024 | Fast, good quality (default) |
| Qwen/Qwen3-Embedding-4B | `DEFAULT_EMBEDDING_MODEL` | 2560 | Higher quality, slower |
| Qwen/Qwen3-Embedding-8B | `DEFAULT_EMBEDDING_MODEL` | 4096 | Best quality, requires more GPU |
| sentence-transformers/all-MiniLM-L6-v2 | `DEFAULT_EMBEDDING_MODEL` | 384 | Very fast, lower quality |
| sentence-transformers/all-mpnet-base-v2 | `DEFAULT_EMBEDDING_MODEL` | 768 | Good balance |

### Quantization Modes

Control memory usage and performance:

- `float32`: Full precision (best quality, highest memory use)
- `float16`: Half precision (recommended - good balance)
- `int8`: 8-bit quantization (lowest memory, slight quality loss)

### Task Instructions

For Qwen3 models, use task-specific instructions:

```python
# Document indexing
"Represent this document for retrieval:"

# Search queries
"Represent this sentence for searching relevant passages:"

# Q&A scenarios
"Represent this question for retrieving supporting answers:"
```

## Service Configuration

### Search API

The Search API (`vecpipe/search_api.py`) uses these environment variables:

- `QDRANT_HOST`, `QDRANT_PORT`: Database connection
- `DEFAULT_COLLECTION`: Default search collection
- `USE_MOCK_EMBEDDINGS`: Enable mock mode
- `DEFAULT_EMBEDDING_MODEL`: Model for query embeddings
- `DEFAULT_QUANTIZATION`: Quantization mode

### Web UI

The Web UI (`webui/main.py`) additionally uses:

- `JWT_SECRET_KEY`: Required for authentication
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Session duration
- Database paths for job management

## Runtime Configuration

### Embedding Service Options

When creating jobs or using the API, you can override defaults:

```python
{
  "model_name": "Qwen/Qwen3-Embedding-4B",
  "quantization": "int8",
  "batch_size": 32,
  "instruction": "Custom instruction"
}
```

### Search Options

Configure search behavior per request:

```python
{
  "search_type": "semantic",  # or "question", "code", "hybrid"
  "include_content": true,    # Include full text in results
  "metadata_filter": {},      # Filter by metadata fields
  "score_threshold": 0.7      # Minimum similarity score
}
```

### Hybrid Search Modes

- `filter`: Use keywords as pre-filter (faster)
- `rerank`: Retrieve more candidates and re-score (more accurate)

## Performance Tuning

### GPU Configuration

The system automatically detects and uses available GPUs. For specific GPU selection:

```bash
# Select specific GPU
CUDA_VISIBLE_DEVICES=0 python vecpipe/search_api.py

# Use CPU only
CUDA_VISIBLE_DEVICES="" python vecpipe/search_api.py
```

### Batch Sizes

Adjust based on available memory:

- GPU with 8GB VRAM: batch_size=96
- GPU with 4GB VRAM: batch_size=32
- CPU mode: batch_size=8

### Adaptive Batch Sizing

The embedding service automatically adjusts batch size on OOM errors. Configure:

```python
# In webui/embedding_service.py
MIN_BATCH_SIZE = 4          # Minimum batch size
BATCH_RESTORE_THRESHOLD = 5  # Successful batches before increasing size
```

## Logging Configuration

### Log Levels

Set logging verbosity:

```bash
# In Python scripts
import logging
logging.basicConfig(level=logging.INFO)  # or DEBUG, WARNING, ERROR
```

### Log Files

Services write to these log files:

- `search_api.log`: Search API logs
- `webui.log`: Web UI logs
- `data/jobs/job_*/output.log`: Individual job logs

## Docker Volume Configuration

When using Docker Compose, the following volumes are created:

### Named Volumes
- `qdrant_storage`: Qdrant vector database storage (persistent)

### Bind Mounts
- `./data`: Application data (jobs, processed files)
- `./logs`: Application logs
- `${DOCUMENT_PATH}`: Your documents directory (read-only)

### Managing Docker Volumes

```bash
# List volumes
docker volume ls

# Inspect a volume
docker volume inspect semantik_qdrant_storage

# Clean up unused volumes
docker volume prune
```

## Docker Networking

Services communicate through the Docker network `semantik-network`:

- WebUI → Search API: `http://vecpipe:8000`
- Services → Qdrant: `qdrant:6333`

To inspect the network:
```bash
docker network inspect semantik-network
```

## Security Configuration

### Authentication

1. **Generate secure JWT secret**:
   ```bash
   openssl rand -hex 32
   ```

2. **Set in .env**:
   ```bash
   JWT_SECRET_KEY=your-generated-secret
   ```

3. **Configure token expiration**:
   ```bash
   ACCESS_TOKEN_EXPIRE_MINUTES=1440  # 24 hours
   ```

### CORS Settings

Configure allowed origins in `webui/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Development Configuration

### Mock Mode

For testing without GPU:

```bash
# Enable mock embeddings
USE_MOCK_EMBEDDINGS=true

# Or use CLI flag
python vecpipe/embed_chunks_unified.py --mock
```

### Debug Mode

Enable detailed logging:

```bash
# Set log level to DEBUG
export LOG_LEVEL=DEBUG

# Enable FastAPI debug mode
export DEBUG=true
```

## Production Configuration Examples

### Production Environment File

Here's a complete production `.env` example:

```bash
# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Database
DATABASE_URL=postgresql://semantik:secure-password@postgres-server:5432/semantik_prod
QDRANT_HOST=qdrant.internal.company.com
QDRANT_PORT=6333
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_USE_TLS=true

# Authentication (CRITICAL - Generate new secrets!)
JWT_SECRET_KEY=<generate-with-openssl-rand-hex-32>
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
DISABLE_AUTH=false

# Models and Processing
DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
DEFAULT_QUANTIZATION=float16
MODEL_UNLOAD_AFTER_SECONDS=600
FORCE_CPU=false
MAX_WORKERS=8
BATCH_SIZE=64
CHUNK_SIZE=512
CHUNK_OVERLAP=128

# Reranking
USE_RERANKER=true
RERANK_CANDIDATE_MULTIPLIER=5

# Storage
DATA_DIR=/data/semantik/data
MODELS_DIR=/data/semantik/models
LOGS_DIR=/var/log/semantik
DOCUMENTS_DIR=/mnt/documents

# Performance
CUDA_VISIBLE_DEVICES=0,1
GPU_MEMORY_FRACTION=0.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Security
CORS_ALLOWED_ORIGINS=https://semantik.company.com
CORS_ALLOW_CREDENTIALS=true
SECURE_HEADERS_ENABLED=true

# Monitoring
METRICS_ENABLED=true
METRICS_PORT=9091
```

### Production Docker Compose Override

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  qdrant:
    restart: always
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"

  vecpipe:
    image: semantik:latest
    restart: always
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          memory: 8G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "10"

  webui:
    image: semantik:latest
    restart: always
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "10"

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - /etc/letsencrypt:/etc/letsencrypt:ro
    depends_on:
      - webui
      - vecpipe
```

## Performance Tuning Guide

### GPU Memory Optimization

#### Model Selection by GPU Memory
```bash
# 6GB GPU (RTX 3060, RTX 2060)
DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
DEFAULT_QUANTIZATION=float16
BATCH_SIZE=32

# 8-12GB GPU (RTX 3070, RTX 4060)
DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
DEFAULT_QUANTIZATION=float16
BATCH_SIZE=64

# 16-24GB GPU (RTX 4080, RTX 4090)
DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
DEFAULT_QUANTIZATION=float16
BATCH_SIZE=128

# Multi-GPU Setup
CUDA_VISIBLE_DEVICES=0,1  # Use multiple GPUs
```

#### Memory Management Strategies

```bash
# Conservative Memory Settings
MODEL_UNLOAD_AFTER_SECONDS=60      # Aggressive unloading
GPU_MEMORY_FRACTION=0.8            # Leave headroom
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Maximum Performance Settings
MODEL_UNLOAD_AFTER_SECONDS=3600    # Keep models loaded
GPU_MEMORY_FRACTION=0.95           # Use most memory
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
```

### CPU Optimization

For CPU-only deployments:

```bash
# CPU Settings
FORCE_CPU=true
USE_MOCK_EMBEDDINGS=false  # Use real models on CPU
DEFAULT_QUANTIZATION=int8  # Smallest size
BATCH_SIZE=8               # Small batches
MAX_WORKERS=16             # More parallel workers

# Use CPU-optimized models
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Storage Performance

#### SSD vs HDD Configuration
```bash
# SSD (fast storage)
BATCH_SIZE=128             # Larger batches
MAX_WORKERS=8              # More parallelism

# HDD (slow storage)
BATCH_SIZE=32              # Smaller batches
MAX_WORKERS=4              # Less parallelism
```

#### Qdrant Performance Tuning
```yaml
# qdrant_config.yaml
service:
  max_request_size_mb: 512
  max_workers: 0  # Use all CPU cores

storage:
  # Storage options
  storage_path: ./storage
  # Performance options
  on_disk_payload: false  # Keep payloads in memory
  
  # HNSW index config
  hnsw_index:
    # Higher = better recall, slower
    m: 16
    # Higher = better quality, slower indexing
    ef_construct: 200
    # Higher = slower building, faster search
    full_scan_threshold: 20000

  # Optimization
  optimizers_config:
    # Segment optimization
    default_segment_number: 8
    indexing_threshold: 20000
```

### Network Optimization

```bash
# For high-latency networks
REQUEST_TIMEOUT=120        # Longer timeouts
CONNECTION_POOL_SIZE=100   # More connections

# For low-latency networks
REQUEST_TIMEOUT=30         # Shorter timeouts
CONNECTION_POOL_SIZE=20    # Fewer connections
```

## Security Hardening Guide

### Authentication Security

```bash
# Generate secure keys
JWT_SECRET_KEY=$(openssl rand -hex 64)  # 512-bit key
DATABASE_ENCRYPTION_KEY=$(openssl rand -hex 32)

# Strict token settings
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15  # Short-lived access
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7     # Reasonable refresh
JWT_ALGORITHM=HS512                  # Stronger algorithm
```

### Network Security

#### Nginx SSL Configuration
```nginx
server {
    listen 443 ssl http2;
    server_name semantik.company.com;
    
    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/semantik.company.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/semantik.company.com/privkey.pem;
    
    # Modern SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000" always;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=100r/s;
    limit_req zone=api burst=20 nodelay;
    
    location / {
        proxy_pass http://webui:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Container Security

```yaml
# docker-compose.security.yml
services:
  webui:
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    read_only: true
    tmpfs:
      - /tmp
      - /app/uploads
```

### File System Security

```bash
# Restrict file permissions
chmod 600 .env
chmod 700 data/
chmod 755 logs/

# Create dedicated user
useradd -r -s /bin/false semantik
chown -R semantik:semantik /app
```

### Environment Variable Security

```bash
# Use Docker secrets for sensitive data
echo "$JWT_SECRET_KEY" | docker secret create jwt_secret -

# Reference in docker-compose.yml
services:
  webui:
    secrets:
      - jwt_secret
    environment:
      JWT_SECRET_KEY_FILE: /run/secrets/jwt_secret
```

## Monitoring and Alerting

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'semantik'
    static_configs:
      - targets: 
        - 'webui:9091'
        - 'vecpipe:9091'
    metrics_path: '/metrics'
```

### Grafana Dashboard

Import dashboard with key metrics:
- Search latency (p50, p95, p99)
- Embedding generation rate
- GPU memory usage
- Model load/unload frequency
- Error rates by endpoint
- Active WebSocket connections

### Health Check Configuration

```bash
# Advanced health checks
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_INTERVAL=30
HEALTH_CHECK_TIMEOUT=5
HEALTH_CHECK_RETRIES=3

# Custom health check endpoints
HEALTH_CHECK_SEARCH_ENABLED=true    # Test search functionality
HEALTH_CHECK_EMBEDDING_ENABLED=true  # Test embedding generation
HEALTH_CHECK_QDRANT_ENABLED=true    # Test Qdrant connection
```

### Log Aggregation

```yaml
# docker-compose with logging
services:
  webui:
    logging:
      driver: "syslog"
      options:
        syslog-address: "tcp://logstash.company.com:5514"
        syslog-format: "rfc5424"
        tag: "semantik-webui"
```

## Backup and Recovery

### Automated Backups

```bash
#!/bin/bash
# backup.sh
BACKUP_DIR="/backups/semantik/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup PostgreSQL database
pg_dump -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -f "$BACKUP_DIR/semantik.sql"
# Or using DATABASE_URL:
# pg_dump $DATABASE_URL -f "$BACKUP_DIR/semantik.sql"

# Backup Qdrant
curl -X POST "http://qdrant:6333/snapshots" \
  -H "api-key: $QDRANT_API_KEY" \
  -d '{"wait": true}'

# Backup configuration
cp .env "$BACKUP_DIR/"
cp docker-compose*.yml "$BACKUP_DIR/"

# Compress and encrypt
tar czf - "$BACKUP_DIR" | \
  openssl enc -aes-256-cbc -salt -pass pass:$BACKUP_PASSWORD \
  > "$BACKUP_DIR.tar.gz.enc"

# Upload to S3 (optional)
aws s3 cp "$BACKUP_DIR.tar.gz.enc" s3://backups/semantik/
```

### Recovery Procedure

```bash
# Restore from backup
openssl enc -d -aes-256-cbc -pass pass:$BACKUP_PASSWORD \
  -in backup.tar.gz.enc | tar xzf -

# Restore PostgreSQL database
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB < backup/semantik.sql
# Or using DATABASE_URL:
# psql $DATABASE_URL < backup/semantik.sql

# Restore Qdrant snapshot
curl -X PUT "http://qdrant:6333/collections/work_docs/snapshots/upload" \
  -H "api-key: $QDRANT_API_KEY" \
  -F "snapshot=@backup/qdrant_snapshot.tar"
```

## Troubleshooting Configuration Issues

### Common Problems and Solutions

1. **JWT Secret Not Set**
   ```bash
   # Error: JWT_SECRET_KEY not configured
   # Solution: Generate and set secret
   export JWT_SECRET_KEY=$(openssl rand -hex 32)
   ```

2. **Qdrant Connection Failed**
   ```bash
   # Check Qdrant is running
   docker compose ps qdrant
   
   # Test connection
   curl http://localhost:6333/health
   ```

3. **GPU Not Available**
   ```bash
   # Check CUDA setup
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi
   ```

4. **Out of Memory**
   ```bash
   # Reduce memory usage
   DEFAULT_QUANTIZATION=int8
   BATCH_SIZE=16
   MODEL_UNLOAD_AFTER_SECONDS=60
   ```

For more troubleshooting tips, see the main [README.md](../README.md#troubleshooting-common-issues).