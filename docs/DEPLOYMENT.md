# Deployment Guide for Semantik

## Overview

This guide covers deploying Semantik in various environments, from development to production. Semantik can be deployed using Docker, systemd services, or manual process management.

## Deployment Options

### 1. Docker Deployment (Recommended)
- **Best for**: Production, staging, consistent environments
- **Benefits**: Isolated dependencies, easy scaling, reproducible builds
- **Requirements**: Docker 20.10+, Docker Compose 2.0+

### 2. Systemd Services
- **Best for**: Linux servers, automatic restart, system integration
- **Benefits**: OS-level management, logging integration, resource limits
- **Requirements**: systemd-based Linux distribution

### 3. Manual Deployment
- **Best for**: Development, debugging, custom setups
- **Benefits**: Direct control, easy debugging, flexible configuration
- **Requirements**: Python 3.12+, CUDA toolkit (optional)

## Quick Start

### Using Docker Compose
```bash
# Clone the repository
git clone https://github.com/your-org/semantik.git
cd semantik

# Copy environment template
cp .env.example .env

# Edit configuration
nano .env

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Using Shell Scripts
```bash
# Install dependencies
poetry install

# Run database migrations
poetry run alembic upgrade head

# Start all services
./start_all_services.sh

# Check status
./status_services.sh
```

## Environment Variables

### Core Configuration
```bash
# Application Settings
ENVIRONMENT=production              # development, staging, production
LOG_LEVEL=INFO                     # DEBUG, INFO, WARNING, ERROR

# Database Configuration
DATABASE_URL=postgresql://user:pass@host:5432/db  # PostgreSQL connection
QDRANT_HOST=localhost             # Qdrant vector database host
QDRANT_PORT=6333                  # Qdrant port

# API Configuration
SEARCH_API_PORT=8000              # Search API port
WEBUI_PORT=8080                   # WebUI port

# Model Configuration
DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
DEFAULT_QUANTIZATION=float16      # float32, float16, int8
USE_MOCK_EMBEDDINGS=false         # Use mock embeddings (no GPU)
MODEL_UNLOAD_AFTER_SECONDS=300    # Auto-unload inactive models

# Authentication
JWT_SECRET_KEY=your-secret-key-here  # Change in production!
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=30
DISABLE_AUTH=false                # Disable auth (development only)

# Performance
MAX_WORKERS=4                     # Parallel processing workers
BATCH_SIZE=32                     # Embedding batch size
CHUNK_SIZE=512                    # Document chunk size
CHUNK_OVERLAP=128                 # Chunk overlap tokens

# Monitoring
METRICS_PORT=9091                 # Prometheus metrics port
ENABLE_METRICS=true               # Enable metrics collection
```

### GPU Configuration
```bash
# CUDA Settings
CUDA_VISIBLE_DEVICES=0            # GPU device ID (comma-separated)
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Memory Management
GPU_MEMORY_FRACTION=0.9           # Fraction of GPU memory to use
FORCE_CPU=false                   # Force CPU mode
```

### Storage Configuration
```bash
# Data Directories
DATA_DIR=/app/data                # Main data directory
MODELS_DIR=/app/models            # Model cache directory
LOGS_DIR=/app/logs                # Log files directory
UPLOAD_DIR=/app/uploads           # Temporary upload directory

# Volume Mounts (Docker)
DOCUMENTS_DIR=/documents          # Document source directory
```

## Docker Deployment

### Production Docker Compose
```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    restart: unless-stopped
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__ENABLE_TLS=false
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  search-api:
    build:
      context: .
      dockerfile: Dockerfile
    image: semantik:latest
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
      - ${DOCUMENTS_DIR}:/documents:ro
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - USE_MOCK_EMBEDDINGS=${USE_MOCK_EMBEDDINGS:-false}
      - DEFAULT_EMBEDDING_MODEL=${DEFAULT_EMBEDDING_MODEL}
      - DEFAULT_QUANTIZATION=${DEFAULT_QUANTIZATION}
      - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    depends_on:
      qdrant:
        condition: service_healthy
    command: ["python", "-m", "vecpipe.search_api"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  webui:
    build:
      context: .
      dockerfile: Dockerfile
    image: semantik:latest
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
      - ${DOCUMENTS_DIR}:/documents:ro
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - SEARCH_API_URL=http://search-api:8000
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - DISABLE_AUTH=${DISABLE_AUTH:-false}
    depends_on:
      - search-api
      - qdrant
    command: ["python", "-m", "webui.main"]

volumes:
  qdrant_storage:
    driver: local

networks:
  default:
    name: semantik-network
```

### Building Docker Images
```bash
# Build standard image
docker build -t semantik:latest .

# Build with CUDA support
docker build -f Dockerfile.cuda -t semantik:cuda .

# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.12 -t semantik:py312 .
```

### Docker Health Checks
```dockerfile
# In Dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1
```

## Volume Management

### Required Volumes
1. **Data Volume**: Application data and job tracking
   ```bash
   ./data:/app/data
   ```

2. **Models Volume**: Cached embedding models
   ```bash
   ./models:/app/models
   ```

3. **Logs Volume**: Application logs
   ```bash
   ./logs:/app/logs
   ```

4. **Documents Volume**: Source documents (read-only)
   ```bash
   /path/to/documents:/documents:ro
   ```

### Volume Permissions
```bash
# Set correct permissions
sudo chown -R 1000:1000 ./data ./models ./logs

# Create directories if needed
mkdir -p data models logs
```

### Backup Strategy
```bash
# Backup script
#!/bin/bash
BACKUP_DIR="/backups/semantik"
DATE=$(date +%Y%m%d_%H%M%S)

# Stop services
docker-compose stop webui

# Backup database
cp data/webui.db "$BACKUP_DIR/webui_${DATE}.db"

# Backup Qdrant
docker exec qdrant qdrant-backup create /qdrant/backup
docker cp qdrant:/qdrant/backup "$BACKUP_DIR/qdrant_${DATE}"

# Restart services
docker-compose start webui
```

## Service Dependencies

### Startup Order
```mermaid
graph LR
    A[Qdrant] --> B[Search API]
    B --> C[WebUI]
    A --> C
```

### Service Dependencies
1. **Qdrant**: Must be healthy before other services start
2. **Search API**: Requires Qdrant, provides embedding/search
3. **WebUI**: Requires both Qdrant and Search API

### Dependency Configuration
```yaml
depends_on:
  qdrant:
    condition: service_healthy
  search-api:
    condition: service_started
```

## Health Checks and Monitoring

### Application Health Endpoints
```bash
# WebUI health check
curl http://localhost:8080/health

# Search API health check
curl http://localhost:8000/

# Qdrant health check
curl http://localhost:6333/health
```

### Prometheus Metrics
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'semantik'
    static_configs:
      - targets: 
        - 'webui:9091'
        - 'search-api:9091'
```

### Logging Configuration
```python
# Structured logging
LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "json": {
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
        }
    },
    "handlers": {
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "/app/logs/semantik.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json"
        }
    }
}
```

## Production Configuration

### Nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name semantik.example.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name semantik.example.com;
    
    ssl_certificate /etc/ssl/certs/semantik.crt;
    ssl_certificate_key /etc/ssl/private/semantik.key;
    
    # WebUI
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Search API
    location /api/search {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Security Hardening
```bash
# Environment variables
JWT_SECRET_KEY=$(openssl rand -hex 32)
DATABASE_ENCRYPTION_KEY=$(openssl rand -hex 32)

# File permissions
chmod 600 .env
chmod 700 data/

# Network isolation
docker network create --internal semantik-internal
```

### Resource Limits
```yaml
# Docker Compose resource limits
services:
  webui:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Systemd Service Files

### Search API Service
```ini
# /etc/systemd/system/semantik-search.service
[Unit]
Description=Semantik Search API
After=network.target

[Service]
Type=simple
User=semantik
Group=semantik
WorkingDirectory=/opt/semantik
Environment="PATH=/opt/semantik/.venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/opt/semantik/.venv/bin/python -m vecpipe.search_api
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### WebUI Service
```ini
# /etc/systemd/system/semantik-webui.service
[Unit]
Description=Semantik WebUI
After=network.target semantik-search.service
Requires=semantik-search.service

[Service]
Type=simple
User=semantik
Group=semantik
WorkingDirectory=/opt/semantik
Environment="PATH=/opt/semantik/.venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/opt/semantik/.venv/bin/python -m webui.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Service Management
```bash
# Enable services
sudo systemctl enable semantik-search semantik-webui

# Start services
sudo systemctl start semantik-search semantik-webui

# Check status
sudo systemctl status semantik-*

# View logs
sudo journalctl -u semantik-search -f
```

## Scaling Strategies

### Horizontal Scaling
```yaml
# Docker Swarm deployment
version: '3.8'

services:
  search-api:
    image: semantik:latest
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
```

### Load Balancing
```nginx
upstream search_api {
    least_conn;
    server search-api-1:8000;
    server search-api-2:8000;
    server search-api-3:8000;
}
```

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
   ```bash
   # Check NVIDIA driver
   nvidia-smi
   
   # Check Docker GPU support
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

2. **Permission Denied**
   ```bash
   # Fix volume permissions
   sudo chown -R $(id -u):$(id -g) ./data ./models ./logs
   ```

3. **Out of Memory**
   ```bash
   # Reduce batch size
   export BATCH_SIZE=16
   
   # Use quantization
   export DEFAULT_QUANTIZATION=int8
   ```

4. **Service Won't Start**
   ```bash
   # Check logs
   docker-compose logs search-api
   
   # Verify dependencies
   docker-compose ps
   ```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run in foreground
docker-compose up search-api

# Interactive shell
docker-compose run --rm search-api bash
```

## Maintenance

### Regular Tasks
1. **Log Rotation**: Configure logrotate or use Docker's json-file driver
2. **Database Backup**: Daily backups of PostgreSQL and Qdrant
3. **Model Cache Cleanup**: Remove unused models periodically
4. **Security Updates**: Regular updates of base images and dependencies

### Update Procedure
```bash
# Pull latest changes
git pull origin main

# Rebuild images
docker-compose build

# Stop services
docker-compose down

# Start with new images
docker-compose up -d

# Verify health
./check_health.sh
```

## Performance Tuning

### GPU Optimization
```bash
# Use TensorRT for inference
export USE_TENSORRT=true

# Optimize memory allocation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable mixed precision
export USE_AMP=true
```

### Database Optimization
```sql
-- PostgreSQL optimizations
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET work_mem = '16MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
-- Restart PostgreSQL after these changes
```

### Caching Strategy
```yaml
# Redis cache for search results
redis:
  image: redis:alpine
  command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
```

## Monitoring and Alerting

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Semantik Monitoring",
    "panels": [
      {
        "title": "Search Latency",
        "targets": [{
          "expr": "histogram_quantile(0.95, search_api_latency_seconds_bucket)"
        }]
      },
      {
        "title": "GPU Memory Usage",
        "targets": [{
          "expr": "gpu_memory_used_mb / gpu_memory_total_mb * 100"
        }]
      }
    ]
  }
}
```

### Alert Rules
```yaml
groups:
  - name: semantik
    rules:
      - alert: HighSearchLatency
        expr: histogram_quantile(0.95, search_api_latency_seconds_bucket) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High search latency detected"
          
      - alert: ServiceDown
        expr: up{job="semantik"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Semantik service is down"
```

## Security Checklist

- [ ] Change default JWT secret key
- [ ] Enable HTTPS with valid certificates
- [ ] Configure firewall rules
- [ ] Set up authentication for all endpoints
- [ ] Enable audit logging
- [ ] Regular security updates
- [ ] Implement rate limiting
- [ ] Configure CORS properly
- [ ] Validate all file uploads
- [ ] Encrypt sensitive data at rest

## Next Steps

1. **Production Readiness**: Complete security hardening
2. **High Availability**: Set up redundant services
3. **Disaster Recovery**: Implement backup/restore procedures
4. **Performance Testing**: Load test your deployment
5. **Monitoring Setup**: Configure comprehensive monitoring

For additional deployment scenarios or enterprise features, consult the advanced deployment guide or contact support.