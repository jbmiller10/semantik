# Docker Infrastructure Component - Cleanroom Documentation

## 1. Component Overview

The Docker infrastructure component provides a complete containerization solution for the Semantik document embedding system, orchestrating six primary services with GPU support, networking isolation, and multi-environment configurations. The infrastructure leverages Docker Compose for service orchestration with distinct configurations for development, production, and CUDA-enabled deployments.

### Core Services Architecture
```
┌──────────────────────────────────────────────────────────┐
│                    External Access Layer                   │
├──────────────────────────────────────────────────────────┤
│  nginx:443 ──► webui:8080 ──► vecpipe:8000               │
│                     │             │                        │
│                     ▼             ▼                        │
│               ┌──────────┐  ┌──────────┐                 │
│               │  Redis   │  │  Qdrant  │                 │
│               │  :6379   │  │  :6333   │                 │
│               └──────────┘  └──────────┘                 │
│                     ▲             ▲                        │
│                     │             │                        │
│                ┌──────────┐       │                        │
│                │  Worker  │───────┘                        │
│                └──────────┘                                │
│                     │                                      │
│                     ▼                                      │
│               ┌──────────┐                                │
│               │ Postgres │                                │
│               │  :5432   │                                │
│               └──────────┘                                │
└──────────────────────────────────────────────────────────┘
```

### Service Manifest
- **webui**: FastAPI backend serving the React frontend and API endpoints
- **vecpipe**: Dedicated embedding and search service with GPU acceleration
- **worker**: Celery-based async task processor for document operations
- **postgres**: PostgreSQL 16 database for metadata and state management
- **redis**: In-memory cache and message broker for Celery
- **qdrant**: Vector database for semantic search operations
- **flower**: (Optional) Celery monitoring dashboard
- **nginx**: (Production) Reverse proxy with TLS termination

## 2. Architecture & Design Patterns

### Multi-Stage Build Pattern
The Dockerfile implements a three-stage build process to optimize image size and security:

```dockerfile
# Stage 1: Frontend Builder
FROM node:20-alpine AS frontend-builder
# Builds React application to static files

# Stage 2: Python Dependencies Builder  
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS python-builder
# Installs Poetry and Python dependencies

# Stage 3: Runtime Image
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS runtime
# Minimal runtime with only necessary components
```

### Service Composition Strategy
Docker Compose layering enables environment-specific configurations:

```yaml
# Base configuration: docker-compose.yml
services:
  webui:
    build:
      context: .
      target: runtime
    # Base service definition

# Development override: docker-compose.dev.yml
services:
  webui:
    volumes:
      - ./packages:/app/packages:ro  # Live code mounting
    environment:
      - WEBUI_RELOAD=true  # Enable hot reload

# Production override: docker-compose.prod.yml
services:
  webui:
    restart: always
    logging:
      driver: "json-file"
    ports: []  # Remove direct exposure
```

### Network Isolation Model
```yaml
networks:
  default:
    name: semantik-network
    driver: bridge
    # Production subnet isolation
    ipam:
      config:
        - subnet: 172.25.0.0/16
```

## 3. Key Interfaces & Contracts

### Service Communication Matrix

| Service | Exposed Ports | Internal Communication | Protocol |
|---------|--------------|------------------------|----------|
| webui | 8080 | vecpipe:8000, postgres:5432, redis:6379 | HTTP/TCP |
| vecpipe | 8000 | qdrant:6333, postgres:5432, redis:6379 | HTTP/gRPC |
| worker | - | postgres:5432, redis:6379, qdrant:6333 | TCP |
| postgres | 5432 | - | PostgreSQL |
| redis | 6379 | - | Redis Protocol |
| qdrant | 6333, 6334 | - | HTTP/gRPC |

### Environment Variable Contract
Critical environment variables required for service operation:

```bash
# Authentication & Security
JWT_SECRET_KEY=${JWT_SECRET_KEY:-CHANGE_THIS_TO_A_STRONG_SECRET_KEY}
ACCESS_TOKEN_EXPIRE_MINUTES=${ACCESS_TOKEN_EXPIRE_MINUTES:-1440}

# Database Configuration
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=${POSTGRES_DB:-semantik}
POSTGRES_USER=${POSTGRES_USER:-semantik}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-CHANGE_THIS_TO_A_STRONG_PASSWORD}

# Service Discovery
QDRANT_HOST=qdrant
QDRANT_PORT=6333
REDIS_URL=redis://redis:6379/0
SEARCH_API_URL=http://vecpipe:8000

# Model Configuration
DEFAULT_EMBEDDING_MODEL=${DEFAULT_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}
DEFAULT_QUANTIZATION=${DEFAULT_QUANTIZATION:-float16}
HF_HOME=/app/.cache/huggingface
```

## 4. Data Flow & Dependencies

### Service Startup Sequence
The entrypoint script (`docker-entrypoint.sh`) implements dependency-aware startup:

```bash
# WebUI startup sequence
1. Wait for vecpipe:8000 (Search API)
2. Run Alembic database migrations
3. Start Uvicorn server

# VecPipe startup sequence
1. Wait for qdrant:6333
2. Initialize embedding models
3. Start FastAPI server

# Worker startup sequence
1. Verify Redis connectivity
2. Initialize Celery application
3. Start worker processes
```

### Health Check Dependencies
```yaml
webui:
  depends_on:
    postgres:
      condition: service_healthy  # Waits for healthy state
    vecpipe:
      condition: service_started
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8080/api/health/readyz"]
    interval: 30s
    start_period: 60s
```

### Volume Mount Architecture
```yaml
volumes:
  # Persistent named volumes
  qdrant_storage:    # Vector index persistence
  postgres_data:     # Database persistence
  redis_data:        # Cache persistence
  
  # Bind mounts for development
  ./data:/app/data                    # Operation data
  ./logs:/app/logs                    # Application logs
  ${DOCUMENT_PATH}:/mnt/docs:ro      # Document source (read-only)
  ${HF_CACHE_DIR}:/app/.cache/huggingface  # Model cache
```

## 5. Critical Implementation Details

### Container Security Configuration

#### Capability Management
```yaml
security_opt:
  - no-new-privileges:true
cap_drop:
  - ALL
cap_add:
  - NET_BIND_SERVICE  # Only for services binding ports
  - CHOWN            # PostgreSQL specific
  - DAC_READ_SEARCH  # PostgreSQL specific
```

#### User Privilege Separation
```dockerfile
# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser
```

### Resource Constraints
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      memory: 2G
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### GPU/CUDA Configuration
```yaml
# docker-compose.cuda.yml overlay
services:
  vecpipe:
    environment:
      - LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
      - CUDA_HOME=/usr/local/cuda
      - BITSANDBYTES_NOWELCOME=1
      - CC=gcc
      - CXX=g++
```

### Logging Configuration
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "100m"
    max-file: "5"
```

## 6. Security Considerations

### Secret Management Requirements

#### JWT Secret Key Generation
```bash
# Required for production deployments
openssl rand -hex 32
# Automatically generated by Makefile if not present
```

#### PostgreSQL Password Security
```bash
# Auto-generated secure password
POSTGRES_PWD=$(openssl rand -hex 32)
# Stored in .env file with restricted permissions
```

### Network Security Model

#### Internal Service Communication
- All services communicate over internal Docker bridge network
- No direct external exposure except webui:8080 (development)
- Production uses nginx reverse proxy for TLS termination

#### Port Exposure Strategy
```yaml
# Development: Direct port exposure
ports:
  - "8080:8080"

# Production: No direct exposure
ports: []  # Access via reverse proxy only
```

### Container Hardening

#### Read-Only Root Filesystem
```yaml
volumes:
  - ./packages:/app/packages:ro  # Read-only source mount
  - ${DOCUMENT_PATH}:/mnt/docs:ro  # Read-only document access
```

#### Security Headers (Production)
```yaml
environment:
  - FORWARDED_ALLOW_IPS=*  # Configure for specific proxy
  - RATE_LIMIT_PER_MINUTE=60
```

## 7. Testing Requirements

### Container Build Tests
```bash
# Verify multi-stage build
docker build --target frontend-builder -t test-frontend .
docker build --target python-builder -t test-python .
docker build --target runtime -t test-runtime .

# Validate CUDA support
docker run --rm test-runtime python -c "import torch; print(torch.cuda.is_available())"
```

### Service Integration Tests
```bash
# Health check validation
curl http://localhost:8080/api/health/readyz
curl http://localhost:8000/health
redis-cli -h localhost ping

# Service communication test
docker compose exec webui curl http://vecpipe:8000/health
docker compose exec vecpipe curl http://qdrant:6333/health
```

### Volume Persistence Tests
```bash
# Database persistence
docker compose down
docker compose up -d
docker compose exec postgres psql -U semantik -c "\dt"

# Vector index persistence
docker compose exec qdrant curl http://localhost:6333/collections
```

## 8. Common Pitfalls & Best Practices

### Directory Permission Issues
```bash
# Problem: Container user (1000) lacks write permissions
# Solution: Set ownership before startup
sudo chown -R 1000:1000 ./models ./data ./logs
```

### Model Download Failures
```bash
# Problem: HuggingFace models fail to download in container
# Solution: Mount cache directory with write permissions
volumes:
  - ${HF_CACHE_DIR:-./models}:/app/.cache/huggingface
```

### Database Migration Failures
```bash
# Problem: Migrations fail on startup
# Solution: Ensure postgres is healthy before webui starts
depends_on:
  postgres:
    condition: service_healthy  # Not just service_started
```

### GPU Memory Issues
```yaml
# Problem: CUDA out of memory errors
# Solution: Set explicit GPU limits
environment:
  - CUDA_VISIBLE_DEVICES=0  # Restrict to single GPU
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
```

### Network Timeout Issues
```bash
# Problem: Services timeout waiting for dependencies
# Solution: Implement retry logic in entrypoint
wait_for_service() {
    local max_attempts=30
    while [ $attempt -le $max_attempts ]; do
        if python -c "import socket; socket.create_connection(('$host', $port), timeout=1)"; then
            return 0
        fi
        sleep 2
    done
}
```

## 9. Configuration & Environment

### Environment File Hierarchy
```
.env                     # Primary configuration (git-ignored)
.env.docker.example      # Docker-specific template
.env.cpu                 # CPU-only configuration
.env.example            # Production template
```

### Configuration Profiles

#### Development Profile
```bash
# docker-compose.yml + docker-compose.dev.yml
ENVIRONMENT=development
WEBUI_RELOAD=true
DB_ECHO=true
LOG_LEVEL=debug
```

#### Production Profile
```bash
# docker-compose.yml + docker-compose.prod.yml
ENVIRONMENT=production
PYTHONOPTIMIZE=2
LOG_LEVEL=WARNING
WEBUI_WORKERS=4
```

#### CUDA Profile
```bash
# docker-compose.yml + docker-compose.cuda.yml
DEFAULT_QUANTIZATION=int8
CUDA_VISIBLE_DEVICES=0
BITSANDBYTES_NOWELCOME=1
```

### Makefile Automation
```bash
# Full stack operations
make docker-up          # Start all services
make docker-down        # Stop all services
make docker-logs        # View all logs
make docker-restart     # Restart services

# Development operations
make docker-dev-up      # Backend only for local development
make dev-local         # Run webui locally with Docker backend

# Database operations
make docker-postgres-backup    # Create timestamped backup
make docker-postgres-restore BACKUP_FILE=path/to/backup.sql

# Maintenance operations
make docker-build-fresh  # Rebuild without cache
make wizard             # Interactive setup wizard
```

## 10. Integration Points

### Database Integration
```python
# Connection URL format
DATABASE_URL = "postgresql://semantik:password@postgres:5432/semantik"

# Connection pool configuration
DB_POOL_SIZE = 20
DB_MAX_OVERFLOW = 40
DB_POOL_TIMEOUT = 30
DB_POOL_RECYCLE = 3600
DB_POOL_PRE_PING = true
```

### Redis Integration
```python
# Celery broker configuration
CELERY_BROKER_URL = "redis://redis:6379/0"
CELERY_RESULT_BACKEND = "redis://redis:6379/0"

# WebSocket pub/sub
REDIS_URL = "redis://redis:6379/0"
```

### Qdrant Integration
```python
# Vector database connection
QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
QDRANT_GRPC_PORT = 6334

# Collection defaults
DEFAULT_COLLECTION = "work_docs"
```

### Inter-Service Communication
```python
# Service discovery URLs
SEARCH_API_URL = "http://vecpipe:8000"
SEARCH_API_HOST = "vecpipe"

# Health check endpoints
/api/health/readyz  # WebUI readiness
/health            # VecPipe health
/health            # Qdrant health
```

### File System Integration
```bash
# Document processing paths
/mnt/docs          # Source documents (read-only)
/app/data          # Processing workspace
/app/logs          # Application logs
/app/.cache/huggingface  # Model cache

# Operation directories
/app/data/operations  # Operation metadata
/app/data/ingest     # Ingestion queue
/app/data/extract    # Extraction workspace
/app/data/loaded     # Processed documents
/app/data/rejects    # Failed documents
```

## Appendix A: Service Command Reference

### WebUI Service
```bash
# Primary command
uvicorn webui.main:app --host 0.0.0.0 --port 8080 --workers 1

# Development mode
uvicorn webui.main:app --host 0.0.0.0 --port 8080 --reload
```

### VecPipe Service
```bash
# Primary command
python -m vecpipe.search_api
```

### Worker Service
```bash
# Primary command
celery -A webui.celery_app worker --loglevel=info --concurrency=1
```

### Flower Service
```bash
# Monitoring command
celery -A webui.celery_app flower --broker=redis://redis:6379/0 --basic_auth=admin:admin
```

## Appendix B: Troubleshooting Guide

### Common Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Permission Denied | Cannot write to /app/data | `sudo chown -R 1000:1000 ./data` |
| Port Already in Use | Cannot bind to port 8080 | Change port in docker-compose.yml |
| Database Connection Failed | psycopg2.OperationalError | Check DATABASE_URL and postgres health |
| CUDA Not Available | torch.cuda.is_available() = False | Verify nvidia-docker runtime |
| Model Download Timeout | HuggingFace connection error | Set HF_HUB_OFFLINE=true with pre-downloaded models |
| Memory Exhaustion | Container OOMKilled | Adjust resource limits in deploy section |
| Slow Startup | Services timeout | Increase start_period in healthcheck |

### Debug Commands
```bash
# Check container logs
docker compose logs -f [service_name]

# Execute commands in container
docker compose exec [service_name] /bin/bash

# Inspect container
docker inspect semantik-[service_name]

# View resource usage
docker stats

# Network debugging
docker network inspect semantik-network
```

## Appendix C: Production Deployment Checklist

- [ ] Generate secure JWT_SECRET_KEY
- [ ] Generate secure POSTGRES_PASSWORD
- [ ] Configure TLS certificates for nginx
- [ ] Set ENVIRONMENT=production
- [ ] Configure FORWARDED_ALLOW_IPS for reverse proxy
- [ ] Set appropriate RATE_LIMIT values
- [ ] Configure logging retention policies
- [ ] Set up automated backups for PostgreSQL
- [ ] Configure monitoring (Prometheus/Grafana)
- [ ] Implement log aggregation (ELK stack)
- [ ] Set up health check monitoring
- [ ] Configure firewall rules
- [ ] Implement secrets management (Docker Secrets/Vault)
- [ ] Document disaster recovery procedures
- [ ] Load test with expected traffic patterns