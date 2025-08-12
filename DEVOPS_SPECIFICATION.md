# Semantik DevOps & Infrastructure Specification

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Docker Architecture](#docker-architecture)
3. [Docker Compose Configuration](#docker-compose-configuration)
4. [Development Workflow](#development-workflow)
5. [Production Deployment](#production-deployment)
6. [Database Management](#database-management)
7. [Performance Optimization](#performance-optimization)
8. [Monitoring & Observability](#monitoring--observability)
9. [Security & Compliance](#security--compliance)
10. [Disaster Recovery](#disaster-recovery)

## Executive Summary

Semantik is a self-hosted semantic search engine built with a microservices architecture using Docker containers. The infrastructure is designed for both development agility and production reliability, supporting GPU acceleration, horizontal scaling, and comprehensive monitoring.

### Key Infrastructure Components
- **Container Orchestration**: Docker Compose with multi-profile support
- **Database Stack**: PostgreSQL 16 (metadata) + Qdrant (vectors)
- **Message Queue**: Redis 7 for Celery tasks and WebSocket management
- **Compute**: NVIDIA CUDA support for GPU-accelerated embeddings
- **Monitoring**: Flower for Celery, custom health checks, partition monitoring

## Docker Architecture

### 1. Dockerfile Structure

The project uses a **multi-stage Dockerfile** with three stages:

#### Stage 1: Frontend Builder
```dockerfile
FROM node:20-alpine AS frontend-builder
```
- **Purpose**: Build React frontend with Vite
- **Optimization**: Separate npm install from source copy for better caching
- **Output**: Static files to `/build/packages/webui/static`

#### Stage 2: Python Dependencies Builder
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS python-builder
```
- **Base Image**: NVIDIA CUDA for GPU support during compilation
- **Python Version**: 3.11 (via deadsnakes PPA)
- **Package Manager**: Poetry 1.8.2
- **Key Features**:
  - Installs build tools (gcc, g++) for native extensions
  - Compiles Python packages with CUDA support
  - No virtual environment (container isolation)

#### Stage 3: Runtime Image
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS runtime
```
- **Base Image**: Lighter CUDA runtime (no build tools)
- **Security**: Non-root user (appuser, UID 1000)
- **Libraries Installed**:
  - Document processing: poppler-utils, tesseract-ocr
  - Database: libpq5 (PostgreSQL client)
  - GPU: CUDA libraries for bitsandbytes INT8 quantization
  - Health checks: wget, curl

### 2. Image Optimization Strategies

- **Layer Caching**: Dependencies installed before source code
- **Multi-stage Build**: Reduces final image size by ~60%
- **Minimal Runtime**: Only essential libraries in production
- **Security Hardening**:
  - Non-root user execution
  - Read-only mounts where possible
  - Capability dropping in compose

### 3. GPU Support Architecture

The Dockerfile includes comprehensive CUDA support:

```dockerfile
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
```

**Quantization Support**:
- float32: Full precision (CPU/GPU)
- float16: Half precision (GPU only, 50% memory reduction)
- int8: 8-bit quantization via bitsandbytes (75% memory reduction)

## Docker Compose Configuration

### 1. Service Architecture

The system consists of 6 primary services:

#### Core Services
1. **qdrant**: Vector database (latest)
2. **postgres**: Metadata database (16-alpine)
3. **redis**: Message broker (7-alpine)
4. **vecpipe**: Search API service
5. **webui**: Web interface and API
6. **worker**: Celery background tasks

#### Optional Services
7. **flower**: Celery monitoring (profile: backend)
8. **nginx**: Reverse proxy (production only)

### 2. Network Configuration

```yaml
networks:
  default:
    name: semantik-network
    driver: bridge
```

**Production Network**:
```yaml
networks:
  default:
    driver: bridge
    driver_opts:
      com.docker.network.bridge.name: semantik_prod
    ipam:
      config:
        - subnet: 172.25.0.0/16
```

### 3. Volume Management

**Named Volumes**:
- `qdrant_storage`: Vector data persistence
- `postgres_data`: Database persistence
- `redis_data`: Cache persistence
- `nginx_cache`: Static file cache (production)

**Bind Mounts**:
- `./data`: Application data
- `./logs`: Application logs
- `./models`: HuggingFace model cache
- `${DOCUMENT_PATH}`: Document processing (read-only)

### 4. Environment Configuration

**Development Override** (`docker-compose.dev.yml`):
- Source code mounted for hot reload
- Debug logging enabled
- DB_ECHO for SQL query logging

**Production Override** (`docker-compose.prod.yml`):
- Specific version tags (not latest)
- Optimized logging (json-file driver)
- Resource limits enforced
- Nginx reverse proxy included

**CUDA Override** (`docker-compose.cuda.yml`):
- Additional environment for bitsandbytes
- JIT compilation support

### 5. Resource Management

Each service has defined resource limits:

| Service | CPU Limit | Memory Limit | Memory Reservation |
|---------|-----------|--------------|-------------------|
| qdrant  | 2 cores   | 4GB          | 2GB               |
| postgres| 2 cores   | 2GB          | 1GB               |
| redis   | 0.5 cores | 512MB        | 256MB             |
| vecpipe | 2 cores   | 4GB + GPU    | 2GB               |
| webui   | 1 core    | 2GB          | 1GB               |
| worker  | 2 cores   | 4GB          | 2GB               |

### 6. Health Checks

All services include health checks:

```yaml
healthcheck:
  test: ["CMD", "command"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

## Development Workflow

### 1. Makefile Automation

The project uses a comprehensive Makefile with 40+ targets:

#### Primary Commands
- `make wizard`: Interactive setup wizard
- `make docker-up`: Start all services
- `make docker-down`: Stop services
- `make check`: Run format, lint, and tests

#### Development Commands
- `make dev-local`: WebUI local + Docker services
- `make docker-dev-up`: Backend services only
- `make frontend-dev`: Frontend development server

### 2. Development Environment Setup

**Prerequisites Check**:
1. Python 3.11+ verification
2. Poetry installation (automated)
3. Docker/Docker Compose availability
4. GPU detection (optional)

**Environment Files**:
- `.env`: Main configuration (auto-generated)
- `.env.local`: Local development overrides
- `.env.docker.example`: Template with defaults

### 3. Hot Reload Configuration

**Backend Hot Reload**:
```python
uvicorn webui.main:app --reload --host 0.0.0.0 --port 8080
```

**Frontend Hot Reload**:
```bash
npm run dev  # Vite dev server on port 5173
```

### 4. Development Scripts

Located in `/scripts/`:

- `dev.sh`: Full stack local development
- `dev-local.sh`: Hybrid development (local webui + Docker services)
- `fix-permissions.sh`: Fix Docker volume permissions
- `download-models.sh`: Pre-download ML models

## Production Deployment

### 1. Production Configuration

**Security Hardening**:
```yaml
security_opt:
  - no-new-privileges:true
cap_drop:
  - ALL
cap_add:
  - NET_BIND_SERVICE
```

**Environment Optimizations**:
```yaml
environment:
  - ENVIRONMENT=production
  - PYTHONOPTIMIZE=2  # Remove docstrings
  - LOG_LEVEL=WARNING
  - WEBUI_WORKERS=4
```

### 2. Deployment Process

**Step 1: Environment Preparation**
```bash
# Generate secure secrets
openssl rand -hex 32  # JWT_SECRET_KEY
openssl rand -hex 32  # POSTGRES_PASSWORD

# Create required directories
mkdir -p ./models ./data ./logs
chown -R 1000:1000 ./models ./data ./logs
```

**Step 2: Database Migration**
```bash
docker compose exec webui alembic upgrade head
```

**Step 3: Service Startup**
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 3. Reverse Proxy Configuration

Production includes Nginx for:
- TLS termination
- Static file caching
- Rate limiting
- Load balancing (future)

### 4. Scaling Strategies

**Horizontal Scaling**:
- Multiple worker instances
- Redis-backed session management
- Stateless service design

**Vertical Scaling**:
- GPU memory optimization
- Connection pool tuning
- Query optimization

## Database Management

### 1. Migration Strategy

**Alembic Configuration**:
- Auto-migrations on container start
- Version control for schema changes
- Rollback capability

**Migration Workflow**:
```bash
# Generate migration
poetry run alembic revision --autogenerate -m "description"

# Apply migration
poetry run alembic upgrade head

# Rollback
poetry run alembic downgrade -1
```

### 2. Partition Management

**Chunk Table Partitioning**:
- 100 LIST partitions by default
- Hash-based distribution
- Monitoring views for skew detection

**Monitoring Script** (`partition_maintenance.py`):
```bash
python scripts/partition_maintenance.py full
```

Provides:
- Health summary
- Hot partition detection
- Skew analysis
- Rebalancing recommendations

### 3. Connection Pooling

**PostgreSQL Pool Configuration**:
```python
DB_POOL_SIZE=20        # Base connections
DB_MAX_OVERFLOW=40     # Burst capacity
DB_POOL_TIMEOUT=30     # Wait timeout
DB_POOL_RECYCLE=3600   # Connection refresh
DB_POOL_PRE_PING=true  # Health check
```

### 4. Backup & Restore

**Automated Backup**:
```bash
make docker-postgres-backup
# Creates: ./backups/semantik_backup_YYYYMMDD_HHMMSS.sql
```

**Restore Process**:
```bash
make docker-postgres-restore BACKUP_FILE=./backups/backup.sql
```

### 5. Maintenance Procedures

**Routine Maintenance**:
- Vacuum operations (automated)
- Index rebuilding
- Statistics updates
- Partition rebalancing

**Performance Monitoring**:
- Query performance tracking
- Lock monitoring
- Connection pool metrics
- Partition distribution analysis

## Performance Optimization

### 1. Caching Strategies

**Redis Caching**:
- Session management
- Celery result backend
- WebSocket state management
- Temporary data storage

**Model Caching**:
- Persistent HuggingFace cache
- Model warm-up on startup
- Lazy loading for memory efficiency

### 2. Resource Optimization

**Memory Management**:
- Streaming document processing
- Chunked file uploads
- Memory pool for large operations
- Garbage collection tuning

**GPU Optimization**:
- Model quantization (INT8/FP16)
- Batch processing
- Memory-mapped models
- CUDA stream management

### 3. Query Optimization

**Database Queries**:
- Prepared statements
- Query plan caching
- Index optimization
- Partition pruning

**Vector Search**:
- HNSW index tuning
- Batch search operations
- Filtered search optimization
- Caching frequent queries

### 4. Load Balancing

**Request Distribution**:
- Round-robin worker selection
- Queue-based task distribution
- Connection pooling
- Rate limiting

## Monitoring & Observability

### 1. Health Check System

**Three-Tier Health Checks**:

**Liveness Probe** (`/api/health/healthz`):
- Process availability
- No dependency checks
- Fast response (<1s)

**Readiness Probe** (`/api/health/readyz`):
- All dependency checks
- Parallel health verification
- Detailed component status

**Service Health** (`/api/health/search-api`):
- Inter-service communication
- Component-specific checks

### 2. Metrics Collection

**Application Metrics**:
- Request latency
- Error rates
- Queue depth
- Active connections

**System Metrics**:
- CPU usage
- Memory consumption
- Disk I/O
- Network traffic

### 3. Logging Architecture

**Log Aggregation**:
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "100m"
    max-file: "5"
```

**Log Levels**:
- Development: INFO
- Production: WARNING
- Debug: DEBUG (on-demand)

### 4. Monitoring Tools

**Flower Dashboard** (port 5555):
- Celery task monitoring
- Worker status
- Queue metrics
- Task history

**Custom Monitoring**:
- Partition health dashboard
- Collection statistics
- Error rate tracking
- Performance profiling

## Security & Compliance

### 1. Container Security

**Security Measures**:
- Non-root user execution
- Read-only filesystem where possible
- Network isolation
- Secret management

**Capability Management**:
```yaml
cap_drop:
  - ALL
cap_add:
  - NET_BIND_SERVICE  # Only what's needed
```

### 2. Authentication & Authorization

**JWT Configuration**:
- Secure key generation
- Token expiration (24h default)
- Refresh token support
- Role-based access control

### 3. Data Protection

**Encryption**:
- TLS for external communication
- Encrypted database connections
- Secure secret storage

**Access Control**:
- Database user separation
- Read-only document mounts
- API rate limiting
- CORS configuration

### 4. Vulnerability Management

**Security Scanning**:
- Trivy vulnerability scanner in CI
- Dependency updates
- Base image updates
- Security patches

## Disaster Recovery

### 1. Backup Strategy

**Automated Backups**:
- Database: Daily PostgreSQL dumps
- Vectors: Qdrant snapshots
- Configuration: Version controlled

**Backup Retention**:
- Daily: 7 days
- Weekly: 4 weeks
- Monthly: 12 months

### 2. Recovery Procedures

**Service Recovery**:
1. Stop affected services
2. Restore from backup
3. Verify data integrity
4. Resume operations

**Full System Recovery**:
1. Provision new infrastructure
2. Restore database
3. Restore vector data
4. Rebuild search indices
5. Verify system health

### 3. High Availability

**Redundancy**:
- Database replication (planned)
- Redis sentinel (planned)
- Multi-node Qdrant (planned)

**Failover Strategy**:
- Health check monitoring
- Automatic restart
- Manual intervention procedures
- Incident response playbook

### 4. Business Continuity

**RTO/RPO Targets**:
- Recovery Time Objective: 4 hours
- Recovery Point Objective: 24 hours

**Testing Schedule**:
- Monthly backup verification
- Quarterly recovery drills
- Annual full DR test

## Appendix A: Quick Reference

### Common Commands

```bash
# Development
make docker-up          # Start all services
make docker-down        # Stop all services
make docker-logs        # View logs
make docker-ps          # Check status

# Database
make docker-postgres-backup     # Backup database
make docker-shell-postgres      # PostgreSQL CLI
docker compose exec webui alembic upgrade head  # Migrate

# Monitoring
docker compose logs -f webui    # Service logs
docker compose exec webui python scripts/partition_maintenance.py full  # Partition health

# Production
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d  # Deploy
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d  # GPU deploy
```

### Port Mappings

| Service | Internal | External | Purpose |
|---------|----------|----------|---------|
| WebUI   | 8080     | 8080     | Main application |
| VecPipe | 8000     | 8000     | Search API |
| Qdrant  | 6333     | 6333     | Vector DB HTTP |
| Qdrant  | 6334     | 6334     | Vector DB gRPC |
| PostgreSQL | 5432  | 5432     | Database |
| Redis   | 6379     | 6379     | Cache/Queue |
| Flower  | 5555     | 5555     | Monitoring |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| JWT_SECRET_KEY | (generated) | Authentication key |
| POSTGRES_PASSWORD | (generated) | Database password |
| DEFAULT_EMBEDDING_MODEL | Qwen/Qwen3-Embedding-0.6B | Embedding model |
| DEFAULT_QUANTIZATION | float16 | Model precision |
| WEBUI_WORKERS | 1 | Uvicorn workers |
| DB_POOL_SIZE | 20 | Connection pool |
| CUDA_VISIBLE_DEVICES | 0 | GPU selection |

## Appendix B: Troubleshooting

### Common Issues

**1. Permission Denied**
```bash
sudo chown -R 1000:1000 ./models ./data ./logs
```

**2. GPU Not Detected**
```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d --build
```

**3. Database Connection Failed**
```bash
docker compose restart postgres
docker compose exec postgres pg_isready
```

**4. Model Download Issues**
```bash
HF_HUB_OFFLINE=false docker compose up -d
```

**5. Memory Issues**
- Reduce DB_POOL_SIZE
- Use INT8 quantization
- Increase Docker memory limit

### Health Check Failures

**Check Service Status**:
```bash
curl http://localhost:8080/api/health/readyz | jq
```

**Individual Service Checks**:
```bash
docker compose exec webui curl http://localhost:8080/api/health
docker compose exec vecpipe curl http://localhost:8000/health
docker compose exec qdrant curl http://localhost:6333/health
```

### Performance Issues

**1. Slow Queries**
- Check partition distribution
- Analyze query plans
- Increase connection pool
- Add appropriate indexes

**2. High Memory Usage**
- Enable model quantization
- Reduce worker concurrency
- Implement pagination
- Clear Redis cache

**3. GPU Memory Errors**
- Switch to INT8 quantization
- Reduce batch size
- Clear GPU cache
- Restart vecpipe service

---

*Last Updated: 2025-01-12*
*Version: 1.0.0*
*Maintained by: Semantik DevOps Team*