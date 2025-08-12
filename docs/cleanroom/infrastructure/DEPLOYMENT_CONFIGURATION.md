# DEPLOYMENT_CONFIGURATION - Cleanroom Documentation

## 1. Component Overview

The DEPLOYMENT_CONFIGURATION component manages the complete deployment lifecycle of the Semantik application, including environment setup, service orchestration, database migrations, model management, and monitoring. It provides multiple deployment strategies (Docker, Kubernetes, manual) with automated configuration management and health verification.

### Core Responsibilities
- **Environment Management**: Configuration of development, staging, and production environments
- **Service Orchestration**: Coordination of microservices startup and dependencies
- **Database Lifecycle**: Migration execution, backup/restore, and health monitoring
- **Model Management**: Download, caching, and configuration of ML models
- **Security Configuration**: Secret generation, permission management, and access control
- **Health Monitoring**: Service health checks, readiness probes, and status verification

### Key Components
- **Docker Orchestration**: docker-compose.yml and environment-specific overrides
- **Build System**: Multi-stage Dockerfile with optimized layer caching
- **Database Migrations**: Alembic-based schema management with safety checks
- **Configuration Wizard**: Interactive TUI for guided setup
- **Automation Scripts**: Make targets and shell scripts for common operations

## 2. Architecture & Design Patterns

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Deployment Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ Environment  │───▶│    Build     │───▶│   Deploy     │   │
│  │   Config     │    │   Process    │    │   Services   │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         │                    │                    │          │
│         ▼                    ▼                    ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   .env       │    │  Docker      │    │   Docker     │   │
│  │   Files      │    │  Images      │    │   Compose    │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Service Dependency Graph

```
PostgreSQL ─┬──▶ WebUI ◀──┬── Redis
            │              │
            ├──▶ VecPipe   │
            │              │
            └──▶ Worker ◀──┘
                   │
                   └──▶ Qdrant
```

### Design Patterns

1. **Multi-Stage Build Pattern**
   - Separate build and runtime stages for smaller images
   - Cached dependency layers for faster rebuilds
   - Security-hardened runtime with non-root user

2. **Environment Override Pattern**
   - Base configuration in docker-compose.yml
   - Environment-specific overrides (.dev.yml, .prod.yml, .cuda.yml)
   - Composition for flexible deployment scenarios

3. **Health Check Pattern**
   - Readiness probes for service startup ordering
   - Liveness checks for automatic recovery
   - Dependency validation before service initialization

4. **Configuration as Code**
   - Environment variables for runtime configuration
   - Volume mounts for persistent data
   - Secret management through environment injection

## 3. Key Interfaces & Contracts

### Environment Variable Contract

```bash
# Required Security Variables
JWT_SECRET_KEY          # 32-byte hex string for JWT signing
POSTGRES_PASSWORD       # Database password (auto-generated if not set)

# Service Discovery (Docker Internal)
QDRANT_HOST=qdrant     # Vector database host
POSTGRES_HOST=postgres # PostgreSQL host
REDIS_URL=redis://redis:6379/0

# Model Configuration
DEFAULT_EMBEDDING_MODEL # HuggingFace model identifier
DEFAULT_QUANTIZATION    # float32|float16|int8
HF_CACHE_DIR           # Model cache directory

# Deployment Environment
ENVIRONMENT            # development|staging|production
LOG_LEVEL             # DEBUG|INFO|WARNING|ERROR
```

### Docker Service Contract

```yaml
service:
  image: <built-from-dockerfile>
  container_name: semantik-<service>
  command: ["<service-name>"]
  environment:
    - REQUIRED_ENV_VARS
  volumes:
    - persistent_data:/path
  depends_on:
    dependency:
      condition: service_healthy
  healthcheck:
    test: ["CMD", "health-check-command"]
    interval: 30s
    timeout: 10s
    retries: 3
  restart: unless-stopped
```

### Migration Contract

```python
# Alembic migration structure
def upgrade() -> None:
    """Apply forward migration"""
    # DDL operations
    # Data migrations
    # Index creation

def downgrade() -> None:
    """Rollback migration"""
    # Reverse operations
```

## 4. Data Flow & Dependencies

### Deployment Pipeline Flow

```
1. Environment Setup
   ├── Load .env configuration
   ├── Generate missing secrets
   └── Validate required variables

2. Build Phase
   ├── Frontend build (Node.js)
   ├── Python dependencies (Poetry)
   └── Docker image assembly

3. Database Initialization
   ├── PostgreSQL startup
   ├── Wait for healthy state
   └── Run Alembic migrations

4. Service Startup Sequence
   ├── Start infrastructure (Qdrant, Redis)
   ├── Start API services (VecPipe)
   ├── Start application (WebUI)
   └── Start workers (Celery)

5. Health Verification
   ├── Check service endpoints
   ├── Validate database connectivity
   └── Confirm model availability
```

### Service Dependencies

```python
SERVICE_DEPENDENCIES = {
    "webui": ["postgres", "vecpipe", "redis"],
    "vecpipe": ["postgres", "qdrant"],
    "worker": ["postgres", "redis", "qdrant"],
    "flower": ["redis"]
}
```

## 5. Critical Implementation Details

### Docker Entrypoint Script

```bash
# docker-entrypoint.sh implementation
#!/bin/bash
set -e

# Environment validation
validate_env_vars() {
    case "$SERVICE" in
        webui)
            if [ "$JWT_SECRET_KEY" = "CHANGE_THIS_TO_A_STRONG_SECRET_KEY" ]; then
                echo "ERROR: JWT_SECRET_KEY must be set"
                exit 1
            fi
            ;;
        vecpipe)
            if [ -z "$QDRANT_HOST" ] || [ -z "$QDRANT_PORT" ]; then
                echo "ERROR: Qdrant configuration missing"
                exit 1
            fi
            ;;
    esac
}

# Service readiness waiting
wait_for_service() {
    local host=$1 port=$2 service=$3
    for i in {1..30}; do
        if python -c "import socket; socket.create_connection(('$host', $port), timeout=1)" 2>/dev/null; then
            echo "$service is ready!"
            return 0
        fi
        sleep 2
    done
    return 1
}

# Service-specific startup
case "$SERVICE" in
    webui)
        wait_for_service "$SEARCH_HOST" 8000 "Search API"
        alembic upgrade head
        exec uvicorn webui.main:app --host 0.0.0.0 --port ${WEBUI_PORT:-8080}
        ;;
    vecpipe)
        wait_for_service "$QDRANT_HOST" 6333 "Qdrant"
        exec python -m vecpipe.search_api
        ;;
    worker)
        exec celery -A webui.celery_app worker --loglevel=info
        ;;
esac
```

### Database Migration Strategy

```python
# alembic/env.py implementation
import os
from alembic import context
from shared.database.models import Base

# Database URL handling
database_url = os.environ.get("DATABASE_URL")
if not database_url:
    raise ValueError("DATABASE_URL not set")

# Convert async to sync driver for Alembic
if "postgresql+asyncpg://" in database_url:
    sync_url = database_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
else:
    sync_url = database_url

config.set_main_option("sqlalchemy.url", sync_url)
target_metadata = Base.metadata

def run_migrations_online():
    """Execute migrations with proper connection handling"""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
```

### Model Download Process

```bash
# scripts/download-models.sh implementation
#!/bin/bash
MODEL_DIR="${HF_CACHE_DIR:-./models}"
MODEL_NAME="${DEFAULT_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}"

# Create cache directory
mkdir -p "$MODEL_DIR"

# Download using Docker container
docker run --rm -it \
    -v "$(realpath "$MODEL_DIR")":/app/.cache/huggingface \
    -e HF_HOME=/app/.cache/huggingface \
    semantik-webui \
    python -c "
from transformers import AutoModel, AutoTokenizer
print(f'Downloading model: $MODEL_NAME')
tokenizer = AutoTokenizer.from_pretrained('$MODEL_NAME')
model = AutoModel.from_pretrained('$MODEL_NAME')
print('✅ Model downloaded successfully!')
"
```

## 6. Security Considerations

### Secret Management

```python
# Automatic secret generation in Makefile
JWT_KEY=$(openssl rand -hex 32)
POSTGRES_PWD=$(openssl rand -hex 32)

# Environment variable validation
SECURITY_CHECKS = {
    "JWT_SECRET_KEY": lambda v: v != "CHANGE_THIS_TO_A_STRONG_SECRET_KEY",
    "POSTGRES_PASSWORD": lambda v: len(v) >= 32,
    "ACCESS_TOKEN_EXPIRE_MINUTES": lambda v: int(v) <= 1440
}
```

### Container Security

```dockerfile
# Non-root user execution
RUN useradd -m -u 1000 appuser
USER appuser

# Security options in docker-compose
security_opt:
  - no-new-privileges:true
cap_drop:
  - ALL
cap_add:
  - NET_BIND_SERVICE  # Only required capability
```

### Permission Management

```bash
# scripts/fix-permissions.sh
#!/bin/bash
DIRS=("./models" "./data" "./logs")

# Set ownership to container user (UID 1000)
for dir in "${DIRS[@]}"; do
    sudo chown -R 1000:1000 "$dir"
done
```

## 7. Testing Requirements

### Deployment Validation Tests

```bash
# validate-docker-setup.sh
#!/bin/bash

# Check Docker installation
docker --version || exit 1

# Validate compose file
docker compose config > /dev/null || exit 1

# Check required directories
for dir in data logs models; do
    [ -d "$dir" ] || mkdir -p "$dir"
done

# Verify service health
docker compose ps --format json | jq '.[] | select(.Health != "healthy")'
```

### Integration Tests

```python
# Test database migrations
async def test_migration_execution():
    """Verify migrations run successfully"""
    result = subprocess.run(
        ["alembic", "upgrade", "head"],
        env={"DATABASE_URL": test_db_url},
        capture_output=True
    )
    assert result.returncode == 0

# Test service connectivity
async def test_service_dependencies():
    """Verify all services can communicate"""
    services = ["postgres:5432", "qdrant:6333", "redis:6379"]
    for service in services:
        host, port = service.split(":")
        assert can_connect(host, int(port))
```

### Smoke Tests

```bash
# Post-deployment verification
curl -f http://localhost:8080/api/health/readyz || exit 1
curl -f http://localhost:8000/health || exit 1
curl -f http://localhost:6333/health || exit 1
```

## 8. Common Pitfalls & Best Practices

### Migration Pitfalls

```python
# WRONG: Running migrations before database is ready
alembic upgrade head  # May fail if postgres isn't ready

# RIGHT: Wait for database before migrations
wait_for_service postgres 5432 "PostgreSQL"
alembic upgrade head
```

### Volume Permissions

```bash
# WRONG: Using host user permissions
mkdir ./data  # Created with current user

# RIGHT: Set container user permissions
mkdir -p ./data
sudo chown -R 1000:1000 ./data
```

### Environment Variables

```bash
# WRONG: Hardcoded secrets
JWT_SECRET_KEY=my-secret-key

# RIGHT: Generated secrets
JWT_SECRET_KEY=$(openssl rand -hex 32)
```

### Service Dependencies

```yaml
# WRONG: No dependency management
services:
  webui:
    image: semantik-webui

# RIGHT: Explicit dependencies with health checks
services:
  webui:
    depends_on:
      postgres:
        condition: service_healthy
```

## 9. Configuration & Environment

### Required Environment Variables

```bash
# Minimal Required Configuration
JWT_SECRET_KEY=<32-byte-hex>      # Authentication key
POSTGRES_PASSWORD=<secure-pass>    # Database password
DATABASE_URL=postgresql://...      # Full database URL

# Service Discovery (Docker handles these)
QDRANT_HOST=qdrant
POSTGRES_HOST=postgres
REDIS_URL=redis://redis:6379/0

# Model Configuration
DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
DEFAULT_QUANTIZATION=float16
HF_CACHE_DIR=./models
```

### Environment-Specific Configurations

```yaml
# Development (docker-compose.dev.yml)
environment:
  - ENVIRONMENT=development
  - LOG_LEVEL=DEBUG
  - WEBUI_RELOAD=true
  - DB_ECHO=true

# Production (docker-compose.prod.yml)
environment:
  - ENVIRONMENT=production
  - LOG_LEVEL=WARNING
  - PYTHONOPTIMIZE=2
  - WEBUI_WORKERS=4
```

### Default Values

```python
CONFIGURATION_DEFAULTS = {
    "ACCESS_TOKEN_EXPIRE_MINUTES": 1440,
    "DEFAULT_COLLECTION": "work_docs",
    "USE_MOCK_EMBEDDINGS": False,
    "DB_POOL_SIZE": 20,
    "DB_MAX_OVERFLOW": 40,
    "DB_POOL_TIMEOUT": 30,
    "MODEL_UNLOAD_AFTER_SECONDS": 300,
    "RATE_LIMIT_PER_MINUTE": 60,
    "CELERY_CONCURRENCY": 1
}
```

## 10. Integration Points

### Cloud Provider Integration

```yaml
# AWS ECS Task Definition
taskDefinition:
  family: semantik
  networkMode: awsvpc
  requiresCompatibilities: [FARGATE]
  cpu: "2048"
  memory: "8192"
  containerDefinitions:
    - name: webui
      image: semantik-webui:latest
      environment:
        - name: DATABASE_URL
          valueFrom: secrets-manager:arn:aws:secretsmanager:...

# Google Cloud Run
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: semantik-webui
spec:
  template:
    spec:
      containers:
        - image: gcr.io/project/semantik-webui
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: db-secret
```

### Monitoring Integration

```yaml
# Prometheus metrics export
services:
  prometheus-exporter:
    image: prom/node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro

# Grafana dashboard
services:
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana-storage:/var/lib/grafana
```

### CI/CD Integration

```yaml
# GitHub Actions deployment
- name: Deploy to Production
  run: |
    docker compose -f docker-compose.yml \
                   -f docker-compose.prod.yml \
                   up -d --build
    
    # Wait for health checks
    ./scripts/wait-for-healthy.sh
    
    # Run smoke tests
    ./scripts/smoke-tests.sh

# GitLab CI deployment
deploy:
  stage: deploy
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker compose pull
    - docker compose up -d
    - ./validate-deployment.sh
```

### Backup Integration

```bash
# PostgreSQL backup to S3
docker compose exec -T postgres pg_dump -U semantik semantik | \
  aws s3 cp - s3://backup-bucket/semantik-$(date +%Y%m%d).sql

# Qdrant snapshot
curl -X POST http://localhost:6333/collections/work_docs/snapshots

# Redis backup
docker compose exec redis redis-cli BGSAVE
docker cp semantik-redis:/data/dump.rdb ./backups/
```

## Deployment Checklist

### Pre-Deployment
- [ ] Generate secure JWT_SECRET_KEY
- [ ] Generate secure POSTGRES_PASSWORD  
- [ ] Configure DOCUMENT_PATH for source documents
- [ ] Set appropriate MODEL configuration
- [ ] Review resource limits in docker-compose.yml
- [ ] Create required directories (data, logs, models)
- [ ] Fix directory permissions (chown 1000:1000)

### Deployment
- [ ] Run environment validation script
- [ ] Execute database migrations
- [ ] Download required models
- [ ] Start services in dependency order
- [ ] Verify health checks pass

### Post-Deployment
- [ ] Run smoke tests
- [ ] Create initial superuser account
- [ ] Configure monitoring alerts
- [ ] Set up backup schedule
- [ ] Document deployment specifics

## Troubleshooting Guide

### Common Issues

1. **Permission Denied Errors**
   ```bash
   # Fix: Set correct ownership
   sudo chown -R 1000:1000 ./models ./data ./logs
   ```

2. **Database Connection Failed**
   ```bash
   # Check PostgreSQL is running
   docker compose ps postgres
   # Check logs
   docker compose logs postgres
   ```

3. **Model Download Failures**
   ```bash
   # Check disk space
   df -h ./models
   # Try offline mode after download
   HF_HUB_OFFLINE=true
   ```

4. **Migration Failures**
   ```bash
   # Check current migration state
   docker compose exec webui alembic current
   # Force specific revision if needed
   docker compose exec webui alembic stamp head
   ```

5. **Service Health Check Failures**
   ```bash
   # View detailed health status
   docker inspect semantik-webui --format='{{json .State.Health}}'
   # Check service logs
   docker compose logs -f webui
   ```

## Performance Tuning

### Resource Optimization

```yaml
# Adjust based on available resources
deploy:
  resources:
    limits:
      cpus: '2'      # Adjust per service needs
      memory: 4G     # Monitor actual usage
    reservations:
      memory: 2G     # Minimum guaranteed
```

### Database Tuning

```bash
# PostgreSQL connection pooling
DB_POOL_SIZE=20           # Active connections
DB_MAX_OVERFLOW=40        # Burst capacity
DB_POOL_RECYCLE=3600      # Connection lifetime
```

### Model Optimization

```bash
# Quantization for memory efficiency
DEFAULT_QUANTIZATION=int8  # 75% memory reduction
MODEL_UNLOAD_AFTER_SECONDS=600  # Auto-unload inactive
```

---

**Document Version**: 1.0.0  
**Last Updated**: 2025-01-12  
**Component**: DEPLOYMENT_CONFIGURATION  
**Status**: Production Ready