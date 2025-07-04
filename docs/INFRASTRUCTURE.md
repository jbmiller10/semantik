# Infrastructure Documentation

## Table of Contents

1. [Infrastructure Overview](#infrastructure-overview)
2. [Development Environment](#development-environment)
3. [Testing Framework](#testing-framework)
4. [Build System](#build-system)
5. [Service Management](#service-management)
6. [Deployment Architecture](#deployment-architecture)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Development Scripts](#development-scripts)
9. [Configuration Management](#configuration-management)
10. [Security & Operations](#security--operations)
11. [CI/CD Pipeline](#cicd-pipeline)
12. [Troubleshooting Guide](#troubleshooting-guide)

---

## Infrastructure Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Document Sources                          │
│                    (/mnt/docs, /var/embeddings)                   │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Semantik Engine                            │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐     │
│  │   Extract   │──│   Embed     │──│   Ingest to Qdrant   │     │
│  │   Chunks    │  │   Service   │  │                      │     │
│  └─────────────┘  └─────────────┘  └──────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Search API (Port 8000)                       │
│                    FastAPI Search Service                         │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      WebUI (Port 8080)                            │
│         ┌────────────────┐    ┌─────────────────────┐           │
│         │  React Frontend │    │   FastAPI Backend   │           │
│         │  (Port 5173)    │    │   (Auth, Jobs API)  │           │
│         └────────────────┘    └─────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.12+
- **Node.js**: 18.0+ (for frontend development)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **RAM**: Minimum 16GB, 32GB+ recommended
- **Storage**: SSD with at least 100GB free space

### Service Dependencies

- **Qdrant**: Vector database (v1.9.0+)
- **PostgreSQL**: Optional for advanced job management
- **Redis**: Optional for caching and rate limiting
- **Prometheus**: Metrics collection (optional)

---

## Development Environment

### Poetry Configuration

The project uses Poetry for Python dependency management. Configuration is defined in `pyproject.toml`:

```toml
[tool.poetry]
name = "document-embedding-system"
version = "2.0.0"
packages = [{include = "vecpipe", from = "packages"}, {include = "webui", from = "packages"}]

[tool.poetry.dependencies]
python = "^3.12"
qdrant-client = "^1.9.0"
sentence-transformers = "^2.5.1"
fastapi = "^0.110.0"
uvicorn = "^0.27.1"
# ... additional dependencies
```

### Package Structure

```
packages/
├── vecpipe/              # Core embedding engine
│   ├── extract_chunks.py # Document parsing
│   ├── embedding_service.py # Embedding generation
│   ├── search_api.py     # Search API service
│   └── config.py         # Configuration
├── webui/                # Web interface
│   ├── main.py          # FastAPI app
│   ├── api/             # API routers
│   ├── database.py      # SQLite management
│   └── static/          # Frontend assets
```

### Development Dependencies

Key development tools:
- **Black**: Code formatting
- **Ruff**: Fast Python linter
- **Mypy**: Static type checking
- **Pytest**: Testing framework
- **Coverage**: Test coverage reporting

### Make Commands

The `Makefile` provides convenient development commands:

```bash
# Installation
make install         # Install production dependencies
make dev-install     # Install development dependencies

# Code Quality
make format          # Format code with Black and isort
make lint           # Run Ruff linter
make type-check     # Run Mypy type checking
make check          # Run all checks (lint, type-check, test)

# Testing
make test           # Run all tests
make test-coverage  # Run tests with coverage report

# Frontend Development
make frontend-install  # Install frontend dependencies
make frontend-build   # Build frontend for production
make frontend-dev     # Start frontend dev server
make frontend-test    # Run frontend tests

# Development
make dev            # Start development environment
make clean          # Clean generated files
```

---

## Testing Framework

### Test Structure

```
tests/
├── conftest.py           # Shared fixtures and configuration
├── test_auth.py         # Authentication tests
├── test_document_api.py # Document API tests
├── test_search.py       # Search functionality tests
├── test_metrics.py      # Metrics collection tests
└── debug/              # Debug utilities
    ├── debug_metrics.py
    └── update_metrics_loop.py
```

### Key Test Fixtures (conftest.py)

```python
@pytest.fixture
def test_client(test_user):
    """FastAPI test client with authentication"""
    
@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing"""
    
@pytest.fixture
def mock_embedding_service():
    """Mock embedding service"""
```

### Mock Mode Testing

The system supports testing without GPU through mock embeddings:

```bash
# Enable mock mode for testing
export USE_MOCK_EMBEDDINGS=true
```

### Coverage Configuration

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=packages.vecpipe --cov=packages.webui --cov-report=html --cov-report=term"
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov

# Run specific test file
poetry run pytest tests/test_search.py

# Run with verbose output
poetry run pytest -v
```

---

## Build System

### Frontend Build Process

The `scripts/build.sh` script handles the frontend build:

```bash
#!/bin/bash
# Build React frontend
cd apps/webui-react
npm install
npm run build
# Assets are output to packages/webui/static/
```

### Build Manifest Generation

The `scripts/build_manifest.sh` creates file lists for processing:

```bash
#!/bin/bash
# Configuration
ROOTS=(
    "/mnt/zfs/docs/dirA"
    "/mnt/zfs/docs/dirB" 
    "/mnt/zfs/docs/dirC"
)
OUTPUT_FILE="/var/embeddings/filelist.null"

# Find eligible files (PDF, DOCX, TXT, etc.)
find "${ROOTS[@]}" -type f \
    \( -iname '*.pdf' -o -iname '*.docx' -o -iname '*.txt' \) \
    -print0 > "$OUTPUT_FILE"
```

### Python Package Building

```bash
# Build Python packages
poetry build

# Output:
# dist/
#   document_embedding_system-2.0.0.tar.gz
#   document_embedding_system-2.0.0-py3-none-any.whl
```

---

## Service Management

### Service Start Script (`start_all_services.sh`)

```bash
#!/bin/bash
# Check if ports are available
lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null && exit 1
lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null && exit 1

# Start Search API
poetry run python -m packages.vecpipe.search_api > search_api.log 2>&1 &
SEARCH_PID=$!

# Start WebUI
poetry run uvicorn packages.webui.app:app --host 0.0.0.0 --port 8080 > webui.log 2>&1 &
WEBUI_PID=$!

# Save PIDs for shutdown
echo $SEARCH_PID > .search_api.pid
echo $WEBUI_PID > .webui.pid
```

### Service Stop Script (`stop_all_services.sh`)

```bash
#!/bin/bash
# Stop services by PID files
stop_service "Search API" ".search_api.pid"
stop_service "WebUI" ".webui.pid"

# Also check ports
kill $(lsof -t -i:8000 -sTCP:LISTEN) 2>/dev/null
kill $(lsof -t -i:8080 -sTCP:LISTEN) 2>/dev/null
```

### Service Status Check (`status_services.sh`)

```bash
#!/bin/bash
# Check service status
check_service "Search API (port 8000)" 8000 ".search_api.pid"
check_service "WebUI (port 8080)" 8080 ".webui.pid"

# Display logs if available
[ -f "search_api.log" ] && echo "Search API log: $(wc -l < search_api.log) lines"
[ -f "webui.log" ] && echo "WebUI log: $(wc -l < webui.log) lines"
```

### Service Orchestration

Services are started in order with health checks:

1. **Search API** starts first (port 8000)
2. Wait for Search API to be healthy
3. **WebUI** starts second (port 8080)
4. Both services log to separate files

---

## Deployment Architecture

### Production Deployment

```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer                          │
│                   (Nginx/HAProxy)                         │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
┌───────▼────────┐                    ┌────────▼────────┐
│   WebUI Node 1 │                    │  WebUI Node 2   │
│   Port 8080    │                    │   Port 8080     │
└────────────────┘                    └─────────────────┘
        │                                       │
        └───────────────────┬───────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                    Search API Cluster                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Search API 1 │  │ Search API 2 │  │ Search API 3 │  │
│  │  Port 8000   │  │  Port 8000   │  │  Port 8000   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                    Qdrant Cluster                         │
│                 (Distributed Mode)                        │
└─────────────────────────────────────────────────────────┘
```

### Docker Deployment (Planned)

While Docker configuration is not yet implemented, the planned structure is:

```yaml
# docker-compose.yml (planned)
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage

  search-api:
    build: ./docker/search-api
    ports:
      - "8000:8000"
    environment:
      - QDRANT_HOST=qdrant
      - USE_MOCK_EMBEDDINGS=false
    depends_on:
      - qdrant

  webui:
    build: ./docker/webui
    ports:
      - "8080:8080"
    environment:
      - SEARCH_API_URL=http://search-api:8000
    depends_on:
      - search-api
```

### Environment Configuration

Production environment variables:

```bash
# Production settings
QDRANT_HOST=qdrant-cluster.internal
QDRANT_PORT=6333
DEFAULT_COLLECTION=production_docs
USE_MOCK_EMBEDDINGS=false
JWT_SECRET_KEY=${SECURE_JWT_SECRET}
ACCESS_TOKEN_EXPIRE_MINUTES=1440
```

---

## Monitoring and Logging

### Prometheus Metrics

The system exposes Prometheus metrics on port 9090:

```python
# Available metrics (packages/vecpipe/metrics.py)
- embedding_jobs_created_total
- embedding_jobs_completed_total
- embedding_jobs_failed_total
- embedding_job_duration_seconds
- embedding_files_processed_total
- embedding_chunks_created_total
- embedding_gpu_memory_used_bytes
- embedding_cpu_utilization_percent
```

### Logging Configuration

Logs are written to separate files:

```
logs/
├── search_api.log      # Search API logs
├── webui.log          # WebUI application logs
├── error_extract.log  # Document extraction errors
└── cleanup.log        # Cleanup service logs
```

### Log Rotation

Configure logrotate for production:

```
/var/embeddings/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 app app
}
```

### Performance Monitoring

Key metrics to monitor:

1. **GPU Utilization**: Track memory usage and compute utilization
2. **API Response Times**: Monitor search latency
3. **Queue Lengths**: Track processing backlogs
4. **Error Rates**: Monitor failed document processing

---

## Development Scripts

### Development Server (`scripts/dev.sh`)

```bash
#!/bin/bash
# Start development environment
# Runs backend and frontend with hot reloading

# Backend (port 8080)
cd packages/webui && python main.py &

# Frontend dev server (port 5173)
cd apps/webui-react && npm run dev &

# Trap exit to clean up processes
trap cleanup EXIT INT TERM
```

### Debug Utilities

```bash
# Debug memory usage
scripts/debug_memory_usage.py

# Benchmark embedding models
scripts/benchmark_qwen3.py

# Test cleanup service
scripts/test_cleanup_service.py

# Clean up temporary images
scripts/cleanup_temp_images.py
```

---

## Configuration Management

### Environment Variables

Core configuration is managed through environment variables:

```python
# packages/vecpipe/config.py
class Settings(BaseSettings):
    # Qdrant Configuration
    QDRANT_HOST: str
    QDRANT_PORT: int = 6333
    DEFAULT_COLLECTION: str = "work_docs"
    
    # Model Configuration
    USE_MOCK_EMBEDDINGS: bool = False
    DEFAULT_EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"
    DEFAULT_QUANTIZATION: str = "float16"
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()
    WEBUI_DB: Path = PROJECT_ROOT / "data" / "webui.db"
    JOBS_DIR: Path = PROJECT_ROOT / "data" / "jobs"
```

### Configuration Files

- `.env`: Local environment configuration
- `pyproject.toml`: Python project configuration
- `package.json`: Frontend dependencies
- `vite.config.ts`: Frontend build configuration

---

## Security & Operations

### Security Configuration

1. **JWT Authentication**:
   ```bash
   # Generate secure secret key
   openssl rand -hex 32
   ```

2. **Input Validation**:
   - File path sanitization
   - SQL injection prevention
   - XSS protection in frontend

3. **Rate Limiting**:
   - API rate limiting with SlowAPI
   - Configurable limits per endpoint

### Backup Procedures

```bash
# Backup WebUI database
cp data/webui.db data/webui.db.backup

# Backup Qdrant data
qdrant-backup --url http://localhost:6333 --output /backup/qdrant

# Backup job metadata
tar -czf jobs-backup.tar.gz data/jobs/
```

### Update Process

1. **Pull latest code**:
   ```bash
   git pull origin main
   ```

2. **Update dependencies**:
   ```bash
   poetry install
   cd apps/webui-react && npm install
   ```

3. **Run migrations** (if any):
   ```bash
   python scripts/migrate.py
   ```

4. **Rebuild frontend**:
   ```bash
   make frontend-build
   ```

5. **Restart services**:
   ```bash
   ./stop_all_services.sh
   ./start_all_services.sh
   ```

---

## CI/CD Pipeline

### GitHub Actions Workflow

The CI/CD pipeline (`.github/workflows/ci.yml`) includes:

1. **Lint Stage**:
   - Black formatting check
   - Ruff linting
   - Mypy type checking

2. **Test Stage**:
   - Unit tests with pytest
   - Integration tests with Qdrant
   - Coverage reporting to Codecov

3. **Build Stage** (planned):
   - Docker image building
   - Frontend asset compilation
   - Package publishing

### CI Environment

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - 6333:6333
```

---

## Troubleshooting Guide

### Common Issues

1. **Port Already in Use**:
   ```bash
   # Find process using port
   lsof -i :8080
   # Kill process
   kill -9 <PID>
   ```

2. **GPU Memory Errors**:
   ```bash
   # Check GPU usage
   nvidia-smi
   # Clear GPU cache
   python -c "import torch; torch.cuda.empty_cache()"
   ```

3. **Database Connection Failed**:
   ```bash
   # Check Qdrant status
   curl http://localhost:6333/
   # Restart Qdrant
   docker restart qdrant
   ```

4. **Search API Not Responding**:
   ```bash
   # Check logs
   tail -f search_api.log
   # Check process
   ps aux | grep search_api
   ```

### Debug Commands

```bash
# Check system resources
htop
nvidia-smi
df -h

# Check service logs
tail -f logs/*.log

# Test API endpoints
curl http://localhost:8000/
curl http://localhost:8080/api/health

# Database queries
sqlite3 data/webui.db "SELECT * FROM jobs;"
```

### Performance Tuning

1. **GPU Memory Optimization**:
   ```python
   # Adjust batch sizes
   EMBEDDING_BATCH_SIZE = 32  # Reduce if OOM
   
   # Enable model unloading
   MODEL_UNLOAD_AFTER_SECONDS = 300
   ```

2. **API Performance**:
   ```python
   # Increase workers
   uvicorn main:app --workers 4
   
   # Enable connection pooling
   QDRANT_CONNECTION_POOL_SIZE = 10
   ```

3. **Frontend Optimization**:
   ```bash
   # Production build with optimizations
   npm run build -- --minify
   ```

---

## Maintenance Schedule

### Daily Tasks
- Monitor service logs for errors
- Check disk space usage
- Verify backup completion

### Weekly Tasks
- Review metrics dashboards
- Clean up old job files
- Update file manifests

### Monthly Tasks
- Security updates
- Performance analysis
- Capacity planning

### Quarterly Tasks
- Dependency updates
- Architecture review
- Disaster recovery testing

---

## Contact and Support

For infrastructure issues:
1. Check this documentation first
2. Review logs in `/logs/` directory
3. Check GitHub issues
4. Contact the development team

For emergency support:
- Infrastructure alerts: [monitoring system]
- On-call rotation: [team schedule]