# Infrastructure & Deployment Architecture

> **Location:** `Dockerfile`, `docker-compose*.yml`, `Makefile`, `alembic/`

## Overview

Semantik uses Docker Compose for orchestration with multi-stage builds for optimization. The architecture supports development, production, and CUDA-specific deployments.

## Docker Architecture

### Multi-Stage Dockerfile

```dockerfile
# Stage 1: Frontend Builder (node:20-alpine)
# Builds React app to /build/packages/webui/static/

# Stage 2: Python Dependencies Builder (nvidia/cuda:12.1.0-cudnn8-devel)
# Installs Python deps via uv, creates /app/.venv

# Stage 3: Runtime (nvidia/cuda:12.1.0-cudnn8-runtime)
# Minimal runtime with venv and compiled frontend
```

**Key Optimizations:**
- Layer caching: package.json/pyproject.toml copied before source
- Non-root user: `appuser` (UID 1000)
- Minimal runtime image (~3GB vs ~5GB dev)

### Docker Entrypoint

`docker-entrypoint.sh` supports five service modes:

| Mode | Command | Description |
|------|---------|-------------|
| webui | `["webui"]` | FastAPI application |
| vecpipe | `["vecpipe"]` | Search/embedding API |
| worker | `["worker"]` | Celery worker |
| beat | `["beat"]` | Celery scheduler |
| flower | `["flower"]` | Celery monitoring |

## Service Topology

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Network                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐      │
│  │ webui   │   │ vecpipe │   │ worker  │   │  beat   │      │
│  │  :8080  │   │  :8000  │   │         │   │         │      │
│  └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘      │
│       │             │             │             │            │
│       └──────┬──────┴──────┬──────┴─────────────┘            │
│              │             │                                  │
│       ┌──────┴──────┐ ┌────┴────┐                            │
│       │  postgres   │ │  redis  │                            │
│       │    :5432    │ │  :6379  │                            │
│       └─────────────┘ └─────────┘                            │
│              │                                                │
│       ┌──────┴──────┐                                        │
│       │   qdrant    │                                        │
│       │ :6333/:6334 │                                        │
│       └─────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

### Service Resources

| Service | CPU | Memory | GPU | Health Check |
|---------|-----|--------|-----|--------------|
| postgres | 2 | 2GB | - | `pg_isready` |
| qdrant | 2 | 4GB | - | HTTP `/health` |
| redis | 0.5 | 512MB | - | `redis-cli ping` |
| webui | 1 | 2GB | - | HTTP `/api/health/readyz` |
| vecpipe | 2 | 4GB | 1 | HTTP `/health` |
| worker | 2 | 4GB | - | Celery inspect |
| beat | 0.5 | 1GB | - | PID file check |
| flower | 0.5 | 512MB | - | - |

## Compose Overlays

### Development (`docker-compose.dev.yml`)
```yaml
services:
  webui:
    volumes:
      - ./packages:/app/packages:ro  # Source mount
    environment:
      - ENVIRONMENT=development
      - DB_ECHO=true
      - WEBUI_RELOAD=true
```

### Production (`docker-compose.prod.yml`)
```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.7.4  # Pinned version
    environment:
      - QDRANT__LOG_LEVEL=WARN
    logging:
      driver: json-file
      options:
        max-size: "100m"
        max-file: "5"

  webui:
    environment:
      - ENVIRONMENT=production
      - PYTHONOPTIMIZE=2
      - WEBUI_WORKERS=4
    ports: []  # No direct exposure (use reverse proxy)
```

### CUDA (`docker-compose.cuda.yml`)
```yaml
services:
  vecpipe:
    environment:
      - LD_LIBRARY_PATH=/usr/local/cuda/lib64
      - CUDA_HOME=/usr/local/cuda
      - BITSANDBYTES_NOWELCOME=1
```

## Volumes

### Named Volumes
| Volume | Mount | Purpose |
|--------|-------|---------|
| postgres_data | /var/lib/postgresql/data | Database files |
| qdrant_storage | /qdrant/storage | Vector index |
| redis_data | /data | Task queue persistence |

### Bind Mounts
| Host | Container | Purpose |
|------|-----------|---------|
| ./data | /app/data | Operations, ingest |
| ./logs | /app/logs | Application logs |
| ./models | /app/.cache/huggingface | Model cache |
| ./documents | /mnt/docs:ro | Document source |

## Makefile Commands

### Core Operations
```bash
make wizard         # Interactive setup
make docker-up      # Start all services
make docker-down    # Stop (keep volumes)
make docker-down-clean  # Stop and delete volumes
make docker-logs    # Stream all logs
make docker-ps      # Show status
```

### Development
```bash
make docker-dev-up      # Backend services only
make dev-local          # Local webui + Docker services
make docker-shell-webui # Container shell
```

### Database
```bash
make docker-postgres-backup   # pg_dump
make docker-postgres-restore BACKUP_FILE=...
```

### Testing
```bash
make test           # All tests
make test-ci        # Exclude E2E
make test-coverage  # With coverage
```

## Environment Configuration

### Required Variables
```bash
DATABASE_URL=postgresql://user:pass@host:5432/db
JWT_SECRET_KEY=<64-hex-chars>
QDRANT_HOST=qdrant
REDIS_URL=redis://redis:6379/0
```

### Embedding Configuration
```bash
DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
DEFAULT_QUANTIZATION=float16  # float32, float16, int8
USE_MOCK_EMBEDDINGS=false
```

### Connection Pool
```bash
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600
DB_POOL_PRE_PING=true
```

## Database Migrations

### Running Migrations
```bash
# In container
docker compose exec webui alembic upgrade head

# Locally
uv run alembic upgrade head
```

### Creating Migrations
```bash
# Auto-generate from model changes
uv run alembic revision --autogenerate -m "description"

# Manual migration
uv run alembic revision -m "description"
```

### Rollback
```bash
uv run alembic downgrade -1  # One step back
uv run alembic downgrade base  # All the way back
```

## CI/CD (GitHub Actions)

### Workflow Triggers
- Push to `main`, `develop`
- All pull requests
- Manual dispatch

### Jobs
1. **Format Check** - black, isort
2. **Linting** - ruff, mypy
3. **Frontend Lint** - npm run lint
4. **Security Scan** - Trivy, Safety
5. **Backend Tests** - Matrix: unit, webui, integration, other
6. **Frontend Tests** - Vitest
7. **Build Validation** - npm run build

### Test Matrix
```yaml
strategy:
  matrix:
    test-group: [unit, webui, integration, other]
```

## Deployment Steps

### Initial Setup
```bash
# Clone and configure
git clone https://github.com/org/semantik.git
cd semantik
cp .env.docker.example .env

# Generate secrets
make wizard

# Create directories
mkdir -p ./models ./data ./logs ./documents

# Start services
make docker-up
```

### Production with Reverse Proxy
```bash
docker compose -f docker-compose.yml \
  -f docker-compose.prod.yml \
  up -d

# Configure nginx to proxy 80/443 → webui:8080
```

### Backup Strategy
```bash
# Daily PostgreSQL backup (add to cron)
make docker-postgres-backup

# Qdrant snapshot
curl -X POST http://localhost:6333/snapshots
```

## Troubleshooting

### Services Won't Start
```bash
docker compose logs <service>
lsof -i :8080  # Check port conflicts
make docker-down-clean && make docker-up  # Reset
```

### Database Issues
```bash
docker compose exec postgres pg_isready -U semantik
docker compose exec postgres psql -U semantik -l
```

### GPU Not Detected
```bash
docker compose exec vecpipe nvidia-smi
# Verify nvidia-docker runtime
```

## Extension Points

### Adding a New Service
1. Add service to `docker-compose.yml`
2. Add entrypoint case to `docker-entrypoint.sh`
3. Add Makefile targets for logs/shell
4. Update documentation

### Adding Environment Variables
1. Add to `.env.docker.example`
2. Update config classes in `packages/shared/config/`
3. Update validation script if required
