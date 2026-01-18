# Configuration Guide

## Quick Start

```bash
# Copy template and run wizard
cp .env.docker.example .env
make wizard  # Auto-detects GPU, generates secrets, creates dirs
make docker-up
```

## Environment Variables

### Core Settings

#### Environment Configuration
```bash
# Environment mode
ENVIRONMENT=development        # Options: development, staging, production

# Logging
LOG_LEVEL=INFO                # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# CORS Configuration
# SECURITY: Wildcards ('*') and 'null' origins are rejected in all environments.
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173  # Comma-separated list
```

#### Database Configuration

##### PostgreSQL (Required)
```bash
# Connection string (overrides individual settings if provided)
DATABASE_URL=postgresql://semantik:password@postgres:5432/semantik

# Individual connection parameters
POSTGRES_HOST=postgres        # PostgreSQL host (container name in Docker)
POSTGRES_PORT=5432           # PostgreSQL port
POSTGRES_DB=semantik         # Database name
POSTGRES_USER=semantik       # Database user
POSTGRES_PASSWORD=CHANGE_THIS_TO_A_STRONG_PASSWORD  # Database password

# Connection pool settings
DB_POOL_SIZE=20              # Connection pool size
DB_MAX_OVERFLOW=40           # Maximum overflow connections
DB_POOL_TIMEOUT=30           # Pool timeout in seconds
DB_POOL_RECYCLE=3600         # Connection recycle time in seconds
DB_POOL_PRE_PING=true        # Test connections before use
```

##### Qdrant (Vector Database)
```bash
# Qdrant connection
QDRANT_HOST=qdrant           # Qdrant host (container name in Docker)
QDRANT_PORT=6333             # Qdrant HTTP port
DEFAULT_COLLECTION=work_docs  # Default collection name

# Qdrant authentication (required in v7.1+)
QDRANT_API_KEY=CHANGE_THIS   # Generate: openssl rand -hex 32
```

##### Redis (Message Broker)
```bash
# Redis connection
REDIS_URL=redis://redis:6379/0              # Redis connection URL
CELERY_BROKER_URL=redis://redis:6379/0      # Celery broker URL
CELERY_RESULT_BACKEND=redis://redis:6379/0  # Celery result backend

# Redis authentication (required in v7.1+)
REDIS_PASSWORD=CHANGE_THIS   # Generate: openssl rand -hex 32
```

#### Authentication & Security
```bash
# JWT Configuration
JWT_SECRET_KEY=CHANGE_THIS_TO_A_STRONG_SECRET_KEY  # Generate: uv run python scripts/generate_jwt_secret.py --write
JWT_ALGORITHM=HS256                                 # JWT signing algorithm
ACCESS_TOKEN_EXPIRE_MINUTES=1440                    # Access token expiration (24 hours)

# Connector Secrets Encryption (for Git/IMAP/etc.)
# - If set, must be a valid Fernet key (44-char base64) or the WebUI will refuse to start.
# - If empty/unset, connectors that require credentials cannot be configured.
CONNECTOR_SECRETS_KEY=CHANGE_THIS_TO_A_FERNET_KEY   # Generate: uv run python scripts/generate_secrets_key.py --write

# Security
DISABLE_AUTH=false           # Disable authentication (NEVER in production!)
```

### Model Configuration

#### Embedding Models
```bash
# Model Selection
DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B  # Default embedding model
USE_MOCK_EMBEDDINGS=false                          # Use mock embeddings for testing

# Available models:
# - Qwen/Qwen3-Embedding-0.6B (default, 1024 dims)
# - Qwen/Qwen3-Embedding-4B (2560 dims)
# - BAAI/bge-large-en-v1.5 (1024 dims)
# - sentence-transformers/all-MiniLM-L6-v2 (384 dims)
```

#### Plugin Loading
```bash
# Global toggle for external plugins (entry points: semantik.plugins)
SEMANTIK_ENABLE_PLUGINS=true

# Per-type toggles (external plugins only; built-ins always load)
SEMANTIK_ENABLE_EMBEDDING_PLUGINS=true
SEMANTIK_ENABLE_CHUNKING_PLUGINS=true
SEMANTIK_ENABLE_CONNECTOR_PLUGINS=true
```

#### Quantization & Performance
```bash
# Quantization modes
DEFAULT_QUANTIZATION=float16  # Options: float32, float16, int8

# Memory usage by quantization:
# float32: Full precision (highest quality, most memory)
# float16: Half precision (recommended, good balance)
# int8: 8-bit quantization (lowest memory, requires CUDA build)

# Model Management
MODEL_UNLOAD_AFTER_SECONDS=300  # Auto-unload models after inactivity
```

#### Local LLM
Local LLMs are GPU-intensive and require CUDA for practical performance. If you do not have a compatible
NVIDIA GPU, set `ENABLE_LOCAL_LLM=false` (the `/llm/health` endpoint will report `disabled`).

```bash
# Enable local LLM inference in VecPipe
ENABLE_LOCAL_LLM=true
# Default quantization for local LLMs (int4, int8, float16)
DEFAULT_LLM_QUANTIZATION=int8
# Idle timeout before unloading local LLM models (seconds)
LLM_UNLOAD_AFTER_SECONDS=300
# KV cache + runtime buffer per loaded LLM (MB)
LLM_KV_CACHE_BUFFER_MB=1024
# Allow models that require remote code (HuggingFace trust_remote_code)
LLM_TRUST_REMOTE_CODE=false
```

#### GPU Configuration
```bash
# GPU Selection
CUDA_VISIBLE_DEVICES=0       # GPU device ID (comma-separated for multiple)
MODEL_MAX_MEMORY_GB=8        # GPU memory limit

# HuggingFace Settings
HF_HOME=/app/.cache/huggingface  # Model cache directory
HF_HUB_OFFLINE=false             # Use offline mode after models downloaded
```

### Storage & Paths

#### Volume Paths (Host Machine)
```bash
# Document Storage
DOCUMENT_PATH=./documents    # Source documents directory

# Model Cache
HF_CACHE_DIR=./models       # HuggingFace model cache
```

#### Operation Paths (Inside Containers)
```bash
# Processing Directories (collection-centric)
EXTRACT_DIR=/app/operations/extract  # Document extraction directory
INGEST_DIR=/app/operations/ingest    # Document ingestion directory

# Data Directories
DATA_DIR=/app/data          # Application data
LOGS_DIR=/app/logs          # Log files
UPLOAD_DIR=/app/uploads     # Temporary uploads
```

### Service Configuration

#### API Services
```bash
# WebUI Service
WEBUI_PORT=8080             # WebUI port
WEBUI_WORKERS=1             # Number of worker processes

# Search API Service
SEARCH_API_PORT=8000        # Search API port
SEARCH_API_URL=http://vecpipe:8000  # Internal URL for WebUI
SEARCH_API_HOST=vecpipe     # Search API hostname

# Service Discovery
WAIT_FOR_QDRANT=true        # Wait for Qdrant on startup
WAIT_FOR_SEARCH_API=true    # Wait for Search API on startup
```

#### Processing Configuration
```bash
# Document Processing
MAX_WORKERS=4               # Parallel processing workers
BATCH_SIZE=32              # Embedding batch size
CHUNK_SIZE=512             # Document chunk size in tokens
CHUNK_OVERLAP=128          # Chunk overlap in tokens
MAX_FILE_SIZE_MB=100       # Maximum file size to process

# Adaptive Batch Sizing
MIN_BATCH_SIZE=4           # Minimum batch size on OOM
BATCH_RESTORE_THRESHOLD=5  # Batches before size increase
ADAPTIVE_BATCH_SIZING=true # Enable adaptive batching
```

#### Resource Limits & Cache

| Variable | Default | Description |
|----------|---------|-------------|
| MAX_COLLECTIONS_PER_USER | 10 | Maximum number of collections per user |
| MAX_STORAGE_GB_PER_USER | 50.0 | Maximum storage per user (GB) |
| CACHE_DEFAULT_TTL_SECONDS | 300 | Default cache TTL (seconds) |

#### Background Cleanup Circuit Breaker (Configurable)

These control the Redis cleanup loop backoff and circuit breaker behavior:

| Variable | Default | Description |
|----------|---------|-------------|
| REDIS_CLEANUP_INTERVAL_SECONDS | 60 | Base cleanup interval in seconds |
| REDIS_CLEANUP_MAX_CONSECUTIVE_FAILURES | 5 | Failures before opening the circuit |
| REDIS_CLEANUP_BACKOFF_MULTIPLIER | 2.0 | Exponential backoff multiplier (>= 1.0) |
| REDIS_CLEANUP_MAX_BACKOFF_SECONDS | 300 | Maximum backoff delay in seconds |

#### Reranking Configuration
```bash
# Reranking Settings
USE_RERANKER=false                 # Enable reranking by default
RERANK_CANDIDATE_MULTIPLIER=5      # Retrieve k*5 candidates
RERANK_MIN_CANDIDATES=20           # Minimum candidates
RERANK_MAX_CANDIDATES=200          # Maximum candidates
DEFAULT_RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B  # Reranker model
```

### Monitoring & Metrics

```bash
# Prometheus Metrics
METRICS_ENABLED=true        # Enable metrics collection
METRICS_PORT=9091          # Metrics port
METRICS_PATH=/metrics      # Metrics endpoint path

# Health Checks
HEALTH_CHECK_ENABLED=true   # Enable health checks
HEALTH_CHECK_INTERVAL=30    # Check interval in seconds

# Flower (Task Monitoring)
FLOWER_PORT=5555           # Flower web UI port
FLOWER_USERNAME=flower_123abc   # Generated by `make wizard`
FLOWER_PASSWORD=sup3r-strong-secret!  # Generated by `make wizard`
```

The setup wizard generates unique Flower credentials and writes them to `.env`. `scripts/validate_env.py` fails if either value is missing or still uses a placeholder such as `admin`. Rerun `make wizard` whenever you need to rotate the Flower username/password.

## Configuration Files

### Docker Environment Files

1. **`.env.docker.example`** - Template for Docker deployments
2. **`.env.example`** - Template for manual deployments
3. **`.env`** - Your actual configuration (git-ignored)

### Service Configuration Examples

#### Development Configuration
```bash
# .env.development
ENVIRONMENT=development
LOG_LEVEL=DEBUG
USE_MOCK_EMBEDDINGS=false
DEFAULT_QUANTIZATION=float16
DISABLE_AUTH=true
```

#### Production Configuration
```bash
# .env.production
ENVIRONMENT=production
LOG_LEVEL=INFO
USE_MOCK_EMBEDDINGS=false
DEFAULT_QUANTIZATION=int8
DISABLE_AUTH=false
JWT_SECRET_KEY=${SECURE_JWT_SECRET}
POSTGRES_PASSWORD=${SECURE_DB_PASSWORD}
```

## Model Configuration Details

### Memory Requirements by Model

| Model | float32 | float16 | int8 |
|-------|---------|---------|------|
| Qwen3-0.6B | ~2.4GB | ~1.2GB | ~0.6GB |
| Qwen3-4B | ~16GB | ~8GB | ~4GB |
| BGE-large | ~1.4GB | ~0.7GB | ~0.35GB |
| MiniLM | ~0.5GB | ~0.25GB | ~0.125GB |

## Runtime Examples

Collection creation:
```json
{"name": "docs", "model_name": "Qwen/Qwen3-Embedding-0.6B", "quantization": "float16"}
```

Search request:
```json
{"query": "...", "collection_id": "uuid", "limit": 10, "search_type": "hybrid"}
```

## Performance Tuning

### GPU Optimization

```bash
# Memory Management
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
GPU_MEMORY_FRACTION=0.9

# Performance
TORCH_CUDNN_V8_API_ENABLED=1
USE_AMP=true  # Automatic Mixed Precision
```

### Batch Size Guidelines

| GPU Memory | Recommended Batch Size |
|------------|----------------------|
| 4GB | 16-32 |
| 8GB | 32-64 |
| 16GB | 64-128 |
| 24GB+ | 128-256 |

### CPU-Only Configuration

```bash
# Force CPU mode
FORCE_CPU=true
CUDA_VISIBLE_DEVICES=""

# Optimize for CPU
DEFAULT_QUANTIZATION=int8
BATCH_SIZE=8
MAX_WORKERS=16
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Security Configuration

### Production Security Checklist

1. **Generate Secure Keys**
   ```bash
   # JWT Secret
   export JWT_SECRET_KEY=$(uv run python scripts/generate_jwt_secret.py)

   # Database Password
   export POSTGRES_PASSWORD=$(openssl rand -hex 32)

   # Redis Password (required in v7.1+)
   export REDIS_PASSWORD=$(openssl rand -hex 32)

   # Qdrant API Key (required in v7.1+)
   export QDRANT_API_KEY=$(openssl rand -hex 32)
   ```

2. **Enable Security Features**
   ```bash
   ENVIRONMENT=production
   DISABLE_AUTH=false
   SECURE_HEADERS_ENABLED=true
   HSTS_MAX_AGE=31536000
   ```

3. **Network Security**
   ```bash
   # Restrict CORS origins (no wildcards)
   CORS_ORIGINS=https://yourdomain.com
   
   # Use internal networks
   QDRANT_HOST=qdrant.internal
   POSTGRES_HOST=postgres.internal
   ```

## Logging Configuration

### Log Levels

```bash
# Development
LOG_LEVEL=DEBUG

# Production
LOG_LEVEL=INFO

# Troubleshooting
LOG_LEVEL=DEBUG
SQLALCHEMY_ECHO=true  # Database query logging
```

### Log Outputs

Services write to these locations:
- WebUI: `/app/logs/webui.log`
- Vecpipe: `/app/logs/vecpipe.log`
- Worker: `/app/logs/worker.log`
- Operations: `/app/logs/operations/`

## Docker-Specific Configuration

### Volume Configuration

```yaml
volumes:
  # Named volumes (persistent)
  - postgres_data:/var/lib/postgresql/data
  - qdrant_storage:/qdrant/storage
  - redis_data:/data
  
  # Bind mounts (host directories)
  - ./data:/app/data
  - ./models:/app/.cache/huggingface
  - ./logs:/app/logs
  - ${DOCUMENT_PATH}:/mnt/docs:ro
```

### Network Configuration

Services communicate via Docker network:
- WebUI → Vecpipe: `http://vecpipe:8000`
- Services → PostgreSQL: `postgres:5432`
- Services → Qdrant: `qdrant:6333`
- Services → Redis: `redis:6379`

## Migration from Legacy Configuration

If migrating from the old job-based system:

### Path Updates
```bash
# Old (job-based)
EXTRACT_DIR=/app/jobs/extract
INGEST_DIR=/app/jobs/ingest

# New (operation-based)
EXTRACT_DIR=/app/operations/extract
INGEST_DIR=/app/operations/ingest
```

### Database Migration
```bash
# Run migrations after updating configuration
docker compose run --rm webui alembic upgrade head
```

## Troubleshooting Configuration

### Common Issues

1. **JWT Secret Not Set**
   ```bash
   # Error: JWT_SECRET_KEY not configured
   # Solution: Generate and set secret
   export JWT_SECRET_KEY=$(uv run python scripts/generate_jwt_secret.py)
   ```

2. **Database Connection Failed**
   ```bash
   # Check PostgreSQL is running
   docker compose ps postgres
   
   # Verify connection string
   echo $DATABASE_URL
   ```

3. **GPU Not Available**
   ```bash
   # Check CUDA configuration
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi
   ```

4. **Model Download Issues**
   ```bash
   # Enable offline mode after download
   HF_HUB_OFFLINE=true
   
   # Check cache directory
   ls -la ./models
   ```

### Debug Configuration

```bash
# Enable all debug options
ENVIRONMENT=development
LOG_LEVEL=DEBUG
DEBUG=true
SQLALCHEMY_ECHO=true
```

## Best Practices

- Never commit `.env` files
- Change all default passwords
- Use strong JWT secrets (64+ chars)
- Restrict CORS in production
- Use GPU when available
- Enable quantization to save memory
- Use SSD for model cache

See also: [DOCKER.md](./DOCKER.md), [DEPLOYMENT.md](./DEPLOYMENT.md)
