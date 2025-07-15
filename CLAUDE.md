You should **NEVER** mention Claude or Anthropic in your PR or Commit messages.


# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Semantik** is a self-hosted semantic search engine (pre-release) that transforms file servers into powerful, private knowledge bases using AI-powered document search. It uses vector embeddings and transformer models to enable semantic search across documents.

## Essential Commands

### Quick Start & Development

```bash
# Interactive setup wizard (recommended for first-time setup)
make wizard

# Start full development environment
make dev                          # Runs backend + frontend dev servers
./scripts/dev.sh                  # Alternative: manual dev server startup

# Docker operations
make docker-up                    # Start all services
make docker-down                  # Stop all services
make docker-logs                  # View container logs
make docker-ps                    # Show container status
make docker-build-fresh           # Rebuild without cache
```

### Code Quality & Testing

```bash
# Python code quality
make format                       # Format with black & isort
make lint                         # Lint with ruff
make type-check                   # Type check with mypy
make test                         # Run all tests
make test-ci                      # Tests excluding E2E
make test-coverage                # Generate coverage report
make check                        # Run all checks (format, lint, type-check)

# Frontend
make frontend-test                # Run React tests
cd apps/webui-react && npm test   # Alternative

# Specific test types
pytest tests/unit                 # Unit tests only
pytest tests/integration          # Integration tests
pytest tests/e2e                  # E2E tests (requires services running)
pytest -m "not e2e"              # All tests except E2E
```

### Building & Installation

```bash
# Backend setup
make install                      # Install production dependencies (uses poetry install --only main)
make dev-install                  # Install development dependencies
poetry install                    # Direct Poetry install

# Frontend setup
make frontend-install             # Install frontend dependencies
make frontend-build               # Production build
make frontend-dev                 # Development server

# Database migrations
alembic upgrade head              # Apply migrations
alembic revision --autogenerate -m "description"  # Create new migration
```

## High-Level Architecture

### Core Components

1. **Vector Pipeline** (`packages/vecpipe/`)
   - Embedding generation using transformer models
   - Vector storage in Qdrant database
   - Search API with semantic/hybrid search
   - Cross-encoder reranking support

2. **Document Processing** (`packages/shared/text_processing/`)
   - Document extraction from various formats
   - Text chunking and preprocessing
   - Executed via Celery job queue

3. **Web Application** (`packages/webui/`)
   - FastAPI backend with JWT authentication
   - Job queue system using Celery + Redis
   - Collection management and search API proxy
   - User management

4. **Frontend** (`apps/webui-react/`)
   - React 19 with TypeScript
   - TailwindCSS for styling
   - Zustand for state management
   - React Query for data fetching

5. **Shared Components** (`packages/shared/`)
   - Database models (SQLAlchemy)
   - Embedding service manager
   - Model management utilities
   - Common configuration

### Service Architecture & Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Frontend  │────▶│   WebUI API  │────▶│  Search API  │
│  (React)    │     │  (FastAPI)   │     │  (FastAPI)   │
└─────────────┘     └──────────────┘     └──────────────┘
                            │                      │
                            ▼                      ▼
                    ┌──────────────┐      ┌──────────────┐
                    │    Redis     │      │   Qdrant     │
                    │   (Queue)    │      │  (Vectors)   │
                    └──────────────┘      └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │   Workers    │
                    │  (Celery)    │
                    └──────────────┘
```

#### Document Processing Pipeline

1. **Upload Phase**
   - User uploads documents via Frontend (React)
   - WebUI API receives files and creates a job
   - Job is queued in Redis for async processing
   - WebSocket connection established for real-time updates

2. **Extraction Phase** (Celery Worker)
   - Worker picks up job from Redis queue
   - Document extraction (`packages/shared/text_processing/extraction.py`)
   - Supports PDF, DOCX, PPTX, TXT, and more formats
   - Extracted text stored temporarily

3. **Chunking Phase** (Celery Worker)
   - Text split into semantic chunks (`packages/shared/text_processing/chunking.py`)
   - Configurable chunk size and overlap
   - Chunks saved to parquet files with metadata

4. **Embedding Phase** (Celery Worker)
   - Chunks processed through embedding model
   - GPU-accelerated when available
   - Embeddings saved to parquet files

5. **Storage Phase** (Celery Worker)
   - Embeddings uploaded to Qdrant vector database
   - Collection created/updated with new vectors
   - Metadata and relationships preserved

6. **Search Phase**
   - User queries through Frontend
   - WebUI proxies to Search API
   - Search API queries Qdrant for similar vectors
   - Optional reranking for better results
   - Results returned with metadata and scores

#### Component Responsibilities

**Frontend (React)**
- User authentication and session management
- Document upload interface
- Search interface with filters
- Real-time job monitoring via WebSocket
- Document preview and navigation

**WebUI API (FastAPI)**
- User authentication (JWT)
- Job orchestration and monitoring
- Collection management
- Search request proxying
- Document serving with range support
- WebSocket connections for real-time updates

**Search API (FastAPI)**
- Embedding model management
- Vector similarity search
- Hybrid search (vector + keyword)
- Cross-encoder reranking
- Model recommendations based on GPU

**Workers (Celery)**
- Async document processing
- Progress reporting via Redis
- Error handling and retries
- GPU resource management

**Databases**
- **SQLite**: User accounts, jobs, file metadata
- **Qdrant**: Vector embeddings and search index
- **Redis**: Job queue, session storage, real-time state

### Key API Endpoints

**Search API (port 8000)**

*Health & Status:*
- `GET /` - Health check with collection info
- `GET /health` - Comprehensive health check
- `GET /model/status` - Get model manager status

*Search Operations:*
- `GET/POST /search` - Semantic search with optional reranking
- `GET /hybrid_search` - Combined vector/keyword search
- `POST /search/batch` - Batch search for multiple queries
- `GET /keyword_search` - Keyword-only search (no vectors)

*Model Management:*
- `GET /models` - List available embedding models
- `POST /models/load` - Load specific embedding model
- `GET /models/suggest` - Suggest optimal model based on GPU memory

*Collection & Embedding Info:*
- `GET /collection/info` - Get vector collection information
- `GET /embedding/info` - Get embedding configuration info

**WebUI API (port 8080)**

*Authentication:*
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login and receive tokens
- `POST /api/auth/refresh` - Refresh access token
- `POST /api/auth/logout` - Logout and revoke token
- `GET /api/auth/me` - Get current user info

*Collections:*
- `GET /api/collections` - List all collections
- `GET /api/collections/{name}` - Get collection details
- `PUT /api/collections/{name}` - Rename collection
- `DELETE /api/collections/{name}` - Delete collection
- `GET /api/collections/{name}/files` - List files (paginated)

*Jobs:*
- `GET /api/jobs/new-id` - Generate new job ID
- `POST /api/jobs` - Create new embedding job
- `POST /api/jobs/add-to-collection` - Add to existing collection
- `GET /api/jobs` - List all jobs
- `GET /api/jobs/{job_id}` - Get job details
- `POST /api/jobs/{job_id}/cancel` - Cancel running job
- `DELETE /api/jobs/{job_id}` - Delete job
- `GET /api/jobs/collection-metadata/{name}` - Get collection metadata
- `POST /api/jobs/check-duplicates` - Check for duplicate files
- `GET /api/jobs/collections-status` - Check Qdrant collections status

*Search Proxy:*
- `POST /api/search` - Unified search endpoint
- `POST /api/preload_model` - Preload model to prevent timeouts
- `POST /api/hybrid_search` - Hybrid search proxy

*Document Viewer:*
- `GET /api/documents/{job_id}/{doc_id}` - Retrieve document (supports range requests)
- `GET /api/documents/{job_id}/{doc_id}/info` - Get document metadata
- `GET /api/documents/temp-images/{session_id}/{filename}` - Serve temporary images

*Additional Endpoints:*
- `GET /api/health/` - Basic health check
- `GET /api/health/ready` - Readiness probe
- `GET /api/metrics` - Prometheus metrics
- `POST /api/scan-directory` - Scan directory for files
- `GET /api/models` - Get available embedding models

**WebSocket Endpoints (Real-time Updates)**

- `WS /ws/{job_id}` - Real-time job status updates
  - Authentication: Pass JWT token as query parameter `?token=<jwt_token>`
  - Messages: Progress updates, status changes, completion notifications
  
- `WS /ws/scan/{scan_id}` - Real-time file scanning progress
  - Authentication: Pass JWT token as query parameter `?token=<jwt_token>`
  - Messages: Files discovered, scan progress, completion status

### Model & Embedding System

- Default model: `Qwen/Qwen3-Embedding-0.6B` (small, efficient)
- Auto-downloads models from HuggingFace on first use
- Supports GPU acceleration (CUDA) with automatic detection
- Models stored in `/models` volume (persisted)
- Automatic model unloading after 5 minutes of inactivity
- Dynamic model loading/unloading based on usage
- GPU memory-based model recommendations via `/models/suggest`

### Document Preview System

The system includes a sophisticated document viewer with:

- **Range Request Support**: Efficient loading of large documents
- **PPTX to Markdown Conversion**: Automatic conversion of PowerPoint presentations
- **Temporary Image Serving**: Extracted images from presentations served via temporary URLs
- **Multiple Format Support**: PDF, DOCX, PPTX, TXT, and more
- **Chunked Document Viewing**: View individual chunks with metadata

Access documents via:
- `/api/documents/{job_id}/{doc_id}` - Full document with range support
- `/api/documents/{job_id}/{doc_id}/info` - Document metadata
- `/api/documents/temp-images/{session_id}/{filename}` - Temporary images

## Key Configuration

### Environment Variables

Create `.env` file with these settings:

```bash
# -- Core Settings --
ENVIRONMENT=development  # or 'production' (affects JWT behavior)
QDRANT_HOST=localhost
QDRANT_PORT=6333
DEFAULT_COLLECTION=work_docs

# -- Model Configuration --
DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
DEFAULT_QUANTIZATION=float16
MODEL_UNLOAD_AFTER_SECONDS=300
USE_MOCK_EMBEDDINGS=false  # Set true for testing without GPU

# -- Authentication --
# CRITICAL: In production, you MUST set a secure JWT_SECRET_KEY
# The application will fail to start in production without this
JWT_SECRET_KEY=your-secret-key-here  # Generate with: openssl rand -hex 32
ACCESS_TOKEN_EXPIRE_MINUTES=1440     # Default: 24 hours
DISABLE_AUTH=false                   # Set true for development only

# -- Redis Configuration --
REDIS_URL=redis://localhost:6379/0

# -- CORS Configuration --
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173  # Frontend URLs

# -- GPU Configuration (Optional) --
# CUDA_VISIBLE_DEVICES=0              # Select specific GPU
# MODEL_MAX_MEMORY_GB=8               # Limit GPU memory usage
# MONITOR_GPU_MEMORY=true             # Enable GPU monitoring

# -- Docker-Specific (when using Docker) --
# DOCUMENT_PATH=./documents           # Path to document volume
# WEBUI_WORKERS=1                     # Number of webui workers
# CELERY_CONCURRENCY=1                # Number of Celery workers
# LOG_LEVEL=INFO                      # Logging level

# -- HuggingFace Configuration --
# HF_HOME=/models                     # Model cache directory
# HF_HUB_OFFLINE=false               # Enable offline mode
```

### Docker Compose Profiles

```bash
# Standard deployment (auto-detects GPU/CPU)
docker compose up -d

# Production deployment
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Force CUDA GPU support
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d

# Note: CPU-only deployment happens automatically when GPU is not available
```

## Development Workflows

### Adding New Features

1. **Backend Feature**:
   - Add models to `packages/shared/database/models/`
   - Add API endpoint in `packages/webui/api/`
   - Create Alembic migration: `alembic revision --autogenerate`
   - Add tests in `tests/`

2. **Frontend Feature**:
   - Components in `apps/webui-react/src/components/`
   - API clients in `apps/webui-react/src/services/`
   - State management in `apps/webui-react/src/stores/`
   - Add tests alongside components

### Database Operations

```bash
# Apply migrations
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "Add user preferences"

# Rollback migration
alembic downgrade -1

# View migration history
alembic history
```

### Testing Strategies

1. **Unit Tests**: Test individual functions/classes in isolation
2. **Integration Tests**: Test service interactions with real databases
3. **E2E Tests**: Test full workflows including API calls
4. **Frontend Tests**: Component tests with React Testing Library

Always run `make check` before committing to ensure code quality.

## Common Issues & Solutions

1. **Model Download Failures**: Check internet connection and HuggingFace availability
2. **GPU Not Detected**: Ensure NVIDIA drivers and CUDA toolkit are installed
3. **Port Conflicts**: Default ports are 8000 (search), 8080 (webui), 6333 (qdrant)
4. **Migration Errors**: Ensure database is running before applying migrations
5. **Redis Connection**: Check Redis is running on port 6379

## Security Considerations

- JWT tokens expire after 30 minutes (configurable)
- Passwords hashed with bcrypt
- CORS configured for frontend development
- API key authentication for internal services
- Rate limiting on sensitive endpoints

## API Examples

### Authentication Flow

```bash
# 1. Register a new user
curl -X POST http://localhost:8080/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "password": "securepassword123"
  }'

# Response:
{
  "id": 1,
  "username": "john_doe"
}

# 2. Login to get tokens
curl -X POST http://localhost:8080/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "password": "securepassword123"
  }'

# Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}

# 3. Use the access token for authenticated requests
export TOKEN="eyJhbGciOiJIUzI1NiIs..."
curl -H "Authorization: Bearer $TOKEN" http://localhost:8080/api/auth/me
```

### Creating and Managing Collections

```bash
# Create a new embedding job
curl -X POST http://localhost:8080/api/jobs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "technical_docs",
    "file_paths": [
      "/documents/manual.pdf",
      "/documents/guide.docx"
    ],
    "chunking_config": {
      "chunk_size": 512,
      "chunk_overlap": 50
    }
  }'

# Response:
{
  "job_id": "job_123456",
  "status": "pending",
  "collection_name": "technical_docs",
  "total_files": 2
}

# Monitor job progress via WebSocket
wscat -c "ws://localhost:8080/ws/job_123456?token=$TOKEN"

# Check job status
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8080/api/jobs/job_123456

# List all collections
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8080/api/collections
```

### Performing Searches

```bash
# Semantic search
curl -X POST http://localhost:8080/api/search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to configure authentication?",
    "collection_name": "technical_docs",
    "limit": 5,
    "score_threshold": 0.7
  }'

# Hybrid search (combines vector and keyword search)
curl -X POST http://localhost:8080/api/hybrid_search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "authentication JWT tokens",
    "collection_name": "technical_docs",
    "limit": 10,
    "alpha": 0.5
  }'

# Batch search for multiple queries
curl -X POST http://localhost:8000/search/batch \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "authentication methods",
      "database configuration",
      "API rate limiting"
    ],
    "collection_name": "technical_docs",
    "limit": 3
  }'
```

### Document Preview

```bash
# Get document content with range support
curl -H "Authorization: Bearer $TOKEN" \
  -H "Range: bytes=0-1024" \
  http://localhost:8080/api/documents/job_123456/doc_789

# Get document metadata
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8080/api/documents/job_123456/doc_789/info

# Response:
{
  "filename": "manual.pdf",
  "content_type": "application/pdf",
  "size": 2048576,
  "chunks": 45,
  "metadata": {
    "title": "User Manual",
    "author": "Tech Team"
  }
}
```

### Model Management

```bash
# Get available models
curl http://localhost:8000/models

# Load a specific model
curl -X POST http://localhost:8000/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "BAAI/bge-large-en-v1.5",
    "device": "cuda"
  }'

# Get model recommendations based on GPU
curl http://localhost:8000/models/suggest

# Response:
{
  "recommended_model": "Qwen/Qwen3-Embedding-0.6B",
  "available_memory_gb": 8.5,
  "reasoning": "Fits comfortably in available GPU memory"
}
```

### Advanced Search Options

```bash
# Search with reranking
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "best practices for API security",
    "collection_name": "technical_docs",
    "search_type": "similarity",
    "limit": 20,
    "rerank": true,
    "rerank_model": "Qwen/Qwen3-Reranker-0.3B",
    "final_limit": 5
  }'

# Code-specific search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "async function implementation",
    "collection_name": "codebase",
    "search_type": "code",
    "limit": 10
  }'
```

## Configuration Reference

### Core Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENVIRONMENT` | string | `development` | Environment mode (`development` or `production`). In production, JWT_SECRET_KEY is required |
| `QDRANT_HOST` | string | `localhost` | Qdrant vector database host |
| `QDRANT_PORT` | int | `6333` | Qdrant vector database port |
| `DEFAULT_COLLECTION` | string | `work_docs` | Default collection name for vectors |
| `INTERNAL_API_KEY` | string | `None` | API key for internal service-to-service communication |

### Authentication & Security

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `JWT_SECRET_KEY` | string | auto-generated | **REQUIRED in production**. Secret key for JWT signing |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | int | `1440` | Access token expiration time (default: 24 hours) |
| `DISABLE_AUTH` | bool | `false` | Disable authentication (development only) |
| `CORS_ORIGINS` | string | `http://localhost:5173,http://127.0.0.1:5173` | Allowed CORS origins |

### Model Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DEFAULT_EMBEDDING_MODEL` | string | `Qwen/Qwen3-Embedding-0.6B` | Default embedding model from HuggingFace |
| `DEFAULT_QUANTIZATION` | string | `float16` | Model quantization (`float16`, `int8`, `float32`) |
| `MODEL_UNLOAD_AFTER_SECONDS` | int | `300` | Auto-unload models after inactivity (5 minutes) |
| `USE_MOCK_EMBEDDINGS` | bool | `false` | Use mock embeddings for testing |

### GPU Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | string | all GPUs | Comma-separated GPU indices to use |
| `MODEL_MAX_MEMORY_GB` | float | no limit | Maximum GPU memory per model |
| `MONITOR_GPU_MEMORY` | bool | `true` | Enable GPU memory monitoring |

### Service Ports

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WEBUI_PORT` | int | `8080` | WebUI API port |
| `SEARCH_API_PORT` | int | `8000` | Search API port |
| `WEBUI_METRICS_PORT` | int | `9092` | WebUI Prometheus metrics port |
| `METRICS_PORT` | int | `9091` | Search API Prometheus metrics port |
| `FLOWER_PORT` | int | `5555` | Flower monitoring UI port |

### Redis & Celery

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `REDIS_URL` | string | `redis://localhost:6379/0` | Redis connection URL |
| `CELERY_BROKER_URL` | string | same as REDIS_URL | Celery broker URL |
| `CELERY_RESULT_BACKEND` | string | same as REDIS_URL | Celery result backend |
| `CELERY_CONCURRENCY` | int | `1` | Number of Celery worker processes |

### Path Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `FILE_TRACKING_DB` | string | `/var/embeddings/file_tracking.json` | File tracking database path |
| `WEBUI_DB` | string | `/var/embeddings/webui.db` | SQLite database path |
| `EXTRACT_DIR` | string | `/opt/semantik/extract` | Document extraction directory |
| `INGEST_DIR` | string | `/var/embeddings/ingest` | Embedding ingestion directory |
| `LOADED_DIR` | string | `/var/embeddings/loaded` | Processed files directory |
| `REJECT_DIR` | string | `/var/embeddings/rejects` | Rejected files directory |

### HuggingFace Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HF_HOME` | string | `~/.cache/huggingface` | HuggingFace cache directory |
| `HF_HUB_OFFLINE` | bool | `false` | Enable offline mode (no model downloads) |

### Docker-Specific

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DOCUMENT_PATH` | string | `./documents` | Host path for document volume |
| `WEBUI_WORKERS` | int | `1` | Number of Gunicorn workers |
| `LOG_LEVEL` | string | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `FLOWER_BASIC_AUTH` | string | `admin:admin` | Flower UI basic auth credentials |

### Development Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WAIT_FOR_QDRANT` | bool | `true` | Wait for Qdrant to be ready on startup |
| `WAIT_FOR_SEARCH_API` | bool | `true` | Wait for Search API to be ready |
| `RATE_LIMIT_PER_MINUTE` | int | `60` | API rate limit per minute |

### Configuration Tips

1. **Minimal Production Setup**: Only `JWT_SECRET_KEY` and `ENVIRONMENT=production` are required
2. **GPU Memory**: Set `MODEL_MAX_MEMORY_GB` to prevent OOM errors on shared GPUs
3. **Performance**: Increase `CELERY_CONCURRENCY` and `WEBUI_WORKERS` for better throughput
4. **Security**: Always use strong `JWT_SECRET_KEY` and change `FLOWER_BASIC_AUTH` in production
5. **Offline Mode**: Set `HF_HUB_OFFLINE=true` after downloading models for air-gapped environments
