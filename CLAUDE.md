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
poetry run pytest tests/unit                 # Unit tests only
poetry run pytest tests/integration          # Integration tests
poetry run pytest tests/e2e                  # E2E tests (requires services running)
poetry run pytest -m "not e2e"              # All tests except E2E
```

### Building & Installation

```bash
# Backend setup
make install                      # Install production dependencies
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
   - Document extraction and chunking
   - Embedding generation using transformer models
   - Vector storage in Qdrant database
   - Search API with semantic/hybrid search

2. **Web Application** (`packages/webui/`)
   - FastAPI backend with JWT authentication
   - Job queue system using Celery + Redis
   - Collection management and search API proxy
   - User management and API key generation

3. **Frontend** (`apps/webui-react/`)
   - React 19 with TypeScript
   - TailwindCSS for styling
   - Zustand for state management
   - React Query for data fetching

4. **Shared Components** (`packages/shared/`)
   - Database models (SQLAlchemy)
   - Embedding service manager
   - Model management utilities
   - Common configuration

### Service Architecture

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

### Key API Endpoints

**Search API (port 8000)**
- `GET/POST /search` - Semantic search
- `GET /hybrid_search` - Combined vector/keyword search
- `POST /search/batch` - Batch search operations

**WebUI API (port 8080)**
- `/api/auth/*` - Authentication (login, register, refresh)
- `/api/collections/*` - Collection CRUD operations
- `/api/jobs/*` - Job queue management
- `/api/search/*` - Search proxy to Search API
- `/api/admin/*` - Admin operations (requires superuser)

### Model & Embedding System

- Default model: `Qwen/Qwen3-Embedding-0.6B` (small, efficient)
- Auto-downloads models from HuggingFace on first use
- Supports GPU acceleration (CUDA) with automatic detection
- Models stored in `/models` volume (persisted)
- Automatic model unloading after 5 minutes of inactivity

## Key Configuration

### Environment Variables

Create `.env` file:

```bash
# Core settings
QDRANT_HOST=localhost
QDRANT_PORT=6333
DEFAULT_COLLECTION=work_docs

# Model configuration
DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
DEFAULT_QUANTIZATION=float16
MODEL_UNLOAD_AFTER_SECONDS=300

# Authentication (generate with: openssl rand -hex 32)
JWT_SECRET_KEY=your-secret-key-here
DISABLE_AUTH=false  # Set true for development

# Redis configuration
REDIS_URL=redis://localhost:6379/0
```

### Docker Compose Profiles

```bash
# Standard deployment (auto-detects GPU/CPU)
docker compose up -d

# Production deployment
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Force CUDA GPU support
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d

# CPU-only deployment
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

## Development Workflows

### Adding New Features

1. **Backend Feature**:
   - Add models to `packages/shared/database/models/`
   - Create service in `packages/webui/services/`
   - Add API endpoint in `packages/webui/api/`
   - Create Alembic migration: `alembic revision --autogenerate`
   - Add tests in `tests/`

2. **Frontend Feature**:
   - Components in `apps/webui-react/src/components/`
   - API clients in `apps/webui-react/src/lib/api/`
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
- API key authentication for programmatic access
- Superuser required for admin operations

##TOOLS/MCP
- context7 for looking up documentation
- puppeteer for viewing webui in browser
