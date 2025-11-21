# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Semantik** is a self-hosted semantic search engine that transforms private file servers into AI-powered knowledge bases without data ever leaving your hardware. Built with privacy-first architecture for technically proficient users.

**Current Status**: Pre-release, undergoing active refactoring from "job-centric" to "collection-centric" architecture.

**Current Branch**: `embedding-viz` - Embedding visualization feature development (recently merged to main)
**Main Branch**: `main` - Stable production branch

## Architecture

### Tech Stack
- **Backend**: Python 3.11+, FastAPI, SQLAlchemy, Celery, Redis
- **Frontend**: React 19, TypeScript, Vite, Zustand, React Query, TailwindCSS
- **Databases**: PostgreSQL (metadata), Qdrant (vectors)
- **DevOps**: Docker, Docker Compose, Alembic (migrations), uv (dependency management)

### Service Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Docker Compose                       │
├─────────────┬─────────────┬──────────────┬──────────────────┤
│   webui     │  vecpipe    │   worker     │   Infrastructure │
│  (Port 8080)│ (Port 8000) │   (Celery)   │   Services       │
│             │             │              │                  │
│ • Auth/API  │ • Embeddings│ • Background │ • PostgreSQL     │
│ • WebSockets│ • Search    │   Tasks      │ • Redis          │
│ • Frontend  │ • Parsing   │ • Indexing   │ • Qdrant         │
└─────────────┴─────────────┴──────────────┴──────────────────┘
```

**webui**: FastAPI service handling authentication, collection management, WebSocket connections, and serving the React frontend.

**vecpipe**: Dedicated FastAPI service for compute-intensive operations: document parsing, embedding generation, and semantic search against Qdrant.

**worker**: Celery worker processing async background tasks (indexing, re-indexing, collection operations).

**shared**: Python library containing database models, repositories, core utilities, and domain logic shared across services.

### Service Startup Flow

All services use `docker-entrypoint.sh` which:
1. **Validates environment** - Runs `scripts/validate_env.py --strict` before startup
2. **Waits for dependencies** - Uses `wait_for_service()` to ensure dependent services are ready
3. **Runs migrations** (webui only) - Executes `alembic upgrade head` automatically
4. **Manages HF cache** - Cleans stale lock files, ensures writable directories
5. **Sets CUDA environment** - Configures paths for bitsandbytes and GPU support

**Critical**: Environment validation is STRICT. Missing or invalid configuration will prevent service startup.

## Development Commands

### Primary Workflow

```bash
# Install dependencies (uses uv with lock file)
uv sync --frozen

# Code quality checks (runs format + lint + type-check + test)
make check

# Run individual checks
make format      # Black + isort
make lint        # Ruff
make type-check  # Mypy
make test        # Pytest with coverage
```

### Docker Operations

```bash
# Interactive setup wizard (recommended for first-time setup)
make wizard

# Start all services
make docker-up

# Start backend services only (for local webui development)
make docker-dev-up

# View logs
make docker-logs
make docker-logs-webui
make docker-logs-vecpipe

# Stop services
make docker-down

# Access service shells
make docker-shell-webui
make docker-shell-vecpipe
```

### Database Migrations

```bash
# Apply migrations
uv run alembic upgrade head

# Create new migration
uv run alembic revision --autogenerate -m "Description"

# Rollback migration
uv run alembic downgrade -1

# PostgreSQL shell access
make docker-shell-postgres

# Database backup/restore
make docker-postgres-backup
make docker-postgres-restore BACKUP_FILE=path/to/backup.sql
```

### Testing

```bash
# Run all tests
make test

# Run tests excluding E2E
make test-ci

# Run only E2E tests (requires running services)
make test-e2e

# Run with coverage report
make test-coverage

# Run specific test file
uv run pytest tests/webui/api/v2/test_chunking_direct.py -v

# Run specific test
uv run pytest tests/webui/api/v2/test_chunking_direct.py::test_function_name -v
```

**Test Database**: Uses dedicated `postgres_test` container (port 55432) activated with `--profile testing` to isolate test data.

### Frontend Development

```bash
# Build frontend
make frontend-build

# Development server (with hot reload)
make frontend-dev

# Frontend tests
cd apps/webui-react
npm run test              # Run Vitest unit tests
npm run test:ui           # Interactive test UI (Vitest)
npm run test:coverage     # Coverage report
npm run test:e2e          # Playwright E2E tests (requires running services)
npm run test:e2e:ui       # Playwright UI mode for debugging

# Backend E2E tests (Playwright in Python)
make test-e2e            # Requires services running via make docker-up
```

**Test Organization:**
- Unit tests: `apps/webui-react/src/**/__tests__/*.test.ts(x)`
- E2E tests (frontend): `apps/webui-react/playwright/`
- E2E tests (backend): `tests/e2e/`
- Cypress tests: `cypress/e2e/` (legacy, being migrated to Playwright)

### Local Development Mode

```bash
# Start backend services in Docker + run webui locally with hot reload
make docker-dev-up
# Then in another terminal:
cd packages/webui && uv run uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

## Code Architecture Patterns

### Three-Layer Architecture (Critical)

**API Layer** (`packages/webui/api/`)
- FastAPI routers ONLY
- No business logic, no direct database calls
- Delegates everything to service layer
- Maps HTTP requests/responses to service calls

**Service Layer** (`packages/webui/services/`)
- ALL business logic lives here
- Orchestrates repository calls
- Manages transactions
- Handles cross-cutting concerns (caching, validation)

**Repository Layer** (`packages/shared/database/repositories/`)
- Database access ONLY
- Abstract SQLAlchemy details
- Provides clean interface for CRUD operations

### Anti-Pattern Example

❌ **BAD - Business logic in router:**
```python
@router.post("/collections")
async def create_collection(request: Request, db: AsyncSession = Depends(get_db)):
    new_collection = CollectionModel(**request.dict())
    db.add(new_collection)
    await db.commit()  # WRONG: Business logic in router
    return new_collection
```

✅ **GOOD - Delegated to service:**
```python
@router.post("/collections")
async def create_collection(
    request: Request,
    service: CollectionService = Depends(get_collection_service)
):
    collection = await service.create_collection(request.dict())
    return collection
```

### Repository Pattern (Critical for Database Access)

**Always use repositories for database access:**

```python
# Get repository from factory
from shared.database import create_collection_repository

collection_repo = create_collection_repository(db_session)

# Use repository methods
collection = await collection_repo.get_by_id(collection_id)
await collection_repo.update(collection_id, updates)
```

**Partition-Aware Queries (Chunks Table)**

The `chunks` table uses 100 LIST partitions based on `collection_id`. ALWAYS include `collection_id` in chunk queries for partition pruning:

```python
# ✅ GOOD - Efficient with partition pruning
chunks = await chunk_repo.get_by_collection_id(collection_id)

# ❌ BAD - Full table scan across all partitions
chunks = await session.execute(select(Chunk))  # Missing collection_id filter
```

### Celery Task Pattern (Critical)

**Transaction BEFORE task dispatch to avoid race conditions:**

```python
async def create_collection_and_index(collection_data: dict):
    # 1. Create operation record in database
    operation = await collection_service.create_operation(
        collection_id=collection_id,
        operation_type="INDEX"
    )

    # 2. Commit transaction FIRST
    await db.commit()

    # 3. THEN dispatch Celery task
    index_collection_task.delay(operation.uuid)

    # 4. Return immediately with operation ID
    return {"operation_id": operation.uuid}
```

❌ **WRONG - Task dispatched before commit causes race condition where worker can't find the operation record.**

### Frontend State Management

**Zustand stores** (`apps/webui-react/src/stores/`) handle all client-side state with optimistic updates:

```typescript
// Optimistic update pattern
updateCollection: async (id, updates) => {
  get().optimisticUpdateCollection(id, updates);  // Update UI immediately
  try {
    await collectionsV2Api.update(id, updates);
    await get().fetchCollectionById(id);  // Re-fetch canonical state
  } catch (error) {
    await get().fetchCollectionById(id);  // Revert on failure
    // Handle error...
  }
}
```

## Key Domain Concepts

### Collection States

Collections progress through states managed by `CollectionStatus` enum:

- `PENDING`: Created, waiting for indexing
- `READY`: Indexed and available for search
- `PROCESSING`: Operation in progress
- `ERROR`: Operation failed
- `DEGRADED`: Partially functional

### Operation Types

Background operations tracked via `OperationType` enum:

- `INDEX`: Initial collection indexing
- `APPEND`: Add new documents to collection
- `REINDEX`: Blue-green reindexing with zero downtime
- `DELETE`: Collection deletion
- `REMOVE_SOURCE`: Remove documents by source path

### Chunking Strategies

Modern chunking system (`packages/shared/chunking/`) with domain-driven design:

- `CHARACTER`: Simple character-based splitting
- `RECURSIVE`: Intelligent hierarchical splitting
- `MARKDOWN`: Markdown-aware preservation
- `SEMANTIC`: Meaning-based boundaries
- `HIERARCHICAL`: Document structure-aware
- `HYBRID`: Combined approach

Use `ChunkingOrchestrator` for all chunking operations. Legacy `ChunkingService` and `text_processing.chunking` modules are deprecated.

## WebSocket Architecture

**Redis Pub/Sub** for horizontal scaling with per-user connection limits:

- Maximum 10 connections per user
- Maximum 10,000 total connections
- Authentication via JWT in first message after connection
- Channels: `operation-progress:{operation_id}`, `collection-updates:{collection_id}`

## Embedding Visualization (Projection Runs)

**Feature**: Interactive 2D visualization of document embeddings using dimensionality reduction.

### Architecture
- **Backend**: Projection computation via UMAP/t-SNE/PCA in Celery tasks
- **Frontend**: React component using `embedding-atlas` library with WebGPU/WebGL fallback
- **Storage**: Projection artifacts (x, y, cat, ids arrays) stored in `data/projection_artifacts/`

### Key Components
- `packages/webui/services/projection_service.py` - Business logic for projection builds
- `packages/webui/tasks/projection.py` - Celery tasks for computing projections
- `apps/webui-react/src/components/EmbeddingVisualizationTab.tsx` - Visualization UI
- `apps/webui-react/src/hooks/useProjectionTooltip.ts` - Tooltip/selection logic

### Database Models
- `ProjectionRun` - Tracks projection metadata, status, and storage paths
- Linked to `Operation` for async processing and progress tracking

### API Endpoints
- `POST /api/v2/collections/{collection_id}/projections` - Start projection build
- `GET /api/v2/collections/{collection_id}/projections` - List projections
- `GET /api/v2/projections/{projection_id}` - Get projection metadata
- `GET /api/v2/projections/{projection_id}/arrays/{artifact}` - Stream projection data
- `POST /api/v2/projections/{projection_id}/select` - Resolve tooltip/selection data

### Critical Patterns
- Projection runs are **immutable** - recompute creates new runs
- Sampling applied during compute to limit point count (default: 10,000)
- WebGPU forced via `embeddingAtlasWebgpuPatch.ts` for compatibility
- Tooltips use LRU cache (`utils/lruCache.ts`) to minimize API calls
- Cluster labels computed from filenames for semantic grouping

**Documentation**: See `docs/EMBEDDING_VISUALIZATION.md` for complete technical details.

## Security Guidelines

1. **Authentication**: JWT with 24h access tokens, 30d refresh tokens
2. **Authorization**: Owner-based collection access control
3. **Input Validation**: Pydantic models for all API inputs
4. **Secrets**: Never commit `JWT_SECRET_KEY` or database passwords
5. **SQL Injection**: Always use SQLAlchemy parameterized queries
6. **Rate Limiting**: Configured per-endpoint via `RateLimitConfig`
7. **Path Traversal**: All file paths validated through `packages.shared.utils.security.validate_safe_path()`
8. **Environment Validation**: `scripts/validate_env.py` enforces required config at startup

### Security Testing

All endpoints must pass security tests in `tests/security/`:
- Path traversal prevention (`test_path_traversal.py`)
- Access control enforcement (`test_access_denied_handlers.py`)
- Input sanitization

## Health Monitoring

**Health Check Endpoints:**
- `/api/health/livez` - Liveness probe (service running)
- `/api/health/readyz` - Readiness probe (service ready to accept traffic)
- `/api/health/startupz` - Startup probe (initialization complete)

**Monitoring Integration:**
- Prometheus metrics exposed via `prometheus-client`
- Resource usage tracked via `psutil`
- GPU metrics via `gputil`

## Testing Requirements

- All new backend services/endpoints require integration tests in `tests/webui/`
- All new frontend components require unit tests in `apps/webui-react/src/components/__tests__/`
- Use `TestClient` for API tests, mock Redis/Celery
- E2E tests marked with `@pytest.mark.e2e` and require running services
- Security tests for all user-controlled inputs
- Run `make check` before committing

## Critical Refactoring Context

**Ongoing Migration**: "job-centric" → "collection-centric" terminology

- ❌ OLD: Jobs, job_id, job tables
- ✅ NEW: Operations, operation_id, operations tables

**When adding features:**
- Use "operation" terminology exclusively
- Never introduce new "job" references
- Update any legacy "job" code you encounter

## Common Pitfalls

1. **Async/Sync Mixing**: Never call blocking I/O in async functions
2. **Missing Commit**: Always commit transaction before Celery dispatch
3. **Collection State**: Check collection status before operations
4. **Partition Pruning**: Always include `collection_id` in chunk queries
5. **Business Logic Location**: Keep it in services, not routers
6. **Frontend Cache Invalidation**: Re-fetch after mutations
7. **Environment Variables**: Use `.env` files, never hardcode
8. **Missing Environment Validation**: Test new required vars in `scripts/validate_env.py`

## Debugging Multi-Service Issues

**Viewing logs across services:**
```bash
# All services
make docker-logs

# Specific service
docker compose logs -f webui
docker compose logs -f vecpipe
docker compose logs -f worker

# Follow specific container
docker logs -f semantik-webui
```

**Debugging database issues:**
```bash
# Access PostgreSQL shell
make docker-shell-postgres

# View active queries
SELECT pid, query, state FROM pg_stat_activity WHERE state != 'idle';

# Check partition info
SELECT tablename FROM pg_tables WHERE schemaname = 'public';
```

**Debugging Celery tasks:**
```bash
# View worker logs
docker compose logs -f worker

# Inspect active tasks
celery -A webui.celery_app inspect active

# Start Flower monitoring UI
docker compose --profile backend up flower
# Access at http://localhost:5555
```

**Debugging WebSocket connections:**
```bash
# Monitor Redis pub/sub
docker compose exec redis redis-cli
> SUBSCRIBE operation-progress:*
> SUBSCRIBE collection-updates:*
```

## File Structure Reference

```
semantik/
├── packages/
│   ├── shared/           # Cross-service models, repos, utilities
│   │   ├── database/     # SQLAlchemy models, repositories
│   │   ├── chunking/     # Domain-driven chunking (NEW)
│   │   ├── config/       # Pydantic settings
│   │   └── managers/     # Qdrant management
│   ├── webui/            # Main API service
│   │   ├── api/v2/       # API routers (thin controllers)
│   │   ├── services/     # Business logic
│   │   ├── middleware/   # HTTP middleware
│   │   └── websocket/    # WebSocket management
│   └── vecpipe/          # Embedding & search service
├── apps/
│   └── webui-react/      # React frontend
│       ├── src/
│       │   ├── stores/   # Zustand state management
│       │   ├── components/
│       │   └── api/      # API client
├── tests/                # Test suite
│   ├── webui/           # WebUI integration tests
│   ├── shared/          # Shared library tests
│   ├── security/        # Security vulnerability tests
│   ├── e2e/             # End-to-end tests
│   └── conftest.py      # Pytest configuration
├── scripts/             # Utility scripts
│   ├── validate_env.py  # Environment validation (used at startup)
│   └── install_uv.sh    # uv installation for Docker
├── alembic/             # Database migrations
├── docker-entrypoint.sh # Service startup orchestration
├── Makefile             # Development commands
└── docker-compose.yml   # Service orchestration
```

## Additional Resources

- [API Reference](docs/API_REFERENCE.md) - Complete REST and WebSocket API documentation
- [Architecture Guide](docs/ARCH.md) - Detailed system design
- [Testing Guide](docs/TESTING.md) - Testing patterns and practices
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

## Migration Note

Semantik is in pre-release. Breaking changes may occur between versions as we optimize the foundation. Always review the changelog when upgrading.
