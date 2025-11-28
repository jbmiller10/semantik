# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Semantik** is a self-hosted semantic search engine that transforms private file servers into AI-powered knowledge bases without data ever leaving your hardware. Built with privacy-first architecture for technically proficient users.

**Status**: Pre-release, actively developed. APIs and defaults may evolve.

## Coding Style & Naming Conventions

**Python**:
- 4-space indentation, 120-character lines, exhaustive typing
- Modules: `snake_case` (e.g., `search_api.py`)
- Pytest fixtures: `snake_case`
- Run `make format` and `make lint` before marking work complete

**Frontend**:
- React components: `PascalCase` (e.g., `SearchResults.tsx`)
- Files in `apps/webui-react/src/`

**Config Files**:
- JSON/YAML: `kebab-case`

## Commit Guidelines

Use Conventional Commit formatting:
- `feat(search): add reranker fallback`
- `fix(webui): guard empty filter`
- Reference PR IDs: `(#212)`
- Squash local WIP commits before merge

## Architecture

### Tech Stack
- **Backend**: Python 3.11+, FastAPI, SQLAlchemy, Celery, Redis
- **Frontend**: React 19, TypeScript, Vite, Zustand, React Query, TailwindCSS
- **Databases**: PostgreSQL (metadata), Qdrant (vectors)
- **DevOps**: Docker Compose, Alembic (migrations), uv (dependency management)

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
make format      # Black + isort (alias: make fix)
make lint        # Ruff
make type-check  # Mypy
make test        # Pytest
```

### Running Locally

```bash
# Run webui API locally with hot reload
make run

# Full development environment (all services)
make dev

# Backend services in Docker + webui locally with hot reload
make docker-dev-up
# Then in another terminal:
make run
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
make docker-down          # Keeps volumes
make docker-down-clean    # Removes volumes (data loss)

# Generate JWT secret (one-time setup, or use make wizard)
uv run python scripts/generate_jwt_secret.py --write

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

**Test Database**: Uses dedicated `postgres_test` container (port 55432) activated with `--profile testing` to isolate test data. Configure via `.env.test` with `POSTGRES_TEST_PORT`, `POSTGRES_TEST_DB`, `POSTGRES_TEST_USER`, `POSTGRES_TEST_PASSWORD`.

```bash
# Start dedicated test database
docker compose --profile testing up -d postgres_test
```

### Frontend Development

```bash
# Install dependencies
make frontend-install

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
```

**Test Organization:**
- Unit tests: `apps/webui-react/src/**/__tests__/*.test.ts(x)`
- E2E tests (Playwright): `apps/webui-react/playwright/`
- E2E tests (backend): `tests/e2e/`
- Cypress tests: `cypress/e2e/` (projection visualization flows)

**Cypress E2E** (for projection visualization):
```bash
# Install Cypress (repo root)
npm install -D cypress

# Interactive mode
npx cypress open

# Headless
npx cypress run --spec cypress/e2e/projection_visualize.cy.ts
```

## Code Architecture Patterns

### Three-Layer Architecture (Critical)

**API Layer** (`packages/webui/api/v2/`)
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

**Zustand stores** (`apps/webui-react/src/stores/`) handle all client-side state:
- `authStore.ts` - Authentication state
- `collectionStore.ts` - Collection management
- `searchStore.ts` - Search state
- `chunkingStore.ts` - Chunking configuration
- `uiStore.ts` - UI state

**Optimistic update pattern:**
```typescript
updateCollection: async (id, updates) => {
  get().optimisticUpdateCollection(id, updates);  // Update UI immediately
  try {
    await collectionsV2Api.update(id, updates);
    await get().fetchCollectionById(id);  // Re-fetch canonical state
  } catch (error) {
    await get().fetchCollectionById(id);  // Revert on failure
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
- `REMOVE_SOURCE`: Remove documents by source path

### Chunking Architecture

Two-level chunking system:

**High-Level (Service Layer)**: `packages/webui/services/chunking/`
- `ChunkingOrchestrator` - Coordinates all chunking operations
- Handles caching, metrics, validation, and operation management
- Use this for application-level chunking needs

**Low-Level (Shared Library)**: `packages/shared/chunking/`
- Domain-driven design with `unified/` factory pattern
- `UnifiedChunkingFactory.create_strategy()` - Creates strategy instances
- Strategies: `CHARACTER`, `RECURSIVE`, `MARKDOWN`, `SEMANTIC`, `HIERARCHICAL`, `HYBRID`

```python
# High-level usage (recommended for services)
from webui.services.chunking import ChunkingOrchestrator
orchestrator = ChunkingOrchestrator(...)
result = await orchestrator.process_document(...)

# Low-level usage (for direct strategy access)
from shared.chunking.unified import UnifiedChunkingFactory
strategy = UnifiedChunkingFactory.create_strategy("recursive", config)
chunks = strategy.chunk(text)
```

**Legacy**: `packages/shared/text_processing/chunking` is deprecated.

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

### API Endpoints
- `POST /api/v2/collections/{collection_id}/projections` - Start projection build
- `GET /api/v2/collections/{collection_id}/projections` - List projections
- `GET /api/v2/projections/{projection_id}` - Get projection metadata
- `GET /api/v2/projections/{projection_id}/arrays/{artifact}` - Stream projection data
- `POST /api/v2/projections/{projection_id}/select` - Resolve tooltip/selection data
- `DELETE /api/v2/projections/{projection_id}` - Delete projection

### Critical Patterns
- Projection runs are **immutable** - recompute creates new runs
- Sampling applied during compute to limit point count (default: 10,000)
- WebGPU forced via `embeddingAtlasWebgpuPatch.ts` for compatibility
- Tooltips use LRU cache (`utils/lruCache.ts`) to minimize API calls

## Security Guidelines

1. **Authentication**: JWT with 24h access tokens, 30d refresh tokens
2. **Authorization**: Owner-based collection access control
3. **Input Validation**: Pydantic models for all API inputs
4. **Secrets**: Never commit `JWT_SECRET_KEY` or database passwords
5. **SQL Injection**: Always use SQLAlchemy parameterized queries
6. **Rate Limiting**: Configured per-endpoint via `RateLimitConfig`
7. **Path Traversal**: All file paths validated through `packages.shared.utils.security.validate_safe_path()`
8. **Environment Validation**: `scripts/validate_env.py` enforces required config at startup
9. **Data Maintenance**: Use `packages/vecpipe/maintenance.py` to rotate embeddings or clear indexes when working with sensitive data

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

## Common Pitfalls

1. **Async/Sync Mixing**: Never call blocking I/O in async functions
2. **Missing Commit**: Always commit transaction before Celery dispatch
3. **Collection State**: Check collection status before operations
4. **Partition Pruning**: Always include `collection_id` in chunk queries
5. **Business Logic Location**: Keep it in services, not routers
6. **Frontend Cache Invalidation**: Re-fetch after mutations
7. **Environment Variables**: Use `.env` files, never hardcode
8. **Missing Environment Validation**: Test new required vars in `scripts/validate_env.py`
9. **Partition Errors After DB Reset**: After resetting Postgres volume, run `uv run alembic upgrade head` to install helper functions like `get_partition_key`. All chunk inserts in tests should use `ChunkRepository` or `PartitionAwareMixin.bulk_insert_partitioned`

## Debugging Multi-Service Issues

**Viewing logs across services:**
```bash
# All services
make docker-logs

# Specific service
docker compose logs -f webui
docker compose logs -f vecpipe
docker compose logs -f worker
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
│   │   ├── chunking/     # Domain-driven chunking
│   │   │   └── unified/  # Factory pattern for strategies
│   │   ├── config/       # Pydantic settings
│   │   └── managers/     # Qdrant management
│   ├── webui/            # Main API service
│   │   ├── api/v2/       # API routers (thin controllers)
│   │   ├── services/     # Business logic
│   │   │   └── chunking/ # ChunkingOrchestrator and sub-services
│   │   ├── middleware/   # HTTP middleware
│   │   ├── tasks/        # Celery tasks
│   │   └── websocket/    # WebSocket management
│   └── vecpipe/          # Embedding & search service
├── apps/
│   └── webui-react/      # React frontend
│       └── src/
│           ├── stores/   # Zustand state management
│           ├── components/
│           ├── hooks/
│           ├── api/      # API client
│           └── utils/    # Utilities (lruCache, etc.)
├── tests/                # Test suite
│   ├── webui/           # WebUI integration tests
│   ├── shared/          # Shared library tests
│   ├── security/        # Security vulnerability tests
│   ├── e2e/             # End-to-end tests
│   ├── domain/          # Domain logic tests
│   ├── fixtures/        # Test fixtures
│   └── conftest.py      # Pytest configuration
├── cypress/             # Cypress E2E tests
│   └── e2e/             # Test specs (projection visualization)
├── scripts/             # Utility scripts
│   ├── validate_env.py  # Environment validation (used at startup)
│   ├── generate_jwt_secret.py
│   └── partition_maintenance.py
├── alembic/             # Database migrations
├── data/                # Sample documents and projection artifacts
├── docker-entrypoint.sh # Service startup orchestration
├── Makefile             # Development commands
└── docker-compose.yml   # Service orchestration
```

## Additional Resources

- [Documentation Index](docs/DOCUMENTATION_INDEX.md) - Complete documentation with reading paths
- [API Reference](docs/API_REFERENCE.md) - REST and WebSocket API documentation
- [Architecture Guide](docs/ARCH.md) - Detailed system design
- [Testing Guide](docs/TESTING.md) - Testing patterns and practices
- [Embedding Visualization](docs/EMBEDDING_VISUALIZATION.md) - Projection feature details
