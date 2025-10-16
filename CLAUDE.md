# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Semantik** is a self-hosted semantic search engine that transforms private file servers into AI-powered knowledge bases without data ever leaving your hardware. Built with privacy-first architecture for technically proficient users.

**Current Status**: Pre-release, undergoing active refactoring from "job-centric" to "collection-centric" architecture.

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

## Development Commands

### Primary Workflow

```bash
# Install dependencies
uv sync --frozen

# Code quality checks (format, lint, test)
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

### Frontend Development

```bash
# Build frontend
make frontend-build

# Development server (with hot reload)
make frontend-dev

# Frontend tests
cd apps/webui-react
npm run test           # Run tests
npm run test:ui        # Interactive test UI
npm run test:coverage  # Coverage report
npm run test:e2e       # Playwright E2E tests
```

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

Use `ChunkingService` for all chunking operations. Legacy `text_processing.chunking` module is deprecated.

## WebSocket Architecture

**Redis Pub/Sub** for horizontal scaling with per-user connection limits:

- Maximum 10 connections per user
- Maximum 10,000 total connections
- Authentication via JWT in first message after connection
- Channels: `operation-progress:{operation_id}`, `collection-updates:{collection_id}`

## Security Guidelines

1. **Authentication**: JWT with 24h access tokens, 30d refresh tokens
2. **Authorization**: Owner-based collection access control
3. **Input Validation**: Pydantic models for all API inputs
4. **Secrets**: Never commit `JWT_SECRET_KEY` or database passwords
5. **SQL Injection**: Always use SQLAlchemy parameterized queries
6. **Rate Limiting**: Configured per-endpoint via `RateLimitConfig`

## Testing Requirements

- All new backend services/endpoints require integration tests in `tests/webui/`
- All new frontend components require unit tests in `apps/webui-react/src/components/__tests__/`
- Use `TestClient` for API tests, mock Redis/Celery
- E2E tests marked with `@pytest.mark.e2e` and require running services
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
│   ├── e2e/             # End-to-end tests
│   └── conftest.py      # Pytest configuration
├── alembic/             # Database migrations
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
