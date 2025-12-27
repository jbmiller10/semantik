# Semantik Architecture Reference

> **Complete Architectural Documentation for Software Architects**
>
> This document enables you to design new functionality for Semantik at a high level without direct access to the codebase. It captures implementation details, patterns, conventions, extension points, and integration requirements.

**Generated:** December 2024 | **Version:** v0.8 (GraphRAG) | **Total Codebase:** ~1.7M tokens

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Component Reference](#component-reference)
4. [Data Model Reference](#data-model-reference)
5. [API Reference](#api-reference)
6. [Patterns Catalog](#patterns-catalog)
7. [Integration Guide](#integration-guide)
8. [Extension Cookbook](#extension-cookbook)
9. [Testing Guide](#testing-guide)
10. [Deployment Reference](#deployment-reference)

---

## Executive Summary

### What is Semantik?

Semantik is a **self-hosted semantic search engine** that enables organizations to index, search, and retrieve documents using AI-powered vector embeddings. It supports multiple data sources (local directories, Git repositories, IMAP mailboxes), multiple chunking strategies, and provides real-time operation tracking via WebSockets.

### Core Capabilities

| Capability | Description |
|------------|-------------|
| **Semantic Search** | Vector similarity search using Qwen3 embedding models |
| **Hybrid Search** | Combined keyword + semantic search with configurable fusion |
| **Multi-Collection** | Organize documents into isolated searchable collections |
| **Continuous Sync** | Automatic re-indexing on schedule (15min+ intervals) |
| **GraphRAG** | Graph-enhanced retrieval for relationship-aware search |
| **Real-time Updates** | WebSocket-based progress tracking for all operations |
| **Reranking** | Cross-encoder reranking for improved result quality |

### Technology Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND                                  │
│  React 19 + TypeScript + Zustand + TanStack Query + TailwindCSS │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      WEBUI SERVICE                               │
│     FastAPI + SQLAlchemy + Celery + Redis + WebSockets          │
└─────────────────────────────────────────────────────────────────┘
                                 │
              ┌──────────────────┴───────────────────┐
              ▼                                      ▼
┌──────────────────────────┐           ┌──────────────────────────┐
│      VECPIPE SERVICE     │           │     CELERY WORKERS       │
│   Embeddings + Search    │           │   Background Processing  │
│   Reranking + GraphRAG   │           │   Document Indexing      │
└──────────────────────────┘           └──────────────────────────┘
              │                                      │
              ▼                                      ▼
┌──────────────────────────┐           ┌──────────────────────────┐
│         QDRANT           │           │       POSTGRESQL         │
│    Vector Database       │           │    Metadata + State      │
└──────────────────────────┘           └──────────────────────────┘
```

### Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **API Layer** | FastAPI | Async support, OpenAPI docs, Pydantic validation |
| **Vector DB** | Qdrant | Performance, filtering, hybrid search support |
| **Task Queue** | Celery + Redis | Reliable async processing, visibility |
| **State Management** | Zustand + TanStack Query | Simple client state, smart server state caching |
| **Embedding Models** | Qwen3-Embedding-0.6B | Balanced quality/speed, quantization support |
| **Partitioning** | 100 PostgreSQL partitions | Chunks table scales to billions of rows |

---

## System Overview

### Package Structure

```
semantik/
├── packages/
│   ├── shared/              # Core models, repositories, utilities
│   │   ├── database/        # SQLAlchemy models, repositories
│   │   ├── chunking/        # Domain-driven chunking system
│   │   ├── embedding/       # Plugin-based embedding providers
│   │   ├── connectors/      # Data source connector system
│   │   └── config/          # Pydantic settings
│   │
│   ├── webui/               # FastAPI web application
│   │   ├── api/v2/          # REST API endpoints
│   │   ├── services/        # Business logic layer
│   │   ├── tasks/           # Celery task definitions
│   │   └── websocket/       # Real-time updates
│   │
│   └── vecpipe/             # Vector search service
│       ├── search_api.py    # Search endpoints
│       ├── embed_chunks_unified.py  # Embedding generation
│       ├── reranker.py      # Cross-encoder reranking
│       └── graphrag/        # Graph-enhanced retrieval
│
└── apps/
    └── webui-react/         # React frontend
        └── src/
            ├── components/  # UI components
            ├── stores/      # Zustand state
            ├── hooks/       # Custom React hooks
            └── services/    # API client layer
```

### Service Communication

```
┌─────────────┐     HTTP/REST      ┌─────────────┐
│   Browser   │ ◄───────────────► │   WebUI     │
│   (React)   │     WebSocket      │  (FastAPI)  │
└─────────────┘                    └──────┬──────┘
                                          │
              ┌───────────────────────────┴───────────────────────────┐
              │                           │                           │
              ▼                           ▼                           ▼
      ┌───────────────┐           ┌───────────────┐           ┌───────────────┐
      │    VecPipe    │           │    Celery     │           │  PostgreSQL   │
      │  (Search API) │           │   (Workers)   │           │   (Metadata)  │
      └───────┬───────┘           └───────┬───────┘           └───────────────┘
              │                           │
              ▼                           ▼
      ┌───────────────┐           ┌───────────────┐
      │    Qdrant     │           │     Redis     │
      │   (Vectors)   │           │    (Queue)    │
      └───────────────┘           └───────────────┘
```

### Request Lifecycle

**Example: Search Request**

```
1. User submits search query in React UI
2. SearchInterface calls searchV2Api.search()
3. WebUI /api/v2/search validates request, checks permissions
4. Service layer resolves collection UUIDs to Qdrant collection names
5. WebUI calls VecPipe /search endpoint with resolved names
6. VecPipe embeds query using same model as indexing
7. VecPipe searches Qdrant for top-k similar vectors
8. Optional: VecPipe reranks results with cross-encoder
9. Results returned through chain to React UI
10. SearchResults component renders hierarchical results
```

---

## Component Reference

Detailed documentation for each component is available in the `/docs/architecture/` directory:

| Component | File | Description |
|-----------|------|-------------|
| Database Layer | [database-layer.md](./architecture/database-layer.md) | SQLAlchemy models, repositories, partitioning |
| Chunking System | [chunking-system.md](./architecture/chunking-system.md) | Domain-driven chunking strategies |
| Embedding System | [embedding-system.md](./architecture/embedding-system.md) | Plugin-based embedding providers |
| VecPipe Service | [vecpipe-service.md](./architecture/vecpipe-service.md) | Search, embedding, reranking |
| WebUI API | [webui-api.md](./architecture/webui-api.md) | REST endpoints, authentication |
| WebUI Services | [webui-services.md](./architecture/webui-services.md) | Business logic, orchestration |
| WebUI Realtime | [webui-realtime.md](./architecture/webui-realtime.md) | Celery tasks, WebSockets |
| Connectors | [connectors.md](./architecture/connectors.md) | Data source connectors |
| Frontend Components | [frontend-components.md](./architecture/frontend-components.md) | React component architecture |
| Frontend State | [frontend-state.md](./architecture/frontend-state.md) | Zustand, React Query, API client |
| Infrastructure | [infrastructure.md](./architecture/infrastructure.md) | Docker, deployment, CI/CD |
| Testing | [testing.md](./architecture/testing.md) | Testing patterns and fixtures |

**Extension Cookbooks:**

| Cookbook | File | Description |
|----------|------|-------------|
| Search Enhancements | [cookbook-search-enhancements.md](./architecture/cookbook-search-enhancements.md) | Step-by-step guides for search features |
| UI Features | [cookbook-ui-features.md](./architecture/cookbook-ui-features.md) | Step-by-step guides for UI development |

---

## Data Model Reference

### Core Entities

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│      User       │────<│   Collection    │────<│    Document     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ id: UUID        │     │ id: UUID        │     │ id: UUID        │
│ username: str   │     │ name: str       │     │ file_name: str  │
│ email: str      │     │ owner_id: FK    │     │ file_path: str  │
│ password_hash   │     │ status: enum    │     │ content_hash    │
│ is_active: bool │     │ embedding_model │     │ chunk_count     │
│ created_at      │     │ vector_store_   │     │ collection_id   │
└─────────────────┘     │   name: str     │     │ status: enum    │
                        │ sync_mode       │     └────────┬────────┘
                        │ sync_interval   │              │
                        └────────┬────────┘              │
                                 │                       ▼
                                 │              ┌─────────────────┐
                                 │              │     Chunk       │
                                 │              ├─────────────────┤
                                 │              │ id: UUID        │
                                 │              │ document_id: FK │
                                 │              │ collection_id   │
                                 │              │ content: text   │
                                 │              │ chunk_index     │
                                 │              │ partition_key   │
                                 │              └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   Operation     │
                        ├─────────────────┤
                        │ uuid: UUID      │
                        │ type: enum      │
                        │ status: enum    │
                        │ collection_id   │
                        │ user_id: FK     │
                        │ progress: int   │
                        │ created_at      │
                        └─────────────────┘
```

### Status Enums

**CollectionStatus:**
- `PENDING` → Collection created, awaiting first index
- `PROCESSING` → Active operation in progress
- `READY` → Fully indexed and searchable
- `ERROR` → Last operation failed
- `DEGRADED` → Partial failure, still searchable

**OperationType:**
- `INDEX` → Initial collection indexing
- `APPEND` → Add new data source
- `REINDEX` → Full re-indexing
- `DELETE` → Collection deletion
- `REMOVE_SOURCE` → Remove specific source

**OperationStatus:**
- `PENDING` → Queued, not started
- `PROCESSING` → Currently executing
- `COMPLETED` → Successfully finished
- `FAILED` → Execution failed
- `CANCELLED` → User cancelled

### Partitioning Strategy

The `chunks` table uses **100 LIST partitions** for horizontal scaling:

```sql
-- Partition key calculation
partition_key = abs(hashtext(collection_id::text)) % 100

-- Partitions: chunks_part_0 through chunks_part_99
-- CRITICAL: Always include collection_id in WHERE clauses for partition pruning
```

---

## API Reference

### REST API (v2)

**Base URL:** `/api/v2`

#### Collections

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/collections` | List user's collections |
| POST | `/collections` | Create new collection |
| GET | `/collections/{uuid}` | Get collection details |
| PUT | `/collections/{uuid}` | Update collection |
| DELETE | `/collections/{uuid}` | Delete collection |
| POST | `/collections/{uuid}/sources` | Add data source |
| DELETE | `/collections/{uuid}/sources` | Remove data source |
| POST | `/collections/{uuid}/reindex` | Trigger reindex |
| GET | `/collections/{uuid}/documents` | List documents |
| GET | `/collections/{uuid}/operations` | List operations |

#### Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/search` | Multi-collection search |

**Search Request:**
```json
{
  "query": "authentication flow",
  "collection_uuids": ["uuid1", "uuid2"],
  "top_k": 10,
  "score_threshold": 0.5,
  "search_type": "hybrid",
  "use_reranker": true,
  "rerank_model": "Qwen/Qwen3-Reranker-0.6B"
}
```

#### Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/operations` | List all operations |
| GET | `/operations/{uuid}` | Get operation status |
| POST | `/operations/{uuid}/cancel` | Cancel operation |

#### Connectors

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/connectors` | Get connector catalog |
| POST | `/connectors/git/preview` | Test Git connection |
| POST | `/connectors/imap/preview` | Test IMAP connection |

### WebSocket API

**Connection:** `ws://host/api/ws`

**Authentication:** Send JWT token after connection:
```json
{"type": "AUTH", "token": "Bearer ..."}
```

**Message Types:**
- `OPERATION_PROGRESS` - Operation status updates
- `OPERATION_COMPLETED` - Operation finished
- `OPERATION_FAILED` - Operation failed
- `COLLECTION_UPDATED` - Collection state changed

---

## Patterns Catalog

### Three-Layer Architecture

All WebUI services follow this pattern:

```
┌─────────────────────────────────────────────────────┐
│ API Layer (routers/)                                 │
│ - Request validation (Pydantic)                     │
│ - Authentication/authorization                       │
│ - Response formatting                                │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ Service Layer (services/)                            │
│ - Business logic orchestration                       │
│ - Cross-cutting concerns                             │
│ - External service coordination                      │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ Repository Layer (database/repositories/)            │
│ - Data access abstraction                            │
│ - Query construction                                 │
│ - Transaction management                             │
└─────────────────────────────────────────────────────┘
```

### Repository Pattern

All database access through repository classes:

```python
class CollectionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, collection_id: str) -> Collection | None:
        result = await self.session.execute(
            select(Collection).where(Collection.id == collection_id)
        )
        return result.scalar_one_or_none()

    async def list_for_user(self, user_id: int, ...) -> list[Collection]:
        # Always include owner_id in queries for security
        ...
```

### Plugin/Factory Pattern (Embeddings)

```python
# Provider registration
@embedding_provider("dense-local")
class DenseLocalProvider(BaseEmbeddingPlugin):
    async def embed_texts(self, texts: list[str], mode: EmbeddingMode) -> list[list[float]]:
        ...

# Provider usage
provider = EmbeddingProviderFactory.create_provider("Qwen/Qwen3-Embedding-0.6B")
embeddings = await provider.embed_texts(texts, mode=EmbeddingMode.QUERY)
```

### State Management Pattern (Frontend)

```typescript
// Zustand for client state (auth, UI preferences)
const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      token: null,
      user: null,
      setAuth: (token, user) => set({ token, user }),
      logout: () => set({ token: null, user: null }),
    }),
    { name: 'auth-storage' }
  )
);

// React Query for server state (collections, search results)
function useCollections() {
  return useQuery({
    queryKey: ['collections'],
    queryFn: () => collectionsApi.list(),
    staleTime: 5000,
    refetchInterval: hasActiveOperations ? 30000 : false,
  });
}
```

### Blue-Green Reindexing

Safe reindexing without downtime:

```
1. Create new_collection_uuid_temp in Qdrant
2. Index all documents into temp collection
3. Atomically swap: old → deleted, temp → active
4. Delete old collection
5. Update collection.vector_store_name to new UUID
```

---

## Integration Guide

### Adding a New API Endpoint

1. **Define Pydantic models** (`schemas/`)
2. **Create router** (`api/v2/my_router.py`)
3. **Implement service** (`services/my_service.py`)
4. **Add repository methods** if needed
5. **Register router** in `api/v2/__init__.py`
6. **Add tests** (`tests/webui/api/v2/test_my_router.py`)

### Adding a New Celery Task

1. **Define task** in `tasks/`:
```python
@celery_app.task(bind=True, name="my_task")
def my_task(self, operation_uuid: str, ...):
    with task_context(operation_uuid) as ctx:
        # Task logic with progress tracking
        ctx.update_progress(50, "Processing...")
```

2. **Queue from service**:
```python
celery_app.send_task("my_task", args=[operation_uuid, ...])
```

### Adding a New React Component

1. **Create component** (`components/MyComponent.tsx`)
2. **Add to parent** or create route
3. **Connect to store** via hooks
4. **Add MSW handler** for API mocking
5. **Write tests** (`components/__tests__/MyComponent.test.tsx`)

### Adding a New Connector

1. **Backend**: Add to `CONNECTOR_DEFINITIONS` in `connectors/definitions.py`
2. **Frontend**:
   - Add icon mapping in `ConnectorTypeSelector.tsx`
   - Add display order
   - Add preview handler if applicable

---

## Extension Cookbook

For detailed step-by-step guides on extending Semantik, see the dedicated cookbook files:

| Cookbook | File | Topics Covered |
|----------|------|----------------|
| **Search Enhancements** | [cookbook-search-enhancements.md](./architecture/cookbook-search-enhancements.md) | New search modes, rerankers, GraphRAG, filters, analytics |
| **UI Features** | [cookbook-ui-features.md](./architecture/cookbook-ui-features.md) | New pages, API endpoints, modals, real-time, forms, tables |

### Quick Reference: Common Extension Tasks

#### Add a New Search Mode
Full guide: [cookbook-search-enhancements.md § Add a New Search Mode](./architecture/cookbook-search-enhancements.md#1-add-a-new-search-mode)

1. Add enum value to `SearchMode` in contracts
2. Implement search logic in VecPipe `SearchService`
3. Add API parameters with validation
4. Update frontend selector and store
5. Write tests

#### Add a New API Endpoint (Full Stack)
Full guide: [cookbook-ui-features.md § Add a New API Endpoint](./architecture/cookbook-ui-features.md#2-add-a-new-api-endpoint-full-stack)

1. Database model + Alembic migration
2. Repository → Service → API router
3. Frontend API client → React Query hook → Component
4. Tests at each layer

#### Add a New Page
Full guide: [cookbook-ui-features.md § Add a New Page](./architecture/cookbook-ui-features.md#1-add-a-new-page)

1. Create page component with `<Layout>` wrapper
2. Add route in `App.tsx`
3. Add navigation link in `Sidebar.tsx`
4. Create data hooks with React Query
5. Add loading/error states

#### Extend GraphRAG
Full guide: [cookbook-search-enhancements.md § Extend GraphRAG](./architecture/cookbook-search-enhancements.md#3-extend-graphrag-capabilities)

1. Define new entity type with attributes
2. Implement entity extractor (spaCy or ML)
3. Create relation extractor if needed
4. Add graph query patterns
5. Update search service and frontend

#### Add Real-Time Updates
Full guide: [cookbook-ui-features.md § Add Real-Time Updates](./architecture/cookbook-ui-features.md#4-add-real-time-updates)

1. WebSocket endpoint (backend)
2. Redis stream publishing
3. `useWebSocket` hook (frontend)
4. Progress component
5. Cleanup and reconnection logic

### Quick Start Checklists

**New Search Feature:**
- [ ] Schema/contract changes
- [ ] VecPipe implementation
- [ ] WebUI service pass-through
- [ ] Frontend component/store
- [ ] Backend tests
- [ ] Frontend tests
- [ ] Documentation

**New UI Feature:**
- [ ] Database model (if needed)
- [ ] Alembic migration (if needed)
- [ ] Repository + Service + Router
- [ ] Frontend API client + hook
- [ ] Component with states
- [ ] Tests (backend + frontend)
- [ ] Update navigation if page

---

## Testing Guide

### Test Stack

| Layer | Framework | Tools |
|-------|-----------|-------|
| Python Backend | pytest | pytest-asyncio, pytest-cov, fakeredis |
| React Frontend | Vitest | React Testing Library, MSW |
| E2E | pytest-playwright | Browser automation |

### Running Tests

```bash
# Backend
make test          # All tests
make test-ci       # Exclude E2E
make test-coverage # With coverage

# Frontend
npm run test       # All tests
npm run test:ui    # Visual debugger
npm run test:coverage
```

### Key Fixtures

**Python:**
- `db_session` - Real async database session
- `api_client` - Authenticated AsyncClient
- `collection_factory` - Create test collections
- `stub_celery_send_task` - Mock Celery (autouse)
- `use_fakeredis` - Opt-in Redis mock

**React:**
- `renderWithProviders()` - Wrapped render with all providers
- `server` (MSW) - API mock server
- `handlers` - Default API response mocks

### Test Patterns

**API Test:**
```python
@pytest.mark.asyncio()
async def test_create_collection(api_client, api_auth_headers):
    response = await api_client.post(
        "/api/v2/collections",
        json={"name": "Test"},
        headers=api_auth_headers
    )
    assert response.status_code == 201
```

**Component Test:**
```typescript
it('should handle search', async () => {
  const { user } = renderWithProviders(<SearchInterface />);
  await user.click(screen.getByRole('button', { name: /search/i }));
  await waitFor(() => expect(screen.getByText(/results/i)).toBeInTheDocument());
});
```

---

## Deployment Reference

### Docker Services

| Service | Port | Purpose |
|---------|------|---------|
| webui | 8080 | FastAPI application |
| vecpipe | 8000 | Search/embedding API |
| postgres | 5432 | PostgreSQL database |
| qdrant | 6333 | Vector database |
| redis | 6379 | Celery broker |
| worker | - | Celery worker |
| beat | - | Celery scheduler |
| flower | 5555 | Celery monitoring |

### Key Environment Variables

```bash
# Required
DATABASE_URL=postgresql://user:pass@host:5432/db
JWT_SECRET_KEY=<64-hex-chars>
QDRANT_HOST=qdrant
REDIS_URL=redis://redis:6379/0

# Embedding
DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
DEFAULT_QUANTIZATION=float16  # float32, float16, int8

# Optional
ENVIRONMENT=production
LOG_LEVEL=WARNING
WEBUI_WORKERS=4
```

### Quick Start

```bash
# Development
make wizard        # Interactive setup
make docker-up     # Start all services

# Production
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## Appendix

### File Quick Reference

| What | Where |
|------|-------|
| API routes | `packages/webui/api/v2/` |
| Database models | `packages/shared/database/models.py` |
| Repositories | `packages/shared/database/repositories/` |
| Celery tasks | `packages/webui/tasks/` |
| React components | `apps/webui-react/src/components/` |
| Zustand stores | `apps/webui-react/src/stores/` |
| API client | `apps/webui-react/src/services/api/v2/` |
| Test fixtures | `tests/conftest.py` |
| Docker config | `docker-compose.yml`, `Dockerfile` |
| Migrations | `alembic/versions/` |

### Common Commands

```bash
# Development
make dev-local           # Local webui + Docker services
make docker-logs-webui   # Stream logs
make docker-shell-webui  # Container shell

# Database
make docker-postgres-backup
uv run alembic upgrade head

# Testing
pytest tests/webui -v -k "test_search"
npm run test -- --watch
```

---

*This document was auto-generated from comprehensive codebase analysis. For the most current information, refer to individual CLAUDE.md files in each package.*
