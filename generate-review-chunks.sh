#!/bin/bash
# Generate Semantik code review chunks with embedded review prompts
# Updated: 2025-10-17 to reflect test refactoring and current structure

set -e

echo "🔍 Generating Semantik review chunks with review prompts..."
echo ""

# Chunk 1: Shared Foundation (Models, Repos, Domain Logic)
echo "[1/7] Generating Shared Foundation chunk..."
repomix \
  --include "packages/shared,alembic,pyproject.toml,uv.lock,CLAUDE.md,README.md" \
  --ignore "**/__pycache__/**,**/.pytest_cache/**,**/*.pyc,**/.mypy_cache/**,**/.ruff_cache/**" \
  --output review-chunk-1-shared-foundation.txt

cat >> review-chunk-1-shared-foundation.txt <<'EOF'

================================================================================
# CODE REVIEW PROMPT - CHUNK 1: SHARED FOUNDATION
================================================================================

## Context
This is **Chunk 1 of 7** for reviewing Semantik, a self-hosted semantic search engine.

**Total Codebase**: ~1.8M tokens across 748 files
**Architecture**: FastAPI microservices (webui, vecpipe, worker) + React frontend
**This Chunk Contains** (~234K tokens):
- Core database models (SQLAlchemy)
- Repository pattern implementations
- Domain logic (chunking strategies, Qdrant management)
- Configuration management (Pydantic settings)
- Database migrations (Alembic)

**Dependencies**: This is the foundation layer. All other chunks depend on models and utilities here.

## Critical Architecture Patterns

### 1. Repository Pattern (STRICT)
All database access MUST go through repositories. Check for:
- ✅ Clean separation: models vs repositories vs services
- ✅ Repositories use async sessions correctly
- ✅ No direct SQLAlchemy queries outside repositories
- ❌ Business logic in repositories (should be in services)
- ❌ Repositories calling other repositories

### 2. Partition-Aware Queries (CRITICAL FOR PERFORMANCE)
The `chunks` table uses 100 LIST partitions by `collection_id`:
- ✅ ALL chunk queries MUST include `collection_id` filter
- ❌ Missing `collection_id` causes full table scans across all 100 partitions
- Check: `packages/shared/database/repositories/chunk_repository.py` methods
- See: `docs/partition-monitoring.md` for monitoring

### 3. Domain-Driven Chunking (NEW ARCHITECTURE)
Modern chunking system in `packages/shared/chunking/`:
- ✅ Strategy pattern with clear interfaces
- ✅ Domain models separate from persistence (no SQLAlchemy in strategies)
- ✅ Unified interface via `ChunkingService`
- ❌ Legacy `text_processing.chunking` module (deprecated, being phased out)

### 4. Migration Safety
Recent migration improvements:
- ✅ Backup manager for critical operations
- ✅ Reversible migrations with down() implemented
- ✅ Partition-aware migration scripts
- See: `docs/SAFE_MIGRATION_GUIDE.md`

## Review Checklist

### Database Models (`packages/shared/database/models/`)
- [ ] Proper relationships and foreign keys
- [ ] Indexes on frequently queried columns
- [ ] Partition keys (`collection_id`) included in unique constraints
- [ ] Cascading deletes configured correctly (`cascade="all, delete-orphan"`)
- [ ] Enums match database types
- [ ] No circular imports

### Repositories (`packages/shared/database/repositories/`)
- [ ] All methods are async
- [ ] Proper exception handling
- [ ] No business logic (pure data access)
- [ ] **Partition pruning**: ALL chunk queries include `collection_id`
- [ ] Proper use of `joinedload`/`selectinload` for relationships
- [ ] No repositories calling other repositories directly

### Migrations (`alembic/versions/`)
- [ ] Reversible (down migrations implemented)
- [ ] Safe for production (no data loss, tested with backup)
- [ ] Partition maintenance scripts included
- [ ] Indexes created with `CONCURRENTLY` for PostgreSQL
- [ ] Migration tested on realistic dataset size

### Chunking System (`packages/shared/chunking/`)
- [ ] Strategy implementations follow `ChunkingStrategy` interface
- [ ] No I/O in chunking logic (pure functions)
- [ ] Proper validation of chunk boundaries
- [ ] Metadata preserved correctly
- [ ] Test coverage for edge cases (see chunk 6)

### Configuration (`packages/shared/config/`)
- [ ] No secrets in code (environment variables only)
- [ ] Validation for required settings (Pydantic validators)
- [ ] Sensible defaults
- [ ] Environment-specific configs (dev, prod)

### Managers (`packages/shared/managers/`)
- [ ] Qdrant client properly initialized
- [ ] Connection pooling configured
- [ ] Error handling for network failures
- [ ] Batch operations for efficiency

## Specific Questions

1. **Data Integrity**: Are there any race conditions in repository methods?
2. **Performance**: Are there missing indexes or inefficient queries? Check with `EXPLAIN ANALYZE`.
3. **Migration Safety**: Can all migrations be rolled back safely? Are backups created?
4. **Partition Strategy**: Is `collection_id` included in ALL chunk queries?
5. **Deprecated Code**: Any remaining uses of legacy `text_processing.chunking`?
6. **Job → Operation**: Flag any remaining "job" terminology (should be "operation")
7. **Async Correctness**: Any blocking I/O in async functions?

## Output Format

Please provide:
1. **Critical Issues** (P0): Security, data loss, race conditions, partition misses
2. **Architecture Violations** (P1): Pattern breaks, coupling issues, business logic in repos
3. **Performance Concerns** (P2): Query optimization, missing indexes, N+1 queries
4. **Code Quality** (P3): Duplication, complexity, naming, missing type hints
5. **Refactoring Opportunities**: Technical debt, modernization, deprecation candidates

For each issue, include:
- File path and line numbers
- Current code snippet
- Problem explanation
- Recommended fix with code example

EOF

# Chunk 2: WebUI API Layer
echo "[2/7] Generating WebUI API Layer chunk..."
repomix \
  --include "packages/webui/api,packages/webui/middleware,packages/webui/websocket,packages/webui/auth.py,packages/webui/rate_limiter.py,CLAUDE.md,docs/API_REFERENCE.md,docs/WEBSOCKET_API.md" \
  --ignore "**/__pycache__/**,**/*.pyc" \
  --output review-chunk-2-webui-api.txt

cat >> review-chunk-2-webui-api.txt <<'EOF'

================================================================================
# CODE REVIEW PROMPT - CHUNK 2: WEBUI API LAYER
================================================================================

## Context
This is **Chunk 2 of 7** for reviewing Semantik.

**This Chunk Contains**:
- API routers (FastAPI endpoints in `api/v2/`)
- Middleware (auth, rate limiting, CORS, request ID)
- WebSocket management (`websocket/` and `websocket_manager.py`)
- Authentication (`auth.py`)
- Rate limiting (`rate_limiter.py`)
- API documentation

**Dependencies**:
- **Chunk 1**: Models, repositories, config
- **Chunk 3**: Services (business logic layer)
- **Chunk 5**: Frontend (consumes these APIs)

## Critical Architecture Pattern: THREE-LAYER SEPARATION

```
┌──────────────┐
│  API Router  │ ← HTTP only, NO business logic
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Service    │ ← ALL business logic
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Repository  │ ← Data access only
└──────────────┘
```

### API Layer Rules (STRICT)

**API routers MUST**:
- Handle HTTP request/response ONLY
- Validate input via Pydantic models
- Delegate ALL logic to services
- Return appropriate HTTP status codes
- Handle exceptions with proper error responses

**API routers MUST NOT**:
- Contain business logic or calculations
- Access database directly (no SQLAlchemy)
- Import repository classes
- Call Celery tasks
- Perform data transformations (belongs in services)

## Review Checklist

### API Routers (`packages/webui/api/v2/`)
- [ ] Zero business logic in routers
- [ ] No direct database access
- [ ] All logic delegated to services via dependency injection
- [ ] Proper Pydantic models for request/response
- [ ] HTTP status codes match REST conventions (200, 201, 204, 400, 404, 500)
- [ ] Consistent error response format
- [ ] OpenAPI documentation (docstrings)

**Anti-Pattern Example**:
```python
# ❌ BAD - Business logic in router
@router.post("/collections")
async def create(data: dict, db: AsyncSession = Depends(get_db)):
    collection = Collection(**data)  # Business logic!
    db.add(collection)
    await db.commit()  # Direct DB access!
    return collection
```

```python
# ✅ GOOD - Thin router delegates to service
@router.post("/collections", status_code=201)
async def create(
    data: CreateCollectionRequest,
    service: CollectionService = Depends(get_collection_service)
):
    """Create a new collection with validation."""
    collection = await service.create_collection(data)
    return collection
```

### WebSocket Management
- [ ] Authentication via JWT in first message after connection
- [ ] Connection limits enforced (10/user, 10,000 total)
- [ ] Proper cleanup on disconnect (no memory leaks)
- [ ] Redis pub/sub for horizontal scaling
- [ ] Channel naming conventions: `{type}:{id}` (e.g., `operation-progress:uuid`)
- [ ] Error handling for malformed messages
- [ ] Progress updates sent correctly to WebSocket clients

### Authentication (`auth.py`)
- [ ] JWT secret from environment (never hardcoded)
- [ ] Token expiration enforced (24h access, 30d refresh)
- [ ] Password hashing with bcrypt
- [ ] No passwords in logs
- [ ] Refresh token rotation implemented

### Middleware (`middleware/`)
- [ ] Auth middleware validates JWTs on protected routes
- [ ] Rate limiting per endpoint (configured in `rate_limiter.py`)
- [ ] CORS configured securely (no `allow_origins=["*"]` in production)
- [ ] Request ID propagated through logs
- [ ] No sensitive data leaked in error responses

## Specific Questions

1. **Architecture Violations**: Any business logic in routers? Direct DB access?
2. **WebSocket Security**: Is authentication enforced? Are connection limits working?
3. **Error Handling**: Are errors properly sanitized before returning to clients?
4. **Rate Limiting**: Are expensive endpoints (search, indexing) rate-limited?
5. **JWT Security**: Are tokens properly validated? Is refresh flow secure?
6. **Dependency Injection**: Are services properly injected via FastAPI Depends?
7. **API Versioning**: Are breaking changes introduced in v2 endpoints properly?

## Output Format

Same as Chunk 1: Critical Issues → Architecture Violations → Performance → Code Quality → Refactoring

EOF

# Chunk 3: WebUI Services & Tasks
echo "[3/7] Generating WebUI Services & Background Tasks chunk..."
repomix \
  --include "packages/webui/services,packages/webui/tasks.py,packages/webui/chunking_tasks.py,packages/webui/background_tasks.py,packages/webui/dependencies.py,packages/webui/celery_app.py,CLAUDE.md" \
  --ignore "**/__pycache__/**,**/*.pyc" \
  --output review-chunk-3-webui-services.txt

cat >> review-chunk-3-webui-services.txt <<'EOF'

================================================================================
# CODE REVIEW PROMPT - CHUNK 3: WEBUI SERVICES & BACKGROUND TASKS
================================================================================

## Context
This is **Chunk 3 of 7** for reviewing Semantik.

**This Chunk Contains**:
- Service layer (`services/`) - ALL business logic
- Celery tasks (`tasks.py`, `chunking_tasks.py`, `background_tasks.py`)
- Dependency injection setup (`dependencies.py`)
- Celery configuration (`celery_app.py`)

**Dependencies**:
- **Chunk 1**: Models and repositories
- **Chunk 2**: API routers call these services
- **Chunk 4**: Vecpipe service (HTTP calls for embeddings/search)

## Critical Pattern: Service Layer + Celery Orchestration

### Service Layer Rules
- **ALL business logic lives here** (not in routers or repositories)
- Transaction management (commit before Celery dispatch)
- Orchestrates repository calls
- Handles errors and validation
- **CRITICAL**: Database commit BEFORE Celery task dispatch (prevent race conditions)

### Transaction Order Pattern (CRITICAL)

```python
# ✅ CORRECT ORDER - No race condition
async def trigger_indexing(collection_id: str):
    # 1. Create operation record in database
    operation = await operation_repo.create(
        collection_id=collection_id,
        operation_type="INDEX",
        status="PENDING"
    )

    # 2. Commit transaction FIRST
    await db.commit()
    await db.refresh(operation)  # Get DB-assigned ID

    # 3. THEN dispatch task (worker can find the record)
    index_collection_task.delay(str(operation.uuid))

    return operation

# ❌ WRONG - Race condition!
async def trigger_indexing(collection_id: str):
    operation = await operation_repo.create(...)
    index_collection_task.delay(str(operation.uuid))  # ❌ Dispatched before commit!
    await db.commit()  # ❌ Worker might query before this completes!
```

**Why this matters**: Celery workers run in separate processes and may start immediately. If the database transaction hasn't committed, the worker's query for the operation record will fail with "not found" error.

## Review Checklist

### Service Layer (`packages/webui/services/`)
- [ ] Contains all business logic (no logic in routers)
- [ ] Proper async/await usage
- [ ] No blocking I/O (use `asyncio.to_thread` if needed)
- [ ] Transaction management correct
- [ ] **CRITICAL**: DB commit BEFORE Celery dispatch in all cases
- [ ] Error handling with proper exceptions
- [ ] Logging at appropriate levels
- [ ] No direct SQLAlchemy usage (use repositories)
- [ ] State machine transitions validated (PENDING → PROCESSING → READY/ERROR)

### Celery Tasks (`tasks.py`, `chunking_tasks.py`)
- [ ] Tasks are idempotent (safe to retry)
- [ ] Proper error handling (try/except with logging)
- [ ] Progress reporting via WebSocket/Redis
- [ ] Database session management (create new session per task)
- [ ] No long-running blocking operations (break into smaller tasks)
- [ ] Timeout configured appropriately (`task_time_limit`)
- [ ] Cleanup on failure (set status to ERROR, clean up temp files)
- [ ] Task names follow convention: `{module}.{function}`

### Background Task Patterns
- [ ] Long operations broken into chunks with progress updates
- [ ] Celery chains for multi-step workflows
- [ ] Error recovery logic (retry with exponential backoff)
- [ ] Orphaned task cleanup (handle worker crashes)

### Progress Update Manager (Recent Addition)
- [ ] WebSocket progress updates sent correctly
- [ ] Progress percentages accurate (0-100)
- [ ] Error states communicated to frontend
- [ ] Cleanup on task completion

## Specific Questions

1. **Race Conditions**: Are ALL instances of task dispatch preceded by `db.commit()`?
2. **Async Correctness**: Any blocking I/O in async service methods?
3. **Collection State Machine**: Are state transitions validated? Can invalid states occur?
4. **Error Recovery**: Do tasks handle failures gracefully? Are retries configured?
5. **Progress Reporting**: Are WebSocket progress updates sent correctly?
6. **Transaction Scope**: Are database sessions properly scoped (no session reuse across tasks)?
7. **Memory Leaks**: Are large objects (embeddings, documents) properly cleaned up after processing?

## Output Format

Same as previous chunks: Critical Issues → Architecture → Performance → Code Quality → Refactoring

EOF

# Chunk 4: Vecpipe Service
echo "[4/7] Generating Vecpipe Service chunk..."
repomix \
  --include "packages/vecpipe,CLAUDE.md" \
  --ignore "**/__pycache__/**,**/*.pyc" \
  --output review-chunk-4-vecpipe.txt

cat >> review-chunk-4-vecpipe.txt <<'EOF'

================================================================================
# CODE REVIEW PROMPT - CHUNK 4: VECPIPE SERVICE
================================================================================

## Context
This is **Chunk 4 of 7** for reviewing Semantik.

**This Chunk Contains**:
- Document parsing (PDF, HTML, Markdown, TXT, DOCX)
- Embedding generation (sentence-transformers)
- Semantic search implementation
- Qdrant vector database operations
- Reranking (if implemented)

**Dependencies**:
- **Chunk 1**: Models, repositories, chunking strategies from shared
- **Chunk 3**: WebUI calls vecpipe via HTTP for embeddings and search

## Critical Patterns

### 1. Stateless Service Design
Vecpipe is compute-intensive and must be stateless for horizontal scaling:
- [ ] No session state stored in service
- [ ] Each request is independent
- [ ] Concurrent requests handled safely (thread-safe embedding model access)
- [ ] Resource cleanup after requests (free GPU memory)

### 2. Embedding Pipeline Optimization
- [ ] Model loaded once on startup and cached (singleton pattern)
- [ ] Batch processing for multiple documents (don't embed one at a time)
- [ ] GPU utilization if available (`device='cuda'` when available)
- [ ] Proper tensor cleanup (avoid GPU memory leaks)
- [ ] Memory management for large documents (chunking before embedding)

### 3. Qdrant Management
- [ ] Collection lifecycle managed correctly
- [ ] Vector dimensions match embedding model
- [ ] Indexing parameters optimized (HNSW config)
- [ ] Batch upserts for performance (not one-by-one)
- [ ] Search filters use `collection_id` for partition efficiency

## Review Checklist

### Document Parsing
- [ ] Support for common formats (PDF, HTML, MD, TXT, DOCX)
- [ ] Character encoding detection (chardet, UTF-8 fallback)
- [ ] Large file handling (streaming, memory limits)
- [ ] Error handling for corrupted/malformed files
- [ ] Text extraction preserves structure (headings, lists)
- [ ] **Security**: No arbitrary code execution (safe PDF parsing)
- [ ] **Security**: Path traversal prevention

### Chunking Integration
- [ ] Uses modern `packages/shared/chunking/` (NOT deprecated `text_processing`)
- [ ] Strategy selection based on document type
- [ ] Chunk boundaries respect semantic meaning
- [ ] Metadata preserved in chunks (source, page number, etc.)
- [ ] Character/token limits enforced (prevent OOM)

### Embedding Generation
- [ ] Model loaded once and cached (singleton)
- [ ] Batching for efficiency (process 32-128 chunks at once)
- [ ] Proper tensor cleanup (`torch.cuda.empty_cache()` after batch)
- [ ] GPU vs CPU handling (fallback if no GPU)
- [ ] Normalization if required by model (L2 norm for cosine similarity)
- [ ] Error handling for embedding failures (individual chunk errors don't fail batch)
- [ ] Model version tracking (for re-indexing when model changes)

### Qdrant Operations
- [ ] Collection creation with proper config (vector size, distance metric)
- [ ] Batch upserts for performance (500-1000 vectors per batch)
- [ ] Search uses filters (`collection_id`) for efficiency
- [ ] Proper error handling for Qdrant API failures
- [ ] Connection pooling/reuse (don't recreate client per request)
- [ ] Blue-green reindexing support (create new collection, swap pointer)

### Search Implementation
- [ ] Efficient vector search (HNSW parameters tuned)
- [ ] Metadata filtering (filter by source, date, etc.)
- [ ] Hybrid search if implemented (combine vector + keyword search)
- [ ] Result ranking and scoring (confidence scores returned)
- [ ] Pagination support (limit/offset)
- [ ] Performance monitoring (search latency tracked)

### Reranking (if implemented)
- [ ] Reranker model loaded correctly
- [ ] Applied after initial vector search (top K results)
- [ ] Significant quality improvement (worth the latency cost)
- [ ] See: `docs/RERANKING.md`

## Specific Questions

1. **Performance**: Is embedding generation batched? What's the batch size?
2. **Memory**: Are there memory leaks in long-running processes? GPU memory freed?
3. **Qdrant**: Are collections created with optimal settings? Batch size for upserts?
4. **Parsing**: Can document parsing handle malicious files safely? Size limits enforced?
5. **Chunking**: Are the right strategies used for different content types?
6. **Error Recovery**: What happens if Qdrant is unavailable? Retries configured?
7. **GPU Usage**: Is GPU properly utilized if available? Fallback to CPU working?

## Output Format

Same as previous chunks: Critical Issues → Architecture → Performance → Code Quality → Refactoring

EOF

# Chunk 5: Frontend
echo "[5/7] Generating Frontend chunk..."
repomix \
  --include "apps/webui-react,CLAUDE.md,docs/FRONTEND_ARCH.md" \
  --ignore "**/node_modules/**,**/dist/**,**/build/**,**/.vite/**,**/coverage/**,**/*.test.tsx.snap,**/.turbo/**" \
  --output review-chunk-5-frontend.txt

cat >> review-chunk-5-frontend.txt <<'EOF'

================================================================================
# CODE REVIEW PROMPT - CHUNK 5: FRONTEND (REACT)
================================================================================

## Context
This is **Chunk 5 of 7** for reviewing Semantik.

**This Chunk Contains**:
- React 19 application
- Zustand state management
- React Query (TanStack Query) for data fetching
- TailwindCSS styling
- API client implementations
- Component library

**Dependencies**:
- **Chunk 2**: Backend API endpoints this frontend consumes

## Tech Stack
- **React**: 19 with hooks
- **TypeScript**: Strict mode
- **State**: Zustand stores (domain-organized)
- **Data Fetching**: React Query (TanStack Query)
- **Routing**: React Router v6
- **Styling**: TailwindCSS + Radix UI primitives
- **Build**: Vite
- **Testing**: Vitest + React Testing Library + Playwright

## Review Checklist

### State Management (`src/stores/`)
- [ ] Zustand stores organized by domain (collections, operations, auth)
- [ ] Optimistic updates implemented correctly (see pattern below)
- [ ] Error rollback on failed mutations
- [ ] No prop drilling (use stores or context)
- [ ] Selectors used for derived state
- [ ] Store actions are async-safe (no race conditions)

**Optimistic Update Pattern**:
```typescript
// ✅ CORRECT - Optimistic update with rollback
updateCollection: async (id, updates) => {
  // 1. Update UI immediately (optimistic)
  get().optimisticUpdateCollection(id, updates);

  try {
    // 2. Make API call
    await collectionsV2Api.update(id, updates);

    // 3. Re-fetch canonical state from server
    await get().fetchCollectionById(id);
  } catch (error) {
    // 4. Rollback on failure (restore previous state)
    await get().fetchCollectionById(id);
    toast.error('Failed to update collection');
    throw error;
  }
}
```

### API Client (`src/api/`)
- [ ] Axios/fetch configured with base URL
- [ ] JWT token attached to requests (Authorization header)
- [ ] Token refresh flow implemented (intercept 401, refresh, retry)
- [ ] Request/response interceptors configured
- [ ] Proper error handling (network errors, API errors)
- [ ] TypeScript types match backend Pydantic schemas
- [ ] API versioning support (`/api/v2/`)

### Components (`src/components/`)
- [ ] Props properly typed (TypeScript interfaces)
- [ ] No inline styles (use Tailwind classes)
- [ ] Accessibility (ARIA labels, keyboard navigation, focus management)
- [ ] Loading states shown (skeletons or spinners)
- [ ] Error boundaries for error handling
- [ ] Memoization where appropriate (`React.memo`, `useMemo`, `useCallback`)
- [ ] Component composition (small, reusable components)

### React Query Integration
- [ ] Query keys organized consistently (`['collections', id]`)
- [ ] Proper cache invalidation (`queryClient.invalidateQueries`)
- [ ] Optimistic updates via `setQueryData`
- [ ] Background refetching configured (`staleTime`, `cacheTime`)
- [ ] Error handling per query (`onError` callbacks)
- [ ] Loading states (`isLoading`, `isFetching`)

### WebSocket Integration
- [ ] Connection established with JWT authentication
- [ ] Reconnection logic (auto-reconnect on disconnect)
- [ ] Message handlers by channel (`operation-progress`, `collection-updates`)
- [ ] UI updates on WebSocket messages (real-time progress)
- [ ] Cleanup on unmount (close connection)
- [ ] Connection state shown to user (connected/disconnected indicator)

### Routing (`src/routes/`)
- [ ] Protected routes require auth (`PrivateRoute` wrapper)
- [ ] 404 handling (catch-all route)
- [ ] Route params properly typed
- [ ] Navigation guards (prevent navigation if unsaved changes)
- [ ] Breadcrumbs/navigation state

### Forms & Validation
- [ ] Form libraries used (React Hook Form recommended)
- [ ] Client-side validation (Zod or Yup schemas)
- [ ] Server validation errors displayed
- [ ] Loading states during submission (disable button)
- [ ] Optimistic updates for better UX

### Performance
- [ ] Code splitting / lazy loading (`React.lazy`)
- [ ] Bundle size optimized (check with `npm run build`)
- [ ] Images optimized (WebP, lazy loading)
- [ ] Virtualization for long lists (react-window or react-virtual)
- [ ] Debouncing for search inputs (300ms recommended)
- [ ] Memo/callback hooks prevent unnecessary re-renders

### Testing (Frontend)
- [ ] Unit tests for stores (Vitest)
- [ ] Component tests with React Testing Library
- [ ] Mock API calls (MSW recommended)
- [ ] E2E tests with Playwright
- [ ] Accessibility tests (jest-axe)
- [ ] **Coverage Target**: ≥75% (per tests/CLAUDE.md)

### TypeScript
- [ ] Strict mode enabled (`tsconfig.json`)
- [ ] No `any` types (use `unknown` if truly unknown)
- [ ] Proper interface definitions
- [ ] Backend DTOs mirrored in frontend types (`src/types/`)

## Specific Questions

1. **State Sync**: Are optimistic updates properly rolled back on errors?
2. **Token Management**: Is JWT refresh handled seamlessly? No 401 errors for users?
3. **WebSocket**: Does reconnection work correctly? Are progress updates shown?
4. **Performance**: Are there unnecessary re-renders? Check with React DevTools Profiler.
5. **Accessibility**: Can the app be used with keyboard only? Screen reader friendly?
6. **Error Handling**: Are error states shown to users? Toast notifications working?
7. **Type Safety**: Do frontend types match backend schemas? Any type mismatches?
8. **Bundle Size**: Is code splitting implemented? What's the main bundle size?

## Output Format

Same format: Critical Issues → Architecture → Performance → Code Quality → Refactoring

EOF

# Chunk 6: Tests (Unit, Integration, E2E, Security)
echo "[6/7] Generating Tests chunk..."
repomix \
  --include "tests,pytest.ini,CLAUDE.md,docs/TESTING.md,docs/TEST_QUALITY_ACTION_PLAN.md" \
  --ignore "**/__pycache__/**,**/.pytest_cache/**,**/htmlcov/**,**/.coverage" \
  --output review-chunk-6-tests.txt

cat >> review-chunk-6-tests.txt <<'EOF'

================================================================================
# CODE REVIEW PROMPT - CHUNK 6: TESTS (ALL TYPES)
================================================================================

## Context
This is **Chunk 6 of 7** for reviewing Semantik.

**This Chunk Contains**:
- Unit tests (`tests/unit/`)
- Integration tests (`tests/integration/`)
- E2E tests (`tests/e2e/`)
- Security tests (`tests/security/`)
- Performance tests (`tests/performance/`)
- Test fixtures (`tests/conftest.py`, `tests/fixtures/`)
- Test documentation (`tests/CLAUDE.md`, `docs/TESTING.md`)

**Dependencies**: Tests reference code from Chunks 1-5

**Coverage Targets** (per tests/CLAUDE.md):
- Backend: ≥80%
- Frontend: ≥75%

## Recent Test Refactoring (October 2025)

Tests were recently reorganized from a monolithic structure to domain-driven organization:
- ✅ Tests now organized by type (unit, integration, e2e, security)
- ✅ Test quality improvements documented in `TEST_QUALITY_ACTION_PLAN.md`
- ✅ Contract tests added for API stability (`test_contracts.py`)
- ✅ Security tests added (OWASP patterns)

## Review Checklist

### Test Organization
- [ ] Tests organized by type and domain
- [ ] Clear separation: unit vs integration vs e2e
- [ ] E2E tests marked with `@pytest.mark.e2e`
- [ ] Test naming convention: `test_{what}_{condition}_{expected}`
- [ ] One test file per module under test
- [ ] Test isolation (no shared state between tests)

### Test Quality (Critical)
- [ ] Comprehensive coverage (≥80% backend, ≥75% frontend)
- [ ] Critical paths fully tested (happy path + error cases)
- [ ] Edge cases covered (empty inputs, max values, special characters)
- [ ] Error scenarios tested (network failures, DB errors, validation errors)
- [ ] Async tests properly awaited (`@pytest.mark.asyncio`)
- [ ] No flaky tests (deterministic, no random values, no sleep())
- [ ] Fast execution (<5 min for unit tests, parallelizable)

### Test Fixtures (`conftest.py`, `fixtures/`)
- [ ] Database fixtures with setup/teardown
- [ ] Mock Celery tasks (no actual background tasks in tests)
- [ ] Mock Redis (`fakeredis` for isolation)
- [ ] Mock Qdrant (don't hit real vector DB in unit tests)
- [ ] HTTP client fixtures (`TestClient`)
- [ ] Auth fixtures (test users, JWT tokens)
- [ ] Proper cleanup after tests (rollback transactions, clear Redis)

### Unit Tests (`tests/unit/`)
- [ ] Test single units in isolation
- [ ] Mock all dependencies (no DB, no Redis, no HTTP)
- [ ] Fast execution (<100ms per test)
- [ ] Test business logic thoroughly
- [ ] Test error handling and validation
- [ ] Test edge cases

### Integration Tests (`tests/integration/`)
- [ ] Test component interactions
- [ ] Use test database (not production!)
- [ ] Transactions rolled back after each test
- [ ] Real repository + service layer tested
- [ ] API endpoints tested with real request/response
- [ ] Test data consistency across components

### E2E Tests (`tests/e2e/`)
- [ ] Test full user workflows end-to-end
- [ ] Require running services (Docker Compose)
- [ ] Test WebSocket functionality
- [ ] Verify background task completion
- [ ] Test cross-service interactions (webui → vecpipe)
- [ ] Realistic test scenarios

### Security Tests (`tests/security/`)
- [ ] Path traversal prevention tested
- [ ] SQL injection prevention (parameterized queries)
- [ ] XSS prevention (input sanitization)
- [ ] CSRF protection (if applicable)
- [ ] Authentication bypass attempts blocked
- [ ] Rate limiting enforced
- [ ] OWASP Top 10 vulnerabilities tested

### Performance Tests (`tests/performance/`)
- [ ] Search latency benchmarks
- [ ] Embedding generation throughput
- [ ] Database query performance
- [ ] WebSocket message throughput
- [ ] Regression detection (alert if performance degrades)

### Contract Tests (`test_contracts.py`)
- [ ] API contract stability verified
- [ ] Breaking changes detected
- [ ] Backward compatibility tested
- [ ] Request/response schemas validated

## Test Anti-Patterns to Avoid

❌ **The Mockery** - Excessive mocking that doesn't test real behavior
❌ **The Giant** - Tests that test too many things at once
❌ **The Slow Poke** - Tests that take too long (no sleep(), use freezegun)
❌ **The Greedy** - Tests that modify global state
❌ **The Secret Catcher** - Tests that pass for the wrong reasons
❌ **The Flickering** - Flaky tests (fix or remove, don't ignore)
❌ **The Inspector** - Tests that inspect internal state instead of behavior

## Specific Questions

1. **Test Coverage**: Are critical paths fully tested? Check with `pytest --cov`.
2. **Test Reliability**: Are there any flaky tests? Run tests 10x to verify.
3. **Mocking**: Are external services properly mocked? No live API calls in tests?
4. **Fixtures**: Are test fixtures reusable and well-organized?
5. **Performance**: Do unit tests run fast (<5 min total)? Parallelize with `pytest-xdist`?
6. **Security**: Are OWASP Top 10 vulnerabilities tested?
7. **E2E Coverage**: Do E2E tests cover main user workflows?
8. **CI Integration**: Do tests pass in CI consistently? Check GitHub Actions.

## Output Format

Same as previous chunks: Critical Issues → Architecture → Performance → Code Quality → Refactoring

EOF

# Chunk 7: Infrastructure, Deployment & Documentation
echo "[7/7] Generating Infrastructure & Documentation chunk..."
repomix \
  --include "docker-compose.yml,docker-compose.dev.yml,Dockerfile,Dockerfile.dev,Dockerfile.vecpipe,.dockerignore,Makefile,.github,docs,.env.example,CLAUDE.md,README.md" \
  --ignore "**/__pycache__/**,**/node_modules/**,docs/api/**" \
  --output review-chunk-7-infrastructure.txt

cat >> review-chunk-7-infrastructure.txt <<'EOF'

================================================================================
# CODE REVIEW PROMPT - CHUNK 7: INFRASTRUCTURE, DEPLOYMENT & DOCUMENTATION
================================================================================

## Context
This is **Chunk 7 of 7** (FINAL) for reviewing Semantik.

**This Chunk Contains**:
- Docker Compose orchestration (`docker-compose*.yml`)
- Dockerfiles for all services
- Makefile (development commands)
- CI/CD workflows (`.github/workflows/`)
- Documentation (`docs/`)
- Environment configuration (`.env.example`)

**This is the final chunk**, so provide cross-cutting recommendations.

## Review Checklist

### Docker Compose (`docker-compose.yml`, `docker-compose.dev.yml`)
- [ ] All services defined (webui, vecpipe, worker, postgres, redis, qdrant)
- [ ] Health checks configured (readiness/liveness probes)
- [ ] Proper network isolation (services on dedicated network)
- [ ] Volume mounts for persistence (postgres data, qdrant data)
- [ ] Environment variables from `.env` file (no hardcoded secrets)
- [ ] No hardcoded secrets (use environment variables)
- [ ] Resource limits set (`mem_limit`, `cpus`)
- [ ] Restart policies configured (`restart: unless-stopped`)

### Dockerfiles (`Dockerfile`, `Dockerfile.dev`, `Dockerfile.vecpipe`)
- [ ] Multi-stage builds for optimization (builder → runtime)
- [ ] Minimal base images (python:3.11-slim, not python:3.11)
- [ ] Proper layer caching (COPY requirements first, then code)
- [ ] Non-root user (`USER appuser`)
- [ ] Health check defined (`HEALTHCHECK`)
- [ ] Dependencies pinned (exact versions in requirements.txt)
- [ ] `.dockerignore` configured (exclude `node_modules`, `.git`, etc.)

### CI/CD (`.github/workflows/`)
- [ ] Run on PR and main branch pushes
- [ ] Test matrix for Python versions (3.11, 3.12)
- [ ] Linting checks (ruff, mypy, black, isort)
- [ ] Format checks (black --check, isort --check)
- [ ] Test execution (pytest with coverage)
- [ ] Coverage reporting (upload to Codecov or similar)
- [ ] Docker build testing (ensure images build successfully)
- [ ] Security scanning (Snyk, Trivy, or GitHub Advanced Security)
- [ ] Deployment automation (if applicable)

### Makefile
- [ ] Common commands documented (`help` target)
- [ ] Consistent naming (snake_case or kebab-case)
- [ ] Proper error handling (use `set -e` in shell commands)
- [ ] Help command available (`make help`)
- [ ] Development vs production targets clearly separated
- [ ] Commands: docker-up, docker-down, test, lint, format, migration

### Database Migrations
- [ ] Alembic configured correctly (`alembic.ini`)
- [ ] Migrations apply cleanly (`alembic upgrade head`)
- [ ] Reversibility tested (`alembic downgrade -1`)
- [ ] Data migration scripts safe (no data loss)
- [ ] Backup/restore documented (`docs/SAFE_MIGRATION_GUIDE.md`)

### Environment Configuration
- [ ] `.env.example` provided with all required variables
- [ ] All required variables documented
- [ ] Sensible defaults for development
- [ ] Production vs development configs clearly separated
- [ ] No secrets committed to git (check `.gitignore`)
- [ ] Secret management documented (how to set JWT_SECRET_KEY, etc.)

### Documentation (`docs/`)
- [ ] Architecture documented (`ARCH.md`)
- [ ] API reference complete (`API_REFERENCE.md`)
- [ ] Deployment guide available (`DEPLOYMENT.md`)
- [ ] Troubleshooting guide (`docs/TROUBLESHOOTING.md`)
- [ ] Testing guide (`TESTING.md`)
- [ ] Database architecture (`DATABASE_ARCH.md`)
- [ ] WebSocket API (`WEBSOCKET_API.md`)
- [ ] Chunking system (`CHUNKING_IMPLEMENTATION_PLAN.md`)
- [ ] **New docs reviewed**: `TEST_QUALITY_ACTION_PLAN.md`, `SAFE_MIGRATION_GUIDE.md`

### Monitoring & Logging
- [ ] Structured logging configured (JSON logs for production)
- [ ] Log levels appropriate (DEBUG in dev, INFO in prod)
- [ ] Sensitive data not logged (passwords, tokens)
- [ ] Health check endpoints (`/health`, `/readiness`)
- [ ] Metrics collection (if implemented - Prometheus, etc.)

## Specific Questions

1. **Docker**: Are images optimized for size and security? Check with `docker images`.
2. **CI/CD**: Does the pipeline catch common issues? Any flaky CI tests?
3. **Deployment**: Is the deployment process documented and automated?
4. **Database Migrations**: Can they be rolled back safely? Tested on staging?
5. **Security**: Are secrets managed properly? No hardcoded credentials?
6. **Monitoring**: How are production issues detected? Alerts configured?
7. **Documentation**: Is the codebase well-documented? README up to date?

## Cross-Cutting Concerns (FINAL CHUNK)

Since this is the final chunk, please also provide recommendations on:

1. **Overall Architecture**: Does the codebase consistently follow the three-layer pattern (Router → Service → Repository)?
2. **Job → Operation Migration**: Any remaining "job" terminology to update? (Should be "operation")
3. **Code Consistency**: Are naming conventions consistent across services? Python (snake_case) vs TypeScript (camelCase)?
4. **Documentation Quality**: Is the codebase well-documented? Inline comments where necessary?
5. **Technical Debt**: What are the top 5 refactoring priorities based on all chunks?
6. **Security Posture**: Any security vulnerabilities discovered across all chunks?
7. **Performance**: Are there performance bottlenecks identified across all chunks?
8. **Maintainability**: Is the code easy to understand and modify? Onboarding documentation sufficient?
9. **Test Coverage**: Are there critical gaps in test coverage across all chunks?
10. **Deprecated Code**: What legacy patterns need to be removed? (`text_processing.chunking`, "job" terminology, etc.)

## Final Output (Comprehensive Summary)

Please provide a comprehensive review summary covering:

1. **Critical Issues** found across all 7 chunks (P0 - fix immediately)
2. **Top 10 Refactoring Priorities** (ordered by impact)
3. **Security Recommendations** (vulnerabilities, hardening)
4. **Performance Optimization Opportunities** (database, API, frontend)
5. **Testing Gaps** (critical untested paths)
6. **Documentation Improvements** (missing or outdated docs)
7. **Overall Code Quality Assessment** (scale 1-10 with detailed justification)
8. **Architectural Recommendations** (long-term improvements)

EOF

echo ""
echo "✅ All chunks generated with review prompts!"
echo ""
echo "📊 Checking sizes..."
echo ""

for file in review-chunk-*.txt; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        # Extract token count from repomix metadata
        tokens=$(grep "Total Tokens:" "$file" | head -1 | awk '{print $3}' | tr -d ',')
        if [ -z "$tokens" ]; then
            # Fallback to word count estimation if repomix metadata not found
            words=$(wc -w < "$file")
            tokens=$(echo "$words * 1.3" | bc 2>/dev/null | cut -d'.' -f1)
            if [ -z "$tokens" ]; then
                tokens="N/A"
            fi
        fi
        echo "📄 $(basename $file)"
        echo "   Size: $size | Tokens: $tokens"

        # Warn if over 400K
        if [ "$tokens" != "N/A" ] && [ "$tokens" -gt 400000 ]; then
            echo "   ⚠️  WARNING: Over 400K token limit!"
        fi
        echo ""
    fi
done

echo "✅ Done! Review chunks are ready for LLM analysis."
echo ""
echo "📋 Chunk Organization:"
echo "1. Shared Foundation - Models, repos, migrations, chunking, Qdrant (~234K tokens)"
echo "2. WebUI API Layer - Routers, middleware, WebSocket, auth"
echo "3. WebUI Services - Business logic, Celery tasks, progress manager"
echo "4. Vecpipe Service - Embeddings, search, parsing, reranking"
echo "5. Frontend - React app, Zustand stores, components, WebSocket client"
echo "6. Tests - Unit, integration, E2E, security, performance (NEW structure)"
echo "7. Infrastructure - Docker, CI/CD, Makefile, docs (includes new test docs)"
echo ""
echo "📝 Recent Updates Reflected:"
echo "   - Test refactoring (unit/integration/e2e/security organization)"
echo "   - Progress update manager"
echo "   - Test quality documentation (TEST_QUALITY_ACTION_PLAN.md)"
echo "   - Safe migration guide (SAFE_MIGRATION_GUIDE.md)"
echo "   - Coverage targets: ≥80% backend, ≥75% frontend"
echo ""
echo "🎯 Send each chunk to your LLM with the embedded review prompt!"
