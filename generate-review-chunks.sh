#!/bin/bash
# Generate Semantik code review chunks with embedded review prompts

set -e

echo "Generating Semantik review chunks with review prompts..."
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

**Architecture**: FastAPI microservices (webui, vecpipe, worker) + React frontend
**This Chunk Contains**:
- Core database models (SQLAlchemy)
- Repository pattern implementations
- Domain logic (chunking, managers)
- Configuration management
- Database migrations (Alembic)

**Dependencies**: This is the foundation layer. All other chunks depend on models and utilities here.

## Critical Architecture Patterns

### 1. Repository Pattern
All database access MUST go through repositories. Check for:
- ✅ Clean separation: models vs repositories
- ✅ Repositories use async sessions correctly
- ✅ No direct SQLAlchemy queries outside repositories
- ❌ Business logic in repositories (should be in services)

### 2. Partition-Aware Queries (CRITICAL)
The `chunks` table uses 100 LIST partitions by `collection_id`:
- ✅ ALL chunk queries include `collection_id` filter
- ❌ Full table scans across all partitions
- Check: `chunk_repository.py` methods

### 3. Domain-Driven Chunking
Modern chunking in `packages/shared/chunking/`:
- ✅ Strategy pattern with clear interfaces
- ✅ Domain models separate from persistence
- ❌ Legacy `text_processing.chunking` usage (deprecated)

## Review Checklist

### Database Models (`packages/shared/database/models/`)
- [ ] Proper relationships and foreign keys
- [ ] Indexes on frequently queried columns
- [ ] Partition keys included in unique constraints
- [ ] Cascading deletes configured correctly
- [ ] Enums match database types

### Repositories (`packages/shared/database/repositories/`)
- [ ] All methods are async
- [ ] Proper exception handling
- [ ] No business logic (pure data access)
- [ ] Partition pruning for chunks table
- [ ] Proper use of `joinedload`/`selectinload` for relationships

### Migrations (`alembic/versions/`)
- [ ] Reversible (down migrations implemented)
- [ ] Safe for production (no data loss)
- [ ] Partition maintenance scripts included
- [ ] Indexes created concurrently for PostgreSQL

### Chunking System (`packages/shared/chunking/`)
- [ ] Strategy implementations follow interface
- [ ] No I/O in chunking logic (pure functions)
- [ ] Proper validation of chunk boundaries
- [ ] Test coverage for edge cases

### Configuration (`packages/shared/config/`)
- [ ] No secrets in code (environment variables)
- [ ] Validation for required settings
- [ ] Sensible defaults

## Specific Questions

1. **Data Integrity**: Are there any race conditions in repository methods?
2. **Performance**: Are there missing indexes or inefficient queries?
3. **Migration Safety**: Can migrations be rolled back safely?
4. **Partition Strategy**: Is the chunks table partitioning optimal?
5. **Deprecated Code**: Any legacy patterns that need refactoring?
6. **Job → Operation**: Flag any remaining "job" terminology (should be "operation")

## Output Format

Please provide:
1. **Critical Issues** (P0): Security, data loss, race conditions
2. **Architecture Violations** (P1): Pattern breaks, coupling issues
3. **Performance Concerns** (P2): Query optimization, indexing
4. **Code Quality** (P3): Duplication, complexity, naming
5. **Refactoring Opportunities**: Technical debt, modernization

For each issue, include:
- File path and line numbers
- Current code snippet
- Problem explanation
- Recommended fix

EOF

# Chunk 2: WebUI API Layer
echo "[2/7] Generating WebUI API Layer chunk..."
repomix \
  --include "packages/webui/api,packages/webui/middleware,packages/webui/websocket,CLAUDE.md" \
  --ignore "**/__pycache__/**,**/*.pyc" \
  --output review-chunk-2-webui-api.txt

cat >> review-chunk-2-webui-api.txt <<'EOF'

================================================================================
# CODE REVIEW PROMPT - CHUNK 2: WEBUI API LAYER
================================================================================

## Context
This is **Chunk 2 of 7** for reviewing Semantik.

**This Chunk Contains**:
- API routers (FastAPI endpoints)
- Middleware (auth, rate limiting, CORS)
- WebSocket management
- Request/response schemas

**Dependencies**:
- **Chunk 1**: Models, repositories, config
- **Chunk 3**: Services (business logic)
- **Chunk 5**: Frontend (consumes these APIs)

## Critical Architecture Pattern: API Layer Rules

**API routers MUST**:
- HTTP request/response handling ONLY
- Input validation via Pydantic models
- Delegate ALL logic to services
- Return HTTP responses

**API routers MUST NOT**:
- Contain business logic
- Access database directly
- Perform calculations or transformations
- Handle Celery tasks

## Review Checklist

### API Routers (`packages/webui/api/v2/`)
- [ ] No business logic in routers
- [ ] No direct database access
- [ ] All logic delegated to services
- [ ] Proper dependency injection
- [ ] Input validation via Pydantic
- [ ] Appropriate HTTP status codes
- [ ] Consistent error responses

**Anti-Pattern Example**:
```python
# ❌ BAD
@router.post("/collections")
async def create(data: dict, db: AsyncSession = Depends(get_db)):
    collection = Collection(**data)
    db.add(collection)
    await db.commit()  # Business logic in router!
    return collection
```

```python
# ✅ GOOD
@router.post("/collections")
async def create(
    data: CreateCollectionRequest,
    service: CollectionService = Depends(get_collection_service)
):
    collection = await service.create_collection(data)
    return collection
```

### WebSocket Management (`packages/webui/websocket/`)
- [ ] Authentication via JWT in first message
- [ ] Connection limits enforced (10/user, 10k total)
- [ ] Proper cleanup on disconnect
- [ ] Redis pub/sub for horizontal scaling
- [ ] Channel naming conventions consistent
- [ ] Error handling for malformed messages

### Middleware (`packages/webui/middleware/`)
- [ ] Authentication middleware validates JWTs
- [ ] Rate limiting per endpoint
- [ ] CORS configured securely
- [ ] Request ID propagation
- [ ] No sensitive data in logs

## Specific Questions

1. **Architecture Violations**: Any business logic in routers? Direct DB access?
2. **WebSocket Security**: Is authentication enforced on all channels?
3. **Error Handling**: Are errors propagated correctly to clients?
4. **Rate Limiting**: Are critical endpoints protected?

## Output Format

Same as Chunk 1: Critical Issues → Architecture Violations → Performance → Code Quality → Refactoring

EOF

# Chunk 3: WebUI Services & Tasks
echo "[3/7] Generating WebUI Services & Tasks chunk..."
repomix \
  --include "packages/webui/services,packages/webui/tasks.py,packages/webui/dependencies.py,CLAUDE.md" \
  --ignore "**/__pycache__/**,**/*.pyc" \
  --output review-chunk-3-webui-services.txt

cat >> review-chunk-3-webui-services.txt <<'EOF'

================================================================================
# CODE REVIEW PROMPT - CHUNK 3: WEBUI SERVICES & TASKS
================================================================================

## Context
This is **Chunk 3 of 7** for reviewing Semantik.

**This Chunk Contains**:
- Service layer (business logic)
- Celery tasks (background operations)
- Dependency injection setup

**Dependencies**:
- **Chunk 1**: Models and repositories
- **Chunk 2**: API routers call these services
- **Chunk 4**: Vecpipe service (HTTP calls)

## Critical Pattern: Service Layer + Celery

### Service Layer Rules
- ALL business logic lives here
- Transaction management
- Orchestrates repository calls
- **CRITICAL**: Commit BEFORE Celery dispatch

### Transaction Order (CRITICAL)
```python
# ✅ CORRECT ORDER
async def trigger_indexing(collection_id: str):
    # 1. Create operation record
    operation = await operation_repo.create(...)

    # 2. Commit transaction FIRST
    await db.commit()

    # 3. THEN dispatch task
    index_collection_task.delay(operation.uuid)

    return operation

# ❌ WRONG - Race condition
async def trigger_indexing(collection_id: str):
    operation = await operation_repo.create(...)
    index_collection_task.delay(operation.uuid)  # Dispatched before commit!
    await db.commit()  # Worker might not find operation
```

## Review Checklist

### Service Layer (`packages/webui/services/`)
- [ ] Contains all business logic
- [ ] Proper async/await usage
- [ ] No blocking I/O
- [ ] Transaction management correct
- [ ] **CRITICAL**: Commit BEFORE Celery dispatch
- [ ] Error handling and validation
- [ ] Proper logging
- [ ] No direct SQLAlchemy usage (use repositories)

### Celery Tasks (`packages/webui/tasks.py`)
- [ ] Idempotent (safe to retry)
- [ ] Proper error handling
- [ ] Progress reporting via WebSocket
- [ ] Database session management
- [ ] No long-running blocking operations
- [ ] Timeout configured appropriately

## Specific Questions

1. **Race Conditions**: Are transactions committed before async task dispatch?
2. **Async Correctness**: Any blocking I/O in async functions?
3. **Collection State Machine**: Are state transitions validated properly?
4. **Error Recovery**: How do tasks handle failures?

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
- Document parsing (PDF, HTML, Markdown, etc.)
- Embedding generation (via sentence-transformers)
- Semantic search implementation
- Qdrant vector database management

**Dependencies**:
- **Chunk 1**: Models, repositories, chunking domain logic
- **Chunk 3**: WebUI calls vecpipe via HTTP

## Critical Patterns

### 1. Stateless Service Design
Vecpipe is compute-intensive and stateless:
- [ ] No session state stored in service
- [ ] Each request is independent
- [ ] Concurrent requests handled safely
- [ ] Resource cleanup after requests

### 2. Embedding Pipeline
- [ ] Model loaded once and cached
- [ ] Batch processing for multiple documents
- [ ] GPU utilization if available
- [ ] Proper error handling for failed embeddings
- [ ] Memory management for large documents

## Review Checklist

### Document Parsing
- [ ] Support for common formats (PDF, HTML, MD, TXT, DOCX)
- [ ] Character encoding detection
- [ ] Large file handling (streaming)
- [ ] Error handling for corrupted files
- [ ] No unsafe file operations

### Embedding Generation
- [ ] Model loaded once and cached
- [ ] Batching for efficiency
- [ ] Proper tensor cleanup (memory leaks)
- [ ] GPU vs CPU handling
- [ ] Error handling for embedding failures

### Qdrant Operations
- [ ] Collection lifecycle managed properly
- [ ] Batch upserts for performance
- [ ] Search uses filters (collection_id)
- [ ] Proper error handling for Qdrant API

## Specific Questions

1. **Performance**: Is embedding generation batched efficiently?
2. **Memory**: Are there memory leaks in long-running processes?
3. **Qdrant**: Is the vector database accessed optimally?
4. **Parsing**: Can document parsing handle malicious files safely?

## Output Format

Same as previous chunks: Critical Issues → Architecture → Performance → Code Quality → Refactoring

EOF

# Chunk 5: Frontend
echo "[5/7] Generating Frontend chunk..."
repomix \
  --include "apps/webui-react,CLAUDE.md" \
  --ignore "**/node_modules/**,**/dist/**,**/build/**,**/.vite/**,**/coverage/**,**/*.test.tsx.snap" \
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
- React Query for data fetching
- TailwindCSS styling
- API client implementations

**Dependencies**:
- **Chunk 2**: Backend API endpoints this frontend consumes

## Tech Stack
- **React**: 19 with hooks
- **TypeScript**: Strict mode
- **State**: Zustand stores
- **Data Fetching**: React Query (TanStack Query)
- **Routing**: React Router v6
- **Styling**: TailwindCSS + Radix UI

## Review Checklist

### State Management (`src/stores/`)
- [ ] Zustand stores organized by domain
- [ ] Optimistic updates implemented correctly
- [ ] Error rollback on failed mutations
- [ ] Store actions are async-safe

**Optimistic Update Pattern**:
```typescript
// ✅ CORRECT
updateCollection: async (id, updates) => {
  get().optimisticUpdateCollection(id, updates);
  try {
    await collectionsV2Api.update(id, updates);
    await get().fetchCollectionById(id);
  } catch (error) {
    await get().fetchCollectionById(id);  // Rollback
    throw error;
  }
}
```

### API Client (`src/api/`)
- [ ] JWT token attached to requests
- [ ] Token refresh flow implemented
- [ ] Proper error handling
- [ ] TypeScript types match backend schemas

### Components (`src/components/`)
- [ ] Props properly typed
- [ ] Accessibility (ARIA labels, keyboard nav)
- [ ] Loading states shown
- [ ] Error boundaries

### Performance
- [ ] Code splitting / lazy loading
- [ ] Virtualization for long lists
- [ ] Debouncing for search inputs

### TypeScript
- [ ] Strict mode enabled
- [ ] No `any` types
- [ ] Backend DTOs mirrored in frontend types

## Specific Questions

1. **State Sync**: Are optimistic updates properly rolled back on errors?
2. **Performance**: Are there unnecessary re-renders?
3. **Accessibility**: Can the app be used with keyboard only?
4. **Type Safety**: Do frontend types match backend schemas?

## Output Format

Same format: Critical Issues → Architecture → Performance → Code Quality → Refactoring

EOF

# Chunk 6: Tests (Split into smaller chunks by excluding heavy integration tests)
echo "[6/7] Generating Tests chunk..."
repomix \
  --include "tests/shared,tests/unit,tests/conftest.py,pytest.ini,CLAUDE.md" \
  --ignore "**/__pycache__/**,**/.pytest_cache/**" \
  --output review-chunk-6-tests.txt

cat >> review-chunk-6-tests.txt <<'EOF'

================================================================================
# CODE REVIEW PROMPT - CHUNK 6: TESTS (UNIT & SHARED)
================================================================================

## Context
This is **Chunk 6 of 7** for reviewing Semantik.

**This Chunk Contains**:
- Unit tests for shared components
- Test fixtures and configuration
- Testing utilities

**Dependencies**: Tests reference code from Chunks 1-5

**Note**: Integration and E2E tests for webui/vecpipe are excluded to keep size manageable. Focus on test quality patterns here.

## Review Checklist

### Test Structure
- [ ] Organized by service (webui, vecpipe, shared)
- [ ] Unit tests vs integration tests clearly separated
- [ ] Test isolation (no shared state)

### Test Quality
- [ ] Critical paths fully tested
- [ ] Edge cases covered
- [ ] Error scenarios tested
- [ ] Async tests properly awaited
- [ ] No flaky tests

### Test Fixtures (`conftest.py`)
- [ ] Database fixtures (setup/teardown)
- [ ] Mock Celery tasks
- [ ] Mock Redis
- [ ] Mock Qdrant
- [ ] Proper cleanup after tests

## Specific Questions

1. **Test Coverage**: Are critical paths fully tested?
2. **Test Reliability**: Are there flaky tests?
3. **Mocking**: Are external services properly mocked?
4. **Fixtures**: Are test fixtures reusable and well-organized?

## Output Format

Same as previous chunks: Critical Issues → Architecture → Performance → Code Quality → Refactoring

EOF

# Chunk 7: Infrastructure & Deployment
echo "[7/7] Generating Infrastructure chunk..."
repomix \
  --include "docker-compose.yml,docker-compose.dev.yml,Dockerfile,Dockerfile.dev,Dockerfile.vecpipe,.dockerignore,Makefile,.github,docs,.env.example,CLAUDE.md" \
  --ignore "**/__pycache__/**" \
  --output review-chunk-7-infrastructure.txt

cat >> review-chunk-7-infrastructure.txt <<'EOF'

================================================================================
# CODE REVIEW PROMPT - CHUNK 7: INFRASTRUCTURE & DEPLOYMENT
================================================================================

## Context
This is **Chunk 7 of 7** for reviewing Semantik.

**This Chunk Contains**:
- Docker Compose orchestration
- Dockerfiles for services
- CI/CD workflows (GitHub Actions)
- Makefile development commands
- Documentation

## Review Checklist

### Docker Compose (`docker-compose.yml`)
- [ ] All services defined
- [ ] Health checks configured
- [ ] Network isolation
- [ ] Volume mounts for persistence
- [ ] Environment variables from .env
- [ ] No hardcoded secrets
- [ ] Resource limits set

### Dockerfiles
- [ ] Multi-stage builds for optimization
- [ ] Minimal base images
- [ ] Proper layer caching
- [ ] Non-root user
- [ ] Health check defined
- [ ] Dependencies pinned

### CI/CD (`.github/workflows/`)
- [ ] Run on PR and main branch
- [ ] Linting checks (ruff, mypy)
- [ ] Test execution
- [ ] Coverage reporting
- [ ] Docker build testing
- [ ] Security scanning

### Makefile
- [ ] Common commands documented
- [ ] Consistent naming
- [ ] Help command available

### Documentation (`docs/`)
- [ ] Architecture documented
- [ ] API reference complete
- [ ] Deployment guide available
- [ ] Troubleshooting guide

## Specific Questions

1. **Docker**: Are images optimized for size and security?
2. **CI/CD**: Does the pipeline catch common issues?
3. **Deployment**: Is the process documented and automated?
4. **Security**: Are secrets managed properly?

## Cross-Cutting Concerns

Since this is the final chunk, please also consider:

1. **Overall Architecture**: Does the codebase follow the three-layer pattern consistently?
2. **Job → Operation Migration**: Any remaining "job" terminology to update?
3. **Code Consistency**: Are naming conventions consistent across services?
4. **Documentation**: Is the codebase well-documented?
5. **Technical Debt**: What are the top refactoring priorities?
6. **Security Posture**: Any security vulnerabilities discovered?
7. **Performance**: Are there performance bottlenecks?

## Final Output

Please provide a comprehensive review summary covering:

1. **Critical Issues** found across all areas
2. **Top 5 Refactoring Priorities**
3. **Security Recommendations**
4. **Performance Optimization Opportunities**
5. **Testing Gaps**
6. **Overall Code Quality Assessment** (scale 1-10 with justification)

EOF

echo ""
echo "✅ All chunks generated with review prompts!"
echo ""
echo "📊 Checking sizes..."
echo ""

for file in review-chunk-*.txt; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        tokens=$(grep "Total Tokens:" "$file" | head -1 | awk '{print $3}' | tr -d ',')
        if [ -z "$tokens" ]; then
            tokens="N/A"
        fi
        echo "📄 $(basename $file)"
        echo "   Size: $size | Tokens: $tokens"
        echo ""
    fi
done

echo "✅ Done! Review chunks are ready for LLM analysis."
echo ""
echo "Chunk Organization:"
echo "1. Shared Foundation - Models, repos, migrations, chunking (~233K tokens)"
echo "2. WebUI API Layer - Routers, middleware, WebSocket"
echo "3. WebUI Services - Business logic, Celery tasks"
echo "4. Vecpipe Service - Embeddings, search, parsing"
echo "5. Frontend - React app, stores, components"
echo "6. Tests - Unit tests and shared test utilities (smaller subset)"
echo "7. Infrastructure - Docker, CI/CD, docs (~177K tokens)"
