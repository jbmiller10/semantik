#!/bin/bash
# Generate Semantik code review chunks with embedded review prompts
# Updated: 2025-10-17 - Larger chunks (~500K tokens) with intentional overlap for better context

set -e

echo "🔍 Generating Semantik review chunks (targeting ~500K tokens each)..."
echo ""

# Chunk 1: Backend Core (Shared + WebUI)
echo "[1/5] Generating Backend Core chunk (Shared + WebUI)..."
repomix \
  --include "packages/shared,packages/webui,alembic,pyproject.toml,uv.lock,CLAUDE.md,README.md,docs/ARCH.md,docs/DATABASE_ARCH.md,docs/API_ARCHITECTURE.md,docs/API_REFERENCE.md,docs/SAFE_MIGRATION_GUIDE.md" \
  --ignore "**/__pycache__/**,**/.pytest_cache/**,**/*.pyc,**/.mypy_cache/**,**/.ruff_cache/**,**/static/**" \
  --output review-chunk-1-backend-core.txt

cat >> review-chunk-1-backend-core.txt <<'EOF'

================================================================================
# CODE REVIEW PROMPT - CHUNK 1: BACKEND CORE (SHARED + WEBUI)
================================================================================

## Context
This is **Chunk 1 of 5** for reviewing Semantik, a self-hosted semantic search engine.

**Total Codebase**: ~1.8M tokens across 748 files
**This Chunk Contains** (~450K-500K tokens):
- **Shared Foundation** (`packages/shared/`)
  - Database models (SQLAlchemy ORM)
  - Repository pattern implementations
  - Chunking strategies (domain-driven design)
  - Qdrant vector DB management
  - Configuration (Pydantic settings)
- **WebUI Service** (`packages/webui/`)
  - API routers (FastAPI endpoints)
  - Service layer (business logic)
  - Celery tasks (background operations)
  - WebSocket management
  - Middleware (auth, rate limiting)
- **Database Migrations** (`alembic/`)
- **Architecture Documentation**

**Note**: This chunk contains the COMPLETE backend - models, services, and APIs together for full context.

## Critical Architecture Patterns

### 1. Three-Layer Architecture (MANDATORY)

```
┌──────────────┐
│  API Router  │ ← HTTP only, delegates to services
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Service    │ ← ALL business logic, orchestrates repos
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Repository  │ ← Data access only
└──────────────┘
```

**Violations to Flag**:
- ❌ Business logic in routers
- ❌ Direct database access in routers (no SQLAlchemy imports)
- ❌ Business logic in repositories
- ❌ Routers calling other routers
- ❌ Repositories calling other repositories

### 2. Partition-Aware Queries (CRITICAL)
The `chunks` table uses **100 LIST partitions** by `collection_id`:
- ✅ **EVERY** chunk query MUST include `collection_id` filter
- ❌ Missing `collection_id` = full scan across all 100 partitions (100x slower!)
- Check ALL methods in: `packages/shared/database/repositories/chunk_repository.py`

### 3. Transaction Order with Celery (CRITICAL)

```python
# ✅ CORRECT - Commit before dispatch
async def trigger_operation():
    operation = await repo.create(...)
    await db.commit()              # 1. Commit FIRST
    task.delay(operation.uuid)     # 2. Then dispatch
    return operation

# ❌ WRONG - Race condition!
async def trigger_operation():
    operation = await repo.create(...)
    task.delay(operation.uuid)     # Worker starts before commit!
    await db.commit()              # Too late - worker can't find record
```

### 4. Domain-Driven Chunking
- ✅ Use `packages/shared/chunking/` (strategy pattern)
- ❌ Avoid `packages/shared/text_processing/chunking/` (deprecated legacy)

## Review Checklist

### Database Models (`packages/shared/database/models/`)
- [ ] Proper relationships and foreign keys (cascading deletes)
- [ ] Indexes on frequently queried columns
- [ ] Partition keys in unique constraints
- [ ] No circular imports

### Repositories (`packages/shared/database/repositories/`)
- [ ] All methods are async
- [ ] **Partition pruning**: ALL chunk queries include `collection_id`
- [ ] No business logic (pure data access)
- [ ] No repositories calling other repositories
- [ ] Proper exception handling

### Services (`packages/webui/services/`)
- [ ] ALL business logic lives here (not in routers)
- [ ] **CRITICAL**: DB commit BEFORE Celery dispatch (check every `.delay()` call)
- [ ] Proper async/await (no blocking I/O)
- [ ] Uses repositories (not direct SQLAlchemy)
- [ ] State machine transitions validated

### API Routers (`packages/webui/api/v2/`)
- [ ] Zero business logic (just HTTP handling)
- [ ] Delegates everything to services
- [ ] No direct database access
- [ ] Proper Pydantic models for validation
- [ ] HTTP status codes correct (200, 201, 204, 400, 404, 500)

### Celery Tasks (`packages/webui/tasks.py`, `chunking_tasks.py`)
- [ ] Tasks are idempotent (safe to retry)
- [ ] Progress reporting via WebSocket
- [ ] Proper error handling
- [ ] Database sessions properly scoped (create new per task)
- [ ] Cleanup on failure

### WebSocket (`packages/webui/websocket/`, `websocket_manager.py`)
- [ ] JWT authentication enforced
- [ ] Connection limits (10/user, 10k total)
- [ ] Proper cleanup on disconnect
- [ ] Redis pub/sub for scaling

### Migrations (`alembic/versions/`)
- [ ] Reversible (down() implemented)
- [ ] Safe for production (backup strategy)
- [ ] Partition-aware scripts

## Specific Review Questions

1. **Race Conditions**: Find ALL `task.delay()` calls - is there a `db.commit()` immediately before each one?
2. **Partition Performance**: Are there any chunk queries missing `collection_id` filter?
3. **Architecture Violations**: Any business logic in routers? Any direct DB access in routers?
4. **Async Correctness**: Any blocking I/O in async functions? (file I/O, sync HTTP calls, etc.)
5. **Job → Operation**: Flag any remaining "job" terminology (should be "operation")
6. **Deprecated Code**: Any usage of `text_processing.chunking` module?
7. **State Machine**: Are collection state transitions validated? Can invalid states occur?

## Output Format

Please provide:
1. **Critical Issues** (P0): Race conditions, partition misses, security, data loss
2. **Architecture Violations** (P1): Three-layer pattern breaks, coupling
3. **Performance Concerns** (P2): Missing indexes, N+1 queries, full table scans
4. **Code Quality** (P3): Duplication, complexity, missing type hints
5. **Refactoring Opportunities**: Technical debt, deprecation candidates

For each issue:
- File path + line numbers
- Code snippet showing the problem
- Clear explanation
- Recommended fix with example

EOF

# Chunk 2: Vecpipe + Shared Context
echo "[2/5] Generating Vecpipe Service chunk (with shared context)..."
repomix \
  --include "packages/vecpipe,packages/shared,CLAUDE.md,docs/CHUNKING_IMPLEMENTATION_PLAN.md,docs/CHUNKING_FEATURE_OVERVIEW.md,docs/RERANKING.md,docs/SEARCH_SYSTEM.md" \
  --ignore "**/__pycache__/**,**/*.pyc" \
  --output review-chunk-2-vecpipe.txt

cat >> review-chunk-2-vecpipe.txt <<'EOF'

================================================================================
# CODE REVIEW PROMPT - CHUNK 2: VECPIPE SERVICE
================================================================================

## Context
This is **Chunk 2 of 5** for reviewing Semantik.

**This Chunk Contains** (~350K-400K tokens):
- **Vecpipe Service** (`packages/vecpipe/`)
  - Document parsing (PDF, HTML, MD, TXT, DOCX)
  - Embedding generation (sentence-transformers)
  - Semantic search implementation
  - Reranking (if implemented)
- **Shared Libraries** (for context)
  - Chunking strategies
  - Qdrant management
  - Database models
  - Configuration

**Note**: Shared code included for full context on how vecpipe integrates with the system.

## Critical Patterns

### 1. Stateless Service Design
Vecpipe MUST be stateless for horizontal scaling:
- [ ] No session state stored
- [ ] Thread-safe embedding model access
- [ ] GPU memory properly managed
- [ ] Resource cleanup after each request

### 2. Embedding Pipeline Optimization
- [ ] Model loaded ONCE on startup (singleton pattern)
- [ ] Batch processing (32-128 chunks at once, not one-by-one)
- [ ] GPU utilization (`device='cuda'` when available)
- [ ] Tensor cleanup after batches (`torch.cuda.empty_cache()`)
- [ ] Memory limits enforced (prevent OOM)

### 3. Qdrant Vector DB Management
- [ ] Collections created with correct vector dimensions
- [ ] Batch upserts (500-1000 vectors per batch)
- [ ] Search filters use `collection_id`
- [ ] HNSW indexing parameters optimized
- [ ] Connection pooling (don't recreate client per request)

## Review Checklist

### Document Parsing
- [ ] Support for all formats (PDF, HTML, MD, TXT, DOCX)
- [ ] Character encoding detection (chardet, UTF-8 fallback)
- [ ] Large file handling (streaming, memory limits)
- [ ] Error handling for corrupted files
- [ ] **Security**: No arbitrary code execution (safe PDF parsing)
- [ ] **Security**: Path traversal prevention

### Chunking Integration
- [ ] Uses modern `packages/shared/chunking/` strategies
- [ ] Strategy selection based on document type
- [ ] Chunk boundaries respect semantic meaning
- [ ] Metadata preserved (source, page number, etc.)
- [ ] Token/character limits enforced

### Embedding Generation
- [ ] Model cached (not reloaded per request)
- [ ] Efficient batching (what's the batch size?)
- [ ] Proper tensor cleanup (check for GPU memory leaks)
- [ ] Fallback to CPU if no GPU
- [ ] Normalization if required (L2 norm for cosine similarity)
- [ ] Model version tracked (for re-indexing)

### Search Implementation
- [ ] Vector search efficient (HNSW parameters)
- [ ] Metadata filtering works
- [ ] Hybrid search (vector + keyword) if implemented
- [ ] Result ranking and scoring
- [ ] Pagination support

### Reranking (if implemented)
- [ ] Reranker model loaded correctly
- [ ] Applied after initial vector search
- [ ] Performance impact acceptable
- [ ] See: `docs/RERANKING.md`

## Specific Questions

1. **Performance**: What's the embedding batch size? Is it optimal?
2. **Memory**: Are there GPU memory leaks? Check cleanup in long-running processes.
3. **Qdrant**: Are collections created with optimal settings? Batch size for upserts?
4. **Parsing**: Can it handle malicious files safely? Size limits enforced?
5. **Chunking**: Are strategies matched to document types correctly?
6. **Error Recovery**: What happens if Qdrant is down? Retry logic?
7. **GPU**: Is GPU utilized when available? CPU fallback working?

## Output Format

Same as Chunk 1: Critical Issues → Architecture → Performance → Code Quality → Refactoring

EOF

# Chunk 3: Frontend + Backend API Context
echo "[3/5] Generating Frontend chunk (with backend API context)..."
repomix \
  --include "apps/webui-react,packages/webui/api,packages/shared/database/models,CLAUDE.md,docs/FRONTEND_ARCH.md,docs/API_REFERENCE.md,docs/WEBSOCKET_API.md" \
  --ignore "**/node_modules/**,**/dist/**,**/build/**,**/.vite/**,**/coverage/**,**/*.test.tsx.snap,**/.turbo/**,**/__pycache__/**" \
  --output review-chunk-3-frontend.txt

cat >> review-chunk-3-frontend.txt <<'EOF'

================================================================================
# CODE REVIEW PROMPT - CHUNK 3: FRONTEND (REACT)
================================================================================

## Context
This is **Chunk 3 of 5** for reviewing Semantik.

**This Chunk Contains** (~400K-450K tokens):
- **React Frontend** (`apps/webui-react/`)
  - React 19 application
  - Zustand state management
  - React Query (TanStack Query)
  - TailwindCSS styling
  - WebSocket client
- **Backend API Context** (for understanding contracts)
  - API routers (`packages/webui/api/v2/`)
  - Database models (for type matching)
  - API documentation

**Note**: Backend API code included so you can verify frontend types match backend schemas.

## Tech Stack
- **React**: 19 with hooks
- **TypeScript**: Strict mode
- **State**: Zustand stores
- **Data Fetching**: React Query
- **Routing**: React Router v6
- **Styling**: TailwindCSS + Radix UI
- **Build**: Vite
- **Testing**: Vitest + React Testing Library + Playwright

## Critical Patterns

### 1. Optimistic Updates with Rollback

```typescript
// ✅ CORRECT
updateCollection: async (id, updates) => {
  // 1. Update UI immediately
  get().optimisticUpdateCollection(id, updates);

  try {
    // 2. Call API
    await collectionsV2Api.update(id, updates);

    // 3. Re-fetch canonical state
    await get().fetchCollectionById(id);
  } catch (error) {
    // 4. Rollback on failure
    await get().fetchCollectionById(id);
    toast.error('Update failed');
    throw error;
  }
}
```

### 2. Type Safety (Frontend ↔ Backend)
- [ ] Frontend types match backend Pydantic models
- [ ] Compare `src/types/` with `packages/shared/database/models/`
- [ ] API response types match backend schemas
- [ ] No `any` types (use `unknown` if truly unknown)

### 3. WebSocket Integration
- [ ] JWT authentication on connection
- [ ] Auto-reconnect on disconnect
- [ ] Progress updates shown in real-time
- [ ] Proper cleanup on unmount

## Review Checklist

### State Management (`src/stores/`)
- [ ] Optimistic updates implemented correctly
- [ ] Error rollback on failure
- [ ] No race conditions in store actions
- [ ] Store organized by domain

### API Client (`src/api/`)
- [ ] JWT token attached to requests
- [ ] Token refresh flow (intercept 401, refresh, retry)
- [ ] Error handling (network + API errors)
- [ ] **Type Safety**: Frontend types match backend

### Components (`src/components/`)
- [ ] Props properly typed
- [ ] Accessibility (ARIA, keyboard nav)
- [ ] Loading states shown
- [ ] Error boundaries
- [ ] Memoization where needed

### React Query
- [ ] Query keys consistent
- [ ] Cache invalidation correct
- [ ] Optimistic updates via `setQueryData`
- [ ] Error handling per query

### WebSocket
- [ ] JWT auth enforced
- [ ] Reconnection logic
- [ ] Real-time progress updates
- [ ] Cleanup on unmount

### Performance
- [ ] Code splitting (`React.lazy`)
- [ ] Virtualization for long lists
- [ ] Debouncing for search (300ms)
- [ ] Bundle size optimized

### Testing
- [ ] Unit tests for stores (Vitest)
- [ ] Component tests (React Testing Library)
- [ ] E2E tests (Playwright)
- [ ] **Coverage Target**: ≥75%

## Specific Questions

1. **Type Safety**: Do frontend types match backend Pydantic models? Check API responses.
2. **Optimistic Updates**: Are rollbacks implemented? Test error scenarios.
3. **Token Refresh**: Is JWT refresh seamless? No 401 errors for users?
4. **WebSocket**: Does reconnection work? Are progress updates shown?
5. **Performance**: Any unnecessary re-renders? Check with React DevTools Profiler.
6. **Accessibility**: Keyboard-only navigation working? Screen reader friendly?
7. **Bundle Size**: What's the main bundle size? Code splitting implemented?

## Output Format

Same as previous chunks: Critical Issues → Architecture → Performance → Code Quality → Refactoring

EOF

# Chunk 4: Tests (All Types)
echo "[4/5] Generating Tests chunk (with shared context)..."
repomix \
  --include "tests,packages/shared/database/models,pytest.ini,CLAUDE.md,docs/TESTING.md,docs/TEST_QUALITY_ACTION_PLAN.md,docs/TEST_QUALITY_TRACKING.md" \
  --ignore "**/__pycache__/**,**/.pytest_cache/**,**/htmlcov/**,**/.coverage" \
  --output review-chunk-4-tests.txt

cat >> review-chunk-4-tests.txt <<'EOF'

================================================================================
# CODE REVIEW PROMPT - CHUNK 4: TESTS (ALL TYPES)
================================================================================

## Context
This is **Chunk 4 of 5** for reviewing Semantik.

**This Chunk Contains** (~450K-500K tokens):
- **All Tests** (`tests/`)
  - Unit tests (`tests/unit/`)
  - Integration tests (`tests/integration/`)
  - E2E tests (`tests/e2e/`)
  - Security tests (`tests/security/`)
  - Performance tests (`tests/performance/`)
  - Test fixtures (`conftest.py`, `fixtures/`)
- **Models** (for understanding what's being tested)
- **Test Documentation**

**Coverage Targets**:
- Backend: ≥80%
- Frontend: ≥75%

## Recent Test Refactoring (October 2025)

Tests reorganized from monolithic structure to domain-driven:
- ✅ Organized by type (unit, integration, e2e, security, performance)
- ✅ Test quality improvements tracked in `TEST_QUALITY_ACTION_PLAN.md`
- ✅ Contract tests added (`test_contracts.py`)
- ✅ OWASP security tests added

## Critical Test Patterns

### Test Anti-Patterns to Flag

❌ **The Mockery** - Excessive mocking that doesn't test real behavior
❌ **The Giant** - Tests that test too many things at once
❌ **The Slow Poke** - Tests that use `sleep()` instead of `freezegun`
❌ **The Greedy** - Tests that modify global state
❌ **The Flickering** - Flaky tests (non-deterministic)
❌ **The Inspector** - Tests that check internal state instead of behavior

### Good Test Pattern

```python
@pytest.mark.asyncio
async def test_collection_creation_success(
    async_session,
    test_client,
    mock_celery
):
    # Arrange
    data = {"name": "Test Collection", "path": "/data"}

    # Act
    response = await test_client.post("/api/v2/collections", json=data)

    # Assert
    assert response.status_code == 201
    assert response.json()["name"] == "Test Collection"
```

## Review Checklist

### Test Organization
- [ ] Clear separation: unit vs integration vs e2e
- [ ] E2E tests marked with `@pytest.mark.e2e`
- [ ] Naming: `test_{what}_{condition}_{expected}`
- [ ] One test file per module
- [ ] Test isolation (no shared state)

### Test Quality
- [ ] Coverage ≥80% backend, ≥75% frontend
- [ ] Critical paths fully tested (happy + error paths)
- [ ] Edge cases covered
- [ ] Error scenarios tested
- [ ] Async tests properly awaited
- [ ] **No flaky tests** (run 10x to verify)
- [ ] Fast execution (<5 min for unit tests)

### Test Fixtures (`conftest.py`)
- [ ] Database setup/teardown
- [ ] Mock Celery tasks
- [ ] Mock Redis (`fakeredis`)
- [ ] Mock Qdrant
- [ ] Auth fixtures (test users, tokens)
- [ ] Proper cleanup

### Unit Tests (`tests/unit/`)
- [ ] Test single units in isolation
- [ ] Mock all dependencies
- [ ] Fast (<100ms per test)
- [ ] Test business logic thoroughly
- [ ] Test error handling

### Integration Tests (`tests/integration/`)
- [ ] Test component interactions
- [ ] Use test database (not production)
- [ ] Transactions rolled back
- [ ] API endpoints tested
- [ ] Data consistency verified

### E2E Tests (`tests/e2e/`)
- [ ] Full user workflows
- [ ] Require running services
- [ ] WebSocket functionality tested
- [ ] Background tasks verified
- [ ] Cross-service interactions

### Security Tests (`tests/security/`)
- [ ] Path traversal prevention
- [ ] SQL injection prevention
- [ ] XSS prevention
- [ ] Auth bypass attempts blocked
- [ ] Rate limiting enforced
- [ ] OWASP Top 10 covered

### Performance Tests (`tests/performance/`)
- [ ] Search latency benchmarks
- [ ] Embedding throughput
- [ ] Database query performance
- [ ] Regression detection

### Contract Tests (`test_contracts.py`)
- [ ] API contract stability
- [ ] Breaking changes detected
- [ ] Backward compatibility

## Specific Questions

1. **Coverage**: Are critical paths fully tested? Run `pytest --cov` to verify.
2. **Flaky Tests**: Any non-deterministic tests? Run tests 10x to find them.
3. **Mocking**: Are external services mocked properly? No live API calls?
4. **Performance**: Do unit tests run fast? Can they be parallelized?
5. **Security**: Are OWASP Top 10 vulnerabilities tested?
6. **E2E Coverage**: Do E2E tests cover main user workflows?
7. **CI**: Do tests pass consistently in CI? Check GitHub Actions.

## Output Format

Same as previous chunks: Critical Issues → Architecture → Performance → Code Quality → Refactoring

EOF

# Chunk 5: Infrastructure + Full Documentation
echo "[5/5] Generating Infrastructure chunk (with full documentation)..."
repomix \
  --include "docker-compose.yml,docker-compose.dev.yml,Dockerfile,Dockerfile.dev,Dockerfile.vecpipe,.dockerignore,Makefile,.github,docs,.env.example,CLAUDE.md,README.md,alembic/alembic.ini" \
  --ignore "**/__pycache__/**,**/node_modules/**,docs/api/**" \
  --output review-chunk-5-infrastructure.txt

cat >> review-chunk-5-infrastructure.txt <<'EOF'

================================================================================
# CODE REVIEW PROMPT - CHUNK 5: INFRASTRUCTURE & DOCUMENTATION (FINAL)
================================================================================

## Context
This is **Chunk 5 of 5** (FINAL) for reviewing Semantik.

**This Chunk Contains** (~400K-450K tokens):
- Docker Compose orchestration
- Dockerfiles for all services
- Makefile (development commands)
- CI/CD workflows (GitHub Actions)
- **Complete Documentation** (`docs/`)
- Environment configuration
- Alembic configuration

**This is the final chunk** - provide comprehensive cross-cutting recommendations.

## Review Checklist

### Docker Compose
- [ ] All services defined (webui, vecpipe, worker, postgres, redis, qdrant)
- [ ] Health checks configured
- [ ] Network isolation
- [ ] Volume mounts for persistence
- [ ] Environment variables from `.env`
- [ ] No hardcoded secrets
- [ ] Resource limits set
- [ ] Restart policies configured

### Dockerfiles
- [ ] Multi-stage builds
- [ ] Minimal base images (python:3.11-slim)
- [ ] Layer caching optimized
- [ ] Non-root user
- [ ] Health checks defined
- [ ] Dependencies pinned

### CI/CD (`.github/workflows/`)
- [ ] Runs on PR + main
- [ ] Linting (ruff, mypy, black, isort)
- [ ] Tests with coverage
- [ ] Docker build testing
- [ ] Security scanning
- [ ] Coverage reporting

### Makefile
- [ ] Commands documented (`make help`)
- [ ] Consistent naming
- [ ] Error handling
- [ ] Dev vs prod targets separated

### Documentation (`docs/`)
- [ ] Architecture (`ARCH.md`)
- [ ] API reference (`API_REFERENCE.md`)
- [ ] Deployment (`DEPLOYMENT.md`)
- [ ] Testing (`TESTING.md`)
- [ ] Database (`DATABASE_ARCH.md`)
- [ ] WebSocket (`WEBSOCKET_API.md`)
- [ ] Chunking (`CHUNKING_IMPLEMENTATION_PLAN.md`)
- [ ] **New docs**: `TEST_QUALITY_ACTION_PLAN.md`, `SAFE_MIGRATION_GUIDE.md`

### Monitoring & Logging
- [ ] Structured logging (JSON for production)
- [ ] Appropriate log levels
- [ ] No sensitive data logged
- [ ] Health check endpoints

## Cross-Cutting Concerns (FINAL CHUNK)

Since this is the final chunk, please provide recommendations across all previous chunks:

1. **Overall Architecture**: Does the codebase follow the three-layer pattern consistently?
2. **Job → Operation Migration**: Any remaining "job" terminology?
3. **Code Consistency**: Naming conventions consistent across services?
4. **Documentation Quality**: Is the codebase well-documented?
5. **Technical Debt**: Top 10 refactoring priorities?
6. **Security Posture**: Any vulnerabilities across all chunks?
7. **Performance**: Bottlenecks identified across all chunks?
8. **Maintainability**: Is code easy to understand? Onboarding docs sufficient?
9. **Test Coverage**: Critical gaps in coverage?
10. **Deprecated Code**: What needs to be removed? (`text_processing.chunking`, "job" terminology)

## Final Comprehensive Output

Please provide:

### 1. Critical Issues (P0) - Across All Chunks
List all P0 issues found in chunks 1-5:
- Race conditions
- Security vulnerabilities
- Data integrity risks
- Performance killers (partition misses, etc.)

### 2. Top 10 Refactoring Priorities
Ordered by impact:
1. [Highest priority refactoring]
2. ...
10. [Lowest priority refactoring]

### 3. Security Recommendations
- Vulnerabilities found
- Hardening suggestions
- OWASP compliance gaps

### 4. Performance Optimization Opportunities
- Database query optimization
- API performance
- Frontend bundle size
- Embedding/search optimization

### 5. Testing Gaps
- Critical paths not tested
- Missing test types
- Low coverage areas

### 6. Documentation Improvements
- Missing documentation
- Outdated docs
- Areas needing better explanation

### 7. Overall Code Quality Assessment
**Score**: X/10

**Justification**:
- Strengths: [what's done well]
- Weaknesses: [what needs improvement]
- Comparison to industry standards

### 8. Architectural Recommendations
Long-term improvements:
- Scalability enhancements
- Technology upgrades
- Architecture evolution

EOF

echo ""
echo "✅ All 5 chunks generated with comprehensive review prompts!"
echo ""
echo "📊 Checking sizes..."
echo ""

for file in review-chunk-*.txt; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        # Extract token count from repomix metadata
        tokens=$(grep "Total Tokens:" "$file" | head -1 | awk '{print $3}' | tr -d ',')
        if [ -z "$tokens" ]; then
            # Fallback to word count estimation
            words=$(wc -w < "$file")
            tokens=$(echo "$words * 1.3" | bc 2>/dev/null | cut -d'.' -f1)
            if [ -z "$tokens" ]; then
                tokens="N/A"
            fi
        fi

        echo "📄 $(basename $file)"
        echo "   Size: $size | Tokens: $tokens"

        # Show status relative to 500K target
        if [ "$tokens" != "N/A" ]; then
            if [ "$tokens" -gt 600000 ]; then
                echo "   ⚠️  Over 600K - consider splitting"
            elif [ "$tokens" -gt 500000 ]; then
                echo "   ✅ Good size (over 500K target)"
            elif [ "$tokens" -gt 400000 ]; then
                echo "   ✅ Acceptable (400K-500K)"
            else
                echo "   ℹ️  Under 400K - could add more context"
            fi
        fi
        echo ""
    fi
done

echo "✅ Done! Review chunks ready for LLM analysis."
echo ""
echo "📋 Chunk Organization (5 Chunks with Overlap):"
echo ""
echo "1️⃣  Backend Core (~450K-500K tokens)"
echo "    • packages/shared (models, repos, chunking, config)"
echo "    • packages/webui (API, services, tasks, websocket)"
echo "    • alembic (migrations)"
echo "    • Architecture docs"
echo ""
echo "2️⃣  Vecpipe Service (~350K-400K tokens)"
echo "    • packages/vecpipe (embeddings, search, parsing)"
echo "    • packages/shared (INCLUDED for context)"
echo "    • Chunking & search documentation"
echo ""
echo "3️⃣  Frontend (~400K-450K tokens)"
echo "    • apps/webui-react (React app, stores, components)"
echo "    • packages/webui/api (INCLUDED to verify type contracts)"
echo "    • packages/shared/database/models (INCLUDED for types)"
echo "    • Frontend & API documentation"
echo ""
echo "4️⃣  Tests (~450K-500K tokens)"
echo "    • tests/ (unit, integration, e2e, security, performance)"
echo "    • packages/shared/database/models (INCLUDED for context)"
echo "    • Test documentation & quality plans"
echo ""
echo "5️⃣  Infrastructure (~400K-450K tokens)"
echo "    • Docker, CI/CD, Makefile"
echo "    • Complete documentation (docs/)"
echo "    • Provides final cross-cutting analysis"
echo ""
echo "✨ Key Features:"
echo "   • Larger chunks (~500K tokens) for better context"
echo "   • Intentional overlap (shared code appears in multiple chunks)"
echo "   • Each chunk can be reviewed independently"
echo "   • CLAUDE.md included in every chunk"
echo ""
echo "🎯 Send each chunk to your LLM with the embedded review prompt!"
