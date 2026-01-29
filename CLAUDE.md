# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Semantik is a self-hosted semantic search engine for documents. It runs locally with no data leaving your machine. The system uses FastAPI + Celery workers + a dedicated embedding/search service, backed by PostgreSQL, Redis, and Qdrant.

## Commands

### Development
```bash
make dev-install          # Install dev dependencies with uv
make run                  # FastAPI hot reload on :8080
make dev                  # Full integrated stack (API + worker + vecpipe)
make dev-local            # Run webui locally with Docker services

make frontend-install     # npm install for React frontend
make frontend-dev         # Vite dev server on :5173
make frontend-build       # Build frontend for production
```

### Testing
```bash
make test                 # Run all tests
make test-ci              # Tests excluding E2E (for CI)
make test-e2e             # E2E tests only (requires running stack)
make test-coverage        # Tests with coverage report

# Single test file
uv run pytest tests/unit/test_specific.py

# Single test function
uv run pytest tests/unit/test_specific.py::test_function_name

# Tests by marker
uv run pytest -m "not e2e"
uv run pytest -m integration

# Frontend tests
make frontend-test
cd apps/webui-react && npm test
```

### Linting & Type Checking
```bash
make lint                 # ruff check
make type-check           # mypy
make format               # black + isort
make check                # lint + type-check + test
```

### Docker
```bash
make wizard               # Interactive setup wizard
make docker-up            # Start all services
make docker-down          # Stop (keep volumes)
make docker-down-clean    # Stop and remove volumes
make docker-dev-up        # Backend services only (for local webui dev)
```

### Database
```bash
uv run alembic upgrade head      # Run migrations
uv run alembic downgrade -1      # Rollback one migration
uv run alembic current           # Check current revision
docker compose --profile testing up -d postgres_test  # Start test database
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          React Frontend (apps/webui-react/)             │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ HTTP/WebSocket
┌────────────────────────────────┴────────────────────────────────────────┐
│                        WebUI (packages/webui/) - Port 8080              │
│   Auth (JWT) │ Collection/Operation APIs │ Search Proxy │ WebSockets   │
└────────┬────────────────────────────────────────┬───────────────────────┘
         │                                        │
         │  ┌─────────────────────────────────────┴─────────────────────┐
         │  │                 Shared (packages/shared/)                 │
         │  │  Database Repos │ Plugin System │ Embedding │ Chunking    │
         │  └───────────────────────────────────────────────────────────┘
         │                                        │
┌────────┴─────────┐                ┌─────────────┴─────────────────────┐
│    PostgreSQL    │                │     VecPipe (packages/vecpipe/)   │
│    (metadata)    │                │          Port 8000                │
└──────────────────┘                │  Search │ Embedding │ Reranking   │
                                    └─────────────┬─────────────────────┘
         ┌──────────────┐                        │
         │    Redis     │                        │
         │  (WebSocket  │           ┌────────────┴────────────────────┐
         │   + Celery)  │           │           Qdrant                │
         └──────────────┘           │      (vector storage)           │
                                    └─────────────────────────────────┘
```

### Package Roles

- **packages/webui/**: FastAPI control plane. Owns PostgreSQL. Handles auth, collection management, operation orchestration, and proxies search to vecpipe.
- **packages/vecpipe/**: Headless embedding/search service. Talks directly to Qdrant. Manages GPU memory for model loading.
- **packages/shared/**: Common utilities - database models/repos, plugin system, chunking strategies, embedding providers, connectors.
- **apps/webui-react/**: React 19 + Vite + TypeScript frontend. Uses Zustand for state, TanStack Query for data fetching.

### Key Patterns

**Collection-Centric Design**: Collections are the primary unit. Each has its own embedding model, chunking config, and sources. Operations (INDEX, APPEND, REINDEX, DELETE) run async via Celery.

**Repository Pattern**: All database access through async repositories in `shared/database/repositories/`. Session management via `get_async_session()`.

**Plugin System**: Extensible via entry points (`semantik.plugins`). Plugin types: embedding, chunking, connector, reranker, sparse_indexer, extractor.

**Chunk Partitioning**: The `chunks` table uses 100 LIST partitions. **Always include `collection_id` in chunk queries** for partition pruning.

### LLM Integration

Per-user LLM configuration with quality tiers for features like HyDE search and document summarization.

**Providers**:
- **anthropic**: Claude models via Anthropic API
- **openai**: GPT models via OpenAI API
- **local**: Local GPU inference via VecPipe (no external API calls)

**Quality Tiers**:
- **HIGH**: Complex tasks (summarization, entity extraction) - uses best models
- **LOW**: Simple tasks (HyDE query expansion, keywords) - uses fast/cheap models

**Local Provider Architecture**:
LocalLLMProvider is an HTTP client to VecPipe (not direct model loading):
```
WebUI → LocalLLMProvider → HTTP → VecPipe → LLMModelManager → GPU
```
This mirrors the embedding pattern where GPU memory is managed by VecPipe's GPUMemoryGovernor.

**Environment Variables**:
- `CONNECTOR_SECRETS_KEY`: Required for API key encryption (Fernet)
- `ENABLE_LOCAL_LLM`: Enable local LLM inference in VecPipe (default: true)
- `DEFAULT_LLM_QUANTIZATION`: int4, int8, or float16 (default: int8)

**Usage Pattern**:
```python
from shared.llm.factory import LLMServiceFactory
from shared.llm.types import LLMQualityTier

factory = LLMServiceFactory(session)
provider = await factory.create_provider_for_tier(user_id, LLMQualityTier.LOW)
async with provider:
    response = await provider.generate(prompt="...", max_tokens=256)
```

**Error Handling**:
- `LLMNotConfiguredError`: User hasn't set up LLM settings
- `LLMAuthenticationError`: Invalid or missing API key
- `LLMRateLimitError`: Provider rate limit (retryable)
- `LLMProviderError`: General provider error (connection, GPU OOM)
- `LLMTimeoutError`: Request timed out
- `LLMContextLengthError`: Input exceeds model context window

**Database Tables**:
- `llm_provider_configs`: Per-user tier configuration (provider + model per tier)
- `llm_provider_api_keys`: Encrypted API keys (per provider, not needed for local)
- `llm_usage_events`: Token usage tracking

### Data Flow

1. User creates collection → PostgreSQL record + Qdrant collection
2. User adds source → Operation created → Celery task queued
3. Worker: extract → chunk → embed (via vecpipe) → upsert to Qdrant
4. Progress updates via Redis → WebSocket → UI

## Critical Implementation Details

### Chunk Queries Must Include collection_id
```python
# CORRECT - partition pruning enabled
select(Chunk).where(Chunk.collection_id == collection_id, Chunk.document_id == doc_id)

# WRONG - scans all 100 partitions
select(Chunk).where(Chunk.document_id == doc_id)
```

### Embedding Mode Matters
```python
from shared.embedding.types import EmbeddingMode
# Use QUERY mode for search, DOCUMENT mode for indexing
query_embeddings = await provider.embed_texts(queries, mode=EmbeddingMode.QUERY)
doc_embeddings = await provider.embed_texts(docs, mode=EmbeddingMode.DOCUMENT)
```

### Async Session Commits
```python
async with get_async_session() as session:
    repo = CollectionRepository(session)
    await repo.create(...)
    await session.commit()  # Don't forget!
```

### Factory Fixtures Require Owner
```python
# In tests, always pass owner_id explicitly
await collection_factory(owner_id=test_user_db.id)
await operation_factory(user_id=test_user_db.id)
```

## Test Organization

```
tests/
├── unit/           # Isolated, mocked dependencies
├── integration/    # Real infrastructure (DB, Redis, Qdrant)
├── e2e/            # Full workflows, requires running stack
├── security/       # OWASP patterns (path traversal, XSS, ReDoS)
├── performance/    # Load and benchmark tests
├── webui/          # WebUI-specific API and service tests
└── shared/         # Shared package tests
```

Key fixtures in `conftest.py`:
- `db_session`: Real async PostgreSQL with rollback
- `test_client` / `async_client`: FastAPI test clients with mocked auth
- `collection_factory` / `document_factory` / `operation_factory`: Entity factories
- `use_fakeredis`: Opt-in Redis mocking

## Environment Variables

Key settings (see `.env.docker.example`):
- `DATABASE_URL`: PostgreSQL connection
- `REDIS_URL`: Redis for Celery/WebSocket
- `QDRANT_HOST/PORT`: Vector database
- `JWT_SECRET_KEY`: Auth secret (generate with `openssl rand -hex 32`)
- `USE_MOCK_EMBEDDINGS`: Skip GPU for testing
- `DEFAULT_EMBEDDING_MODEL`: Model for new collections
- `CONNECTOR_SECRETS_KEY`: Fernet key for credential encryption

## LSP Tool Usage

### When to Use LSP vs Grep

| Task | Tool |
|------|------|
| Find string literals, comments, log messages | Grep |
| Find files by name/pattern | Glob |
| Regex pattern matching | Grep |
| Find where a symbol is defined | LSP `goToDefinition` |
| Find all usages of a symbol | LSP `findReferences` |
| Understand types/documentation | LSP `hover` |
| Trace call hierarchy | LSP `incomingCalls` / `outgoingCalls` |
| Find interface implementations | LSP `goToImplementation` |
| List symbols in a file | LSP `documentSymbol` |
| Search symbols across workspace | LSP `workspaceSymbol` |

### LSP Operations Reference

Available operations:
- `goToDefinition` - Jump to where a symbol is defined
- `findReferences` - Find all places a symbol is used
- `hover` - Get type info and documentation
- `documentSymbol` - List all symbols in a file
- `workspaceSymbol` - Search symbols across the project
- `goToImplementation` - Find implementations of interfaces/abstract methods
- `prepareCallHierarchy` - Get call hierarchy item at position
- `incomingCalls` - Find what calls this function
- `outgoingCalls` - Find what this function calls

**Not available:** `rename`, `signatureHelp`, `typeDefinition`, `diagnostics`

### Common Patterns

```
"What is this?"           → hover → goToDefinition if needed
"Where is this used?"     → findReferences
"What calls this?"        → incomingCalls
"What does this call?"    → outgoingCalls
"What implements this?"   → goToImplementation
"What's in this file?"    → documentSymbol
```

### Refactoring Procedure

1. **Anchor:** `goToDefinition` to confirm the symbol location
2. **Impact:** `findReferences` to identify all usages
3. **Change:** Use Edit tool (no LSP rename available)
4. **Validate:** Run repo typecheck/tests (no LSP diagnostics)

### Position Format

- Line and character are **1-based** (as shown in editors)
- Get positions from Read tool output (line numbers on left) or prior LSP results

### Efficiency Tips

- Summarize key findings (symbol → file:line, ref count, top callers) to avoid repeat calls
- Don't over-verify obvious symbols from context
