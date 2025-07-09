# Semantik Codebase Dependency Analysis

## 1. Component Dependencies

### Cross-Package Dependencies (High Risk)

1. **`webui.embedding_service` → Used by both packages**
   - `vecpipe/model_manager.py` imports from `webui.embedding_service`
   - `vecpipe/search_api.py` imports from `webui.embedding_service`
   - This is the most critical shared component - changes here affect both packages

2. **`vecpipe.config` → Used by webui**
   - `webui/auth.py` imports `vecpipe.config.settings`
   - `webui/database.py` imports `vecpipe.config.settings`
   - Configuration changes affect both packages

3. **Database Access Patterns**
   - `vecpipe/cleanup.py` directly accesses webui's SQLite database
   - This creates a direct dependency on webui's database schema

### High-Impact Components (Used by Many Others)

1. **`embedding_service.py`** (CRITICAL)
   - Used by: vecpipe search/model management, webui job processing
   - External deps: torch, sentence-transformers, transformers, CUDA
   - Testing considerations: Requires heavy mocking, GPU availability checks

2. **`database.py`** (webui)
   - Used by: All webui API endpoints
   - External deps: SQLite
   - Testing considerations: Needs in-memory database for isolation

3. **`qdrant_manager.py`**
   - Used by: Jobs API, search operations
   - External deps: Qdrant vector database
   - Testing considerations: Requires Qdrant mock or test instance

4. **`model_manager.py`** (vecpipe)
   - Manages GPU memory and model lifecycle
   - External deps: GPU memory management, embedding service
   - Testing considerations: Complex GPU state management

### Integration Points Requiring Integration Testing

1. **Search API Proxy**
   - `webui/api/search.py` → HTTP calls to → `vecpipe/search_api.py`
   - Requires both services running or careful mocking

2. **WebSocket Connections**
   - `webui/api/jobs.py` - Job progress updates
   - `webui/api/files.py` - File scanning progress
   - Testing considerations: Async testing, connection lifecycle

3. **File Processing Pipeline**
   - File scanning → Text extraction → Chunking → Embedding → Vector storage
   - Involves: files.py, jobs.py, extract_chunks.py, embedding_service.py, Qdrant
   - Testing considerations: End-to-end integration tests needed

## 2. External Dependencies

### Critical External Services
1. **Qdrant** - Vector database
   - Used in: search operations, job processing, cleanup
   - Testing: Mock or containerized test instance

2. **CUDA/GPU**
   - Used in: embedding_service.py, model_manager.py
   - Testing: Need CPU fallback, mock GPU operations

3. **File System**
   - Used in: File scanning, job outputs, temp files
   - Testing: Use temp directories, mock file operations

## 3. Circular Dependencies
- No direct circular imports detected
- However, tight coupling exists between vecpipe and webui through shared components

## 4. Testing Gaps Analysis

### Modules WITHOUT Unit Tests:
1. **vecpipe/**
   - `cleanup.py` - No tests for cleanup service
   - `config.py` - Configuration module
   - `embed_chunks_unified.py` - Embedding logic
   - `hybrid_search.py` - Hybrid search implementation
   - `ingest_qdrant.py` - Data ingestion
   - `memory_utils.py` - Memory management utilities
   - `search_api.py` - Main search API (only integration tests exist)
   - `search_utils.py` - Search utilities
   - `validate_search_setup.py` - Validation utilities
   - `qwen3_search_config.py` - Model-specific config

2. **webui/**
   - `app.py` - Application entry point
   - `main.py` - FastAPI app configuration
   - `rate_limiter.py` - Rate limiting logic
   - `schemas.py` - Pydantic schemas
   - **api/** endpoints missing tests:
     - `collection_metadata.py`
     - `collections.py`
     - `files.py` (partial coverage)
     - `metrics.py`
     - `models.py`
     - `root.py`
     - `search.py`
     - `settings.py`
   - **utils/**
     - `qdrant_manager.py`
     - `retry.py`

### Modules WITH Tests:
1. **Unit Tests:**
   - `test_database.py` → `webui/database.py`
   - `test_extract_chunks.py` → `vecpipe/extract_chunks.py`
   - `test_model_manager.py` → `vecpipe/model_manager.py`
   - `test_embedding_service.py` → `webui/embedding_service.py`
   - `test_reranker.py` → `vecpipe/reranker.py`

2. **Integration Tests:**
   - `test_auth_api.py` → Auth flow
   - `test_jobs_api.py` → Job creation flow
   - `test_search.py` → Search functionality

## 5. Optimal Testing Order

### Phase 1: Foundation Components (No Dependencies)
1. `vecpipe/config.py` - Pure configuration
2. `webui/schemas.py` - Data models
3. `webui/rate_limiter.py` - Standalone utility
4. `vecpipe/memory_utils.py` - Utility functions
5. `webui/utils/retry.py` - Utility decorator

### Phase 2: Core Services (Mock External Deps)
1. `webui/database.py` ✓ (already tested)
2. `vecpipe/extract_chunks.py` ✓ (already tested)
3. `webui/embedding_service.py` ✓ (already tested)
4. `vecpipe/model_manager.py` ✓ (already tested)

### Phase 3: Service Layer (Mock Dependencies)
1. `webui/utils/qdrant_manager.py` - Mock Qdrant client
2. `vecpipe/search_utils.py` - Mock Qdrant operations
3. `vecpipe/hybrid_search.py` - Mock search components
4. `vecpipe/cleanup.py` - Mock database and Qdrant

### Phase 4: API Endpoints (Mock Service Layer)
1. `webui/api/models.py` - Model listing endpoint
2. `webui/api/settings.py` - Settings endpoint
3. `webui/api/collections.py` - Collection management
4. `webui/api/files.py` - File operations
5. `webui/api/search.py` - Search proxy endpoint

### Phase 5: Integration Tests
1. File processing pipeline (scan → extract → chunk → embed)
2. Search pipeline (query → embed → search → rerank)
3. WebSocket communication (job updates, file scanning)
4. Authentication flow with all endpoints
5. End-to-end job creation and search

## 6. Testing Recommendations

### High Priority (Core Functionality):
1. `vecpipe/search_api.py` - Main search service
2. `webui/api/search.py` - Search proxy
3. `webui/utils/qdrant_manager.py` - Vector DB operations
4. `vecpipe/cleanup.py` - Data consistency

### Medium Priority (Important Features):
1. `webui/api/files.py` - File management
2. `webui/api/collections.py` - Collection management
3. `vecpipe/hybrid_search.py` - Advanced search
4. WebSocket handlers - Real-time updates

### Low Priority (Utilities/Config):
1. Configuration modules
2. Simple utility functions
3. Schema definitions

### Testing Strategies:
1. **Mock Heavy Components**: GPU operations, ML models, external services
2. **Use Test Doubles**: In-memory SQLite, mock Qdrant client
3. **Isolate Network Calls**: Mock HTTP clients for API proxying
4. **Async Testing**: Use pytest-asyncio for async endpoints
5. **Fixtures**: Reusable test data, mock services, temp directories