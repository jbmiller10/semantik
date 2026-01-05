# VecPipe Service Architecture

> **Location:** `packages/vecpipe/`

## Overview

VecPipe is the vector search and embedding service. It handles:
- Query and document embedding generation
- Vector similarity search against Qdrant
- Hybrid + keyword search (keyword extraction + Qdrant text filters + weighted scoring)
- Optional cross-encoder reranking
- Model lifecycle management (including memory governor integration)

## Key Modules

### search/app.py + search/router.py
FastAPI app and HTTP routes.

**Primary endpoints (protected by `X-Internal-Api-Key` unless noted):**
- `GET /health` – health check
- `GET /model/status` – model manager status
- `POST /search` and `GET /search` – semantic/question/code/hybrid search
- `POST /search/batch` – batch search
- `GET /hybrid_search` – hybrid search
- `GET /keyword_search` – keyword-only search
- `POST /embed` – embed texts
- `POST /upsert` – upsert vectors
- `GET /models` / `POST /models/load` – list/load models

### search/service.py
Core search pipeline and request handling.

Key behaviors:
- `perform_search()` routes `search_type=hybrid` to hybrid search and applies reranking if enabled.
- `SEARCH_INSTRUCTIONS` provides per-type embedding instructions (semantic/question/code).
- Collection resolution supports explicit collection names or `operation_uuid`.
- `filters` (Qdrant filter dicts) are passed directly to Qdrant when provided.

### embed_chunks_unified.py
CLI utility for offline embedding generation and parquet output.

### hybrid_search.py
Keyword extraction + hybrid scoring logic used by hybrid and keyword searches.

### reranker.py
Cross-encoder reranking implementation (Qwen3-Reranker).

### model_manager.py + governed_model_manager.py
Model lifecycle, load/unload, and memory-governed eviction for embedding and reranking models.

## API Contracts
- `SearchRequest` / `SearchResponse` live in `packages/shared/contracts/search.py`.
- Supported `search_type`: `semantic`, `question`, `code`, `hybrid` (plus legacy alias `vector`).
- `filters` is a raw Qdrant filter dict passed through to the search call.

## Security
- Protected endpoints require `X-Internal-Api-Key` (see `packages/vecpipe/search/router.py`).
- The key is generated/validated at startup (see `packages/vecpipe/search/lifespan.py`).

## Error Handling Notes
- Dimension mismatch returns `400` with expected vs actual dimensions.
- Qdrant API errors surface as `502/503` depending on call path.
