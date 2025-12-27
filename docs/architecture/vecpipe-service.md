# VecPipe Service Architecture

> **Location:** `packages/vecpipe/`

## Overview

VecPipe is the vector search and embedding service. It handles:
- Query and document embedding generation
- Vector similarity search against Qdrant
- Hybrid (keyword + semantic) search
- Cross-encoder reranking
- GraphRAG (graph-enhanced retrieval)

## Key Modules

### embed_chunks_unified.py
Unified embedding generation for all documents.

**Batch Sizes by Model:**
| Model Size | Quantization | Batch Size | VRAM |
|------------|--------------|------------|------|
| 0.6B | int8 | 256 | ~0.6 GB |
| 0.6B | float16 | 128 | ~1.2 GB |
| 4B | float32 | 16 | ~16 GB |
| 8B | float16 | 8 | ~16 GB |

```python
async def embed_chunks(
    texts: list[str],
    model_name: str,
    mode: EmbeddingMode = EmbeddingMode.DOCUMENT
) -> list[list[float]]:
    """Generate embeddings for text chunks."""
    provider = EmbeddingProviderFactory.create_provider(model_name)
    return await provider.embed_texts(texts, mode=mode)
```

### search_api.py
FastAPI search endpoints.

**Endpoints:**
- `POST /search` - Vector/hybrid search
- `POST /hybrid_search` - Explicit hybrid search
- `GET /health` - Health check

**Search Flow:**
```
1. Receive search request
2. Resolve collection UUID → Qdrant collection name
3. If hybrid: perform_hybrid_search()
4. Embed query with same model as indexing
5. Search Qdrant for top-k vectors
6. Apply score_threshold filter (BEFORE reranking)
7. Optional: Rerank with cross-encoder
8. Return results with scores
```

### reranker.py
Cross-encoder reranking for improved result quality.

**Strategy:**
- 5x candidate multiplication (request 50 for top_k=10)
- Bounds: 20-200 candidates
- Models: Qwen3-Reranker variants

```python
async def rerank_results(
    query: str,
    results: list[SearchResult],
    top_k: int,
    model: str = "Qwen/Qwen3-Reranker-0.6B"
) -> list[SearchResult]:
    """Rerank search results with cross-encoder."""
    # Score each (query, result.content) pair
    scores = await cross_encoder.score(
        [(query, r.content) for r in results]
    )
    # Sort by reranked score
    reranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [r for r, _ in reranked[:top_k]]
```

### model_manager.py
Model lifecycle management.

**Features:**
- Lazy loading on first use
- Auto-unload after 300s idle
- GPU memory checking before load
- CPU fallback if GPU unavailable

```python
class ModelManager:
    def __init__(self):
        self._models: dict[str, Model] = {}
        self._last_used: dict[str, float] = {}
        self._unload_after_seconds = 300

    async def get_model(self, model_name: str) -> Model:
        if model_name not in self._models:
            await self._check_gpu_memory()
            self._models[model_name] = await self._load_model(model_name)
        self._last_used[model_name] = time.time()
        return self._models[model_name]
```

### graphrag/
Graph-enhanced retrieval for relationship-aware search.

**Components:**
- Entity extraction (spaCy NER)
- Relationship extraction
- Graph storage (Neo4j or in-memory)
- Graph-augmented search

## API Contracts

### Search Request
```python
class SearchRequest(BaseModel):
    query: str
    collection: str | None = None
    collection_uuids: list[str] | None = None
    operation_uuid: str | None = None
    top_k: int = 10
    score_threshold: float = 0.0
    search_type: str = "semantic"  # semantic, hybrid, question, code
    use_reranker: bool = False
    rerank_model: str | None = None
    hybrid_alpha: float = 0.5  # 0=keyword, 1=semantic
```

### Search Response
```python
class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
    search_type: str
    reranking_used: bool
    reranking_time_ms: float | None
```

### SearchResult
```python
class SearchResult(BaseModel):
    doc_id: str
    chunk_id: str
    content: str
    score: float
    file_path: str
    file_name: str
    chunk_index: int
    total_chunks: int
    collection_id: str | None
    collection_name: str | None
    original_score: float | None  # Pre-reranking
    reranked_score: float | None  # Post-reranking
```

## Collection Resolution

**Priority Order:**
1. Explicit `collection` parameter
2. `operation_uuid` → database lookup → `collection.vector_store_name`
3. `DEFAULT_COLLECTION` from settings

**Error Handling:**
- If `operation_uuid` provided but not found: HTTP 404
- If no collection resolved: HTTP 400

## Error Handling

| Scenario | HTTP Status | Error |
|----------|-------------|-------|
| Dimension mismatch | 400 | Clear error message |
| GPU memory exhausted | 507 | Insufficient Storage |
| Search timeout | 504 | Gateway Timeout |
| Collection not found | 404 | Not Found |

## Performance Targets

| Operation | Target |
|-----------|--------|
| Single search | <500ms p95 |
| Batch embedding | >100 texts/second |
| Reranking (100 candidates) | <200ms |

## Extension Points

### Adding a New Search Type
1. Add to `SearchType` enum
2. Handle in search endpoint
3. Apply type-specific preprocessing
4. Update frontend selector

### Adding a New Reranking Model
1. Add to model registry
2. Ensure compatible tokenizer
3. Add quantization support if needed
4. Update model_manager loading logic
