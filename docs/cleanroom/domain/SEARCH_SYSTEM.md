# SEARCH_SYSTEM - Cleanroom Documentation

## 1. Component Overview

The SEARCH_SYSTEM is a sophisticated semantic search infrastructure that provides high-performance vector similarity search, hybrid search capabilities, and intelligent result reranking. It serves as the core retrieval mechanism for the Semantik application, enabling users to find relevant information across their indexed collections using natural language queries.

### Core Capabilities
- **Multi-Collection Search**: Search across up to 10 collections simultaneously with result aggregation
- **Hybrid Search**: Combines vector similarity with keyword matching for improved recall
- **Cross-Encoder Reranking**: Uses Qwen3-Reranker models for state-of-the-art relevance scoring
- **Query Type Optimization**: Specialized handling for semantic, question-answering, and code searches
- **Real-time Performance**: Optimized for low-latency responses with adaptive batch processing
- **Fault Tolerance**: Graceful degradation with partial failure handling and timeout retry logic

### Service Components
- **Search Service** (`packages/webui/services/search_service.py`): Business logic orchestration
- **Search API** (`packages/webui/api/v2/search.py`): RESTful endpoints for search operations
- **VecPipe Search** (`packages/vecpipe/search_api.py`): Vector search engine and embedding service
- **Hybrid Search Engine** (`packages/vecpipe/hybrid_search.py`): Keyword and vector fusion
- **Cross-Encoder Reranker** (`packages/vecpipe/reranker.py`): Neural reranking for relevance
- **Frontend Store** (`apps/webui-react/src/stores/searchStore.ts`): Client-side state management

## 2. Architecture & Design Patterns

### Layered Architecture

```
┌─────────────────────────────────────────────────┐
│            Frontend (React/TypeScript)           │
│         searchStore.ts / searchValidation.ts     │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│              Search API Layer (v2)               │
│    /api/v2/search (FastAPI Routers)              │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│           Search Service Layer                   │
│  SearchService (Business Logic & Orchestration)  │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│            VecPipe Search Engine                 │
│   Embedding Service │ Hybrid Search │ Reranker   │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│              Vector Database                     │
│                  Qdrant                          │
└─────────────────────────────────────────────────┘
```

### Search Pipeline Architecture

```
Query → Validation → Embedding Generation → Vector Search
                                              ↓
Result Aggregation ← Reranking ← Candidate Retrieval
        ↓
Response Formatting → Client
```

### Design Patterns

1. **Repository Pattern**: Collection access through `CollectionRepository`
2. **Service Layer Pattern**: Business logic isolated in `SearchService`
3. **Async/Await Pattern**: Non-blocking I/O for concurrent searches
4. **Factory Pattern**: Service instantiation via `get_search_service()`
5. **Strategy Pattern**: Different search types (semantic, hybrid, keyword)
6. **Chain of Responsibility**: Request processing through middleware layers

## 3. Key Interfaces & Contracts

### Search Request Schema

```python
# packages/webui/api/v2/schemas.py
class CollectionSearchRequest(BaseModel):
    collection_uuids: list[str]  # 1-10 collection UUIDs
    query: str                    # 1-1000 characters
    k: int = 10                   # 1-100 results
    search_type: str = "semantic" # semantic|question|code|hybrid
    use_reranker: bool = True     # Enable neural reranking
    rerank_model: str | None      # Override reranker model
    score_threshold: float = 0.0  # 0.0-1.0 minimum score
    metadata_filter: dict | None  # Qdrant filter conditions
    include_content: bool = True  # Include document content
    # Hybrid search parameters
    hybrid_alpha: float = 0.7     # Vector vs keyword weight
    hybrid_mode: str = "rerank"   # filter|rerank
    keyword_mode: str = "any"     # any|all keyword matching
```

### Search Response Schema

```python
class CollectionSearchResponse(BaseModel):
    query: str
    results: list[CollectionSearchResult]
    total_results: int
    collections_searched: list[dict]
    search_type: str
    reranking_used: bool
    reranker_model: str | None
    # Timing metrics
    embedding_time_ms: float | None
    search_time_ms: float
    reranking_time_ms: float | None
    total_time_ms: float
    # Failure information
    partial_failure: bool
    failed_collections: list[dict] | None
```

### VecPipe Search Contract

```python
# shared/contracts/search.py
class SearchRequest(BaseModel):
    query: str
    k: int  # Aliased as "top_k"
    collection: str | None
    model_name: str | None
    quantization: str | None
    search_type: str = "semantic"
    filters: dict | None
    include_content: bool = False
    use_reranker: bool = False
    rerank_model: str | None
    score_threshold: float = 0.0
    hybrid_alpha: float = 0.7
    hybrid_search_mode: str = "weighted"
```

## 4. Data Flow & Dependencies

### Search Request Flow

1. **Frontend Initiation**
   ```typescript
   // apps/webui-react/src/stores/searchStore.ts
   validateAndUpdateSearchParams() → API call
   ```

2. **API Layer Processing**
   ```python
   # packages/webui/api/v2/search.py
   @router.post("")
   async def multi_collection_search():
       - Rate limiting (30/minute)
       - User authentication
       - Request validation
       - Service delegation
   ```

3. **Service Layer Orchestration**
   ```python
   # packages/webui/services/search_service.py
   async def multi_collection_search():
       - Collection access validation
       - Parallel search execution
       - Result aggregation
       - Error handling
   ```

4. **VecPipe Processing**
   ```python
   # packages/vecpipe/search_api.py
   async def search_post():
       - Query embedding generation
       - Dimension validation
       - Vector search execution
       - Optional reranking
   ```

### Embedding Generation Flow

```
Query Text → Instruction Selection → Model Loading
    ↓              ↓                      ↓
Tokenization → Embedding Model → Vector Output
    ↓              ↓                      ↓
Normalization → Dimension Check → Search Ready
```

### Reranking Pipeline

```
Initial Results (k * multiplier candidates)
            ↓
    Document Extraction
            ↓
    Cross-Encoder Scoring
            ↓
    Score-based Reordering
            ↓
    Top-k Selection
```

## 5. Critical Implementation Details

### Vector Search Implementation

```python
# packages/vecpipe/search_utils.py
async def search_qdrant(
    qdrant_host: str,
    qdrant_port: int,
    collection_name: str,
    query_vector: list[float],
    k: int,
    with_payload: bool = True
) -> list[dict]:
    client = AsyncQdrantClient(url=f"http://{qdrant_host}:{qdrant_port}")
    results = await client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=k,
        with_payload=with_payload
    )
    return [{"id": point.id, "score": point.score, 
             "payload": point.payload} for point in results]
```

### Hybrid Search Fusion

```python
# packages/vecpipe/hybrid_search.py
class HybridSearchEngine:
    def hybrid_search(self, query_vector, query_text, limit, mode):
        if mode == "filter":
            # Qdrant filter-based approach
            keywords = self.extract_keywords(query_text)
            text_filter = self.build_text_filter(keywords)
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=text_filter,
                limit=limit
            )
        else:  # mode == "rerank"
            # Post-processing fusion
            candidates = self.client.search(limit=limit * 3)
            # Score fusion: 0.7 * vector_score + 0.3 * keyword_score
            return self.rerank_by_keywords(candidates, keywords)
```

### Reranking with Cross-Encoders

```python
# packages/vecpipe/reranker.py
class CrossEncoderReranker:
    def compute_relevance_scores(self, query, documents):
        # Format inputs for Qwen3-Reranker
        inputs = [f"<Instruct>: {instruction}\n"
                  f"<Query>: {query}\n"
                  f"<Document>: {doc}" for doc in documents]
        
        # Get yes/no token probabilities
        outputs = self.model(**encoded)
        yes_logits = last_token_logits[:, yes_token_id]
        no_logits = last_token_logits[:, no_token_id]
        
        # Compute P(yes) as relevance score
        yes_no_logits = torch.stack([no_logits, yes_logits], dim=1)
        probs = torch.nn.functional.softmax(yes_no_logits, dim=1)
        return probs[:, 1].cpu().tolist()  # P(yes)
```

### Adaptive Timeout Handling

```python
# packages/webui/services/search_service.py
async def search_single_collection(self, collection, query, k, search_params):
    try:
        # Initial attempt with default timeout (30s)
        response = await client.post(url, json=params, timeout=self.default_timeout)
    except httpx.ReadTimeout:
        # Retry with extended timeout (4x multiplier)
        extended_timeout = httpx.Timeout(
            timeout=120.0,
            connect=20.0,
            read=120.0,
            write=20.0
        )
        response = await client.post(url, json=params, timeout=extended_timeout)
```

## 6. Security Considerations

### Input Validation

```typescript
// apps/webui-react/src/utils/searchValidation.ts
export function validateQuery(query: string): ValidationError | null {
    const trimmed = query.trim();
    if (trimmed.length > 1000) {
        return { field: 'query', 
                 message: 'Query must not exceed 1000 characters' };
    }
    // Sanitization
    return sanitizeQuery(query);  // Remove excessive whitespace
}
```

### Collection Access Control

```python
# packages/webui/services/search_service.py
async def validate_collection_access(self, collection_uuids, user_id):
    for uuid in collection_uuids:
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=uuid, 
            user_id=user_id
        )
        # Raises AccessDeniedError if unauthorized
```

### Rate Limiting

```python
# packages/webui/api/v2/search.py
@limiter.limit("30/minute")  # Multi-collection search
@limiter.limit("60/minute")  # Single collection search
```

### Metadata Filter Validation

- Filters are passed directly to Qdrant but should be validated
- Prevent injection attacks through filter construction
- Limit filter complexity to prevent DoS

## 7. Testing Requirements

### Unit Tests

```python
# tests/unit/test_search_service.py
- Test collection access validation
- Test search parameter validation
- Test result aggregation logic
- Test error handling paths
- Test timeout retry mechanism
```

### Integration Tests

```python
# tests/integration/test_search_api_integration.py
- Test end-to-end search flow
- Test embedding generation integration
- Test Qdrant connectivity
- Test reranking pipeline
- Test multi-collection aggregation
```

### Performance Tests

```python
# Required performance benchmarks:
- Single search latency < 500ms (p95)
- Multi-collection search < 1500ms (p95)
- Reranking overhead < 200ms for 100 candidates
- Concurrent search handling (10 req/s minimum)
```

### Search Quality Tests

```python
# Relevance testing requirements:
- Semantic similarity validation
- Reranking improvement measurement
- Hybrid search recall testing
- Query type specialization validation
```

## 8. Common Pitfalls & Best Practices

### Pitfalls to Avoid

1. **Dimension Mismatch**
   ```python
   # WRONG: Not validating embedding dimensions
   results = await search_qdrant(query_vector, k)
   
   # RIGHT: Validate before search
   validate_dimension_compatibility(
       expected_dimension=collection_dim,
       actual_dimension=len(query_vector)
   )
   ```

2. **Unbounded Result Sets**
   ```python
   # WRONG: No limit on candidates
   candidates = search_k * multiplier  # Could be huge
   
   # RIGHT: Apply bounds
   search_k = max(min_candidates, min(k * multiplier, max_candidates))
   ```

3. **Synchronous Blocking**
   ```python
   # WRONG: Sequential collection searches
   for collection in collections:
       results.append(search(collection))
   
   # RIGHT: Parallel execution
   search_tasks = [search(c) for c in collections]
   results = await asyncio.gather(*search_tasks)
   ```

### Best Practices

1. **Query Preprocessing**
   - Sanitize and validate input early
   - Apply query expansion for better recall
   - Use appropriate search instructions

2. **Result Caching**
   - Cache frequent queries (15-minute TTL)
   - Cache embeddings for repeated queries
   - Invalidate on collection updates

3. **Error Recovery**
   - Implement exponential backoff for retries
   - Provide partial results on failure
   - Log detailed error context

4. **Performance Optimization**
   - Use batch embedding generation
   - Implement connection pooling
   - Enable GPU acceleration when available

## 9. Configuration & Environment

### Search Configuration

```python
# packages/vecpipe/qwen3_search_config.py
RERANK_CONFIG = {
    "enabled": True,
    "candidate_multiplier": 5,    # Retrieve 5x candidates
    "min_candidates": 20,          # Minimum to retrieve
    "max_candidates": 200,         # Maximum cap
    "default_model": "Qwen/Qwen3-Reranker-0.6B",
    "use_hybrid_scoring": True,
    "hybrid_weight": 0.3           # 0.3 vector + 0.7 rerank
}

SEARCH_OPTIMIZATIONS = {
    "enable_instruction_tuning": True,
    "normalize_embeddings": True,
    "use_last_token_pooling": True,
    "enable_caching": True,
    "parallel_encoding": True,
    "adaptive_batch_sizing": True
}
```

### Environment Variables

```bash
# Core search settings
SEARCH_API_URL="http://vecpipe:8100"
SEARCH_API_PORT=8100
DEFAULT_COLLECTION="work_docs"

# Embedding configuration
DEFAULT_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B"
DEFAULT_QUANTIZATION="float32"
USE_MOCK_EMBEDDINGS=false
MODEL_UNLOAD_AFTER_SECONDS=300

# Qdrant configuration
QDRANT_HOST="qdrant"
QDRANT_PORT=6333

# Performance tuning
SEARCH_TIMEOUT_SECONDS=30
SEARCH_RETRY_MULTIPLIER=4.0
MAX_SEARCH_BATCH_SIZE=256
```

### Model Selection by Use Case

```python
# Optimal configurations for different scenarios
QWEN3_MODEL_RECOMMENDATIONS = {
    "high_quality": {
        "model": "Qwen/Qwen3-Embedding-8B",
        "quantization": "int8",
        "description": "Best quality, MTEB #1"
    },
    "balanced": {
        "model": "Qwen/Qwen3-Embedding-4B",
        "quantization": "float16",
        "description": "Great balance"
    },
    "fast": {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "quantization": "float16",
        "description": "Fast inference"
    }
}
```

## 10. Integration Points

### Frontend Integration

```typescript
// Search initiation from frontend
const performSearch = async (params: SearchParams) => {
    const response = await fetch('/api/v2/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
            collection_uuids: params.selectedCollections,
            query: params.query,
            k: params.topK,
            search_type: params.searchType,
            use_reranker: params.useReranker
        })
    });
    return response.json();
};
```

### Collection Service Integration

```python
# Collection status verification
if collection.status != CollectionStatus.READY:
    return (collection, None, f"Collection {collection.name} not ready")

# Collection metadata retrieval
metadata = get_collection_metadata(client, collection_name)
collection_model = metadata.get("model_name")
collection_quantization = metadata.get("quantization")
```

### VecPipe Service Communication

```python
# Embedding generation
async with httpx.AsyncClient() as client:
    response = await client.post(
        f"{settings.SEARCH_API_URL}/embed",
        json={
            "texts": texts,
            "model_name": model,
            "quantization": quantization,
            "batch_size": 32
        }
    )

# Vector upload
response = await client.post(
    f"{settings.SEARCH_API_URL}/upsert",
    json={
        "collection_name": collection_name,
        "points": points,
        "wait": True
    }
)
```

### Metrics & Monitoring Integration

```python
# Prometheus metrics
search_latency = Histogram(
    "search_api_latency_seconds",
    "Search API request latency",
    ["endpoint", "search_type"]
)

search_requests = Counter(
    "search_api_requests_total",
    "Total search API requests",
    ["endpoint", "search_type"]
)

# Usage tracking
search_latency.labels(
    endpoint="/search",
    search_type=request.search_type
).observe(elapsed_time)
```

## Additional Implementation Notes

### Query Expansion Strategies

1. **Synonym Expansion**: Expand queries with synonyms for better recall
2. **Acronym Resolution**: Expand acronyms to full forms
3. **Stemming/Lemmatization**: Normalize word forms
4. **Context Enhancement**: Add domain-specific context to queries

### Result Highlighting

```python
# Snippet generation with query term highlighting
def generate_snippet(content: str, query: str, context_size: int = 150):
    query_terms = query.lower().split()
    # Find best matching section
    best_position = find_best_match_position(content, query_terms)
    # Extract context window
    snippet = extract_context(content, best_position, context_size)
    # Highlight matching terms
    return highlight_terms(snippet, query_terms)
```

### Search History & Analytics

```python
# Track search patterns for optimization
class SearchAnalytics:
    async def log_search(self, user_id, query, results_count, latency):
        await self.db.insert({
            "user_id": user_id,
            "query": query,
            "timestamp": datetime.utcnow(),
            "results_count": results_count,
            "latency_ms": latency,
            "search_type": search_type
        })
    
    async def get_popular_queries(self, timeframe):
        # Aggregate most common queries for caching
        return await self.db.aggregate_top_queries(timeframe)
```

### Performance Optimization Techniques

1. **Connection Pooling**
   ```python
   # Reuse HTTP connections
   class SearchClient:
       def __init__(self):
           self.client = httpx.AsyncClient(
               limits=httpx.Limits(
                   max_keepalive_connections=20,
                   max_connections=100
               )
           )
   ```

2. **Result Prefetching**
   ```python
   # Prefetch additional results for pagination
   actual_k = min(k * 1.5, max_results)
   results = await search(query, actual_k)
   cache[query_hash] = results[k:]  # Cache extras
   return results[:k]
   ```

3. **Adaptive Batching**
   ```python
   # Adjust batch size based on GPU memory
   def get_optimal_batch_size(model_size, available_memory):
       base_sizes = {"0.6B": 128, "4B": 32, "8B": 16}
       return adjust_for_memory(base_sizes[model_size], available_memory)
   ```

This documentation provides a comprehensive reference for LLM agents working on the SEARCH_SYSTEM component, covering all critical aspects from architecture to implementation details, security considerations, and optimization strategies.