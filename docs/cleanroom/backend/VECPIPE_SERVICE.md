# VECPIPE_SERVICE - Vector Pipeline Service Documentation

## 1. Component Overview

### Purpose
The VECPIPE_SERVICE (Vector Pipeline Service) is the core vector processing engine of the Semantik application. It handles all vector-related operations including embedding generation, vector search, hybrid search, reranking, and vector database management.

### Responsibilities
- **Embedding Generation**: Convert text chunks into high-dimensional vector representations using Qwen3 and other embedding models
- **Vector Search**: Perform similarity search operations against Qdrant vector database
- **Hybrid Search**: Combine vector similarity with keyword matching for improved search results
- **Reranking**: Use cross-encoder models to rerank search results for better relevance
- **Model Management**: Lazy loading and automatic unloading of ML models to optimize GPU memory
- **Vector Ingestion**: Bulk upload of vectors to Qdrant collections
- **Maintenance**: Clean up orphaned vectors and maintain database integrity

### Role in Vector Processing
The vecpipe service acts as the bridge between document content and semantic search capabilities. It transforms textual data into searchable vector representations and provides high-performance retrieval with advanced features like reranking and hybrid search.

## 2. Architecture & Design Patterns

### Pipeline Architecture
```
Document → Chunks → Embeddings → Vectors → Qdrant Storage
                ↓
         Search Query → Query Embedding → Vector Search → Reranking → Results
```

### Model Loading Strategy
The service implements a **lazy loading pattern** with automatic unloading:

```python
# packages/vecpipe/model_manager.py
class ModelManager:
    def __init__(self, unload_after_seconds: int = 300):
        self.embedding_service: EmbeddingService | None = None
        self.reranker: CrossEncoderReranker | None = None
        self.unload_after_seconds = unload_after_seconds
        self.last_used: float = 0
        
    def ensure_model_loaded(self, model_name: str, quantization: str) -> bool:
        """Load model only when needed"""
        if self.current_model_key == model_key:
            self._update_last_used()
            return True
        # Load new model...
```

### Service Architecture
- **FastAPI Application** (`search_api.py`): RESTful API for search operations
- **Model Manager** (`model_manager.py`): Lifecycle management for ML models
- **Hybrid Search Engine** (`hybrid_search.py`): Combines vector and keyword search
- **Cross-Encoder Reranker** (`reranker.py`): Improves search relevance
- **Memory Manager** (`memory_utils.py`): GPU memory monitoring and optimization

## 3. Key Interfaces & Contracts

### Search API Endpoints

#### POST /search
```python
class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(alias="k", default=10)
    collection: str | None = None
    search_type: str = "semantic"  # semantic, question, code, hybrid
    model_name: str | None = None
    quantization: str | None = None
    use_reranker: bool = False
    rerank_model: str | None = None
    filters: dict[str, Any] | None = None
    include_content: bool = False

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    num_results: int
    search_type: str
    model_used: str
    embedding_time_ms: float | None
    search_time_ms: float | None
    reranking_used: bool = False
    reranker_model: str | None = None
    reranking_time_ms: float | None = None
```

#### POST /embed
```python
class EmbedRequest(BaseModel):
    texts: list[str]  # Max 1000 texts
    model_name: str
    quantization: str = "float32"
    instruction: str | None = None
    batch_size: int = 32

class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model_used: str
    embedding_time_ms: float | None
    batch_count: int
```

#### POST /upsert
```python
class UpsertRequest(BaseModel):
    collection_name: str
    points: list[UpsertPoint]  # Max 1000 points
    wait: bool = True

class UpsertPoint(BaseModel):
    id: str
    vector: list[float]
    payload: PointPayload

class PointPayload(BaseModel):
    doc_id: str
    chunk_id: str
    path: str
    content: str | None = None
    metadata: dict[str, Any] | None = None
```

### Data Formats

#### Parquet Schema for Embeddings
```python
# Input (from chunking stage)
{
    "doc_id": str,
    "chunk_id": str,
    "path": str,
    "text": str,
    "metadata": json_string
}

# Output (after embedding)
{
    "id": str (UUID),
    "vector": list[float],
    "payload": {
        "doc_id": str,
        "chunk_id": str,
        "path": str,
        "content": str,
        "metadata": dict
    }
}
```

## 4. Data Flow & Dependencies

### Document Processing Pipeline
```
1. Document Extraction (external) → Parquet files in extract_dir
2. embed_chunks_unified.py reads Parquet files
3. EmbeddingService generates embeddings (batch processing)
4. Embeddings saved to Parquet files in ingest_dir
5. ingest_qdrant.py uploads vectors to Qdrant
6. Files moved to loaded_dir on success, reject_dir on failure
```

### Search Flow
```
1. User Query → search_api.py endpoint
2. Query → Embedding Generation (using same model as collection)
3. Vector Search in Qdrant (with optional filters)
4. Optional: Retrieve more candidates for reranking
5. Optional: Cross-encoder reranking for relevance
6. Return top-k results to user
```

### Dependencies
- **Qdrant**: Vector database (external service)
- **shared.embedding**: Unified embedding service
- **shared.config**: Configuration management
- **shared.contracts.search**: Data contracts
- **PostgreSQL**: Collection metadata (via shared)

## 5. Critical Implementation Details

### Embedding Generation

#### Batch Processing with Qwen3
```python
# packages/vecpipe/embed_chunks_unified.py
async def process_document_async(
    file_path: str, 
    output_dir: str, 
    embedding_service: EmbeddingService,
    args: argparse.Namespace
) -> str | None:
    # Read input data
    data = await read_parquet_async(file_path)
    texts = data["texts"]
    
    # Generate embeddings with timing
    with TimingContext(embedding_batch_duration):
        embeddings = embedding_service.generate_embeddings(
            texts=texts,
            model_name=args.model,
            quantization=args.quantization,
            batch_size=args.batch_size
        )
    
    # Generate unique IDs and prepare output
    point_ids = [str(uuid.uuid4()) for _ in range(len(texts))]
    output_data = {
        "id": point_ids,
        "vector": embeddings.tolist(),
        "payload": [...]
    }
```

#### Model-Specific Instructions
```python
# packages/vecpipe/qwen3_search_config.py
DOMAIN_INSTRUCTIONS = {
    "general": {
        "index": "Represent this document for retrieval:",
        "query": "Represent this sentence for searching relevant passages:"
    },
    "technical": {
        "index": "Represent this technical documentation for retrieval:",
        "query": "Represent this technical question for finding relevant documentation:"
    },
    "code": {
        "index": "Represent this code snippet for similarity search:",
        "query": "Represent this code query for finding similar implementations:"
    }
}
```

### GPU Memory Management

#### Memory Requirement Calculation
```python
# packages/vecpipe/memory_utils.py
MODEL_MEMORY_REQUIREMENTS = {
    ("Qwen/Qwen3-Embedding-0.6B", "float32"): 2400,  # MB
    ("Qwen/Qwen3-Embedding-0.6B", "float16"): 1200,
    ("Qwen/Qwen3-Embedding-0.6B", "int8"): 600,
    ("Qwen/Qwen3-Embedding-4B", "float32"): 16000,
    ("Qwen/Qwen3-Embedding-4B", "float16"): 8000,
    ("Qwen/Qwen3-Embedding-4B", "int8"): 4000,
    # ... more models
}

def check_memory_availability(
    model_name: str, 
    quantization: str,
    current_models: dict[str, tuple[str, str]] | None = None
) -> tuple[bool, str]:
    free_mb, total_mb = get_gpu_memory_info()
    required_mb = get_model_memory_requirement(model_name, quantization)
    # Check and return availability status
```

#### Automatic Model Unloading
```python
# packages/vecpipe/model_manager.py
async def _schedule_unload(self) -> None:
    """Schedule model unloading after inactivity"""
    async def unload_after_delay() -> None:
        await asyncio.sleep(self.unload_after_seconds)
        with self.lock:
            if time.time() - self.last_used >= self.unload_after_seconds:
                logger.info(f"Unloading model after {self.unload_after_seconds}s")
                self.unload_model()
```

### Hybrid Search Implementation

#### Keyword Extraction and Filtering
```python
# packages/vecpipe/hybrid_search.py
class HybridSearchEngine:
    def extract_keywords(self, query: str) -> list[str]:
        """Extract meaningful keywords from query"""
        stop_words = {"a", "an", "the", "is", "are", ...}
        words = re.findall(r"\w+", query.lower())
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def build_text_filter(self, keywords: list[str], mode: str = "any") -> Filter | None:
        """Build Qdrant filter for text matching"""
        conditions = []
        for keyword in keywords:
            condition = FieldCondition(key="content", match=MatchText(text=keyword))
            conditions.append(condition)
        
        if mode == "all":
            return Filter(must=conditions)  # All keywords must match
        return Filter(should=conditions)    # Any keyword can match
```

### Reranking with Cross-Encoders

#### Qwen3 Reranker Integration
```python
# packages/vecpipe/reranker.py
class CrossEncoderReranker:
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
        instruction: str | None = None
    ) -> list[tuple[int, float]]:
        """Rerank documents using cross-encoder"""
        # Prepare query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get reranking scores in batches
        all_scores = []
        for batch in self._batch_iterator(pairs, self.get_batch_size()):
            scores = self._compute_scores_batch(batch, instruction)
            all_scores.extend(scores)
        
        # Sort and return top-k
        indexed_scores = list(enumerate(all_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:top_k]
```

### Batch Processing for Efficiency

#### Optimized Batch Sizes
```python
# packages/vecpipe/qwen3_search_config.py
BATCH_CONFIGS = {
    "Qwen/Qwen3-Embedding-0.6B": {
        "float32": {"batch_size": 64, "max_length": 32768},
        "float16": {"batch_size": 128, "max_length": 32768},
        "int8": {"batch_size": 256, "max_length": 32768}
    },
    "Qwen/Qwen3-Embedding-4B": {
        "float32": {"batch_size": 16, "max_length": 8192},
        "float16": {"batch_size": 32, "max_length": 8192},
        "int8": {"batch_size": 64, "max_length": 8192}
    }
}
```

## 6. Security Considerations

### Input Sanitization
- All user queries are validated for length and content
- Batch sizes are capped to prevent resource exhaustion
- File paths are validated before processing

### Resource Limits
```python
# API endpoint limits
class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=1000)
    batch_size: int = Field(32, ge=1, le=256)

class SearchRequest(BaseModel):
    top_k: int = Field(alias="k", default=10, ge=1, le=100)
```

### Error Handling
```python
# Specific error types for different failures
class InsufficientMemoryError(Exception):
    """Raised when there's not enough GPU memory"""

class DimensionMismatchError(Exception):
    """Raised when embedding dimensions don't match collection"""

# HTTP error responses with details
except InsufficientMemoryError as e:
    raise HTTPException(
        status_code=507,  # Insufficient Storage
        detail={
            "error": "insufficient_memory",
            "message": str(e),
            "suggestion": "Try using a smaller model or different quantization"
        }
    )
```

## 7. Testing Requirements

### Embedding Tests
```python
# Test embedding generation with different models
def test_embedding_generation():
    service = EmbeddingService(mock_mode=False)
    service.load_model("Qwen/Qwen3-Embedding-0.6B", "float16")
    
    texts = ["test document", "another document"]
    embeddings = service.generate_embeddings(texts, batch_size=2)
    
    assert embeddings.shape == (2, 1024)  # 0.6B model has 1024 dimensions
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)  # Normalized
```

### Search Tests
```python
# Test search with reranking
async def test_search_with_reranking():
    request = SearchRequest(
        query="technical documentation",
        k=5,
        use_reranker=True,
        rerank_model="Qwen/Qwen3-Reranker-0.6B"
    )
    
    response = await search_post(request)
    assert response.reranking_used == True
    assert len(response.results) <= 5
    assert response.reranking_time_ms is not None
```

### Memory Management Tests
```python
# Test automatic model unloading
async def test_model_unloading():
    manager = ModelManager(unload_after_seconds=1)
    manager.ensure_model_loaded("test-model", "float32")
    
    assert manager.current_model_key is not None
    await asyncio.sleep(2)
    assert manager.current_model_key is None  # Should be unloaded
```

## 8. Common Pitfalls & Best Practices

### OOM (Out of Memory) Handling

**Pitfall**: Loading multiple large models simultaneously
```python
# BAD: Loading without checking memory
embedding_service.load_model("Qwen/Qwen3-Embedding-8B", "float32")
reranker.load_model("Qwen/Qwen3-Reranker-8B", "float32")  # May OOM!
```

**Best Practice**: Check memory before loading
```python
# GOOD: Check memory availability first
can_load, message = check_memory_availability(
    "Qwen/Qwen3-Reranker-8B", 
    "float32",
    current_models={"embedding": ("Qwen/Qwen3-Embedding-8B", "float32")}
)
if not can_load:
    # Use smaller model or different quantization
    model_name = "Qwen/Qwen3-Reranker-0.6B"
```

### Batch Processing

**Pitfall**: Processing one document at a time
```python
# BAD: Individual processing
for text in texts:
    embedding = generate_embedding(text)  # Inefficient!
```

**Best Practice**: Process in batches
```python
# GOOD: Batch processing
embeddings = embedding_service.generate_embeddings(
    texts=texts,
    batch_size=optimal_batch_size
)
```

### Dimension Mismatch

**Pitfall**: Using different models for indexing and searching
```python
# BAD: Mismatch between index and query embeddings
# Index with: Qwen/Qwen3-Embedding-0.6B (1024d)
# Search with: Qwen/Qwen3-Embedding-4B (2560d)  # Error!
```

**Best Practice**: Store and use collection metadata
```python
# GOOD: Use collection metadata to ensure consistency
metadata = get_collection_metadata(client, collection_name)
model_name = metadata.get("model_name")
quantization = metadata.get("quantization")
query_embedding = generate_embedding(query, model_name, quantization)
```

### Error Recovery

**Pitfall**: Silent failures
```python
# BAD: Swallowing errors
try:
    results = search(query)
except:
    results = []  # User gets no feedback!
```

**Best Practice**: Informative error handling
```python
# GOOD: Detailed error responses
try:
    results = search(query)
except DimensionMismatchError as e:
    return {
        "error": "dimension_mismatch",
        "expected": e.expected_dimension,
        "actual": e.actual_dimension,
        "suggestion": f"Use model that outputs {e.expected_dimension}d vectors"
    }
```

## 9. Configuration & Environment

### Environment Variables
```python
# Core configuration from shared.config.settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
DEFAULT_COLLECTION = "work_docs"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_QUANTIZATION = "float16"
USE_MOCK_EMBEDDINGS = False
MODEL_UNLOAD_AFTER_SECONDS = 300
SEARCH_API_PORT = 8080
METRICS_PORT = 9090

# Directory paths
extract_dir = "/data/extract"    # Input chunks
ingest_dir = "/data/ingest"      # Embeddings to upload
loaded_dir = "/data/loaded"      # Successfully uploaded
reject_dir = "/data/rejects"     # Failed uploads
```

### Model Configuration
```python
# packages/vecpipe/qwen3_search_config.py
QWEN3_MODEL_RECOMMENDATIONS = {
    "high_quality": {
        "model": "Qwen/Qwen3-Embedding-8B",
        "quantization": "int8",
        "description": "Best quality, MTEB #1, 4096d embeddings"
    },
    "balanced": {
        "model": "Qwen/Qwen3-Embedding-4B",
        "quantization": "float16",
        "description": "Great balance of quality and speed, 2560d"
    },
    "fast": {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "quantization": "float16",
        "description": "Fast inference, good quality, 1024d"
    }
}
```

### GPU Settings
```python
# Automatic GPU detection and configuration
if torch.cuda.is_available():
    device = "cuda"
    # Get GPU memory for optimization
    free_mb, total_mb = torch.cuda.mem_get_info()
    config = suggest_model_configuration(free_mb)
else:
    device = "cpu"
    # Use lightweight models for CPU
    config = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_quantization": "float32"
    }
```

### Qdrant Configuration
```python
# Collection creation parameters
{
    "vectors": {
        "size": 1024,  # Must match embedding dimension
        "distance": "Cosine"  # For normalized vectors
    },
    "payload_schema": {
        "doc_id": "keyword",
        "chunk_id": "keyword",
        "path": "text",
        "content": "text",  # For hybrid search
        "metadata": "json"
    },
    "optimizers_config": {
        "indexing_threshold": 20000,
        "memmap_threshold": 50000
    }
}
```

## 10. Integration Points

### WebUI Service Integration

The WebUI service calls vecpipe for search operations:

```python
# packages/webui/services/search_service.py
async def perform_search(self, request: SearchRequest) -> SearchResponse:
    async with httpx.AsyncClient() as client:
        # Call vecpipe search API
        response = await client.post(
            f"http://vecpipe:8080/search",
            json=request.model_dump(),
            timeout=httpx.Timeout(30.0)
        )
        return SearchResponse(**response.json())
```

### Worker Service Integration

The worker service uses vecpipe for embedding generation during indexing:

```python
# Worker calls vecpipe /embed endpoint for batch embedding
async def generate_embeddings_for_collection(
    texts: list[str],
    model_name: str,
    quantization: str
) -> list[list[float]]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://vecpipe:8080/embed",
            json={
                "texts": texts,
                "model_name": model_name,
                "quantization": quantization,
                "batch_size": 32
            }
        )
        return response.json()["embeddings"]
```

### Qdrant Direct Integration

Vecpipe communicates directly with Qdrant for vector operations:

```python
# Direct Qdrant operations
client = QdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}")

# Search
results = client.search(
    collection_name=collection_name,
    query_vector=query_embedding,
    limit=k,
    query_filter=filters
)

# Upsert
client.upsert(
    collection_name=collection_name,
    points=[PointStruct(id=id, vector=vec, payload=data)]
)
```

### Metrics & Monitoring

Prometheus metrics exposed on port 9090:

```python
# Metrics tracked
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

embedding_generation_latency = Histogram(
    "search_api_embedding_latency_seconds",
    "Embedding generation latency"
)
```

## Usage Examples

### Basic Search
```bash
# Simple semantic search
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to implement authentication?",
    "k": 10,
    "search_type": "semantic"
  }'
```

### Search with Reranking
```bash
# Advanced search with reranking for better relevance
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "database optimization techniques",
    "k": 5,
    "use_reranker": true,
    "rerank_model": "Qwen/Qwen3-Reranker-4B",
    "rerank_quantization": "float16"
  }'
```

### Hybrid Search
```bash
# Combine vector similarity with keyword matching
curl -X GET "http://localhost:8080/hybrid_search?q=Python+async+programming&k=10&mode=filter&keyword_mode=all"
```

### Batch Embedding Generation
```bash
# Generate embeddings for multiple texts
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Document 1", "Document 2", "Document 3"],
    "model_name": "Qwen/Qwen3-Embedding-0.6B",
    "quantization": "float16",
    "batch_size": 32
  }'
```

## Troubleshooting Guide

### Common Issues

1. **Dimension Mismatch Error**
   - Cause: Using different embedding models for indexing vs searching
   - Solution: Check collection metadata and use the same model

2. **OOM Errors**
   - Cause: Insufficient GPU memory for model
   - Solution: Use smaller model or more aggressive quantization (int8)

3. **Slow Search Performance**
   - Cause: Large result sets without optimization
   - Solution: Enable reranking, use appropriate batch sizes

4. **Connection Refused to Qdrant**
   - Cause: Qdrant service not running or wrong host/port
   - Solution: Verify Qdrant is running and configuration is correct

### Debug Commands

```bash
# Check model status
curl http://localhost:8080/model/status

# Get GPU memory suggestions
curl http://localhost:8080/models/suggest

# Health check
curl http://localhost:8080/health

# Collection info
curl http://localhost:8080/collection/info
```

## Performance Optimization Tips

1. **Use appropriate quantization**: float16 offers good balance, int8 for memory-constrained environments
2. **Enable batch processing**: Process multiple documents/queries together
3. **Configure model unloading**: Set appropriate timeout based on usage patterns
4. **Use reranking selectively**: Only for queries requiring high precision
5. **Optimize Qdrant settings**: Adjust indexing and memmap thresholds based on collection size
6. **Monitor metrics**: Use Prometheus metrics to identify bottlenecks
7. **Cache frequently used embeddings**: Implement caching for common queries

## Future Enhancements

- Support for additional embedding models (OpenAI, Cohere, etc.)
- Multi-GPU support for larger models
- Streaming embeddings for real-time processing
- Advanced caching strategies
- Query expansion and reformulation
- Feedback loop for search quality improvement
- A/B testing framework for model selection