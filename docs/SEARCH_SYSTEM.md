# Semantik - Search Architecture Documentation

## Table of Contents
1. [Search System Overview](#search-system-overview)
2. [Unified Search Implementation](#unified-search-implementation)
3. [Search API Implementation](#search-api-implementation)
4. [Hybrid Search](#hybrid-search)
5. [Embedding Search](#embedding-search)
6. [Qwen3 Search Optimization](#qwen3-search-optimization)
7. [Batch Search](#batch-search)
8. [Search Types](#search-types)
9. [Search Configuration](#search-configuration)
10. [Search Validation](#search-validation)
11. [Performance Analysis](#performance-analysis)
12. [Usage Examples](#usage-examples)

## 1. Search System Overview

Semantik implements a sophisticated search architecture that combines vector similarity search with keyword-based search capabilities. The system is designed for high performance, scalability, and flexibility.

### Architecture Design

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend UI   │────▶│  WebUI Search   │────▶│  Search API v2  │
│  (React 19)     │     │    Service      │     │  (FastAPI)      │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                        ┌─────────────────────────────────────────┐
                        │         Search Components              │
                        ├─────────────────┬───────────────────────┤
                        │ Search Utils    │  Hybrid Search Engine │
                        │                 │                       │
                        │ - Qdrant Client │  - Keyword Extract   │
                        │ - Result Parse  │  - Filter/Rerank     │
                        │ - Vector Search │  - Score Combine     │
                        └─────────────────┴───────────────────────┘
                                          │
                                          ▼
                        ┌─────────────────────────────────────────┐
                        │      Reranking & Model Management      │
                        ├─────────────────────────────────────────┤
                        │ - Cross-Encoder Reranker               │
                        │ - Model Manager (Lazy Loading)         │
                        │ - Memory-Aware Loading                 │
                        │ - Auto-Unloading After 5min            │
                        └─────────────────────────────────────────┘
                                          │
                                          ▼
                        ┌─────────────────────────────────────────┐
                        │         Embedding Service              │
                        ├─────────────────────────────────────────┤
                        │ - Multi-Model Support                  │
                        │ - Quantization (float32/16, int8)     │
                        │ - Qwen3 Optimization                   │
                        │ - Batch Processing                     │
                        └─────────────────────────────────────────┘
                                          │
                                          ▼
                        ┌─────────────────────────────────────────┐
                        │         Qdrant Vector DB               │
                        ├─────────────────────────────────────────┤
                        │ - Collection-Based Organization        │
                        │ - HNSW Index (Optimized)              │
                        │ - Blue-Green Deployments              │
                        │ - Metadata Storage                    │
                        └─────────────────────────────────────────┘
```

### Key Features

- **High Performance**: Optimized for low-latency search with lazy model loading and connection pooling
- **Flexible Search Types**: Vector, hybrid, keyword-only, and cross-encoder reranked search
- **Model Agnostic**: Support for multiple embedding models with automatic selection
- **Quantization Support**: float32, float16, and int8 quantization for memory efficiency
- **Collection-Centric**: Each collection maintains its own embedding model metadata
- **Reranking**: Optional two-stage retrieval with cross-encoder reranking for 20%+ relevance improvement
- **Monitoring**: Comprehensive Prometheus metrics for observability

### Performance Characteristics

- Vector search latency: ~50-200ms (depending on collection size)
- Hybrid search latency: ~100-300ms (with keyword filtering)
- Reranked search latency: ~200-1200ms (model dependent)
- Batch search: Up to 256 queries in parallel
- Model loading: 2-30 seconds (cached after first load)
- Automatic model unloading after 5 minutes of inactivity
- Memory-aware loading prevents OOM errors

## 2. Unified Search Implementation

The core search logic is implemented in `vecpipe/search_utils.py`, providing shared utilities for both the search API and web UI.

### Core Components

```python
# vecpipe/search_utils.py

async def search_qdrant(
    qdrant_host: str,
    qdrant_port: int,
    collection_name: str,
    query_vector: list[float],
    k: int,
    with_payload: bool = True,
) -> list[dict]:
    """
    Perform vector search in Qdrant
    
    - Uses AsyncQdrantClient for non-blocking operations
    - Returns results sorted by similarity score
    - Includes document payload by default
    """

def parse_search_results(qdrant_results: list[dict]) -> list[dict]:
    """
    Parse Qdrant search results into a standard format
    
    - Extracts path, chunk_id, score, doc_id, content, metadata
    - Provides consistent format across different search types
    """
```

### Search Flow

1. **Query Processing**: Clean and prepare search query
2. **Embedding Generation**: Convert query to vector using appropriate model
3. **Vector Search**: Query Qdrant for similar vectors
4. **Result Parsing**: Convert raw results to standard format
5. **Response Generation**: Format results for client consumption

## 3. Search API Implementation

The search API (`vecpipe/search_api.py`) provides a RESTful interface for all search operations.

### API Endpoints

#### GET/POST `/search`
Primary search endpoint supporting multiple search types.

```python
@app.post("/search", response_model=SearchResponse)
async def search_post(request: SearchRequest):
    """
    Parameters:
    - query: Search query text
    - k: Number of results (1-100)
    - search_type: semantic, question, code, hybrid
    - model_name: Override embedding model
    - quantization: Override quantization
    - collection: Target collection name
    - filters: Metadata filters
    - include_content: Include chunk content
    """
```

#### GET `/hybrid_search`
Specialized endpoint for hybrid search combining vector and keyword matching.

```python
@app.get("/hybrid_search", response_model=HybridSearchResponse)
async def hybrid_search(
    q: str,
    k: int,
    collection: str,
    mode: str = "filter",  # filter or rerank
    keyword_mode: str = "any",  # any or all
    score_threshold: float = None,
):
    """
    Hybrid search with keyword filtering or reranking
    """
```

#### POST `/search/batch`
Batch processing for multiple queries.

```python
@app.post("/search/batch", response_model=BatchSearchResponse)
async def batch_search(request: BatchSearchRequest):
    """
    Process multiple search queries in parallel
    - Efficient batch embedding generation
    - Parallel Qdrant queries
    - Consolidated response
    """
```

### Key Features

- **Model Management**: Lazy loading with automatic unloading
- **Collection Metadata**: Automatic model/quantization detection
- **Error Handling**: Comprehensive error responses with fallbacks
- **Metrics**: Prometheus metrics for all operations
- **Mock Mode**: Development mode with simulated embeddings

## 3.5. Search API v2 Endpoints

The Search API v2 provides enhanced functionality for collection-based search operations.

### Core Endpoints

#### POST `/api/v2/search/collections/{collection_id}`
Enhanced search with reranking support:
```python
{
    "query": "search query",
    "top_k": 10,
    "search_type": "semantic",  # semantic, question, code, hybrid
    "filters": {
        "metadata_field": "value"
    },
    "use_reranker": true,
    "rerank_model": "auto",  # auto-selects based on embedding model
    "include_content": true,
    "score_threshold": 0.5
}
```

#### POST `/api/v2/search/hybrid/{collection_id}`
Hybrid search with keyword and vector matching:
```python
{
    "query": "machine learning algorithms",
    "top_k": 20,
    "hybrid_mode": "filter",  # filter or rerank
    "keyword_mode": "any",    # any or all
    "hybrid_alpha": 0.7,      # weight for vector score (0-1)
    "use_reranker": false
}
```

#### POST `/api/v2/search/batch`
Batch search across multiple queries:
```python
{
    "collection_id": "uuid-here",
    "queries": [
        {"query": "query1", "top_k": 5},
        {"query": "query2", "top_k": 10}
    ],
    "search_type": "semantic",
    "model_override": null  # uses collection's model
}
```

### New Features in v2

1. **Collection-Aware Search**:
   - Automatically uses the embedding model the collection was created with
   - Warns if searching with a different model than indexed
   - Maintains search consistency

2. **Reranking Integration**:
   - Seamless cross-encoder reranking
   - Auto-selects appropriate reranker model
   - Memory-aware loading prevents OOM

3. **Enhanced Metadata**:
   - Returns collection metadata with results
   - Includes model information
   - Provides performance metrics

4. **Improved Error Handling**:
   - Detailed error messages
   - Fallback strategies
   - Resource limit warnings

## 4. Hybrid Search

The hybrid search implementation (`vecpipe/hybrid_search.py`) combines vector similarity with keyword matching.

### Hybrid Search Engine

```python
class HybridSearchEngine:
    """Hybrid search engine combining vector and text search"""
    
    def extract_keywords(self, query: str) -> list[str]:
        """
        Extract meaningful keywords from query
        - Removes stop words
        - Filters short words
        - Returns cleaned keyword list
        """
    
    def build_text_filter(self, keywords: list[str], mode: str = "any") -> Filter:
        """
        Build Qdrant filter for text matching
        - "any": Match any keyword (OR)
        - "all": Match all keywords (AND)
        """
    
    def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        limit: int = 10,
        keyword_mode: str = "any",
        score_threshold: float = None,
        hybrid_mode: str = "filter",
    ):
        """
        Perform hybrid search
        - filter: Use Qdrant filters (faster)
        - rerank: Post-process and rerank (more flexible)
        """
```

### Search Modes

#### Filter Mode (Default)
- Uses Qdrant's built-in filtering
- Keywords are applied as filters during vector search
- More efficient for large collections
- Results already filtered by keywords

#### Rerank Mode
- Retrieves more candidates (3x limit)
- Scores based on keyword matches
- Combines vector and keyword scores
- Default weighting: 70% vector, 30% keywords
- More flexible scoring

### Keyword Matching

- **Stop Word Removal**: Common words filtered out
- **Minimum Length**: Words must be > 2 characters
- **Case Insensitive**: All matching is case-insensitive
- **Modes**:
  - `any`: Match documents containing any keyword
  - `all`: Match documents containing all keywords

## 5. Embedding Search

The embedding search system is powered by the `EmbeddingService` and `ModelManager`.

### Model Manager

```python
class ModelManager:
    """Manages embedding model lifecycle with lazy loading"""
    
    def ensure_model_loaded(self, model_name: str, quantization: str) -> bool:
        """Load model if not already loaded"""
    
    async def generate_embedding_async(
        self, text: str, model_name: str, 
        quantization: str, instruction: str = None
    ) -> list:
        """Generate embedding with automatic model management"""
    
    def unload_model(self):
        """Unload model to free memory"""
```

### Key Features

- **Lazy Loading**: Models loaded only when needed
- **Automatic Unloading**: After 5 minutes of inactivity
- **Memory Management**: Aggressive garbage collection
- **GPU Support**: CUDA acceleration when available
- **Quantization**: Dynamic quantization support

### Embedding Generation Process

1. **Model Selection**: Based on collection metadata or request
2. **Instruction Formatting**: Task-specific instructions
3. **Tokenization**: Model-specific tokenization
4. **Embedding**: Generate dense vectors
5. **Normalization**: L2 normalization for cosine similarity

## 6. Qwen3 Search Optimization

The system includes specific optimizations for Qwen3 embedding models.

### Configuration (`qwen3_search_config.py`)

```python
QWEN3_MODEL_RECOMMENDATIONS = {
    "high_quality": {
        "model": "Qwen/Qwen3-Embedding-8B",
        "quantization": "int8",
        "description": "Best quality, MTEB #1, 4096d embeddings",
        "use_cases": ["research", "legal", "medical"],
    },
    "balanced": {
        "model": "Qwen/Qwen3-Embedding-4B",
        "quantization": "float16",
        "description": "Great balance of quality and speed, 2560d",
        "use_cases": ["general", "documentation"],
    },
    "fast": {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "quantization": "float16",
        "description": "Fast inference, good quality, 1024d",
        "use_cases": ["real-time", "chat", "high-volume"],
    },
}
```

### Optimizations

- **Last Token Pooling**: Qwen3-specific pooling strategy
- **Instruction Tuning**: Domain-specific instructions
- **Adaptive Batch Sizing**: Based on model and GPU memory
- **Quantization**: Optimized int8 quantization for large models

### Batch Processing Configuration

```python
BATCH_CONFIGS = {
    "Qwen/Qwen3-Embedding-8B": {
        "float32": {"batch_size": 8, "max_length": 8192},
        "float16": {"batch_size": 16, "max_length": 8192},
        "int8": {"batch_size": 32, "max_length": 8192},
    },
    # ... configurations for other models
}
```

## 7. Batch Search

Batch search enables efficient processing of multiple queries.

### Implementation

```python
async def batch_search(request: BatchSearchRequest):
    """
    Process multiple queries efficiently:
    1. Batch embedding generation
    2. Parallel Qdrant queries
    3. Consolidated response
    """
    # Generate embeddings for all queries in batch
    embedding_tasks = [
        generate_embedding_async(query, model_name, quantization, instruction)
        for query in request.queries
    ]
    query_vectors = await asyncio.gather(*embedding_tasks)
    
    # Create search tasks
    search_tasks = [
        search_qdrant(host, port, collection, vector, k)
        for vector in query_vectors
    ]
    
    # Execute searches in parallel
    all_results = await asyncio.gather(*search_tasks)
```

### Performance Benefits

- **Parallel Processing**: All queries processed concurrently
- **Batch Embedding**: Efficient GPU utilization
- **Reduced Latency**: ~40% faster than sequential processing
- **Resource Efficiency**: Better memory and compute usage

## 8. Search Types

The system supports multiple search types optimized for different use cases.

### Semantic Search
```python
"semantic": "Represent this sentence for searching relevant passages:"
```
- General-purpose semantic similarity
- Best for document retrieval
- Focuses on meaning and context

### Question-Based Search
```python
"question": "Represent this question for retrieving supporting documents:"
```
- Optimized for Q&A systems
- Retrieves documents that answer questions
- Better for information seeking

### Code Search
```python
"code": "Represent this code query for finding similar code snippets:"
```
- Specialized for code similarity
- Understands programming concepts
- Preserves syntactic patterns

### Hybrid Search
```python
"hybrid": "Generate a comprehensive embedding for multi-modal search:"
```
- Combines vector and keyword matching
- Best for precision requirements
- Supports filtering and reranking

## 9. Search Configuration

The system provides extensive configuration options.

### Environment Variables
```bash
# Embedding Configuration
USE_MOCK_EMBEDDINGS=false
DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
DEFAULT_QUANTIZATION=float16

# Model Management
MODEL_UNLOAD_AFTER_SECONDS=300

# Search Settings
DEFAULT_COLLECTION=work_docs
SEARCH_API_PORT=8000
```

### Collection Metadata

Collections store metadata about their creation:
```python
{
    "collection_name": "collection_<uuid>",
    "model_name": "Qwen/Qwen3-Embedding-0.6B",
    "quantization": "float16",
    "vector_dim": 1024,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "instruction": "Represent this document for retrieval:",
    "created_at": "2025-01-20T10:30:00Z",
    "distance_metric": "cosine",
    "optimizer_config": {
        "indexing_threshold": 20000,
        "memmap_threshold": 0
    }
}
```

### Collection Naming Conventions

- **Production Collections**: `collection_<uuid>` - Main collections for document storage
- **Staging Collections**: `staging_collection_<uuid>_<timestamp>` - Used during blue-green reindexing
- **Legacy Collections**: May contain `work_docs` or operation-specific names

### Search Parameters

- **k**: Number of results (1-100)
- **score_threshold**: Minimum similarity score
- **filters**: Metadata-based filtering
- **include_content**: Return chunk content
- **model_name**: Override default model
- **quantization**: Override default quantization
- **use_reranker**: Enable cross-encoder reranking
- **rerank_model**: Override reranker model selection
- **rerank_quantization**: Override reranker quantization

## 9.5. Qdrant Configuration Best Practices

### HNSW Index Configuration

```python
# Optimal HNSW parameters for different collection sizes
{
    "small_collection": {  # < 100K vectors
        "hnsw_config": {
            "m": 16,
            "ef_construct": 100,
            "ef": 100,
            "full_scan_threshold": 10000
        }
    },
    "medium_collection": {  # 100K - 1M vectors
        "hnsw_config": {
            "m": 32,
            "ef_construct": 200,
            "ef": 150,
            "full_scan_threshold": 20000
        }
    },
    "large_collection": {  # > 1M vectors
        "hnsw_config": {
            "m": 64,
            "ef_construct": 400,
            "ef": 200,
            "full_scan_threshold": 40000
        }
    }
}
```

### Memory Configuration

```python
# Optimizer configuration for performance
{
    "optimizers_config": {
        "indexing_threshold": 20000,  # Build index after this many vectors
        "memmap_threshold": 0,        # Always use memory-mapped storage
        "max_segment_size": 200_000,  # Maximum vectors per segment
        "default_segment_number": 4   # Parallel segments for better performance
    }
}
```

### Storage Configuration

```python
# On-disk vs in-memory configuration
{
    "on_disk": True,  # Store vectors on disk for large collections
    "wal_config": {
        "wal_capacity_mb": 32,
        "wal_segments_ahead": 0
    },
    "quantization_config": {
        "scalar": {
            "type": "int8",
            "quantile": 0.99,
            "always_ram": True  # Keep quantized vectors in RAM
        }
    }
}
```

### Performance Tuning Tips

1. **Index Building**:
   - Set `indexing_threshold` based on expected collection size
   - Higher `ef_construct` improves recall but increases indexing time
   - Balance `m` parameter: higher values = better recall, more memory

2. **Search Performance**:
   - Adjust `ef` parameter at search time for speed/accuracy tradeoff
   - Use `full_scan_threshold` to bypass index for small result sets
   - Enable quantization for large collections to reduce memory usage

3. **Memory Management**:
   - Use `memmap_threshold` to control memory-mapped file usage
   - Enable `on_disk` storage for collections > available RAM
   - Configure `always_ram` for frequently accessed data

4. **Concurrent Operations**:
   - Use multiple segments for better parallel performance
   - Configure appropriate `wal_capacity_mb` for write-heavy workloads
   - Consider read replicas for high-query loads

## 10. Search Validation

The `validate_search_setup.py` script helps diagnose configuration issues.

### Validation Steps

1. **Environment Check**: Verify configuration variables
2. **Dependencies**: Check required packages
3. **Hardware**: Verify GPU/CPU capabilities
4. **Model Loading**: Test model initialization
5. **Recommendations**: Suggest fixes for issues

### Example Output
```
Search API Configuration Validation
====================================

1. Environment Variables:
   USE_MOCK_EMBEDDINGS: false
   DEFAULT_EMBEDDING_MODEL: Qwen/Qwen3-Embedding-0.6B
   DEFAULT_QUANTIZATION: float16

2. Dependencies:
   ✓ transformers: 4.51.0
   ✓ torch: 2.1.0
   ✓ sentence_transformers: 2.2.2

3. Hardware:
   ✓ CUDA available: NVIDIA GeForce RTX 3090
   ✓ GPU memory: 20.5GB free / 24.0GB total

4. Testing Model Loading:
   ✓ Model loaded successfully
   ✓ Test embedding generated (dimension: 1024)
```

## 11. Performance Analysis

### Metrics Collection

The system uses Prometheus metrics for comprehensive monitoring:

```python
# Search latency by endpoint and type
search_latency = Histogram(
    "search_api_latency_seconds",
    "Search API request latency",
    ["endpoint", "search_type"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

# Embedding generation latency
embedding_generation_latency = Histogram(
    "search_api_embedding_latency_seconds",
    "Embedding generation latency",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2),
)
```

### Performance Benchmarks

#### Vector Search (1M documents)
- P50 latency: 80ms
- P95 latency: 200ms
- P99 latency: 500ms
- Throughput: 100 queries/second

#### Hybrid Search (Filter Mode)
- P50 latency: 120ms
- P95 latency: 300ms
- P99 latency: 700ms
- Throughput: 60 queries/second

#### Batch Search (100 queries)
- Total time: 2-5 seconds
- Per-query average: 20-50ms
- GPU utilization: 80-95%

### Optimization Strategies

1. **Caching**: Frequently used embeddings cached
2. **Connection Pooling**: Reuse Qdrant connections
3. **Batch Processing**: Group operations for efficiency
4. **Lazy Loading**: Load models only when needed
5. **Quantization**: Reduce memory usage with int8

## 12. Usage Examples

### Basic Vector Search
```python
import requests

response = requests.post(
    "http://localhost:8000/search",
    json={
        "query": "machine learning algorithms",
        "k": 10,
        "search_type": "semantic"
    }
)
results = response.json()
```

### Hybrid Search with Keywords
```python
response = requests.get(
    "http://localhost:8000/hybrid_search",
    params={
        "q": "python async programming",
        "k": 20,
        "mode": "filter",
        "keyword_mode": "all"
    }
)
```

### Batch Search
```python
response = requests.post(
    "http://localhost:8000/search/batch",
    json={
        "queries": [
            "neural networks",
            "deep learning",
            "transformers"
        ],
        "k": 5,
        "search_type": "semantic"
    }
)
```

### Search with Specific Model
```python
response = requests.post(
    "http://localhost:8000/search",
    json={
        "query": "quantum computing",
        "k": 10,
        "model_name": "Qwen/Qwen3-Embedding-8B",
        "quantization": "int8"
    }
)
```

### Keyword-Only Search
```python
response = requests.get(
    "http://localhost:8000/keyword_search",
    params={
        "q": "python django rest",
        "k": 15,
        "mode": "any"
    }
)
```

## Conclusion

Semantik's search architecture provides a robust, scalable, and flexible solution for semantic search. With support for multiple search types, advanced hybrid search capabilities, and comprehensive optimization for modern embedding models like Qwen3, the system is well-suited for production deployments requiring high-performance search functionality.

Key strengths include:
- Flexible architecture supporting various search paradigms
- Performance optimization through lazy loading and quantization
- Comprehensive monitoring and observability
- Easy integration with existing systems
- Production-ready error handling and fallbacks

The modular design allows for easy extension and customization while maintaining high performance and reliability.