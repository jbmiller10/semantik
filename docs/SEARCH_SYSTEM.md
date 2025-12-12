# Semantik - Search Architecture Documentation

## Table of Contents
1. [Search System Overview](#1-search-system-overview)
2. [Unified Search Implementation](#2-unified-search-implementation)
3. [Search API Implementation](#3-search-api-implementation)
   - [3.5 WebUI Search API v2 Endpoints](#35-webui-search-api-v2-endpoints)
4. [Hybrid Search](#4-hybrid-search)
5. [Embedding Search](#5-embedding-search)
   - [Embedding Provider Plugin Architecture](#embedding-provider-plugin-architecture)
6. [Qwen3 Search Optimization](#6-qwen3-search-optimization)
7. [Batch Search](#7-batch-search)
8. [Search Types](#8-search-types)
9. [Search Configuration](#9-search-configuration)
   - [9.5 Qdrant Configuration Recommendations](#95-qdrant-configuration-recommendations)
10. [Search Validation](#10-search-validation)
11. [Performance Analysis](#11-performance-analysis)
12. [Usage Examples](#12-usage-examples)
13. [Error Handling](#13-error-handling)

## 1. Search System Overview

Semantik combines vector similarity search with keyword-based search. Fast, scalable, and flexible.

### Architecture Design

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend UI   │────▶│  WebUI Search   │────▶│  Search API v2  │
│  (React 19.1)   │     │    Service      │     │  (FastAPI)      │
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

- Low-latency search with lazy model loading and connection pooling
- Vector, hybrid, keyword-only, and cross-encoder reranked search
- Multiple embedding models with automatic selection
- float32, float16, and int8 quantization
- Collection-centric design with embedded model metadata
- Optional cross-encoder reranking (20%+ relevance improvement)
- Prometheus metrics

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

The search API is implemented using FastAPI's router-based architecture in `packages/vecpipe/search/router.py`, with business logic in `packages/vecpipe/search/service.py`.

### API Endpoints

#### GET/POST `/search`
Primary search endpoint supporting multiple search types.

```python
@router.post("/search", response_model=SearchResponse)
async def search_post(request: SearchRequest):
    """
    Parameters:
    - query: Search query text (required, 1-1000 chars)
    - k: Number of results (1-100), alias: top_k
    - search_type: semantic, question, code, hybrid, vector
      Note: "vector" is mapped to "semantic" for backward compatibility
    - model_name: Override embedding model
    - quantization: Override quantization (float32, float16, int8)
    - collection: Target collection name
    - operation_uuid: Operation UUID for collection inference
    - filters: Metadata filters
    - include_content: Include chunk content in results
    - use_reranker: Enable cross-encoder reranking
    - rerank_model: Override reranker model
    - rerank_quantization: Override reranker quantization
    - score_threshold: Minimum score threshold (0.0-1.0)
    - hybrid_alpha: Vector vs keyword weight (0.0-1.0, default 0.7)
    - hybrid_mode: 'filter' or 'weighted'
    - keyword_mode: 'any' or 'all'
    """
```

#### GET `/hybrid_search`
Specialized endpoint for hybrid search combining vector and keyword matching.

```python
@router.get("/hybrid_search", response_model=HybridSearchResponse)
async def hybrid_search(
    q: str,
    k: int = 10,
    collection: str | None = None,
    mode: str = "filter",  # filter or weighted
    keyword_mode: str = "any",  # any or all
    score_threshold: float | None = None,
    model_name: str | None = None,
    quantization: str | None = None,
):
    """
    Hybrid search with keyword filtering or weighted blending.
    Supports model_name and quantization overrides.
    """
```

#### GET `/keyword_search`
Keyword-only search without vector similarity.

```python
@router.get("/keyword_search", response_model=HybridSearchResponse)
async def keyword_search(
    q: str,
    k: int = 10,
    collection: str | None = None,
    mode: str = "any",  # any or all keyword matching
):
    """
    Pure keyword-based search without vector embeddings.
    Returns results matching the extracted keywords.
    """
```

#### POST `/search/batch`
Batch processing for multiple queries.

```python
@router.post("/search/batch", response_model=BatchSearchResponse)
async def batch_search(request: BatchSearchRequest):
    """
    Process multiple search queries in parallel
    - Efficient batch embedding generation
    - Parallel Qdrant queries
    - Consolidated response
    - Up to 100 queries per batch
    """
```

### Additional Endpoints

#### Health and Status Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check with collection info and embedding status |
| `/health` | GET | Comprehensive health check with component status |
| `/model/status` | GET | Current model manager status |
| `/collection/info` | GET | Detailed collection information |

#### Model Management Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/models` | GET | List available embedding models with provider info |
| `/models/load` | POST | Explicitly load a specific model (eager loading) |
| `/models/suggest` | GET | Suggest optimal model based on GPU memory |
| `/embedding/info` | GET | Current embedding configuration and status |

#### Embedding and Vector Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/embed` | POST | Generate embeddings for a batch of texts |
| `/upsert` | POST | Upsert vector points into Qdrant |

### Key Features

- **Model Management**: Lazy loading with automatic unloading
- **Collection Metadata**: Automatic model/quantization detection
- **Error Handling**: Comprehensive error responses with fallbacks
- **Metrics**: Prometheus metrics for all operations
- **Mock Mode**: Development mode with simulated embeddings

## 3.5. WebUI Search API v2 Endpoints

The WebUI v2 search API adds multi-collection search and proxies requests to the Vecpipe Search API.

### Core Endpoints

#### POST `/api/v2/search`
Search across multiple collections (up to 10) with optional reranking and hybrid scoring:

```python
# CollectionSearchRequest schema
{
    "collection_uuids": ["uuid-1", "uuid-2"],  # Required, 1-10 UUIDs
    "query": "search query",                    # Required, 1-1000 chars
    "k": 20,                                    # 1-100, default 10
    "search_type": "semantic",                  # semantic, question, code, hybrid
    "use_reranker": True,                       # default True
    "rerank_model": None,                       # Optional reranker override
    "score_threshold": 0.5,                     # 0.0-1.0, default 0.0
    "metadata_filter": {"mime_type": "text/markdown"},
    "include_content": True,                    # default True
    "hybrid_alpha": 0.7,                        # Vector vs keyword weight (0-1)
    "hybrid_mode": "weighted",                  # 'filter' or 'weighted'
    "keyword_mode": "any"                       # 'any' or 'all'
}
```

#### POST `/api/v2/search/single`
Single collection search (backward compatibility for older clients):

```python
# SingleCollectionSearchRequest schema
{
    "collection_id": "uuid-here",  # Required UUID
    "query": "search query",       # Required, 1-1000 chars
    "k": 10,                       # 1-100, default 10
    "search_type": "semantic",     # semantic, question, code, hybrid
    "use_reranker": False,         # default False
    "score_threshold": 0.0,        # 0.0-1.0
    "metadata_filter": None,       # Optional metadata filters
    "include_content": True        # default True
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

The embedding search system is powered by the plugin-aware `ModelManager` and the `EmbeddingProviderFactory`.

### Model Manager

The `ModelManager` (`packages/vecpipe/model_manager.py`) manages embedding model lifecycle using the plugin-aware provider system:

```python
class ModelManager:
    """Manages embedding model lifecycle with lazy loading.

    Uses EmbeddingProviderFactory for auto-detection of appropriate
    providers, enabling support for built-in and third-party plugins.
    """

    def __init__(self, unload_after_seconds: int = 300):
        """Initialize with configurable unload timeout (default 5 min)"""

    async def generate_embedding_async(
        self, text: str, model_name: str,
        quantization: str, instruction: str = None,
        mode: str = None  # 'query' or 'document'
    ) -> list[float]:
        """Generate embedding with automatic model management.

        Args:
            mode: Embedding mode - 'query' for search queries,
                  'document' for indexing. Affects prefix handling.
        """

    async def generate_embeddings_batch_async(
        self, texts: list[str], model_name: str,
        quantization: str, instruction: str = None,
        batch_size: int = 32, mode: str = None
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently"""

    async def unload_model_async(self):
        """Unload model to free memory (async)"""
```

### Key Features

- **Lazy Loading**: Models loaded only when needed
- **Automatic Unloading**: After 5 minutes of inactivity (configurable)
- **Memory Management**: Aggressive garbage collection and GPU cache clearing
- **GPU Support**: CUDA acceleration when available
- **Quantization**: Dynamic quantization support (float32, float16, int8)
- **Plugin Support**: Auto-detection of appropriate provider via factory

### Embedding Generation Process

1. **Model Selection**: Based on collection metadata or request
2. **Instruction Formatting**: Task-specific instructions
3. **Tokenization**: Model-specific tokenization
4. **Embedding**: Generate dense vectors
5. **Normalization**: L2 normalization for cosine similarity

### Embedding Provider Plugin Architecture

Semantik uses a plugin-based architecture for embedding providers, allowing both built-in and third-party providers.

#### Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `EmbeddingProviderFactory` | `packages/shared/embedding/factory.py` | Central dispatch for creating providers |
| `BaseEmbeddingPlugin` | `packages/shared/embedding/plugin_base.py` | Abstract base class for all providers |
| `EmbeddingProviderDefinition` | `packages/shared/embedding/plugin_base.py` | Provider metadata definition |
| `plugin_loader.py` | `packages/shared/embedding/plugin_loader.py` | Entry point discovery |

#### Factory Pattern

```python
from shared.embedding.factory import EmbeddingProviderFactory

# Auto-detect provider for a model
provider = EmbeddingProviderFactory.create_provider("Qwen/Qwen3-Embedding-0.6B")

# Create provider by explicit name
provider = EmbeddingProviderFactory.create_provider_by_name("dense_local")

# Check if model is supported
is_supported = EmbeddingProviderFactory.is_model_supported("my-model")

# List available providers
providers = EmbeddingProviderFactory.list_available_providers()
```

#### Built-in Providers

| Provider | Internal Name | Description |
|----------|---------------|-------------|
| `DenseLocalEmbeddingProvider` | `dense_local` | Local models via sentence-transformers, supports Qwen |
| `MockEmbeddingProvider` | `mock` | Deterministic embeddings for testing |

#### Third-Party Plugin Registration

Plugins are discovered via Python entry points in `pyproject.toml`:

```toml
[project.entry-points."semantik.embedding_providers"]
my_provider = "my_package.embedding:MyEmbeddingProvider"
```

#### Asymmetric Embedding Modes

Many retrieval models need different processing for queries vs documents:

```python
from shared.embedding.types import EmbeddingMode

# For search queries - applies query-specific prefixes/instructions
embeddings = await provider.embed_texts(texts, mode=EmbeddingMode.QUERY)

# For document indexing - typically no prefix
embeddings = await provider.embed_texts(texts, mode=EmbeddingMode.DOCUMENT)
```

**Mode behavior by model type:**
- **E5 models**: `"query: {text}"` vs `"passage: {text}"`
- **BGE models**: Instruction prefix for queries, none for documents
- **Qwen models**: `"Instruct: {task}\nQuery:{text}"` vs raw text

#### Creating a Custom Provider

```python
from shared.embedding.plugin_base import BaseEmbeddingPlugin, EmbeddingProviderDefinition

class MyEmbeddingProvider(BaseEmbeddingPlugin):
    INTERNAL_NAME = "my_provider"
    API_ID = "my-provider"
    PROVIDER_TYPE = "local"  # or "remote", "hybrid"

    @classmethod
    def get_definition(cls) -> EmbeddingProviderDefinition:
        return EmbeddingProviderDefinition(
            api_id="my-provider",
            internal_id="my_provider",
            display_name="My Custom Embeddings",
            description="Custom embedding provider",
            provider_type="local",
            supports_asymmetric=True,
        )

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return model_name.startswith("my-org/")

    async def embed_texts(self, texts, mode=None, **kwargs):
        # Implementation
        pass
```

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

# Search Settings
DEFAULT_COLLECTION=work_docs
SEARCH_API_PORT=8000
```

### Model Unloading Configuration

The model unload timeout is configured in `ModelManager` with a default of 300 seconds (5 minutes). This is currently a hardcoded value in the `ModelManager.__init__` method, not an environment variable:

```python
class ModelManager:
    def __init__(self, unload_after_seconds: int = 300):  # 5 minutes default
        self.unload_after_seconds = unload_after_seconds
```

To customize this value, you would need to pass it when instantiating `ModelManager`.

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

## 9.5. Qdrant Configuration Recommendations

Recommended settings for different use cases. Not system defaults - tune based on your needs.

### Recommended HNSW Index Configuration

```python
# Recommended HNSW parameters for different collection sizes
# Apply these when creating collections based on expected size
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

### Recommended Memory Configuration

```python
# Recommended optimizer configuration for performance
{
    "optimizers_config": {
        "indexing_threshold": 20000,  # Build index after this many vectors
        "memmap_threshold": 0,        # Always use memory-mapped storage
        "max_segment_size": 200_000,  # Maximum vectors per segment
        "default_segment_number": 4   # Parallel segments for better performance
    }
}
```

### Recommended Storage Configuration

```python
# Recommended on-disk vs in-memory configuration
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

## 13. Error Handling

The search API provides comprehensive error handling with meaningful HTTP status codes and detailed error responses.

### HTTP Status Codes

| Code | Error Type | Description |
|------|------------|-------------|
| 400 | Bad Request | Invalid request parameters, dimension mismatch |
| 404 | Not Found | Collection not found |
| 500 | Internal Server Error | Unexpected server errors |
| 502 | Bad Gateway | Qdrant/vector database errors |
| 503 | Service Unavailable | Embedding service errors, model not initialized |
| 507 | Insufficient Storage | GPU out of memory for model loading |

### Error Response Format

All error responses include detailed information for debugging:

```python
# Dimension Mismatch Error (400)
{
    "detail": {
        "error": "dimension_mismatch",
        "message": "Query dimension 384 does not match collection dimension 1024",
        "expected_dimension": 1024,
        "actual_dimension": 384,
        "suggestion": "Use the same model that was used to create the collection, or ensure the model outputs 1024-dimensional vectors"
    }
}

# Insufficient Memory Error (507)
{
    "detail": {
        "error": "insufficient_memory",
        "message": "Cannot load reranker due to insufficient GPU memory",
        "suggestion": "Try using a smaller model or different quantization (float16/int8)"
    }
}

# Qdrant Error (502)
{
    "detail": "Vector database error"
}

# Embedding Service Error (503)
{
    "detail": "Embedding service error: Model manager not initialized. Check logs for details."
}
```

### Error Categories

#### Validation Errors (400)
- **Dimension Mismatch**: Query embedding dimension does not match collection
- **Invalid Parameters**: Invalid search type, k value out of range, etc.
- **Invalid Collection UUID**: Malformed UUID format

#### Resource Errors (404, 507)
- **Collection Not Found**: Specified collection does not exist
- **Insufficient Memory**: Not enough GPU memory to load model

#### Service Errors (502, 503)
- **Qdrant Unavailable**: Cannot connect to vector database
- **Model Manager Not Initialized**: Embedding service not ready
- **Model Load Failure**: Failed to load embedding or reranker model

### Error Handling Best Practices

1. **Check Response Status**: Always verify HTTP status code before parsing response
2. **Parse Error Details**: Error responses include `detail` field with specifics
3. **Implement Retries**: For 502/503 errors, implement exponential backoff
4. **Validate Dimensions**: Ensure embedding model matches collection configuration
5. **Monitor Memory**: Use `/models/suggest` endpoint to check GPU memory before loading large models

### Example Error Handling

```python
import requests

response = requests.post(
    "http://localhost:8000/search",
    json={"query": "test", "k": 10}
)

if response.status_code == 200:
    results = response.json()
elif response.status_code == 400:
    error = response.json()["detail"]
    if error.get("error") == "dimension_mismatch":
        print(f"Dimension mismatch: expected {error['expected_dimension']}")
elif response.status_code == 507:
    print("GPU out of memory - try a smaller model")
elif response.status_code >= 500:
    print(f"Server error: {response.status_code}")
```

## Conclusion

Key strengths:
- Multiple search types (vector, hybrid, keyword, reranked)
- Plugin-based embedding providers
- Lazy loading and quantization for efficiency
- Prometheus monitoring
- Asymmetric embedding support (query vs document)
- Production-ready error handling

Modular design makes it easy to extend while keeping performance high.
