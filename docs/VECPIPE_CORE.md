# VecPipe Core Engine Documentation

## Table of Contents
1. [Core Architecture Overview](#core-architecture-overview)
2. [Document Extraction Pipeline](#document-extraction-pipeline)
3. [Embedding Service Architecture](#embedding-service-architecture)
4. [Vector Database Integration](#vector-database-integration)
5. [Model Management System](#model-management-system)
6. [Configuration and Utilities](#configuration-and-utilities)
7. [Hybrid Search Implementation](#hybrid-search-implementation)
8. [Qwen3 Optimization](#qwen3-optimization)

## Core Architecture Overview

### Purpose and Design Philosophy

VecPipe is a high-performance, resource-efficient semantic search engine designed for technical users who prioritize performance and control. The core engine follows a modular pipeline architecture that separates concerns into distinct stages:

1. **Document Extraction** - Parsing and chunking documents into processable units
2. **Embedding Generation** - Converting text chunks into vector representations
3. **Vector Storage** - Ingesting vectors into Qdrant for similarity search
4. **Search API** - Exposing search functionality via REST endpoints

### Key Components and Their Interactions

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Document Files  │────▶│ extract_chunks   │────▶│ Parquet Files   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Qdrant Database │◀────│ ingest_qdrant    │◀────│ embed_chunks    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                                                 │
         │                                                 ▼
         │                                        ┌─────────────────┐
         │                                        │ Model Manager   │
         │                                        └─────────────────┘
         ▼
┌─────────────────┐     ┌──────────────────┐
│   Search API    │────▶│  Hybrid Search   │
└─────────────────┘     └──────────────────┘
```

### Data Flow Through the Pipeline

1. **Input Stage**: Raw documents are read from the filesystem
2. **Extraction Stage**: Documents are parsed using the `unstructured` library and chunked by token count
3. **Embedding Stage**: Text chunks are converted to dense vectors using transformer models
4. **Storage Stage**: Vectors with metadata are stored in Qdrant collections
5. **Search Stage**: Query vectors are compared against stored vectors for similarity search

## Document Extraction Pipeline

### extract_chunks.py: Document Parsing and Chunking

The extraction module provides robust document parsing with change tracking and metadata preservation.

#### Core Classes

**TokenChunker**
```python
class TokenChunker:
    def __init__(self, model_name: str = "cl100k_base", chunk_size: int = 600, chunk_overlap: int = 200):
        """Initialize tokenizer for chunking"""
```

Key features:
- Uses tiktoken for accurate token counting
- Configurable chunk size and overlap
- Smart sentence boundary detection
- Prevents infinite loops with validation

**FileChangeTracker**
```python
class FileChangeTracker:
    def __init__(self, db_path: str = None):
        """Track file changes using SHA256 and SCD-like approach"""
```

Features:
- SHA256 hash-based change detection
- JSON-based persistence
- Tracks processing history and statistics
- Identifies removed files for cleanup

#### Supported File Formats

The system uses the `unstructured` library which supports:
- PDF documents
- Microsoft Office files (DOCX, XLSX, PPTX)
- Plain text files
- HTML/XML documents
- Markdown files
- Images with OCR
- Email formats (EML, MSG)

#### Chunking Strategies

1. **Token-Based Chunking**: Primary strategy using tiktoken
   - Default: 600 tokens per chunk with 200 token overlap
   - Ensures consistent chunk sizes for model input

2. **Sentence Boundary Preservation**: 
   - Searches for natural break points in last 10% of chunk
   - Looks for sentence endings (., !, ?, \n\n)
   - Prevents mid-sentence splits when possible

3. **Metadata Preservation**:
   - Page numbers for PDFs
   - Element types (title, paragraph, table, etc.)
   - File metadata (name, type, path)

#### Error Handling

- Graceful fallback for unsupported formats
- Error logging to `error_extract.log`
- Resume capability via output file checking
- Validation of existing outputs before skipping

## Embedding Service Architecture

### embed_chunks_unified.py: Unified Embedding Generation

The embedding service provides a unified interface for generating embeddings with support for multiple models and quantization levels.

#### Architecture

```python
async def process_file_async(file_path: str, output_dir: str, embedding_service: EmbeddingService, args) -> str | None:
    """Process a single file asynchronously"""
```

Key features:
- Asynchronous processing with configurable concurrency
- Adaptive batch sizing for OOM prevention
- Progress tracking with tqdm
- Metrics collection for monitoring

#### Model Loading and Management

The `EmbeddingService` class (from webui package) handles:

1. **Model Types**:
   - Standard Sentence Transformers models
   - Qwen3 embedding models with special handling
   - Support for instruction-based embeddings

2. **Quantization Support**:
   - **float32**: Full precision (default)
   - **float16**: Half precision for GPU memory savings
   - **int8**: 8-bit quantization using bitsandbytes

3. **Memory Management**:
   - Automatic model unloading
   - GPU cache clearing
   - Lazy loading on demand

#### Batch Processing Strategies

1. **Adaptive Batching**:
   ```python
   if torch.cuda.OutOfMemoryError:
       current_batch_size = max(self.min_batch_size, current_batch_size // 2)
   ```
   - Automatically reduces batch size on OOM
   - Restores batch size after successful runs
   - Minimum batch size of 4 to prevent stalls

2. **Parallel File Processing**:
   - Concurrent file I/O with semaphore limiting
   - Asynchronous parquet reading/writing
   - Progress tracking across all files

#### GPU Utilization

- Automatic device selection (CUDA/CPU)
- Mixed precision training with autocast
- Memory usage tracking and reporting
- Peak memory logging for optimization

## Vector Database Integration

### ingest_qdrant.py: Qdrant Vector Storage

The ingestion module handles bulk loading of vectors into Qdrant with reliability features.

#### Key Features

1. **Batch Upload Strategy**:
   ```python
   BATCH_SIZE = 4000  # Optimized for Qdrant performance
   ```
   - Large batches for efficient network usage
   - Point deduplication via UUIDs
   - Progress tracking with tqdm

2. **Retry Mechanism**:
   ```python
   MAX_RETRIES = 5
   RETRY_DELAY = 2  # Exponential backoff
   ```
   - Exponential backoff for transient failures
   - Per-batch retry logic
   - Failure isolation to prevent data loss

3. **File Management**:
   - Successful files moved to `loaded/` directory
   - Failed files moved to `rejects/` directory
   - Preserves original filenames for traceability

#### Metadata Handling

Each vector point includes:
```python
payload = {
    "doc_id": doc_id,
    "chunk_id": chunk_id,
    "path": file_path,
    "content": full_text,  # For hybrid search
    "metadata": {
        "page_number": page_num,
        "element_type": element_type,
        "filename": filename,
        # ... additional metadata
    }
}
```

## Model Management System

### model_manager.py: GPU Memory Management

The ModelManager provides intelligent model lifecycle management with lazy loading and automatic unloading.

#### Core Functionality

```python
class ModelManager:
    def __init__(self, unload_after_seconds: int = 300):
        """Manages embedding model lifecycle with lazy loading"""
```

Features:
1. **Lazy Loading**: Models loaded only when needed
2. **Automatic Unloading**: Frees GPU memory after inactivity
3. **Model Switching**: Seamless switching between models/quantizations
4. **Thread Safety**: Lock-based synchronization for concurrent access

#### Resource Optimization

1. **Memory Management**:
   - Tracks last usage timestamp
   - Schedules unloading with asyncio tasks
   - Forces garbage collection on unload
   - Clears CUDA cache explicitly

2. **Model Caching**:
   - Caches loaded models by key (model_name + quantization)
   - Avoids redundant loading
   - Validates model state before use

## Configuration and Utilities

### config.py: Centralized Configuration

The configuration system uses Pydantic for validation and environment variable support:

```python
class Settings(BaseSettings):
    # Qdrant Configuration
    QDRANT_HOST: str
    QDRANT_PORT: int = 6333
    DEFAULT_COLLECTION: str = "work_docs"
    
    # Embedding Configuration
    DEFAULT_EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"
    DEFAULT_QUANTIZATION: str = "float16"
```

Key aspects:
- Environment variable override via `.env` file
- Automatic directory creation
- Path resolution relative to project root
- Type validation and defaults

### search_utils.py: Search Utilities

Shared utilities for search operations:

```python
async def search_qdrant(
    qdrant_host: str,
    qdrant_port: int,
    collection_name: str,
    query_vector: list[float],
    k: int,
    with_payload: bool = True,
) -> list[dict]:
```

### metrics.py: Performance Monitoring

Comprehensive metrics using Prometheus:

1. **Job Metrics**:
   - Jobs created/completed/failed
   - Job duration histogram

2. **Pipeline Metrics**:
   - Files processed per stage
   - Chunks created
   - Embeddings generated

3. **Resource Metrics**:
   - GPU memory usage
   - GPU utilization
   - CPU/Memory usage

4. **Performance Metrics**:
   - Extraction duration
   - Embedding generation time
   - Ingestion latency

### cleanup.py: Resource Cleanup

Handles removal of vectors for deleted documents:

```python
class QdrantCleanupService:
    def cleanup_removed_files(self, current_files: list[str], dry_run: bool = False) -> dict:
```

Features:
- Tracks removed files via FileChangeTracker
- Deletes from all collections (including job collections)
- Dry-run mode for safety
- Detailed logging and statistics

## Hybrid Search Implementation

### hybrid_search.py: Combined Vector and Keyword Search

The hybrid search engine provides multiple search strategies:

#### Search Modes

1. **Filter Mode**: Uses Qdrant's built-in text filtering
   ```python
   def build_text_filter(self, keywords: list[str], mode: str = "any") -> Filter | None:
   ```
   - Efficient for large datasets
   - Native Qdrant filtering
   - AND/OR keyword matching

2. **Rerank Mode**: Post-processes results with keyword scoring
   ```python
   # Weighted combination (70% vector, 30% keywords)
   combined_score = 0.7 * hit.score + 0.3 * normalized_keyword_score
   ```
   - Better for nuanced ranking
   - Combines vector and keyword relevance
   - Configurable weight balance

#### Keyword Extraction

- Removes common stop words
- Filters short words (<3 characters)
- Case-insensitive matching
- Configurable stop word list

## Qwen3 Optimization

### qwen3_search_config.py: Qwen3-Specific Configurations

Optimized configurations for Qwen3 embedding models:

#### Model Recommendations

```python
QWEN3_MODEL_RECOMMENDATIONS = {
    "high_quality": {
        "model": "Qwen/Qwen3-Embedding-8B",
        "quantization": "int8",
        "description": "Best quality, MTEB #1, 4096d embeddings",
    },
    "balanced": {
        "model": "Qwen/Qwen3-Embedding-4B",
        "quantization": "float16",
        "description": "Great balance of quality and speed, 2560d embeddings",
    },
    "fast": {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "quantization": "float16",
        "description": "Fast inference, good quality, 1024d embeddings",
    }
}
```

#### Domain-Specific Instructions

```python
DOMAIN_INSTRUCTIONS = {
    "technical": {
        "index": "Represent this technical documentation for retrieval:",
        "query": "Represent this technical question for finding relevant documentation:",
    },
    "code": {
        "index": "Represent this code snippet for similarity search:",
        "query": "Represent this code query for finding similar implementations:",
    }
}
```

#### Batch Processing Optimization

Model-specific batch configurations:
- Qwen3-8B: 8-32 batches depending on quantization
- Qwen3-4B: 16-64 batches
- Qwen3-0.6B: 64-256 batches

#### Search Optimizations

```python
SEARCH_OPTIMIZATIONS = {
    "enable_instruction_tuning": True,
    "normalize_embeddings": True,
    "use_last_token_pooling": True,  # Qwen3's optimal strategy
    "enable_caching": True,
    "parallel_encoding": True,
    "adaptive_batch_sizing": True,
}
```

## Search API Implementation

### search_api.py: REST API Service

The search API provides multiple endpoints for different search strategies:

#### Endpoints

1. **GET /search**: Basic semantic search
2. **POST /search**: Advanced search with filters
3. **GET /hybrid_search**: Combined vector and keyword search
4. **POST /search/batch**: Batch search for multiple queries
5. **GET /keyword_search**: Keyword-only search

#### Key Features

1. **Model Manager Integration**: Lazy loading and automatic unloading
2. **Collection Metadata**: Automatic model/quantization detection
3. **Instruction Support**: Task-specific embeddings for better results
4. **Mock Mode**: Testing without GPU requirements
5. **Comprehensive Metrics**: Latency, errors, and resource tracking

#### Error Handling

- Graceful degradation for model loading failures
- Automatic batch size reduction on OOM
- Detailed error messages and logging
- HTTP status codes for different error types

## Performance Characteristics

### Memory Requirements

| Model | float32 | float16 | int8 |
|-------|---------|---------|------|
| Qwen3-0.6B | 2.4GB | 1.2GB | 0.6GB |
| Qwen3-4B | 16GB | 8GB | 4GB |
| Qwen3-8B | 32GB | 16GB | 8GB |

### Processing Speeds

- Document extraction: ~50-100 docs/minute (varies by size)
- Embedding generation: ~1000-5000 chunks/minute (GPU-dependent)
- Vector ingestion: ~10,000 vectors/minute to Qdrant

### Optimization Tips

1. Use float16 or int8 quantization for larger models
2. Adjust batch sizes based on GPU memory
3. Enable instruction tuning for domain-specific tasks
4. Use hybrid search for better precision
5. Monitor metrics to identify bottlenecks

## Conclusion

The VecPipe core engine provides a robust, scalable foundation for semantic search applications. Its modular architecture allows for easy customization while maintaining high performance through careful resource management and optimization strategies.