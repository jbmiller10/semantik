# Semantik Document Processing & Vector Pipeline Architecture Specification

## Executive Summary

Semantik's document processing and vector pipeline (vecpipe) architecture implements a sophisticated, memory-bounded streaming system capable of processing documents of unlimited size while maintaining strict resource constraints. The system employs a multi-stage pipeline that includes document parsing, intelligent chunking, embedding generation, vector storage, and hybrid search capabilities.

## Architecture Overview

### Core Design Principles

1. **Memory-Bounded Processing**: All operations maintain memory usage under 100MB regardless of document size
2. **Stream-First Architecture**: Documents are processed as streams with UTF-8 boundary safety
3. **Checkpoint/Resume Capability**: All long-running operations support interruption and resumption
4. **Horizontal Scalability**: Processing can be distributed across multiple workers
5. **Real-Time Progress Updates**: WebSocket streams provide live processing feedback
6. **Zero-Copy Operations**: Memory pooling and buffer reuse minimize allocations

## 1. Document Parsing Architecture

### 1.1 Supported File Formats

The system leverages the **Unstructured** library for unified document parsing, supporting:

- **Text Documents**: TXT, MD, RST, LOG
- **Office Documents**: DOC, DOCX, XLS, XLSX, PPT, PPTX
- **PDF Documents**: With OCR support for scanned documents
- **Web Documents**: HTML, XML, JSON, YAML
- **Code Files**: PY, JS, TS, GO, RUST, etc.
- **Email**: EML, MSG
- **Images**: PNG, JPG (with OCR)

### 1.2 Parsing Pipeline Stages

```python
def extract_and_serialize(filepath: str) -> list[tuple[str, dict[str, Any]]]:
    """
    Stage 1: Document Ingestion
    - Uses unstructured.partition.auto for intelligent format detection
    - Strategy: "auto" allows library to choose optimal parsing method
    
    Stage 2: Element Extraction
    - Extracts structured elements (paragraphs, tables, lists)
    - Preserves document structure and metadata
    
    Stage 3: Metadata Enrichment
    - Page numbers, element types, coordinates
    - File metadata (name, type, size)
    
    Returns: List of (text, metadata) tuples
    """
```

**Key Features:**
- **Automatic Strategy Selection**: The system uses `strategy="auto"` for optimal parsing
- **Table Structure Inference**: `infer_table_structure=True` preserves tabular data
- **Page Break Detection**: `include_page_breaks=True` maintains document structure
- **Coordinate Preservation**: Spatial information for advanced processing

### 1.3 Text Extraction Strategies

1. **Direct Extraction**: For simple text formats
2. **OCR-Based Extraction**: For scanned documents and images
3. **Structure-Preserving Extraction**: For complex documents with tables/formatting
4. **Metadata-Rich Extraction**: Preserves all document metadata

## 2. Chunking Algorithms and Strategies

### 2.1 Base Chunking Architecture

```python
class BaseChunker(ABC):
    """Abstract base for all chunking strategies"""
    
    @abstractmethod
    def chunk_text(self, text: str, doc_id: str, metadata: dict) -> list[ChunkResult]:
        """Synchronous chunking method"""
    
    @abstractmethod
    async def chunk_text_async(self, text: str, doc_id: str, metadata: dict) -> list[ChunkResult]:
        """Asynchronous chunking for I/O bound operations"""
    
    @abstractmethod
    def estimate_chunks(self, text_length: int, config: dict) -> int:
        """Capacity planning estimation"""
```

### 2.2 Implemented Chunking Strategies

#### 2.2.1 Character Chunker
- **Algorithm**: Fixed-size character windows with overlap
- **Parameters**: 
  - `chunk_size`: 1000 characters default
  - `chunk_overlap`: 200 characters default
- **Use Case**: Simple, predictable chunking for uniform text

#### 2.2.2 Recursive Chunker
- **Algorithm**: Hierarchical splitting by separators
- **Separators**: `["\n\n", "\n", ". ", " ", ""]`
- **Features**: Preserves sentence and paragraph boundaries
- **Use Case**: General-purpose text chunking

#### 2.2.3 Semantic Chunker
- **Algorithm**: Embedding-based semantic boundary detection
- **Implementation**: LlamaIndex SemanticSplitterNodeParser
- **Parameters**:
  - `breakpoint_percentile_threshold`: 95 (topic change sensitivity)
  - `buffer_size`: 1 (sentence grouping)
  - `max_chunk_size`: 1000 tokens
- **Process**:
  1. Generate embeddings for sentence groups
  2. Calculate cosine similarity between adjacent groups
  3. Identify breakpoints at similarity valleys
  4. Split at semantic boundaries
- **Use Case**: Topic-coherent chunks for better retrieval

#### 2.2.4 Markdown Chunker
- **Algorithm**: Structure-aware markdown parsing
- **Features**:
  - Respects heading hierarchy
  - Preserves code blocks intact
  - Maintains list structure
- **Use Case**: Technical documentation and markdown files

#### 2.2.5 Hierarchical Chunker
- **Algorithm**: Multi-level document structure preservation
- **Levels**: Document → Section → Subsection → Paragraph
- **Features**: Parent-child chunk relationships
- **Use Case**: Complex documents with clear structure

#### 2.2.6 Hybrid Chunker
- **Algorithm**: Combines multiple strategies
- **Process**:
  1. Apply structural chunking (markdown/hierarchical)
  2. Refine with semantic boundaries
  3. Optimize chunk sizes
- **Use Case**: Best overall performance for diverse content

### 2.3 Streaming Chunking Implementation

```python
class StreamingDocumentProcessor:
    """Memory-bounded streaming processor"""
    
    # Memory Management Constants
    BUFFER_SIZE = 64 * 1024          # 64KB read buffer
    WINDOW_SIZE = 256 * 1024         # 256KB processing window
    MAX_MEMORY = 100 * 1024 * 1024   # 100MB total limit
    
    # Backpressure Control
    HIGH_WATERMARK = 0.8  # Pause at 80% memory
    LOW_WATERMARK = 0.6   # Resume at 60% memory
```

**Key Features:**
- **UTF-8 Boundary Safety**: Never splits multi-byte characters
- **Sliding Window Processing**: Maintains context across boundaries
- **Memory Pool Management**: Reuses buffers to minimize allocations
- **Checkpoint Support**: Can resume from any byte position

## 3. Embedding Generation

### 3.1 Embedding Service Architecture

```python
class DenseEmbeddingService(BaseEmbeddingService):
    """Production embedding service with model management"""
    
    def __init__(self):
        self.model_cache = {}
        self.tokenizer_cache = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
```

### 3.2 Supported Embedding Models

#### Qwen3 Series (Primary)
- **Qwen3-Embedding-0.6B**: Lightweight, 2400MB (float32)
- **Qwen3-Embedding-4B**: Balanced, 16000MB (float32)
- **Qwen3-Embedding-8B**: High-quality, 32000MB (float32)

#### Quantization Support
- **float32**: Full precision
- **float16**: Half precision (50% memory reduction)
- **int8**: 8-bit quantization (75% memory reduction)

### 3.3 Batch Processing Optimization

```python
class BatchManager:
    """Intelligent batch processing with memory management"""
    
    async def process_batches(self, texts: list[str], batch_size: int = 32):
        # Dynamic batch sizing based on available memory
        # Automatic retry with smaller batches on OOM
        # Progress tracking and cancellation support
```

**Optimization Strategies:**
1. **Dynamic Batch Sizing**: Adjusts based on text length and memory
2. **Padding Optimization**: Minimizes wasted computation
3. **Mixed Precision**: Uses float16 for inference when possible
4. **Model Caching**: Keeps frequently used models in memory

### 3.4 Memory Management

```python
MODEL_MEMORY_REQUIREMENTS = {
    ("Qwen3-Embedding-0.6B", "float32"): 2400,   # MB
    ("Qwen3-Embedding-0.6B", "float16"): 1200,
    ("Qwen3-Embedding-0.6B", "int8"): 600,
    # ... additional models
}

def check_memory_availability(model_name, quantization):
    # GPU memory checking
    # Automatic model unloading suggestions
    # Fallback to CPU if needed
```

## 4. Vector Storage and Indexing

### 4.1 Qdrant Integration Architecture

```python
class QdrantManager:
    """High-level Qdrant operations manager"""
    
    def __init__(self):
        self.client = QdrantClient(url=f"http://{host}:{port}")
        self.batch_size = 4000  # Optimal for Qdrant
```

### 4.2 Vector Indexing Strategies

#### Collection Configuration
```python
{
    "vector_config": {
        "size": 1024,  # Dimension
        "distance": "Cosine",
        "hnsw_config": {
            "m": 16,                # Connections per node
            "ef_construct": 100,     # Build-time accuracy
            "full_scan_threshold": 10000
        }
    },
    "optimizers_config": {
        "deleted_threshold": 0.2,
        "vacuum_min_vector_number": 1000,
        "default_segment_number": 4
    }
}
```

#### Indexing Pipeline
1. **Batch Accumulation**: Groups vectors into optimal batch sizes
2. **Parallel Upload**: Multiple workers for high throughput
3. **Retry Logic**: Exponential backoff on failures
4. **Progress Tracking**: Real-time indexing metrics

### 4.3 Metadata Storage Patterns

```python
class PointPayload:
    doc_id: str           # Document identifier
    chunk_id: str         # Chunk identifier
    path: str             # File path
    content: str          # Original text
    metadata: dict        # Additional metadata
    
    # Indexed fields for filtering
    collection_id: str    # Collection identifier
    timestamp: datetime   # Processing time
    chunk_index: int      # Position in document
```

### 4.4 Partitioning Strategy

The system uses PostgreSQL partitioning for chunks table:

```sql
-- Hash partitioning by collection_id
CREATE TABLE chunks_p0 PARTITION OF chunks
    FOR VALUES WITH (modulus 16, remainder 0);
```

Benefits:
- **Improved Query Performance**: Partition pruning
- **Parallel Processing**: Operations on different partitions
- **Maintenance Efficiency**: Per-partition vacuum and reindex

## 5. Search and Retrieval

### 5.1 Search Query Processing Pipeline

```python
async def search_pipeline(query: str, config: SearchConfig):
    # Stage 1: Query Enhancement
    enhanced_query = apply_search_instruction(query, config.mode)
    
    # Stage 2: Embedding Generation
    query_vector = await generate_embedding(enhanced_query)
    
    # Stage 3: Vector Search
    candidates = await qdrant_search(query_vector, limit=config.k * 3)
    
    # Stage 4: Reranking (optional)
    if config.use_reranker:
        results = await rerank_results(query, candidates)
    
    # Stage 5: Result Aggregation
    return aggregate_results(results, config)
```

### 5.2 Ranking and Scoring Algorithms

#### Vector Similarity Scoring
- **Cosine Similarity**: Default for normalized vectors
- **Score Range**: [0, 1] where 1 is perfect match
- **Threshold Filtering**: Configurable minimum score

#### Cross-Encoder Reranking
```python
class CrossEncoderReranker:
    """Qwen3-Reranker implementation"""
    
    def rerank(self, query: str, documents: list[str]):
        # Pairwise query-document scoring
        # Uses language model for contextual understanding
        # Returns relevance probabilities
```

### 5.3 Hybrid Search Implementation

```python
class HybridSearchEngine:
    """Combines vector and keyword search"""
    
    def hybrid_search(self, query_vector, query_text, mode="filter"):
        if mode == "filter":
            # Qdrant filtering with keyword conditions
            filter = build_text_filter(extract_keywords(query_text))
            return qdrant.search(query_vector, query_filter=filter)
        
        elif mode == "rerank":
            # Post-processing combination
            vector_results = qdrant.search(query_vector)
            keyword_scores = calculate_keyword_scores(query_text, vector_results)
            return combine_scores(vector_results, keyword_scores, weights=[0.7, 0.3])
```

#### Keyword Extraction
- **Stop Word Removal**: Filters common words
- **Minimum Length**: 3 characters
- **Mode Options**:
  - `any`: Match any keyword (OR)
  - `all`: Match all keywords (AND)

### 5.4 Result Aggregation Strategies

1. **Score-Based Aggregation**: Weighted combination of similarity scores
2. **Reciprocal Rank Fusion**: Combines rankings from multiple searches
3. **Diversity Optimization**: MMR (Maximal Marginal Relevance) for result diversity

## 6. Performance Optimizations

### 6.1 Caching Strategies

#### Model Caching
```python
class ModelManager:
    """Lazy loading with automatic unloading"""
    
    unload_after_seconds = 300  # 5 minutes
    
    async def ensure_model_loaded(self, model_name):
        if self.current_model != model_name:
            await self.unload_current()
            await self.load_model(model_name)
        self.schedule_unload()
```

#### Result Caching
- **Redis-Based**: Distributed cache for search results
- **Content Hash Keys**: Deduplication of identical queries
- **TTL Management**: Automatic expiration
- **Cache Invalidation**: On collection updates

### 6.2 Batch Processing Optimizations

```python
# Optimal batch sizes by operation
EMBEDDING_BATCH_SIZE = 100       # GPU memory limited
VECTOR_UPLOAD_BATCH_SIZE = 4000  # Qdrant optimal
DOCUMENT_REMOVAL_BATCH_SIZE = 100 # Database transaction size
```

### 6.3 Memory-Bounded Processing Patterns

#### Stream Processing with Backpressure
```python
class StreamingWindow:
    """Sliding window for stream processing"""
    
    def __init__(self, size=256*1024):  # 256KB
        self.buffer = bytearray(size)
        self.position = 0
    
    def slide(self):
        # Keep overlap for context
        overlap = self.size // 4
        self.buffer[:overlap] = self.buffer[-overlap:]
        self.position = overlap
```

#### Memory Pool Management
```python
class MemoryPool:
    """Buffer pooling for zero-allocation processing"""
    
    def __init__(self, buffer_size=64*1024, pool_size=10):
        self.buffers = [bytearray(buffer_size) for _ in range(pool_size)]
        self.available = queue.Queue()
        
    async def acquire_async(self, timeout=10):
        # Context manager for automatic release
        buffer = await self.available.get(timeout=timeout)
        try:
            yield buffer
        finally:
            self.available.put(buffer)
```

### 6.4 Concurrent Processing

#### Parallel Document Processing
```python
executor = ThreadPoolExecutor(max_workers=8)

async def process_documents_parallel(documents):
    tasks = []
    for doc in documents:
        task = asyncio.create_task(
            process_in_executor(executor, doc)
        )
        tasks.append(task)
    
    # Process with controlled concurrency
    semaphore = asyncio.Semaphore(4)
    async with semaphore:
        results = await asyncio.gather(*tasks)
```

#### GPU Utilization Optimization
- **Mixed Precision Training**: float16 for inference
- **Dynamic Batching**: Adjusts to GPU memory
- **Multi-GPU Support**: Model parallelism for large models

## 7. Monitoring and Metrics

### 7.1 Performance Metrics

```python
# Prometheus metrics
search_latency = Histogram(
    "search_api_latency_seconds",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
)

embedding_generation_latency = Histogram(
    "embedding_latency_seconds",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2)
)

collection_metrics = {
    "documents_processed": Counter(),
    "chunks_created": Counter(),
    "vectors_indexed": Counter(),
    "processing_time": Histogram(),
    "memory_usage": Gauge()
}
```

### 7.2 Real-Time Progress Tracking

```python
class OperationProgressTracker:
    """Redis stream-based progress updates"""
    
    async def send_update(self, operation_id, progress):
        stream_key = f"operation-progress:{operation_id}"
        message = {
            "timestamp": datetime.now().isoformat(),
            "progress": progress,
            "stage": self.current_stage,
            "metrics": self.collect_metrics()
        }
        await redis.xadd(stream_key, {"message": json.dumps(message)})
```

## 8. Error Handling and Recovery

### 8.1 Checkpoint/Resume System

```python
class CheckpointManager:
    """Persistent checkpoints for long operations"""
    
    async def save_checkpoint(self, operation_id, state):
        checkpoint = {
            "byte_position": state.position,
            "chunks_processed": state.chunks,
            "pending_bytes": state.buffer,
            "timestamp": datetime.now()
        }
        await self.store.save(operation_id, checkpoint)
    
    async def resume_from_checkpoint(self, operation_id):
        checkpoint = await self.store.load(operation_id)
        if checkpoint:
            logger.info(f"Resuming from {checkpoint['byte_position']}")
            return checkpoint
```

### 8.2 Graceful Degradation

1. **Model Fallback**: Smaller models on memory pressure
2. **Batch Size Reduction**: Automatic on OOM errors
3. **CPU Fallback**: When GPU unavailable
4. **Quality Degradation**: Trade accuracy for availability

### 8.3 Error Recovery Patterns

```python
async def resilient_operation(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await func()
        except MemoryError:
            # Reduce batch size and retry
            await reduce_memory_pressure()
        except NetworkError:
            # Exponential backoff
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Retry {attempt + 1}: {e}")
```

## 9. Security Considerations

### 9.1 Input Validation
- **File Type Verification**: Magic number checking
- **Size Limits**: Configurable per-operation limits
- **Content Sanitization**: XSS and injection prevention
- **Rate Limiting**: Per-user and per-IP limits

### 9.2 Resource Protection
- **Memory Limits**: Hard caps on allocation
- **CPU Throttling**: Priority-based scheduling
- **Disk Space Management**: Automatic cleanup
- **Network Isolation**: Separate processing networks

## 10. Future Enhancements

### 10.1 Planned Optimizations
1. **Sparse Embeddings**: Hybrid dense-sparse representations
2. **Learned Indexing**: ML-optimized index structures
3. **Adaptive Chunking**: Content-aware dynamic strategies
4. **Multi-Modal Support**: Image and audio embeddings

### 10.2 Scalability Improvements
1. **Distributed Processing**: Multi-node coordination
2. **Streaming Replication**: Real-time vector sync
3. **Federated Search**: Cross-instance queries
4. **Edge Deployment**: Lightweight inference models

## Conclusion

Semantik's document processing and vector pipeline represents a sophisticated, production-ready system designed for scalability, reliability, and performance. The architecture's emphasis on streaming processing, memory management, and intelligent chunking strategies ensures it can handle diverse workloads while maintaining consistent quality and performance characteristics.

The system's modular design allows for easy extension and optimization, while comprehensive monitoring and error handling ensure production reliability. This architecture provides a solid foundation for building advanced semantic search applications at scale.