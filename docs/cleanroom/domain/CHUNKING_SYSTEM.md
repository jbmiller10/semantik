# CHUNKING_SYSTEM - Document Chunking Component

## 1. Component Overview

The CHUNKING_SYSTEM is a sophisticated document processing component in Semantik that breaks down large documents into semantically meaningful chunks for vector embedding and search. It implements multiple chunking strategies, supports streaming processing for large documents, and provides robust error handling and recovery mechanisms.

### Core Purpose
- Transform documents of arbitrary size into optimally-sized chunks for embedding
- Preserve semantic coherence and document structure in chunks
- Handle documents up to 100MB while maintaining bounded memory usage (100MB limit)
- Support multiple chunking strategies optimized for different content types
- Enable streaming processing with checkpoint/resume capability

### Key Capabilities
- **6 Chunking Strategies**: Character, Recursive, Semantic, Markdown, Hierarchical, Hybrid
- **Streaming Processing**: Process arbitrarily large documents with bounded memory
- **UTF-8 Safety**: Never splits UTF-8 character boundaries
- **ReDoS Protection**: Safe regex execution with timeout protection
- **Memory Management**: Automatic memory pool management with leak detection
- **Error Recovery**: Comprehensive error handling with retry strategies
- **Progress Tracking**: Real-time progress updates via Redis streams

## 2. Architecture & Design Patterns

### Domain-Driven Design Structure
```
packages/shared/chunking/
├── domain/                    # Core business logic
│   ├── entities/              # Chunk, ChunkCollection, ChunkingOperation
│   ├── services/              # Chunking strategies
│   │   ├── chunking_strategies/
│   │   └── streaming_strategies/
│   └── value_objects/         # ChunkConfig, ChunkMetadata, OperationStatus
├── application/               # Use cases and DTOs
│   ├── dto/                   # Request/Response DTOs
│   └── use_cases/            # Preview, Process, Compare, etc.
└── infrastructure/           # Technical implementations
    ├── streaming/            # Memory pool, checkpoints, processor
    └── repositories/         # Partition manager
```

### Design Patterns Employed

#### 1. Strategy Pattern
All chunking strategies implement the `ChunkingStrategy` base class:
```python
class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, content: str, config: ChunkConfig, 
              progress_callback: Callable[[float], None] | None = None) -> list[Chunk]
    
    @abstractmethod
    def validate_content(self, content: str) -> tuple[bool, str | None]
    
    @abstractmethod
    def estimate_chunks(self, content_length: int, config: ChunkConfig) -> int
```

#### 2. Factory Pattern
Strategy creation is handled by factories:
- `ChunkingStrategyFactory` - Creates strategy instances
- `ChunkingConfigBuilder` - Builds validated configurations

#### 3. Repository Pattern
Data access is abstracted through repositories:
- `CollectionRepository` - Collection operations
- `DocumentRepository` - Document operations
- `OperationRepository` - Operation tracking

#### 4. Circuit Breaker Pattern
External service calls use circuit breakers to prevent cascading failures:
```python
CHUNKING_CIRCUIT_BREAKER = ChunkingCircuitBreaker(
    FAILURE_THRESHOLD=5,
    SUCCESS_THRESHOLD=2,
    TIMEOUT_SECONDS=60,
    HALF_OPEN_REQUESTS=3
)
```

## 3. Key Interfaces & Contracts

### ChunkingStrategy Interface
```python
# Base interface for all strategies
class ChunkingStrategy(ABC):
    def chunk(self, content: str, config: ChunkConfig, 
              progress_callback: Callable[[float], None] | None = None) -> list[Chunk]
    def validate_content(self, content: str) -> tuple[bool, str | None]
    def estimate_chunks(self, content_length: int, config: ChunkConfig) -> int
    def count_tokens(self, text: str) -> int
    def calculate_overlap_size(self, chunk_size: int, overlap_percentage: float) -> int
    def find_sentence_boundary(self, text: str, target_position: int, prefer_before: bool = True) -> int
    def find_word_boundary(self, text: str, target_position: int, prefer_before: bool = True) -> int
    def clean_chunk_text(self, text: str) -> str
```

### StreamingChunkingStrategy Interface
```python
class StreamingChunkingStrategy(ABC):
    async def process_window(self, window: StreamingWindow, config: ChunkConfig, 
                           is_final: bool = False) -> list[Chunk]
    def get_buffer_size(self) -> int
    def get_max_buffer_size(self) -> int
    def reset(self) -> None
    def validate_memory_constraint(self) -> bool
```

### ChunkConfig Schema
```python
@dataclass
class ChunkConfig:
    strategy_name: str
    min_tokens: int = 100
    max_tokens: int = 1000
    overlap_tokens: int = 50
    separator: str = " "
    preserve_structure: bool = True
    semantic_threshold: float = 0.7  # For semantic chunking
    hierarchy_levels: int = 3        # For hierarchical chunking
    strategies: list[str] | None     # For hybrid chunking
    weights: dict[str, float] | None # For hybrid chunking
    
    # Security: Whitelist of allowed additional parameters
    ALLOWED_ADDITIONAL_PARAMS = {
        "chunk_size", "chunk_overlap", "encoding", "language",
        "preserve_whitespace", "max_retries", "timeout", "batch_size",
        "metadata", "similarity_threshold", "hierarchy_level",
        "strategies", "weights", "adaptive_weights"
    }
```

### Chunk Entity
```python
@dataclass
class Chunk:
    content: str
    metadata: ChunkMetadata
    min_tokens: int
    max_tokens: int
    
    # Validation in __post_init__:
    # - Token count must be between min_tokens and max_tokens
    # - Content cannot be empty
    # - Metadata must be valid
```

### ChunkMetadata
```python
@dataclass(frozen=True)
class ChunkMetadata:
    chunk_id: str
    document_id: str
    chunk_index: int
    start_offset: int
    end_offset: int
    token_count: int
    strategy_name: str
    created_at: datetime
    
    # Optional fields
    semantic_score: float | None = None
    semantic_density: float | None = None
    confidence_score: float | None = None
    hierarchy_level: int | None = None
    section_title: str | None = None
    custom_attributes: dict[str, Any] | None = None
```

## 4. Data Flow & Dependencies

### Document Processing Pipeline
```
1. Document Input
   ├── Validation (size, format, encoding)
   ├── Strategy Selection (based on content type)
   └── Configuration Building

2. Chunking Process
   ├── For Small Documents (<10MB)
   │   └── Direct Strategy Processing
   └── For Large Documents (>10MB)
       └── Streaming Processing
           ├── Window Creation (256KB windows)
           ├── Strategy Processing per Window
           ├── Checkpoint Management
           └── Memory Pool Management

3. Chunk Production
   ├── Token Counting
   ├── Metadata Generation
   ├── Validation
   └── Storage/Streaming Output

4. Post-Processing
   ├── Embedding Generation (external service)
   ├── Vector Storage (Qdrant)
   └── Metadata Storage (PostgreSQL)
```

### Service Dependencies
```python
ChunkingService
├── ChunkingErrorHandler      # Error recovery
├── ChunkingInputValidator    # Input validation
├── ChunkingConfigBuilder     # Config construction
├── ChunkingStrategyFactory   # Strategy creation
├── CacheManager             # Preview caching
├── CollectionRepository     # Collection access
├── DocumentRepository       # Document access
└── Redis Client            # Progress tracking
```

## 5. Critical Implementation Details

### 5.1 Character Chunking Strategy
**File**: `packages/shared/chunking/domain/services/chunking_strategies/character.py`

Fixed-size chunking with overlap:
```python
def chunk(self, content: str, config: ChunkConfig, ...) -> list[Chunk]:
    # Calculate sizes
    chars_per_token = 4  # Domain approximation
    chunk_size_chars = config.max_tokens * chars_per_token
    overlap_chars = config.overlap_tokens * chars_per_token
    
    # Process with boundary preservation
    while position < total_chars:
        end = min(position + chunk_size_chars, total_chars)
        
        # Adjust to sentence/word boundaries
        if end < total_chars:
            sentence_boundary = self.find_sentence_boundary(content, end, prefer_before=True)
            if sentence_boundary > start:
                end = sentence_boundary
            else:
                end = self.find_word_boundary(content, end, prefer_before=True)
        
        # Create chunk with metadata
        chunk_text = content[start:end]
        # ... create Chunk entity
```

### 5.2 Recursive Chunking Strategy
**File**: `packages/shared/chunking/domain/services/chunking_strategies/recursive.py`

Hierarchical separator-based splitting:
```python
SEPARATORS = [
    "\n\n\n",  # Major sections
    "\n\n",    # Paragraphs
    "\n",      # Lines
    ". ",      # Sentences
    "! ", "? ", "; ", ", ",  # Punctuation
    " ",       # Words
    "",        # Characters (last resort)
]

def _recursive_split(self, text: str, max_tokens: int, ...):
    # Try separators in order
    for separator in SEPARATORS:
        parts = text.split(separator)
        # Recursively split large parts
        # Group small parts to meet min_tokens
```

### 5.3 Semantic Chunking Strategy
**File**: `packages/shared/chunking/domain/services/chunking_strategies/semantic.py`

Groups semantically similar content:
```python
def _create_semantic_clusters(self, sentences: list[dict], max_tokens: int, threshold: float):
    clusters = []
    current_cluster = {"sentences": [], "tokens": 0}
    
    for sentence in sentences:
        # Calculate similarity with last sentence
        if current_cluster["sentences"]:
            similarity = self._calculate_similarity(
                current_cluster["sentences"][-1], 
                sentence["text"]
            )
            
            # Start new cluster if similarity below threshold
            if similarity < threshold:
                clusters.append(current_cluster)
                current_cluster = {"sentences": [sentence["text"]], ...}
            else:
                current_cluster["sentences"].append(sentence["text"])
```

### 5.4 Markdown Chunking Strategy
**File**: `packages/shared/chunking/domain/services/chunking_strategies/markdown.py`

Structure-aware chunking with ReDoS protection:
```python
def __init__(self):
    self.safe_regex = SafeRegex(timeout=1.0)
    self.patterns = {
        "heading": r"^#{1,6}\s+\S.*$",      # Bounded pattern
        "code_block_start": r"^```[^`\n]*$", # Safe pattern
        "list_item": r"^[\*\-\+]\s+\S.*$",  # Non-greedy
        # ... other safe patterns
    }

def _parse_document_structure(self, content: str):
    # Parse with timeout protection
    sections = []
    for line in content.split("\n"):
        # Check patterns with timeout
        if self.safe_regex.match_with_timeout(pattern, line, timeout=0.05):
            # Process match
```

### 5.5 Hierarchical Chunking Strategy
**File**: `packages/shared/chunking/domain/services/chunking_strategies/hierarchical.py`

Multi-level chunking with parent-child relationships:
```python
def chunk(self, content: str, config: ChunkConfig, ...):
    levels = min(config.hierarchy_levels, 3)  # Max 3 levels
    
    for level in range(levels):
        # Each level has progressively smaller chunks
        scale_factor = 2**level
        level_max_tokens = base_config.max_tokens // scale_factor
        
        # Create chunks with parent references
        for chunk in level_chunks:
            custom_attrs = {
                "hierarchy_level": level,
                "parent_id": parent_chunk.id if level > 0 else None,
                "summary": self._generate_summary(chunk_text) if level == 0 else None
            }
```

### 5.6 Hybrid Chunking Strategy
**File**: `packages/shared/chunking/domain/services/chunking_strategies/hybrid.py`

Intelligent strategy combination:
```python
def chunk(self, content: str, config: ChunkConfig, ...):
    # Analyze content
    content_analysis = self._analyze_content(content)
    
    if content_analysis["is_mixed"]:
        # Apply different strategies to different sections
        for section in sections:
            if section["type"] == "code":
                strategy = self._character_strategy
            elif section["type"] in ["header", "list", "table"]:
                strategy = self._markdown_strategy
            elif section["type"] == "prose":
                strategy = self._semantic_strategy
            
            section_chunks = strategy.chunk(section["content"], config)
    
    elif config.strategies and len(config.strategies) > 1:
        # Build consensus from multiple strategies
        chunks = self._build_consensus(strategy_results)
```

## 6. Streaming Processing

### StreamingDocumentProcessor
**File**: `packages/shared/chunking/infrastructure/streaming/processor.py`

Handles large documents with bounded memory:
```python
class StreamingDocumentProcessor:
    BUFFER_SIZE = 64 * 1024         # 64KB read buffer
    WINDOW_SIZE = 256 * 1024        # 256KB processing window
    MAX_MEMORY = 100 * 1024 * 1024  # 100MB total limit
    
    async def process_document(self, file_path: str, strategy: ChunkingStrategy, ...):
        async with aiofiles.open(file_path, 'rb') as file:
            window = StreamingWindow(size=self.WINDOW_SIZE)
            
            while True:
                # Read with UTF-8 boundary safety
                chunk = await self._read_safe_chunk(file)
                if not chunk:
                    break
                
                # Process window when full
                if window.is_full():
                    chunks = await strategy.process_window(window, config)
                    async for chunk in chunks:
                        yield chunk
                    
                # Checkpoint periodically
                if bytes_processed > last_checkpoint + CHECKPOINT_INTERVAL:
                    await self.checkpoint_manager.save_checkpoint(...)
```

### Memory Pool Management
**File**: `packages/shared/chunking/infrastructure/streaming/memory_pool.py`

Efficient buffer reuse with leak detection:
```python
class MemoryPool:
    def __init__(self, max_size: int = 100 * 1024 * 1024):
        self._buffers: dict[str, bytearray] = {}
        self._allocations: dict[str, BufferAllocation] = {}
        self._leak_detector = asyncio.create_task(self._detect_leaks())
    
    @contextmanager
    def allocate(self, size: int) -> Generator[ManagedBuffer, None, None]:
        buffer_id = str(uuid4())
        buffer = bytearray(size)
        
        # Track allocation
        self._allocations[buffer_id] = BufferAllocation(
            buffer_id=buffer_id,
            size=size,
            allocated_at=datetime.now(UTC),
            stack_trace=traceback.format_stack()
        )
        
        try:
            yield ManagedBuffer(buffer_id, buffer, self)
        finally:
            self.release(buffer_id)
```

### Checkpoint Management
**File**: `packages/shared/chunking/infrastructure/streaming/checkpoint.py`

Resume capability for interrupted processing:
```python
@dataclass
class Checkpoint:
    operation_id: str
    position: int
    chunks_created: int
    timestamp: datetime
    strategy_state: dict[str, Any]

class CheckpointManager:
    async def save_checkpoint(self, checkpoint: Checkpoint):
        # Save to persistent storage (Redis/PostgreSQL)
        await self.redis_client.hset(
            f"checkpoint:{checkpoint.operation_id}",
            mapping=checkpoint.to_dict()
        )
    
    async def load_checkpoint(self, operation_id: str) -> Checkpoint | None:
        # Load and validate checkpoint
        data = await self.redis_client.hgetall(f"checkpoint:{operation_id}")
        if data and self._validate_checkpoint(data):
            return Checkpoint.from_dict(data)
```

## 7. Security Considerations

### 7.1 ReDoS Protection
**File**: `packages/shared/chunking/utils/safe_regex.py`

Protection against Regular Expression Denial of Service:
```python
class SafeRegex:
    DEFAULT_TIMEOUT = 1.0  # 1 second max execution
    
    def compile_safe(self, pattern: str, use_re2: bool = True):
        # Check pattern complexity
        if self._is_pattern_dangerous(pattern):
            raise ValueError(f"Pattern rejected as potentially dangerous")
        
        if use_re2 and HAS_RE2:
            # RE2 guarantees linear time complexity
            compiled = re2.compile(pattern)
        else:
            # Fall back with timeout protection
            compiled = re.compile(pattern)
    
    def match_with_timeout(self, pattern: str, text: str, timeout: float = None):
        # Execute in separate thread with timeout
        future = self.executor.submit(_match)
        return future.result(timeout=timeout)
```

### 7.2 Input Validation
**File**: `packages/webui/services/chunking_security.py`

Comprehensive security validation:
```python
class ChunkingSecurityValidator:
    MAX_CHUNK_SIZE = 10000
    MIN_CHUNK_SIZE = 50
    MAX_DOCUMENT_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_CHUNKS_PER_DOCUMENT = 50000
    
    @staticmethod
    def validate_chunk_params(params: dict[str, Any]):
        # Type validation
        if not isinstance(chunk_size, int):
            raise ValidationError("chunk_size must be an integer")
        
        # Range validation
        if not (MIN_CHUNK_SIZE <= chunk_size <= MAX_CHUNK_SIZE):
            raise ValidationError(f"chunk_size out of range")
        
        # Relationship validation
        if chunk_overlap >= chunk_size:
            raise ValidationError("overlap must be less than chunk_size")
    
    @staticmethod
    def validate_file_path(path: str) -> Path:
        # Prevent path traversal
        safe_path = Path(path).resolve()
        if ".." in safe_path.parts:
            raise ValidationError("Path traversal detected")
```

### 7.3 Parameter Whitelisting
Configuration parameters are strictly whitelisted:
```python
ALLOWED_ADDITIONAL_PARAMS = {
    "chunk_size", "chunk_overlap", "encoding", "language",
    "preserve_whitespace", "max_retries", "timeout", "batch_size",
    "metadata", "similarity_threshold", "hierarchy_level",
    "strategies", "weights", "adaptive_weights"
}

# Reject unknown parameters
for key in kwargs:
    if key not in ALLOWED_ADDITIONAL_PARAMS:
        raise InvalidConfigurationError(f"Unknown parameter: {key}")
```

## 8. Error Handling

### Error Handler Framework
**File**: `packages/webui/services/chunking_error_handler.py`

Comprehensive error recovery:
```python
class ChunkingErrorHandler:
    async def handle_error(self, error: Exception, context: ErrorContext) -> ErrorHandlingResult:
        error_type = self._classify_error(error)
        
        if error_type == ChunkingErrorType.MEMORY_ERROR:
            return await self._handle_memory_error(error, context)
        elif error_type == ChunkingErrorType.TIMEOUT_ERROR:
            return await self._handle_timeout_error(error, context)
        elif error_type == ChunkingErrorType.PARTIAL_FAILURE:
            return await self._handle_partial_failure(error, context)
    
    async def _handle_memory_error(self, error: Exception, context: ErrorContext):
        # Resource cleanup
        await self._cleanup_resources(context.operation_id)
        
        # Recovery strategy
        if context.retry_count < 2:
            return ErrorHandlingResult(
                recovery_action="retry_with_reduced_batch",
                retry_after=10,
                recommendations=["Reduce batch size", "Use streaming mode"]
            )
```

### Circuit Breaker Implementation
```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_count = 0
        self.success_count = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func: Callable, *args, **kwargs):
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenError()
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
```

## 9. Testing Requirements

### Unit Tests
```python
# Test each strategy independently
async def test_character_chunking_respects_boundaries():
    strategy = CharacterChunkingStrategy()
    content = "This is a test. This is another sentence."
    config = ChunkConfig(strategy_name="character", max_tokens=5)
    
    chunks = strategy.chunk(content, config)
    
    # Verify no partial words
    for chunk in chunks:
        assert not chunk.content.startswith(" ")
        assert not chunk.content.endswith(" ")

# Test memory constraints
async def test_streaming_memory_bounded():
    processor = StreamingDocumentProcessor()
    large_file = create_temp_file(size=200 * 1024 * 1024)  # 200MB
    
    memory_usage = []
    async for chunk in processor.process_document(large_file, ...):
        memory_usage.append(processor.memory_pool.get_current_usage())
    
    assert max(memory_usage) <= 100 * 1024 * 1024  # Never exceeds 100MB
```

### Integration Tests
```python
# Test end-to-end pipeline
async def test_chunking_pipeline():
    service = ChunkingService(db_session, ...)
    
    # Upload document
    document = await service.upload_document(file_path)
    
    # Apply chunking
    operation_id = await service.apply_chunking(
        document_id=document.id,
        strategy="semantic",
        config_overrides={"semantic_threshold": 0.8}
    )
    
    # Wait for completion
    operation = await wait_for_operation(operation_id)
    assert operation.status == OperationStatus.COMPLETED
    
    # Verify chunks
    chunks = await service.get_chunks(document.id)
    assert len(chunks) > 0
    assert all(chunk.metadata.strategy_name == "semantic" for chunk in chunks)
```

### Streaming Tests
```python
# Test checkpoint/resume
async def test_checkpoint_resume():
    processor = StreamingDocumentProcessor()
    
    # Process partially
    chunks_before = []
    async for chunk in processor.process_document(file_path, ...):
        chunks_before.append(chunk)
        if len(chunks_before) == 10:
            break  # Simulate interruption
    
    # Resume from checkpoint
    checkpoint = await processor.checkpoint_manager.load_checkpoint(operation_id)
    chunks_after = []
    async for chunk in processor.resume_from_checkpoint(checkpoint):
        chunks_after.append(chunk)
    
    # Verify no duplicates or gaps
    all_chunks = chunks_before + chunks_after
    assert len(set(c.metadata.chunk_index for c in all_chunks)) == len(all_chunks)
```

### Memory Leak Tests
```python
# Test buffer pool cleanup
async def test_memory_pool_no_leaks():
    pool = MemoryPool(max_size=10 * 1024 * 1024)
    
    # Allocate and release many times
    for _ in range(1000):
        with pool.allocate(1024 * 1024) as buffer:
            buffer.data[:100] = b'test'
    
    # Check for leaks
    leaks = await pool.detect_leaks()
    assert len(leaks) == 0
    assert pool.get_allocated_count() == 0
```

## 10. Common Pitfalls & Best Practices

### Pitfalls to Avoid

1. **Memory Exhaustion**
   - Never load entire document into memory for large files
   - Always use streaming for documents > 10MB
   - Monitor memory usage with metrics

2. **UTF-8 Boundary Splitting**
   ```python
   # BAD: Can split UTF-8 characters
   chunk = content[position:position+size]
   
   # GOOD: Ensure complete characters
   chunk = await self._read_safe_chunk(file)
   ```

3. **ReDoS Vulnerabilities**
   ```python
   # BAD: Unbounded backtracking
   pattern = r"(a+)+" 
   
   # GOOD: Use safe patterns or RE2
   pattern = r"a+"
   ```

4. **Ignoring Chunk Size Validation**
   ```python
   # Always validate chunk sizes
   if token_count < config.min_tokens:
       # Handle appropriately - merge or skip
   ```

### Best Practices

1. **Strategy Selection**
   ```python
   def select_strategy(content_type: str, has_structure: bool) -> str:
       if content_type == "markdown":
           return "markdown"
       elif has_structure:
           return "recursive"
       elif content_type == "code":
           return "character"
       else:
           return "semantic"
   ```

2. **Error Recovery**
   ```python
   # Always provide fallback strategies
   try:
       chunks = semantic_strategy.chunk(content, config)
   except Exception:
       chunks = character_strategy.chunk(content, config)  # Fallback
   ```

3. **Progress Tracking**
   ```python
   # Report progress for long operations
   async def chunk_with_progress(content, config):
       total = len(content)
       processed = 0
       
       for chunk in chunks:
           processed += len(chunk.content)
           progress = (processed / total) * 100
           await redis.publish(f"progress:{operation_id}", progress)
   ```

4. **Resource Cleanup**
   ```python
   # Always use context managers
   async with memory_pool.allocate(size) as buffer:
       # Use buffer
       pass  # Automatically released
   ```

## 11. Configuration & Environment

### Default Configuration
**File**: `packages/webui/services/chunking_config.py`

```python
# Resource Limits
CHUNKING_LIMITS = ChunkingLimits(
    PREVIEW_MEMORY_LIMIT_BYTES=512 * 1024 * 1024,      # 512MB
    OPERATION_MEMORY_LIMIT_BYTES=2 * 1024 * 1024 * 1024, # 2GB
    MAX_DOCUMENT_SIZE_BYTES=100 * 1024 * 1024,         # 100MB
    MAX_CHUNKS_PER_DOCUMENT=10000,
    MAX_CHUNKS_PER_OPERATION=100000
)

# Timeouts
CHUNKING_TIMEOUTS = ChunkingTimeouts(
    PREVIEW_TIMEOUT_SECONDS=30.0,
    OPERATION_SOFT_TIMEOUT_SECONDS=3600,  # 1 hour
    OPERATION_HARD_TIMEOUT_SECONDS=7200,  # 2 hours
    EMBEDDING_SERVICE_TIMEOUT_SECONDS=60.0
)

# Retry Configuration
CHUNKING_RETRY = ChunkingRetry(
    DEFAULT_MAX_RETRIES=3,
    MEMORY_ERROR_MAX_RETRIES=2,
    TIMEOUT_ERROR_MAX_RETRIES=3,
    NETWORK_ERROR_MAX_RETRIES=5,
    RETRY_BACKOFF_ENABLED=True
)
```

### Strategy-Specific Defaults
```python
STRATEGY_DEFAULTS = {
    "character": {
        "min_tokens": 100,
        "max_tokens": 1000,
        "overlap_tokens": 50
    },
    "recursive": {
        "min_tokens": 100,
        "max_tokens": 1000,
        "overlap_tokens": 50,
        "separators": ["\n\n", "\n", ". ", " "]
    },
    "semantic": {
        "min_tokens": 100,
        "max_tokens": 1000,
        "semantic_threshold": 0.7,
        "similarity_method": "word_overlap"
    },
    "markdown": {
        "min_tokens": 100,
        "max_tokens": 1000,
        "preserve_structure": True
    },
    "hierarchical": {
        "hierarchy_levels": 3,
        "level_scaling": 2
    },
    "hybrid": {
        "strategies": ["semantic", "recursive"],
        "weights": {"semantic": 0.6, "recursive": 0.4}
    }
}
```

### Environment Variables
```bash
# Memory limits
CHUNKING_MAX_MEMORY_MB=100
CHUNKING_BUFFER_SIZE_KB=64
CHUNKING_WINDOW_SIZE_KB=256

# Timeouts
CHUNKING_REGEX_TIMEOUT_MS=1000
CHUNKING_OPERATION_TIMEOUT_S=3600

# Redis configuration
REDIS_PROGRESS_CHANNEL="chunking:progress"
REDIS_CHECKPOINT_TTL=86400

# Feature flags
ENABLE_STREAMING_MODE=true
ENABLE_RE2_REGEX=true
ENABLE_MEMORY_PROFILING=false
```

## 12. Integration Points

### Service Layer Integration
**File**: `packages/webui/services/chunking_service.py`

```python
class ChunkingService:
    def __init__(self, db_session: AsyncSession, 
                 collection_repo: CollectionRepository,
                 document_repo: DocumentRepository,
                 redis_client: Redis | None = None):
        self.error_handler = ChunkingErrorHandler()
        self.validator = ChunkingInputValidator()
        self.config_builder = ChunkingConfigBuilder()
        self.strategy_factory = ChunkingStrategyFactory()
        self.cache_manager = CacheManager(redis_client)
    
    async def apply_chunking(self, document_id: str, strategy: str, 
                           config_overrides: dict | None = None) -> str:
        # Validate access
        await self._validate_document_access(document_id, user_id)
        
        # Build configuration
        config_result = self.config_builder.build_config(strategy, config_overrides)
        
        # Create operation
        operation_id = await self.start_chunking_operation(...)
        
        # Queue for processing
        await self.queue_chunking_task(operation_id, document_id, config_result)
        
        return operation_id
```

### Celery Task Integration
**File**: `packages/webui/chunking_tasks.py`

```python
@celery_app.task(
    bind=True,
    max_retries=3,
    soft_time_limit=3600,
    time_limit=7200,
    acks_late=True
)
async def process_chunking_operation(self, operation_id: str, 
                                   document_id: str, config: dict):
    """Background task for chunking operations."""
    async with AsyncSessionLocal() as session:
        service = ChunkingService(session, ...)
        
        try:
            # Track progress
            chunking_tasks_started.labels(operation_type="chunking").inc()
            
            # Process document
            chunks = await service.process_document(document_id, config)
            
            # Generate embeddings
            embeddings = await embedding_service.generate_embeddings(chunks)
            
            # Store in Qdrant
            await vector_store.upsert_chunks(chunks, embeddings)
            
            # Update operation status
            await service.complete_operation(operation_id)
            
        except SoftTimeLimitExceeded:
            # Graceful shutdown
            await service.pause_operation(operation_id)
            self.retry(countdown=60)
```

### API Endpoint Integration
```python
@router.post("/documents/{document_id}/chunk")
async def chunk_document(
    document_id: str,
    request: ChunkingRequest,
    service: ChunkingService = Depends(get_chunking_service)
):
    """Apply chunking to a document."""
    operation_id = await service.apply_chunking(
        document_id=document_id,
        strategy=request.strategy,
        config_overrides=request.config
    )
    
    return {
        "operation_id": operation_id,
        "status": "processing",
        "estimated_time": service.estimate_processing_time(document_id)
    }
```

### Database Integration
```python
# Operation tracking in PostgreSQL
class Operation(Base):
    __tablename__ = "operations"
    
    id = Column(UUID, primary_key=True)
    type = Column(Enum(OperationType))
    status = Column(Enum(OperationStatus))
    config = Column(JSON)
    progress = Column(Float)
    error_details = Column(JSON)
    created_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    # Relationships
    collection_id = Column(UUID, ForeignKey("collections.id"))
    collection = relationship("Collection", back_populates="operations")
```

## 13. Performance Optimizations

### Memory Optimization
1. **Streaming for Large Documents**: Documents > 10MB processed in windows
2. **Buffer Pooling**: Reusable buffers to reduce allocation overhead
3. **Lazy Loading**: Chunks generated on-demand, not all at once
4. **Memory Monitoring**: Automatic backpressure when approaching limits

### Processing Optimization
1. **Parallel Window Processing**: Multiple windows processed concurrently
2. **Batch Token Counting**: Count tokens for multiple chunks at once
3. **Strategy Caching**: Compiled regex patterns and strategy instances cached
4. **Preview Caching**: Cache preview results for 15 minutes

### Database Optimization
1. **Bulk Inserts**: Insert chunks in batches of 100
2. **Async Operations**: All database operations are async
3. **Connection Pooling**: Reuse database connections
4. **Indexed Queries**: Proper indexes on operation_id, document_id

## 14. Monitoring & Metrics

### Prometheus Metrics
```python
# Operation metrics
chunking_tasks_started_total{operation_type="chunking"}
chunking_tasks_completed_total{operation_type="chunking", status="success"}
chunking_tasks_failed_total{operation_type="chunking", error_type="memory"}
chunking_task_duration_seconds{operation_type="chunking"}

# Resource metrics
chunking_operation_memory_usage_bytes{operation_id="..."}
chunking_buffer_pool_usage_ratio
chunking_active_operations_count

# Performance metrics
chunking_chunks_produced_total
chunking_average_chunk_size_bytes
chunking_processing_rate_bytes_per_second
```

### Logging
```python
# Structured logging with correlation IDs
logger.info("Starting chunking operation", extra={
    "operation_id": operation_id,
    "document_id": document_id,
    "strategy": strategy_name,
    "correlation_id": correlation_id
})

# Error logging with context
logger.error("Chunking failed", extra={
    "operation_id": operation_id,
    "error_type": error_type,
    "retry_count": retry_count,
    "stack_trace": traceback.format_exc()
})
```

### Health Checks
```python
async def check_chunking_health() -> dict:
    return {
        "status": "healthy" if all_checks_pass else "degraded",
        "checks": {
            "memory_usage": memory_pool.get_usage_percentage() < 0.8,
            "active_operations": active_ops < MAX_CONCURRENT_OPERATIONS,
            "error_rate": error_rate < 0.1,
            "circuit_breaker": circuit_breaker.state != "OPEN"
        }
    }
```

## 15. Future Considerations

### Planned Enhancements
1. **ML-Based Strategy Selection**: Use machine learning to automatically select optimal strategy
2. **Custom Strategies**: Allow users to define custom chunking strategies
3. **Distributed Processing**: Support multi-node processing for very large datasets
4. **Smart Caching**: Predictive caching based on usage patterns

### Technical Debt
1. **Strategy Interface Consistency**: Some strategies have slightly different interfaces
2. **Test Coverage**: Need more edge case testing for streaming mode
3. **Documentation**: API documentation needs updating for new strategies

### Known Limitations
1. **Maximum Document Size**: 100MB hard limit
2. **Strategy Flexibility**: Limited customization for some strategies
3. **Language Support**: Optimized for English text primarily
4. **Embedding Integration**: Tightly coupled to specific embedding service

---

*This documentation represents the current state of the CHUNKING_SYSTEM component as of the latest codebase analysis. It should be updated as the system evolves.*