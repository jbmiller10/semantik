# WORKER_SERVICE (Celery Worker) - Cleanroom Documentation

## 1. Component Overview

### Purpose
The WORKER_SERVICE is Semantik's asynchronous task processing engine built on Celery. It handles all long-running operations including document indexing, reindexing, chunking, and deletion operations. The worker operates as a separate container/process that consumes tasks from Redis queues and executes them in the background while providing real-time progress updates.

### Core Responsibilities
- **Document Processing**: Extract text, generate embeddings, store vectors in Qdrant
- **Collection Operations**: INDEX, APPEND, REINDEX, REMOVE_SOURCE operations
- **Chunking Operations**: Apply various chunking strategies to documents
- **Progress Reporting**: Send real-time updates via Redis streams and WebSocket channels
- **Resource Management**: Monitor and limit memory/CPU usage per task
- **Error Recovery**: Implement retry logic with exponential backoff
- **Task Lifecycle**: Manage operation states from PENDING through COMPLETED/FAILED
- **Background Maintenance**: Clean up old results, refresh materialized views, monitor partitions

### Role in Async Processing
The worker service decouples heavy computational tasks from the web API, allowing immediate HTTP responses while processing continues in the background. Users can monitor progress through WebSocket connections that receive updates from Redis streams populated by the workers.

## 2. Architecture & Design Patterns

### Celery Configuration
```python
# packages/webui/celery_app.py
celery_app = Celery(
    "webui",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
    include=["webui.tasks"],
)

# Key Configuration Settings
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    
    # Time Limits
    task_soft_time_limit=3600,  # 1 hour soft limit (graceful shutdown)
    task_time_limit=7200,       # 2 hour hard limit (forced termination)
    
    # Reliability
    task_acks_late=True,                    # Acknowledge after execution
    task_reject_on_worker_lost=True,        # Reject on worker shutdown
    worker_prefetch_multiplier=1,           # No prefetching for long tasks
    worker_max_tasks_per_child=100,         # Restart after 100 tasks (prevent memory leaks)
    
    # Connection Retry
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,
    broker_connection_retry_delay=1.0,
    broker_connection_retry_max_delay=30.0,
    broker_connection_retry_backoff_factor=2.0,
    
    # Task Retry
    task_default_retry_delay=60,
    task_max_retries=3,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)
```

### Task Organization Structure

#### Main Task Entry Points
1. **process_collection_operation** - Unified entry for all collection operations
2. **process_chunking_operation** - Dedicated chunking task with enhanced error handling
3. **cleanup_old_collections** - Deferred cleanup of old Qdrant collections
4. **Periodic Tasks** - Scheduled maintenance tasks

#### Task Hierarchy
```
celery_app
├── tasks.py
│   ├── process_collection_operation (main task)
│   ├── cleanup_old_results (periodic)
│   ├── refresh_collection_chunking_stats (periodic)
│   ├── monitor_partition_health (periodic)
│   └── cleanup_old_collections (deferred)
├── chunking_tasks.py
│   ├── process_chunking_operation (ChunkingTask base)
│   ├── retry_failed_chunks
│   └── monitor_dead_letter_queue
└── background_tasks.py
    └── RedisCleanupTask (Redis memory management)
```

### Design Patterns

#### 1. **Unified Task Entry Pattern**
All collection operations flow through a single `process_collection_operation` task for consistency:
```python
@celery_app.task(
    bind=True,
    name="webui.tasks.process_collection_operation",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
    soft_time_limit=3600,
    time_limit=7200,
)
def process_collection_operation(self, operation_id: str) -> dict[str, Any]:
    # Single entry point for INDEX, APPEND, REINDEX, REMOVE_SOURCE
```

#### 2. **Context Manager Pattern for Resources**
```python
class CeleryTaskWithOperationUpdates:
    """Context manager for Redis stream updates"""
    async def __aenter__(self):
        # Initialize Redis connection
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Ensure cleanup even on exceptions
```

#### 3. **Circuit Breaker Pattern (Chunking)**
```python
class ChunkingTask(Task):
    _circuit_breaker_failures = 0
    _circuit_breaker_state = "closed"  # closed, open, half_open
    CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 300
```

## 3. Key Interfaces & Contracts

### Task Signatures

#### process_collection_operation
```python
# Input
operation_id: str  # UUID of the operation record

# Return Value
{
    "status": "success" | "failed",
    "operation_id": str,
    "operation_type": "INDEX" | "APPEND" | "REINDEX" | "REMOVE_SOURCE",
    "collection_id": str,
    "duration": float,  # seconds
    "documents_processed": int,
    "vectors_created": int,
    "error": str | None,
    "metrics": {
        "cpu_seconds": float,
        "memory_peak_mb": float,
        "extraction_time": float,
        "embedding_time": float,
        "storage_time": float
    }
}
```

#### process_chunking_operation
```python
# Input
operation_id: str       # Operation UUID
correlation_id: str     # Request correlation ID

# Return Value
{
    "status": "success" | "partial_success" | "failed",
    "operation_id": str,
    "chunks_created": int,
    "documents_processed": int,
    "documents_failed": int,
    "error_details": list[dict],
    "metrics": {
        "duration_seconds": float,
        "memory_usage_mb": float,
        "cpu_time": float
    }
}
```

### Redis Stream Message Format
```python
# Stream Key: operation-progress:{operation_id}
{
    "timestamp": "2024-01-15T10:30:00Z",
    "type": "progress" | "status_change" | "error" | "completed",
    "data": {
        "status": str,
        "progress": float,  # 0.0 to 1.0
        "current_step": str,
        "message": str,
        "details": dict  # Type-specific data
    }
}
```

### Operation Status Transitions
```
PENDING → PROCESSING → COMPLETED
                    ↘ FAILED
                    ↘ CANCELLED
```

## 4. Data Flow & Dependencies

### Task Queuing Flow
```
API Request
    ↓
CollectionService.create_operation()
    ↓
Database: Create Operation (status=PENDING)
    ↓
celery_app.send_task("webui.tasks.process_collection_operation")
    ↓
Redis Queue (broker)
    ↓
Worker pulls task
    ↓
Task execution begins
```

### Task Execution Flow
```
process_collection_operation(operation_id)
    ↓
1. Load operation from database
2. Update status to PROCESSING
3. Store Celery task_id
    ↓
4. Based on operation.type:
   - INDEX: Process new documents
   - APPEND: Add to existing collection
   - REINDEX: Blue-green deployment
   - REMOVE_SOURCE: Delete documents
    ↓
5. For each document:
   - Extract text (shared.text_processing)
   - Generate chunks
   - Create embeddings (vecpipe service)
   - Store in Qdrant
    ↓
6. Send progress updates via Redis
7. Update operation status
8. Record metrics
```

### Result Handling
```
Task Completion
    ↓
Update Operation (status=COMPLETED/FAILED)
    ↓
Send final Redis stream message
    ↓
WebSocket notifies connected clients
    ↓
Result stored in Redis backend (TTL: 1 hour)
```

### Dependencies
- **Database**: PostgreSQL via SQLAlchemy for operation state
- **Redis**: Message broker and result backend
- **Qdrant**: Vector storage operations
- **Vecpipe Service**: Embedding generation
- **Shared Library**: Text extraction, database models

## 5. Critical Implementation Details

### Task Routing
```python
# No explicit routing - all tasks go to default queue
# Future: Can implement priority queues
celery_app.conf.task_routes = {
    'webui.tasks.cleanup_old_results': {'queue': 'maintenance'},
    'webui.tasks.process_collection_operation': {'queue': 'default'},
}
```

### Retry Logic
```python
# Automatic retry for specific exceptions
@celery_app.task(
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_backoff_max=600,
)
```

### Error Handling Hierarchy
```python
try:
    # Main task logic
except SoftTimeLimitExceeded:
    # Graceful shutdown - save progress
    operation.status = OperationStatus.FAILED
    operation.error_message = "Task exceeded time limit"
except (ValueError, TypeError):
    # Don't retry programming errors
    raise
except Exception as exc:
    # Retry transient failures
    raise self.retry(exc=exc, countdown=60)
finally:
    # Always update operation status
    # Always close connections
```

### Memory Management
```python
# Batch processing to control memory
EMBEDDING_BATCH_SIZE = 100
VECTOR_UPLOAD_BATCH_SIZE = 100

# Forced garbage collection after large operations
gc.collect()

# Worker restart after N tasks
worker_max_tasks_per_child=100
```

### Progress Reporting Granularity
```python
async def send_progress(current: int, total: int, step: str):
    if total > 0:
        progress = current / total
        await updater.send_update("progress", {
            "progress": progress,
            "current": current,
            "total": total,
            "step": step,
            "message": f"{step}: {current}/{total}"
        })
```

## 6. Security Considerations

### Task Validation
- **Operation Ownership**: Tasks verify user owns the collection
- **Input Sanitization**: All file paths and configs validated
- **Resource Limits**: Memory and CPU limits enforced

### Resource Limits
```python
# Memory limit enforcement
CHUNKING_MEMORY_LIMIT_GB = 4
process = psutil.Process()
if process.memory_info().rss > CHUNKING_MEMORY_LIMIT_GB * 1024**3:
    raise ChunkingMemoryError("Memory limit exceeded")

# CPU time limit
CHUNKING_CPU_TIME_LIMIT = 1800  # 30 minutes
if cpu_time > CHUNKING_CPU_TIME_LIMIT:
    raise ChunkingTimeoutError("CPU time limit exceeded")
```

### Message Security
- **JSON Serialization Only**: No pickle for security
- **Redis AUTH**: Password-protected Redis instance
- **TLS**: Redis TLS in production

### Task Isolation
- **Container Isolation**: Worker runs in separate container
- **Process Isolation**: worker_max_tasks_per_child prevents memory leaks
- **Queue Isolation**: Separate queues for different task types (future)

## 7. Testing Requirements

### Unit Tests
```python
# Test task logic without Celery
async def test_process_operation_logic():
    # Mock dependencies
    mock_db = AsyncMock()
    mock_qdrant = MagicMock()
    
    # Test operation processing
    result = await _process_collection_operation_async(
        operation_id="test-op",
        celery_task=MagicMock()
    )
    assert result["status"] == "success"
```

### Integration Tests with Mock Celery
```python
# Use Celery's test utilities
from celery.contrib.testing import worker
from webui.celery_app import celery_app

@pytest.fixture
def celery_worker(celery_session_worker):
    with worker.start_worker(celery_app):
        yield

def test_task_execution(celery_worker):
    result = process_collection_operation.delay("op-123")
    assert result.get(timeout=10)["status"] == "success"
```

### Task State Tests
```python
# Test all status transitions
def test_operation_status_transitions():
    # PENDING → PROCESSING
    # PROCESSING → COMPLETED
    # PROCESSING → FAILED
    # Verify database updates at each step
```

### Progress Reporting Tests
```python
# Verify Redis stream messages
async def test_progress_updates():
    updater = CeleryTaskWithOperationUpdates("op-123")
    await updater.send_update("progress", {"progress": 0.5})
    
    # Verify message in Redis stream
    messages = await redis.xread({"operation-progress:op-123": 0})
    assert json.loads(messages[0]["message"])["data"]["progress"] == 0.5
```

## 8. Common Pitfalls & Best Practices

### Task Idempotency
**Pitfall**: Tasks that can't be safely retried
```python
# BAD: Non-idempotent operation
def process_task(operation_id):
    operation.retry_count += 1  # Will increment on each retry
    
# GOOD: Idempotent operation
def process_task(operation_id):
    if operation.status != OperationStatus.PENDING:
        return  # Already processed
```

### Timeout Handling
**Pitfall**: Not handling soft time limits gracefully
```python
# GOOD: Graceful shutdown
try:
    # Long-running operation
except SoftTimeLimitExceeded:
    # Save progress
    operation.meta = {"progress": current_progress}
    await session.commit()
    raise  # Let Celery handle the retry
```

### Database Session Management
**Pitfall**: Keeping database sessions open across async boundaries
```python
# BAD: Session crosses async boundary
session = AsyncSessionLocal()
await long_running_task()
await session.commit()  # Session may be stale

# GOOD: Fresh session per operation
async with AsyncSessionLocal() as session:
    await quick_db_operation()
```

### Memory Leaks
**Pitfall**: Not clearing large objects from memory
```python
# GOOD: Explicit cleanup
large_embeddings = generate_embeddings(documents)
process_embeddings(large_embeddings)
del large_embeddings  # Explicit cleanup
gc.collect()  # Force garbage collection
```

### Progress Update Frequency
**Pitfall**: Too frequent updates overwhelming Redis/WebSocket
```python
# GOOD: Throttled updates
last_update = 0
for i, doc in enumerate(documents):
    if time.time() - last_update > 0.5:  # Max 2 updates/second
        await send_progress(i, total)
        last_update = time.time()
```

## 9. Configuration & Environment

### Environment Variables
```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Worker Configuration
CELERY_WORKER_CONCURRENCY=4        # Number of worker processes
CELERY_WORKER_PREFETCH_MULTIPLIER=1  # Tasks per worker
CELERY_TASK_SOFT_TIME_LIMIT=3600    # Soft limit in seconds
CELERY_TASK_TIME_LIMIT=7200         # Hard limit in seconds

# Database
DATABASE_URL=postgresql://user:pass@localhost/semantik

# Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Monitoring
PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
```

### Celery Worker Start Command
```bash
# Development
celery -A webui.celery_app worker --loglevel=info --concurrency=2

# Production
celery -A webui.celery_app worker \
    --loglevel=warning \
    --concurrency=4 \
    --max-tasks-per-child=100 \
    --time-limit=7200 \
    --soft-time-limit=3600
```

### Worker Pool Types
```bash
# Prefork (default) - Good for CPU-bound tasks
celery worker --pool=prefork --concurrency=4

# Gevent - Good for I/O-bound tasks
celery worker --pool=gevent --concurrency=100

# Solo - Single-threaded for debugging
celery worker --pool=solo
```

### Periodic Tasks (Beat Scheduler)
```python
beat_schedule={
    "cleanup-old-results": {
        "task": "webui.tasks.cleanup_old_results",
        "schedule": 86400.0,  # Daily
        "args": (7,),  # Keep 7 days
    },
    "refresh-collection-chunking-stats": {
        "task": "webui.tasks.refresh_collection_chunking_stats",
        "schedule": 3600.0,  # Hourly
    },
    "monitor-partition-health": {
        "task": "webui.tasks.monitor_partition_health",
        "schedule": 21600.0,  # Every 6 hours
    },
}
```

## 10. Integration Points

### WebSocket Notifications
```python
# Task sends updates to Redis
await redis_client.xadd(
    f"operation-progress:{operation_id}",
    {"message": json.dumps(update)}
)

# WebSocketManager consumes from Redis
async def _consume_updates(self, operation_id: str):
    while True:
        messages = await redis.xread({
            f"operation-progress:{operation_id}": last_id
        })
        for msg in messages:
            await self._broadcast(operation_id, msg)
```

### Database Updates
```python
# Atomic operation status updates
async with AsyncSessionLocal() as session:
    operation = await operation_repo.get_by_uuid(operation_id)
    operation.status = OperationStatus.PROCESSING
    operation.started_at = datetime.now(UTC)
    operation.task_id = celery_task.request.id
    await session.commit()
```

### Qdrant Operations
```python
# Batch vector uploads
with QdrantOperationTimer("upload_vectors"):
    qdrant_client.upsert(
        collection_name=collection.vector_store_name,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=metadata
            )
            for embedding, metadata in batch
        ]
    )
```

### Metrics Collection
```python
# Prometheus metrics
from shared.metrics.collection_metrics import (
    collection_operations_total,
    collection_operation_duration_seconds,
    collection_documents_processed_total,
)

# Record metrics
collection_operations_total.labels(
    operation_type=operation.type.value,
    status="completed"
).inc()

collection_operation_duration_seconds.labels(
    operation_type=operation.type.value
).observe(duration)
```

### Service Communication
```python
# Call vecpipe service for embeddings
async with httpx.AsyncClient() as client:
    response = await client.post(
        f"http://vecpipe:8001/embed",
        json={
            "texts": chunk_texts,
            "model": collection.embedding_model
        },
        timeout=30.0
    )
    embeddings = response.json()["embeddings"]
```

## Task Types Deep Dive

### INDEX Operation
```python
# Process new documents for a collection
1. Extract text from source files
2. Apply chunking strategy
3. Generate embeddings via vecpipe
4. Store vectors in Qdrant
5. Update document records in PostgreSQL
```

### REINDEX Operation  
```python
# Blue-green reindexing with validation
1. Create new Qdrant collection (blue)
2. Process all documents into new collection
3. Validate vector counts (±10% tolerance)
4. Test search quality (sample queries)
5. Atomic swap: update collection.vector_store_name
6. Schedule cleanup of old collection (5-30 min delay based on size)
```

### APPEND Operation
```python
# Add new documents to existing collection
1. Verify collection exists and is active
2. Process new documents only
3. Append vectors to existing Qdrant collection
4. Update collection statistics
```

### REMOVE_SOURCE Operation
```python
# Remove documents by source
1. Query documents by source_id
2. Batch delete vectors from Qdrant
3. Delete document records from PostgreSQL
4. Update collection statistics
```

### Chunking Operations
```python
# Apply chunking strategies with error recovery
1. Load strategy configuration
2. Check for idempotency (operation fingerprint)
3. Process documents with resource monitoring
4. Handle partial failures gracefully
5. Support retry for failed documents
6. Dead letter queue for unrecoverable failures
```

## Error Recovery & Monitoring

### Retry Strategy
```python
# Exponential backoff with jitter
retry_delay = min(
    base_delay * (2 ** attempt) + random.uniform(0, 10),
    max_delay
)
```

### Dead Letter Queue (Chunking)
```python
# Failed tasks after max retries
if task.request.retries >= max_retries:
    await redis.lpush(
        "chunking:dead_letter_queue",
        json.dumps({
            "operation_id": operation_id,
            "error": str(exc),
            "timestamp": datetime.now(UTC).isoformat()
        })
    )
```

### Health Checks
```bash
# Celery inspect ping
celery -A webui.celery_app inspect ping

# Check active tasks
celery -A webui.celery_app inspect active

# Monitor with Flower (web UI)
celery -A webui.celery_app flower --port=5555
```

### Monitoring Metrics
- Tasks started/completed/failed by type
- Task duration histograms
- Memory usage per operation
- CPU time consumed
- Queue lengths and processing rates
- Worker availability and health

## Performance Considerations

### Batch Sizes
- **Embedding Generation**: 100 texts per batch
- **Vector Upload**: 100 vectors per batch  
- **Document Removal**: 100 documents per batch

### Concurrency Limits
- **Default Workers**: 4 processes
- **Max Tasks per Child**: 100 (prevent memory leaks)
- **Prefetch Multiplier**: 1 (no prefetching for long tasks)

### Memory Optimization
- Stream large files instead of loading into memory
- Process documents in batches
- Explicit garbage collection after large operations
- Worker restart after N tasks

### Time Limits
- **Soft Limit**: 1 hour (graceful shutdown)
- **Hard Limit**: 2 hours (forced termination)
- **Chunking CPU Limit**: 30 minutes of CPU time

## Future Enhancements

### Planned Improvements
1. **Priority Queues**: Separate queues for quick vs long operations
2. **Task Chaining**: Pipeline operations for complex workflows  
3. **Partial Progress Persistence**: Resume interrupted operations
4. **Dynamic Scaling**: Auto-scale workers based on queue depth
5. **Enhanced Monitoring**: Custom Celery event consumers
6. **Task Result Caching**: Avoid reprocessing identical requests
7. **Distributed Locking**: Prevent duplicate operations
8. **Advanced Retry Policies**: Per-operation-type retry configuration

### Architecture Evolution
- Migration to Celery 5.x task protocol
- Integration with Kubernetes for container orchestration
- Support for multiple Redis instances (sharding)
- GraphQL subscriptions as alternative to WebSockets
- Event sourcing for operation history