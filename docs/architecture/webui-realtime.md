# WebUI Realtime Architecture

> **Location:** `packages/webui/tasks/`, `websocket/`

## Overview

Real-time functionality powered by:
- **Celery** for distributed task processing
- **Redis Streams** for message queuing
- **WebSockets** for client push updates

## Celery Configuration

### Task Queue Setup
```python
# celery_app.py
from celery import Celery

celery_app = Celery(
    "semantik",
    broker=os.environ["REDIS_URL"],
    backend=os.environ["REDIS_URL"]
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_default_queue="semantik",
    worker_prefetch_multiplier=1,  # One task at a time per worker
)
```

### Beat Schedule (Periodic Tasks)
```python
celery_app.conf.beat_schedule = {
    "check-continuous-sync": {
        "task": "webui.tasks.sync.check_continuous_sync",
        "schedule": crontab(minute="*/5"),  # Every 5 minutes
    },
    "cleanup-old-operations": {
        "task": "webui.tasks.maintenance.cleanup_operations",
        "schedule": crontab(hour=3, minute=0),  # Daily at 3 AM
    },
}
```

## Task Definitions

### Indexing Tasks

**process_source** - Main indexing task:
```python
@celery_app.task(bind=True, name="webui.tasks.indexing.process_source")
def process_source(self, operation_uuid: str) -> dict:
    """Process documents from a data source."""
    with task_context(operation_uuid) as ctx:
        operation = ctx.get_operation()

        # Update status
        ctx.update_status(OperationStatus.PROCESSING)

        try:
            # Get source configuration
            source_config = SourceConfig(**operation.config["source_config"])

            # Process documents
            processed = 0
            for doc in iterate_source_documents(source_config):
                # Extract content
                content = extract_content(doc)

                # Chunk content
                chunks = chunk_content(content, operation.collection_id)

                # Generate embeddings
                embeddings = generate_embeddings(chunks, operation.collection.embedding_model)

                # Store in Qdrant
                upsert_vectors(operation.collection.vector_store_name, chunks, embeddings)

                # Store metadata
                store_document_metadata(operation.collection_id, doc, chunks)

                processed += 1
                ctx.update_progress(
                    progress=calculate_progress(processed),
                    message=f"Processed {processed} documents"
                )

            ctx.complete(
                documents_processed=processed,
                chunks_created=total_chunks
            )

        except Exception as e:
            ctx.fail(str(e))
            raise
```

**generate_embeddings_batch** - Batch embedding task:
```python
@celery_app.task(bind=True)
def generate_embeddings_batch(
    self,
    texts: list[str],
    model: str,
    operation_uuid: str
) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    provider = EmbeddingProviderFactory.create_provider(model)
    return provider.embed_texts(texts, mode=EmbeddingMode.DOCUMENT)
```

### Deletion Tasks

**delete_collection** - Collection deletion:
```python
@celery_app.task(bind=True)
def delete_collection(self, collection_id: str) -> None:
    """Delete collection and all associated data."""
    with task_context(collection_id) as ctx:
        # Get collection
        collection = ctx.get_collection()

        # Delete from Qdrant
        qdrant_client.delete_collection(collection.vector_store_name)

        # Delete chunks (partition-aware)
        delete_chunks_for_collection(collection_id)

        # Delete documents
        delete_documents_for_collection(collection_id)

        # Delete collection record
        delete_collection_record(collection_id)

        ctx.complete()
```

### Sync Tasks

**check_continuous_sync** - Periodic sync checker:
```python
@celery_app.task
def check_continuous_sync() -> None:
    """Check for collections needing re-sync."""
    collections = get_collections_for_sync()

    for collection in collections:
        if should_sync(collection):
            # Queue sync operation
            celery_app.send_task(
                "webui.tasks.sync.sync_collection",
                args=[collection.id]
            )
```

**sync_collection** - Collection sync:
```python
@celery_app.task(bind=True)
def sync_collection(self, collection_id: str) -> None:
    """Sync collection with source, detecting changes."""
    with task_context(collection_id) as ctx:
        collection = ctx.get_collection()

        # Get current document hashes
        current_hashes = get_document_hashes(collection_id)

        # Scan source for changes
        source_files = scan_source(collection.source_config)

        # Find changes
        added, modified, deleted = diff_sources(current_hashes, source_files)

        # Process changes
        for doc in added + modified:
            process_document(collection_id, doc)

        for doc in deleted:
            remove_document(collection_id, doc)

        ctx.complete(
            added=len(added),
            modified=len(modified),
            deleted=len(deleted)
        )
```

## Task Context Manager

```python
@contextmanager
def task_context(operation_uuid: str):
    """Context manager for task execution with progress tracking."""
    db = get_sync_session()

    try:
        operation = db.query(Operation).filter_by(uuid=operation_uuid).one()

        ctx = TaskContext(db, operation)
        yield ctx

    except Exception as e:
        operation.status = OperationStatus.FAILED
        operation.error_message = str(e)
        db.commit()
        raise

    finally:
        db.close()


class TaskContext:
    def __init__(self, db: Session, operation: Operation):
        self.db = db
        self.operation = operation
        self.redis = get_redis_client()

    def update_status(self, status: OperationStatus) -> None:
        self.operation.status = status
        if status == OperationStatus.PROCESSING:
            self.operation.started_at = datetime.utcnow()
        self.db.commit()
        self._broadcast()

    def update_progress(self, progress: int, message: str = "") -> None:
        self.operation.progress = progress
        self.db.commit()
        self._broadcast(message=message)

    def complete(self, **metadata) -> None:
        self.operation.status = OperationStatus.COMPLETED
        self.operation.progress = 100
        self.operation.completed_at = datetime.utcnow()
        self.operation.metadata = metadata
        self.db.commit()
        self._broadcast()

    def fail(self, error_message: str) -> None:
        self.operation.status = OperationStatus.FAILED
        self.operation.error_message = error_message
        self.operation.completed_at = datetime.utcnow()
        self.db.commit()
        self._broadcast()

    def _broadcast(self, message: str = "") -> None:
        """Publish update to Redis for WebSocket broadcast."""
        self.redis.xadd(
            f"operations:{self.operation.uuid}",
            {
                "type": "OPERATION_PROGRESS",
                "status": self.operation.status.value,
                "progress": str(self.operation.progress),
                "message": message,
                "collection_id": self.operation.collection_id
            }
        )
```

## WebSocket Architecture

### Connection Manager
```python
class WebSocketManager:
    def __init__(self):
        self.connections: dict[str, set[WebSocket]] = defaultdict(set)
        self.redis = get_async_redis_client()

    async def connect(self, websocket: WebSocket, operation_id: str) -> None:
        """Accept WebSocket connection and subscribe to operation."""
        await websocket.accept()

        # Authenticate
        token = await self._authenticate(websocket)
        if not token:
            await websocket.close(code=4001)
            return

        self.connections[operation_id].add(websocket)

        # Start listening to Redis stream
        asyncio.create_task(
            self._listen_to_stream(websocket, operation_id)
        )

    async def _listen_to_stream(
        self,
        websocket: WebSocket,
        operation_id: str
    ) -> None:
        """Listen to Redis stream and forward to WebSocket."""
        stream_key = f"operations:{operation_id}"
        last_id = "0"

        while True:
            try:
                messages = await self.redis.xread(
                    {stream_key: last_id},
                    count=1,
                    block=5000  # 5 second timeout
                )

                for stream, message_list in messages:
                    for message_id, data in message_list:
                        await websocket.send_json(data)
                        last_id = message_id

            except WebSocketDisconnect:
                self.connections[operation_id].discard(websocket)
                break
            except Exception as e:
                logger.error(f"Stream error: {e}")
                break
```

### WebSocket Endpoints
```python
@router.websocket("/ws/operations/{operation_id}")
async def operation_websocket(
    websocket: WebSocket,
    operation_id: str
):
    """WebSocket endpoint for operation progress."""
    await ws_manager.connect(websocket, operation_id)

    try:
        while True:
            # Keep connection alive, handle client messages
            data = await websocket.receive_json()
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, operation_id)


@router.websocket("/ws/operations")
async def global_operations_websocket(websocket: WebSocket):
    """Global WebSocket for all operations (authenticated user only)."""
    await ws_manager.connect_global(websocket)
```

### Message Types
```python
class WebSocketMessage(BaseModel):
    type: str  # OPERATION_PROGRESS, OPERATION_COMPLETED, OPERATION_FAILED
    operation_id: str
    status: str
    progress: int
    message: str | None = None
    collection_id: str | None = None
    metadata: dict | None = None
```

## Worker Configuration

### Concurrency Control
```python
# Environment variables
EMBEDDING_CONCURRENCY_PER_WORKER = 1  # Protect GPU VRAM
CELERY_MAX_CONCURRENCY = 4  # Safety cap

# In entrypoint
concurrency = min(
    os.cpu_count() - 1,
    int(os.environ.get("CELERY_MAX_CONCURRENCY", 4))
)
```

### Resource Management
```python
@celery_app.task(bind=True, max_retries=3)
def gpu_intensive_task(self, ...):
    try:
        # Check GPU memory before starting
        if not check_gpu_memory_available():
            raise Retry(countdown=60)

        # Process with GPU
        ...

    except CudaOutOfMemoryError:
        # Clear cache and retry
        torch.cuda.empty_cache()
        raise self.retry(countdown=30)
```

## Extension Points

### Adding a New Task
1. Create task function with `@celery_app.task` decorator
2. Use `task_context` for progress tracking
3. Handle errors and update status
4. Queue via `celery_app.send_task()`

### Adding Periodic Task
```python
# In celery configuration
celery_app.conf.beat_schedule["my-periodic-task"] = {
    "task": "webui.tasks.my_module.my_task",
    "schedule": crontab(minute="*/10"),
}
```

### Adding WebSocket Message Type
1. Define message structure
2. Add to WebSocket handler
3. Update frontend to handle message type
4. Add to Redis stream publishing
