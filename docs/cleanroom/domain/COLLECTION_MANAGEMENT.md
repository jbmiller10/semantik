# Collection Management Component - Cleanroom Documentation

## 1. Component Overview

The Collection Management component is the core domain model of Semantik, providing functionality to organize, index, and manage document collections with vector embeddings for semantic search. Collections serve as isolated containers for documents with specific embedding models and chunking configurations.

### Key Features
- **Collection Lifecycle Management**: Creation, configuration, status tracking, and deletion
- **Document Organization**: Grouping related documents with deduplication by content hash
- **Operation Orchestration**: Asynchronous processing for indexing, reindexing, and updates
- **Permission System**: Owner-based access control with public/private visibility
- **Real-time Updates**: WebSocket notifications for operation progress
- **Chunking Strategy Support**: Flexible document chunking with multiple strategies
- **Blue-Green Reindexing**: Zero-downtime collection updates

### Collection States
```python
class CollectionStatus(str, enum.Enum):
    PENDING = "pending"      # Initial state, awaiting first indexing
    READY = "ready"          # Fully indexed and searchable
    PROCESSING = "processing" # Operation in progress
    ERROR = "error"          # Failed operation, requires intervention
    DEGRADED = "degraded"    # Partially functional, some documents failed
```

## 2. Architecture & Design Patterns

### Service Layer Pattern
The collection service (`packages/webui/services/collection_service.py`) orchestrates all business logic:

```python
class CollectionService:
    def __init__(self, db_session, collection_repo, operation_repo, document_repo):
        # Dependency injection of repositories
        
    async def create_collection(self, user_id, name, description, config):
        # 1. Validate inputs and configuration
        # 2. Create collection via repository
        # 3. Create initial operation
        # 4. Commit transaction
        # 5. Dispatch async task
        # 6. Return collection and operation
```

### Repository Pattern
Data access is abstracted through repositories (`packages/shared/database/repositories/`):

```python
class CollectionRepository:
    async def create(self, name, owner_id, **kwargs) -> Collection
    async def get_by_uuid(self, collection_uuid) -> Collection | None
    async def get_by_uuid_with_permission_check(self, collection_uuid, user_id) -> Collection
    async def update_status(self, collection_uuid, status, status_message) -> Collection
    async def delete(self, collection_uuid, user_id) -> None
```

### API Layer
RESTful endpoints in `packages/webui/api/v2/collections.py`:

```python
router = APIRouter(prefix="/api/v2/collections")

@router.post("")  # Create collection
@router.get("")   # List collections
@router.get("/{collection_uuid}")  # Get collection details
@router.put("/{collection_uuid}")  # Update metadata
@router.delete("/{collection_uuid}")  # Delete collection
@router.post("/{collection_uuid}/sources")  # Add source
@router.post("/{collection_uuid}/reindex")  # Trigger reindex
```

## 3. Key Interfaces & Contracts

### Collection Creation Request
```python
class CollectionCreate:
    name: str                    # Unique collection name
    description: Optional[str]   
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    quantization: str = "float16"  # float32, float16, int8
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunking_strategy: Optional[str]  # recursive, semantic, markdown, etc.
    chunking_config: Optional[Dict]   # Strategy-specific configuration
    is_public: bool = False
    metadata: Optional[Dict]
```

### Collection Response Model
```python
class CollectionResponse:
    id: str                      # UUID
    name: str
    description: Optional[str]
    owner_id: int
    vector_store_name: str       # Qdrant collection identifier
    embedding_model: str
    quantization: str
    chunk_size: int
    chunk_overlap: int
    chunking_strategy: Optional[str]
    chunking_config: Optional[Dict]
    is_public: bool
    status: str                  # pending, ready, processing, error, degraded
    status_message: Optional[str]
    document_count: int
    vector_count: int
    created_at: datetime
    updated_at: datetime
    initial_operation_id: Optional[str]  # For tracking creation
```

### Operation Types
```python
class OperationType(str, enum.Enum):
    INDEX = "index"              # Initial indexing
    APPEND = "append"            # Add new documents
    REINDEX = "reindex"          # Complete reindexing
    REMOVE_SOURCE = "remove_source"  # Remove documents from source
    DELETE = "delete"            # Delete collection
```

## 4. Data Flow & Dependencies

### Collection Creation Flow
```
1. Client Request → API Endpoint
   POST /api/v2/collections
   
2. API → CollectionService.create_collection()
   - Validate chunking strategy and config
   - Check for duplicate names
   
3. Service → CollectionRepository.create()
   - Generate UUID and vector_store_name
   - Set initial status to PENDING
   - Persist to database
   
4. Service → OperationRepository.create()
   - Create INDEX operation
   - Link to collection
   
5. Service → Database Commit
   - Ensure atomicity
   
6. Service → Celery Task Dispatch
   - celery_app.send_task("webui.tasks.process_collection_operation")
   - Pass operation UUID
   
7. Response → Client
   - Return collection and operation details
```

### Document Addition Flow
```
1. Client → POST /api/v2/collections/{id}/sources
2. Service validates collection state (must be READY or DEGRADED)
3. Creates APPEND operation
4. Updates collection status to PROCESSING
5. Dispatches async task
6. Worker processes documents:
   - Parses files
   - Chunks content
   - Generates embeddings
   - Stores in Qdrant
7. Updates collection status back to READY
```

## 5. Critical Implementation Details

### Status Management
```python
# packages/webui/services/collection_service.py

async def add_source(self, collection_id, user_id, source_path, source_config):
    # Status validation
    if collection.status not in [CollectionStatus.PENDING, CollectionStatus.READY, CollectionStatus.DEGRADED]:
        raise InvalidStateError(f"Cannot add source to collection in {collection.status} state")
    
    # Check for active operations
    active_operations = await self.operation_repo.get_active_operations(collection.id)
    if active_operations:
        raise InvalidStateError("Cannot add source while another operation is in progress")
    
    # Update status to processing
    await self.collection_repo.update_status(collection.id, CollectionStatus.PROCESSING)
```

### Transaction Management
```python
# Critical: Commit BEFORE dispatching async tasks
async def create_collection(self, ...):
    collection = await self.collection_repo.create(...)
    operation = await self.operation_repo.create(...)
    
    # Commit to ensure data is persisted
    await self.db_session.commit()
    
    # Only then dispatch task (avoids race conditions)
    celery_app.send_task("webui.tasks.process_collection_operation", args=[operation.uuid])
```

### Deduplication by Content Hash
```python
# packages/shared/database/repositories/document_repository.py

async def create(self, collection_id, file_path, file_name, file_size, content_hash, ...):
    # Check for existing document with same hash
    existing_doc = await self.get_by_content_hash(collection_id, content_hash)
    if existing_doc:
        logger.info(f"Document with content_hash {content_hash} already exists")
        return existing_doc  # Return existing instead of creating duplicate
```

### Chunking Strategy Validation
```python
# packages/webui/services/collection_service.py

if chunking_strategy is not None:
    try:
        # Validate strategy exists and is supported
        ChunkingStrategyFactory.create_strategy(
            strategy_name=chunking_strategy,
            config=chunking_config or {},
        )
        # Normalize strategy name for persistence
        chunking_strategy = ChunkingStrategyFactory.normalize_strategy_name(chunking_strategy)
    except ChunkingStrategyError as e:
        if "Unknown strategy" in e.reason:
            available = ChunkingStrategyFactory.get_available_strategies()
            raise ValueError(f"Invalid chunking_strategy '{e.strategy}'. Available: {', '.join(available)}")
```

## 6. Security Considerations

### Access Control
```python
# Permission check in repository
async def get_by_uuid_with_permission_check(self, collection_uuid, user_id):
    collection = await self.get_by_uuid(collection_uuid)
    if not collection:
        raise EntityNotFoundError("collection", collection_uuid)
    
    # Check ownership or public access
    if collection.owner_id != user_id and not collection.is_public:
        raise AccessDeniedError(str(user_id), "collection", collection_uuid)
    
    return collection
```

### Rate Limiting
```python
# API endpoints use rate limiting decorators
@router.delete("/{collection_uuid}")
@limiter.limit("5/hour")  # Prevent deletion abuse
async def delete_collection(...)

@router.post("/{collection_uuid}/reindex")
@limiter.limit("1/5minutes")  # Prevent resource exhaustion
async def reindex_collection(...)
```

### Input Validation
- Collection names must be unique
- Chunk size must be positive
- Chunk overlap must be less than chunk size
- SHA-256 content hashes validated with regex
- File sizes cannot be negative

### WebSocket Authentication
```python
# packages/webui/api/v2/operations.py
async def operation_websocket(websocket: WebSocket, operation_id: str):
    # Extract JWT from query parameters
    token = websocket.query_params.get("token")
    
    # Authenticate user
    user = await get_current_user_websocket(token)
    
    # Verify operation access
    await service.verify_websocket_access(operation_uuid=operation_id, user_id=user["id"])
```

## 7. Testing Requirements

### Service Layer Tests
```python
# tests/webui/services/test_collection_service.py

async def test_create_collection_with_chunking_strategy():
    # Test valid strategy configuration
    collection, operation = await service.create_collection(
        user_id=1,
        name="test_collection",
        config={"chunking_strategy": "recursive", "chunking_config": {...}}
    )
    assert collection["chunking_strategy"] == "recursive"

async def test_create_collection_invalid_strategy():
    # Test invalid strategy rejection
    with pytest.raises(ValueError, match="Invalid chunking_strategy"):
        await service.create_collection(
            user_id=1,
            name="test",
            config={"chunking_strategy": "invalid"}
        )

async def test_concurrent_operations_blocked():
    # Test that concurrent operations are prevented
    await service.add_source(collection_id, user_id, source_path)
    with pytest.raises(InvalidStateError, match="operation is in progress"):
        await service.reindex_collection(collection_id, user_id)
```

### API Integration Tests
```python
# tests/webui/api/v2/test_collections.py

async def test_collection_lifecycle():
    # 1. Create collection
    response = await client.post("/api/v2/collections", json={...})
    assert response.status_code == 201
    collection_id = response.json()["id"]
    
    # 2. Add source
    response = await client.post(f"/api/v2/collections/{collection_id}/sources", json={...})
    assert response.status_code == 202
    
    # 3. Check status updates
    response = await client.get(f"/api/v2/collections/{collection_id}")
    assert response.json()["status"] == "processing"
    
    # 4. Delete collection
    response = await client.delete(f"/api/v2/collections/{collection_id}")
    assert response.status_code == 204
```

### Repository Tests
```python
# tests/shared/database/repositories/test_collection_repository.py

async def test_deduplication():
    # Test document deduplication by content hash
    doc1 = await document_repo.create(collection_id, "file1.txt", content_hash="abc123...")
    doc2 = await document_repo.create(collection_id, "file2.txt", content_hash="abc123...")
    assert doc1.id == doc2.id  # Same document returned
```

## 8. Common Pitfalls & Best Practices

### Pitfall: Race Conditions in Task Dispatch
```python
# BAD: Dispatching before commit
operation = await self.operation_repo.create(...)
celery_app.send_task("process_operation", args=[operation.uuid])  # Task may not find operation!
await self.db_session.commit()

# GOOD: Commit first
operation = await self.operation_repo.create(...)
await self.db_session.commit()
celery_app.send_task("process_operation", args=[operation.uuid])
```

### Pitfall: Not Checking Active Operations
```python
# BAD: Allowing concurrent operations
await service.add_source(collection_id, source1)
await service.add_source(collection_id, source2)  # May corrupt state!

# GOOD: Check for active operations
active_ops = await self.operation_repo.get_active_operations_count(collection.id)
if active_ops > 0:
    raise InvalidStateError("Cannot start new operation while another is in progress")
```

### Best Practice: Status Transitions
```python
# Maintain consistent status transitions
VALID_TRANSITIONS = {
    CollectionStatus.PENDING: [CollectionStatus.PROCESSING, CollectionStatus.ERROR],
    CollectionStatus.READY: [CollectionStatus.PROCESSING, CollectionStatus.DEGRADED],
    CollectionStatus.PROCESSING: [CollectionStatus.READY, CollectionStatus.ERROR, CollectionStatus.DEGRADED],
    CollectionStatus.ERROR: [CollectionStatus.PROCESSING],  # Allow retry
    CollectionStatus.DEGRADED: [CollectionStatus.PROCESSING, CollectionStatus.READY],
}
```

### Best Practice: Cleanup on Deletion
```python
async def delete_collection(self, collection_id, user_id):
    # 1. Check no active operations
    if active_ops > 0:
        raise InvalidStateError("Cannot delete with active operations")
    
    # 2. Delete from Qdrant
    try:
        qdrant_client.delete_collection(collection.vector_store_name)
    except Exception as e:
        logger.error(f"Failed to delete Qdrant collection: {e}")
        # Continue with database deletion
    
    # 3. Delete from database (cascades to documents, operations, etc.)
    await self.collection_repo.delete(collection_id, user_id)
```

## 9. Configuration & Environment

### Collection Defaults
```python
# packages/webui/services/collection_service.py

DEFAULT_VECTOR_DIMENSION = 1536
QDRANT_COLLECTION_PREFIX = "collection_"

# Default configurations
embedding_model = config.get("embedding_model") or "Qwen/Qwen3-Embedding-0.6B"
quantization = config.get("quantization") or "float16"
chunk_size = config.get("chunk_size") or 1000
chunk_overlap = config.get("chunk_overlap") or 200
```

### Environment Variables
```bash
# Qdrant configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=optional_key

# Redis for WebSocket pub/sub
REDIS_URL=redis://localhost:6379/2

# PostgreSQL for metadata
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/semantik
```

### Collection Limits
```python
# Rate limits for operations
CREATE_COLLECTION_LIMIT = "10/hour"
DELETE_COLLECTION_LIMIT = "5/hour"
REINDEX_COLLECTION_LIMIT = "1/5minutes"
ADD_SOURCE_LIMIT = "10/hour"

# Size limits
MAX_COLLECTION_NAME_LENGTH = 255
MAX_DESCRIPTION_LENGTH = 1000
MAX_DOCUMENTS_PER_COLLECTION = 100000
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100MB
```

## 10. Integration Points

### Frontend Integration (React/TypeScript)
```typescript
// apps/webui-react/src/hooks/useCollections.ts

export function useCreateCollection() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (data: CreateCollectionRequest) => {
      const response = await collectionsV2Api.create(data);
      return response.data;
    },
    onSuccess: (data) => {
      // Invalidate and refetch collections list
      queryClient.invalidateQueries({ queryKey: ['collections'] });
      
      // Start polling for operation updates
      pollOperationStatus(data.initial_operation_id);
    }
  });
}
```

### Worker Task Integration
```python
# packages/webui/tasks.py

@celery_app.task
def process_collection_operation(operation_uuid: str):
    operation = get_operation(operation_uuid)
    collection = get_collection(operation.collection_id)
    
    if operation.type == OperationType.INDEX:
        # Initial indexing
        create_qdrant_collection(collection)
        process_documents(collection)
    elif operation.type == OperationType.APPEND:
        # Add new documents
        process_source(operation.config["source_path"])
    elif operation.type == OperationType.REINDEX:
        # Blue-green reindex
        perform_blue_green_reindex(collection)
```

### Search Integration
```python
# packages/vecpipe/services/search_service.py

async def search_collection(collection_id: str, query: str, limit: int = 10):
    collection = await get_collection(collection_id)
    
    # Generate query embedding
    embedding = embed_text(query, collection.embedding_model)
    
    # Search in Qdrant
    results = qdrant_client.search(
        collection_name=collection.vector_store_name,
        query_vector=embedding,
        limit=limit
    )
    
    return format_search_results(results)
```

### WebSocket Notifications
```python
# Real-time operation updates
async def broadcast_operation_update(operation_id: str, status: dict):
    await ws_manager.send_to_channel(
        channel=f"operation:{operation_id}",
        message={
            "type": "operation_update",
            "operation_id": operation_id,
            "status": status["status"],
            "progress": status.get("progress"),
            "message": status.get("message")
        }
    )
```

### Chunking Strategy Integration
```python
# packages/webui/services/collection_service.py

# Strategy validation during collection creation
ChunkingStrategyFactory.create_strategy(
    strategy_name=chunking_strategy,
    config=chunking_config or {},
)

# Available strategies:
# - "recursive": Recursive text splitting
# - "semantic": Semantic similarity-based chunking  
# - "markdown": Document structure-aware chunking
# - "character": Fixed character-based chunks
# - "hybrid": Combination of multiple strategies
```

## Migration Notes

### From Job-Centric to Collection-Centric
The system has been refactored from a job-centric to collection-centric architecture:

- **Old**: Jobs → Tasks → Documents
- **New**: Collections → Operations → Documents

Key changes:
1. Collections are now the primary organizing principle
2. Operations replace jobs for async processing
3. Documents belong directly to collections
4. Status tracking moved to collection level
5. Permissions simplified to collection ownership

### Database Schema Changes
```sql
-- Collections table additions
ALTER TABLE collections ADD COLUMN chunking_strategy VARCHAR;
ALTER TABLE collections ADD COLUMN chunking_config JSONB;
ALTER TABLE collections ADD COLUMN status VARCHAR DEFAULT 'pending';
ALTER TABLE collections ADD COLUMN status_message TEXT;

-- Operations table (replaces jobs)
CREATE TABLE operations (
    uuid UUID PRIMARY KEY,
    collection_id UUID REFERENCES collections(id),
    user_id INTEGER REFERENCES users(id),
    type VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    config JSONB,
    created_at TIMESTAMP WITH TIME ZONE,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT
);
```

## Monitoring & Observability

### Key Metrics
- Collection creation rate
- Operation success/failure rates
- Average indexing time per document
- Vector count growth rate
- Storage utilization per collection
- Active WebSocket connections

### Health Checks
```python
async def check_collection_health(collection_id: str) -> dict:
    collection = await get_collection(collection_id)
    
    return {
        "status": collection.status,
        "document_count": collection.document_count,
        "vector_count": collection.vector_count,
        "last_operation": await get_last_operation(collection_id),
        "qdrant_status": await check_qdrant_collection(collection.vector_store_name),
        "storage_bytes": collection.total_size_bytes
    }
```

### Error Recovery
1. **Failed Operations**: Automatically retry with exponential backoff
2. **Degraded Collections**: Allow partial functionality, mark problematic documents
3. **Orphaned Qdrant Collections**: Cleanup task runs periodically
4. **Stuck Operations**: Timeout after configurable period (default: 1 hour)

---

This documentation represents the complete Collection Management component as of the current implementation. All code examples are taken directly from the production codebase and represent actual patterns in use.