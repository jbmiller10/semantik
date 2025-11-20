# Collection Service Integration Guide

This guide helps developers integrate the Collection Service into API endpoints and understand the overall architecture.

## Architecture Overview

The Collection Service acts as the orchestration layer between:
- **API Endpoints** (FastAPI routes)
- **Repository Layer** (data access)
- **Celery Tasks** (async processing)
- **Qdrant** (vector storage)

```
API Layer → Collection Service → Repositories → Database
                ↓
            Celery Tasks → Qdrant
```

## Quick Start: Using Collection Service in API Endpoints

### 1. Dependency Injection Pattern

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from shared.database import get_db, InvalidStateError, EntityNotFoundError, AccessDeniedError
from webui.services import create_collection_service, CollectionService
from webui.auth import get_current_user

router = APIRouter(prefix="/api/collections", tags=["collections"])

# Dependency function
async def get_collection_service(db: AsyncSession = Depends(get_db)) -> CollectionService:
    return create_collection_service(db)
```

### 2. Create Collection Endpoint

```python
@router.post("/", response_model=CollectionResponse)
async def create_collection(
    request: CreateCollectionRequest,
    current_user: dict = Depends(get_current_user),
    service: CollectionService = Depends(get_collection_service),
):
    """Create a new collection."""
    try:
        collection, operation = await service.create_collection(
            user_id=int(current_user["id"]),
            name=request.name,
            description=request.description,
            config=request.config,
        )
        
        return CollectionResponse(
            collection=collection,
            operation=operation,
            message="Collection creation started",
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### 3. Add Source Endpoint

```python
@router.post("/{collection_id}/sources", response_model=OperationResponse)
async def add_source(
    collection_id: str,
    request: AddSourceRequest,
    current_user: dict = Depends(get_current_user),
    service: CollectionService = Depends(get_collection_service),
):
    """Add a source to an existing collection."""
    try:
        operation = await service.add_source(
            collection_id=collection_id,
            user_id=int(current_user["id"]),
            source_path=request.source_path,
            source_config=request.config,
        )
        
        return OperationResponse(
            operation=operation,
            message="Source addition started",
            monitor_url=f"/api/operations/{operation['uuid']}/status",
        )
        
    except InvalidStateError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
```

## Service Methods

### 1. `create_collection`
- Creates a new collection and dispatches INDEX operation
- Returns: `(collection, operation)` tuple
- Exceptions: `ValueError`, `AccessDeniedError`

### 2. `add_source`
- Adds documents from a source to existing collection
- Validates collection state (must be READY or DEGRADED)
- Returns: operation dictionary
- Exceptions: `InvalidStateError`, `EntityNotFoundError`, `AccessDeniedError`

### 3. `reindex_collection`
- Performs blue-green reindexing with optional config updates
- Zero downtime operation
- Returns: operation dictionary
- Exceptions: `InvalidStateError`, `EntityNotFoundError`, `AccessDeniedError`

### 4. `delete_collection`
- Deletes collection and all associated data
- Requires owner permission
- Cleans up Qdrant collection
- Exceptions: `InvalidStateError`, `EntityNotFoundError`, `AccessDeniedError`

### 5. `remove_source`
- Removes documents from a specific source
- Returns: operation dictionary
- Exceptions: `InvalidStateError`, `EntityNotFoundError`, `AccessDeniedError`

## Exception Handling

Use this decorator pattern for consistent error handling:

```python
from functools import wraps
from fastapi import HTTPException

def handle_service_errors(func):
    """Decorator to handle collection service exceptions."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except InvalidStateError as e:
            raise HTTPException(status_code=409, detail=str(e))
        except EntityNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except AccessDeniedError as e:
            raise HTTPException(status_code=403, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    return wrapper

# Usage
@router.post("/{collection_id}/reindex")
@handle_service_errors
async def reindex_collection(
    collection_id: str,
    request: ReindexRequest,
    current_user: dict = Depends(get_current_user),
    service: CollectionService = Depends(get_collection_service),
):
    operation = await service.reindex_collection(
        collection_id=collection_id,
        user_id=int(current_user["id"]),
        config_updates=request.config_updates,
    )
    return {"operation": operation}
```

## Operation Monitoring

All service methods that trigger background work return an operation object with:
- `uuid`: Operation identifier
- `type`: Operation type (INDEX, APPEND, REINDEX, REMOVE_SOURCE)
- `status`: Current status (PENDING, PROCESSING, COMPLETED, FAILED)
- `created_at`: Timestamp

### Real-time Updates via Redis Streams

Operations send updates to Redis streams at key: `operation:updates:{operation_uuid}`

Update types:
- `operation_started`
- `operation_completed`
- `operation_failed`
- Type-specific updates (e.g., `index_completed`, `append_completed`)

### WebSocket Integration (for next developer)

```python
@router.websocket("/operations/{operation_id}/ws")
async def operation_updates(
    websocket: WebSocket,
    operation_id: str,
    current_user: dict = Depends(get_current_user_ws),
):
    """Stream operation updates via WebSocket."""
    await websocket.accept()
    
    # Subscribe to Redis stream
    stream_key = f"operation:updates:{operation_id}"
    
    # Stream updates to client
    # Implementation needed...
```

## State Transitions

Collections follow these state transitions:

```
EMPTY → INDEXING → READY
         ↓    ↑
       FAILED  ↓
         ↑    ↓
      DEGRADED ← PROCESSING
```

The service enforces these rules:
- Can only add sources when READY or DEGRADED
- Cannot perform operations during INDEXING
- Cannot reindex FAILED collections (must delete and recreate)

## TODOs for Full Implementation

1. **Document Processing (APPEND operation)**
   - Implement file scanning
   - Add content extraction
   - Generate embeddings
   - Store in Qdrant

2. **Reindexing Logic**
   - Fetch all documents
   - Reprocess with new config
   - Atomic collection switch

3. **Document-Vector Mapping**
   - Store document ID in Qdrant payload
   - Enable proper deletion

## Testing the Service

```python
# Example integration test
async def test_create_collection():
    async with get_test_db() as db:
        service = create_collection_service(db)
        
        collection, operation = await service.create_collection(
            user_id=1,
            name="Test Collection",
            description="Test description",
            config={"vector_dim": 768},
        )
        
        assert collection["name"] == "Test Collection"
        assert operation["type"] == OperationType.INDEX
        assert operation["status"] == OperationStatus.PENDING
```

## Configuration Constants

- `DEFAULT_VECTOR_DIMENSION`: 768 (default embedding dimension)
- `QDRANT_COLLECTION_PREFIX`: "collection_" (prefix for Qdrant collections)

## Next Steps for API Developer

1. Create request/response models (Pydantic)
2. Implement operation status endpoint
3. Add WebSocket support for real-time updates
4. Create collection list/search endpoints
5. Add operation history endpoint
6. Implement batch operations
## Chunking Composition Root

`packages/webui/services/chunking/container.py` is the single composition root for chunking workflows. The container assembles cache, metrics, validator, processor, and config manager collaborators around the orchestrator.

- **FastAPI**: depend on `packages.webui.dependencies.get_chunking_orchestrator_dependency`.
- **Celery/tasks**: call `packages.webui.services.chunking.container.resolve_celery_chunking_orchestrator` so workers receive a cache-less orchestrator without additional wiring.
- **Tests**: patch these resolver functions instead of instantiating services manually; fixtures stay aligned with production wiring.

Chunking now flows exclusively through the orchestrator; legacy adapters and the monolithic `ChunkingService` have been removed. External plugins should register a strategy definition via `register_strategy_definition` and bind an implementation with `ChunkingStrategyFactory.register_strategy`; see `docs/api/CHUNKING_API.md` for a minimal plugin example.
