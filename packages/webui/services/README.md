# Collection Service

Orchestrates collections, sources, operations, and Qdrant integration.

## Usage

```python
from webui.services import create_collection_service, CollectionService

async def get_collection_service(db: AsyncSession = Depends(get_db)) -> CollectionService:
    return create_collection_service(db)

@router.post("/")
async def create_collection(
    request: CreateCollectionRequest,
    current_user: dict = Depends(get_current_user),
    service: CollectionService = Depends(get_collection_service),
):
    collection, operation = await service.create_collection(
        user_id=int(current_user["id"]),
        name=request.name,
        config=request.config,
    )
    return CollectionResponse(collection=collection, operation=operation)
```

## Methods

| Method | Returns | Exceptions |
|--------|---------|------------|
| create_collection | (collection, operation) | ValueError, AccessDeniedError |
| add_source | operation | InvalidStateError, EntityNotFoundError, AccessDeniedError |
| reindex_collection | operation | InvalidStateError, EntityNotFoundError, AccessDeniedError |
| delete_collection | None | InvalidStateError, EntityNotFoundError, AccessDeniedError |
| remove_source | operation | InvalidStateError, EntityNotFoundError, AccessDeniedError |

## State Rules

- Add sources: READY only
- No operations during PROCESSING
- Cannot reindex FAILED (must delete and recreate)

## Chunking

Container: `packages/webui/services/chunking/container.py`

- **FastAPI**: `get_chunking_orchestrator_dependency`
- **Celery**: `resolve_celery_chunking_orchestrator`
- **Plugins**: `register_strategy_definition` + `ChunkingStrategyFactory.register_strategy`

See `docs/api/CHUNKING_API.md` for plugin examples.
