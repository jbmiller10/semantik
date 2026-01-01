# WebUI API Layer Architecture

> **Location:** `packages/webui/api/`, `routers/`

## Overview

FastAPI-based REST API implementing the v2 endpoint specification. Uses JWT authentication, Pydantic validation, and follows three-layer architecture.

## API Structure

```
packages/webui/
├── api/
│   ├── v2/
│   │   ├── __init__.py      # Router registration
│   │   ├── collections.py   # Collection endpoints
│   │   ├── operations.py    # Operation endpoints
│   │   ├── search.py        # Search endpoints
│   │   ├── documents.py     # Document endpoints
│   │   ├── connectors.py    # Connector catalog
│   │   ├── chunking.py      # Chunking preview
│   │   ├── projections.py   # Embedding projections
│   │   └── system.py        # System status
│   ├── auth.py              # Authentication endpoints
│   ├── health.py            # Health checks
│   └── deps.py              # Dependency injection
├── schemas/                  # Pydantic models
│   ├── collections.py
│   ├── operations.py
│   ├── search.py
│   └── ...
└── middleware/
    ├── auth.py              # JWT middleware
    ├── rate_limit.py        # Rate limiting
    └── logging.py           # Request logging
```

## Authentication

### JWT Token Flow
```
1. POST /api/auth/login (username, password)
2. Validate credentials against database
3. Generate access_token (24h) + refresh_token
4. Return tokens to client
5. Client sends: Authorization: Bearer {token}
6. Middleware validates token on each request
7. Token refresh via POST /api/auth/refresh
```

### Token Configuration
```python
JWT_SECRET_KEY = os.environ["JWT_SECRET_KEY"]  # 64-hex-chars
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours
ALGORITHM = "HS256"
```

### Dependency Injection
```python
# deps.py
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
    user = await UserRepository(db).get_by_username(payload["sub"])
    if not user or not user.is_active:
        raise HTTPException(status_code=401)
    return user

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session
```

## Core Endpoints

### Collections

**GET /api/v2/collections**
```python
@router.get("/collections")
async def list_collections(
    status: CollectionStatus | None = None,
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> CollectionsResponse:
    repo = CollectionRepository(db)
    collections, total = await repo.list_for_user(
        user_id=user.id, status=status, limit=limit, offset=offset
    )
    return CollectionsResponse(collections=collections, total=total)
```

**POST /api/v2/collections**
```python
@router.post("/collections", status_code=201)
async def create_collection(
    request: CreateCollectionRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> CollectionResponse:
    service = CollectionService(db)
    collection = await service.create(
        name=request.name,
        owner_id=user.id,
        embedding_model=request.embedding_model,
        chunking_config=request.chunking_config,
        source_config=request.source_config
    )

    # Queue initial indexing if source provided
    if request.source_config:
        operation = await service.queue_index_operation(
            collection_id=collection.id,
            user_id=user.id,
            source_config=request.source_config
        )
        return CollectionResponse(
            **collection.dict(),
            initial_operation_id=operation.uuid
        )

    return CollectionResponse(**collection.dict())
```

**DELETE /api/v2/collections/{uuid}**
```python
@router.delete("/collections/{uuid}", status_code=204)
async def delete_collection(
    uuid: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    service = CollectionService(db)
    await service.delete(collection_id=uuid, user_id=user.id)
    # Deletion is async - actual cleanup via Celery task
```

### Search

**POST /api/v2/search**
```python
@router.post("/search")
async def search(
    request: SearchRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> SearchResponse:
    # Validate collection access
    repo = CollectionRepository(db)
    collections = await repo.get_accessible(
        collection_ids=request.collection_uuids,
        user_id=user.id
    )

    # Resolve to Qdrant collection names
    qdrant_collections = [c.vector_store_name for c in collections]

    # Call VecPipe search API
    vecpipe_response = await vecpipe_client.search(
        query=request.query,
        collections=qdrant_collections,
        top_k=request.top_k,
        search_type=request.search_type,
        use_reranker=request.use_reranker,
        rerank_model=request.rerank_model
    )

    # Map results back to collection UUIDs
    return SearchResponse(
        results=map_results(vecpipe_response, collections),
        query=request.query,
        reranking_used=vecpipe_response.reranking_used
    )
```

### Operations

**GET /api/v2/operations**
```python
@router.get("/operations")
async def list_operations(
    status: str | None = Query(None),  # comma-separated
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> OperationsResponse:
    repo = OperationRepository(db)
    statuses = status.split(",") if status else None
    operations = await repo.list_for_user(
        user_id=user.id,
        statuses=statuses
    )
    return OperationsResponse(operations=operations)
```

**POST /api/v2/operations/{uuid}/cancel**
```python
@router.post("/operations/{uuid}/cancel")
async def cancel_operation(
    uuid: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> OperationResponse:
    service = OperationService(db)
    operation = await service.cancel(operation_uuid=uuid, user_id=user.id)
    return OperationResponse(**operation.dict())
```

### Connectors

**GET /api/v2/connectors**
```python
@router.get("/connectors")
async def get_connector_catalog() -> ConnectorCatalogResponse:
    """Return all available connector types with field definitions."""
    return ConnectorCatalogResponse(connectors=get_connector_catalog())
```

**POST /api/v2/connectors/git/preview**
```python
@router.post("/connectors/git/preview")
async def preview_git(
    request: GitPreviewRequest,
    user: User = Depends(get_current_user)
) -> GitPreviewResponse:
    """Test Git repository connection and list branches."""
    git_service = GitConnectorService()
    return await git_service.preview(
        url=request.url,
        branch=request.branch,
        credentials=request.credentials
    )
```

## Request/Response Schemas

### CreateCollectionRequest
```python
class CreateCollectionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    quantization: str = "float16"
    chunk_size: int = Field(512, ge=100, le=4000)
    chunk_overlap: int = Field(64, ge=0, le=500)
    chunking_strategy: str = "recursive"
    is_public: bool = False
    source_config: SourceConfig | None = None
    sync_mode: str = "one_time"
    sync_interval_minutes: int | None = None
```

### SearchRequest
```python
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection_uuids: list[str]
    top_k: int = Field(10, ge=1, le=100)
    score_threshold: float = Field(0.0, ge=0.0, le=1.0)
    search_type: str = "semantic"  # semantic, hybrid, question, code
    use_reranker: bool = False
    rerank_model: str | None = None
    hybrid_alpha: float | None = Field(None, ge=0.0, le=1.0)
```

### SearchResponse
```python
class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
    search_type: str
    reranking_used: bool
    reranking_time_ms: float | None = None
    failed_collections: list[FailedCollection] | None = None
```

## Middleware Stack

### Request Processing Order
```
1. Rate Limiting (if enabled)
2. Request Logging
3. Authentication (for protected routes)
4. Request Validation (Pydantic)
5. Route Handler
6. Response Serialization
7. Error Handling
```

### Rate Limiting
```python
# Configurable per-minute limits
RATE_LIMIT_PER_MINUTE = 60  # default

@router.get("/search")
@rate_limit(requests_per_minute=RATE_LIMIT_PER_MINUTE)
async def search(...):
    ...
```

### Error Handling
```python
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )
```

## Health Endpoints

```python
@router.get("/health/livez")
async def liveness():
    """Kubernetes liveness probe."""
    return {"status": "alive"}

@router.get("/health/readyz")
async def readiness(db: AsyncSession = Depends(get_db)):
    """Kubernetes readiness probe."""
    # Check database connection
    await db.execute(text("SELECT 1"))

    # Check VecPipe connectivity
    await vecpipe_client.health()

    return {"status": "ready"}
```

## Extension Points

### Adding a New Endpoint
1. Create route function in appropriate module
2. Define Pydantic request/response schemas
3. Implement service layer logic
4. Add to router registration in `__init__.py`
5. Write API tests

### Adding Authentication to Endpoint
```python
@router.get("/my-endpoint")
async def my_endpoint(
    user: User = Depends(get_current_user),  # Add this
    db: AsyncSession = Depends(get_db)
):
    # user is now the authenticated user
    ...
```

### Adding Rate Limiting
```python
from webui.middleware.rate_limit import rate_limit

@router.post("/expensive-operation")
@rate_limit(requests_per_minute=10)
async def expensive_operation(...):
    ...
```
