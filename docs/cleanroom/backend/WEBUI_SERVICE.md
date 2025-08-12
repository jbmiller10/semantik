# WEBUI_SERVICE - FastAPI Backend Service Documentation

## 1. Component Overview

### Purpose
The WEBUI_SERVICE is the central FastAPI backend service that orchestrates all web-based interactions for the Semantik application. It serves as the primary API gateway, handles authentication, manages WebSocket connections for real-time updates, and coordinates asynchronous operations through Celery task dispatching.

### Core Responsibilities
- **API Gateway**: Exposes RESTful endpoints for collection, document, operation, and search management
- **Authentication & Authorization**: JWT-based authentication with user session management
- **WebSocket Management**: Real-time bidirectional communication for operation progress tracking
- **Rate Limiting**: Distributed rate limiting with Redis backend and circuit breaker pattern
- **Static File Serving**: Hosts the React frontend application
- **Task Orchestration**: Dispatches asynchronous tasks to Celery workers
- **Request Correlation**: Tracks requests across the system with correlation IDs

### Role in System Architecture
The WEBUI_SERVICE acts as the primary entry point for all client interactions, sitting between the React frontend and the backend services (worker, vecpipe). It maintains the web session state, enforces security policies, and coordinates distributed operations across the system.

## 2. Architecture & Design Patterns

### Service Layer Architecture
The service implements a strict three-layer architecture:

```python
# Layer 1: API Routers (packages/webui/api/v2/)
# - Handle HTTP request/response
# - Input validation via Pydantic
# - Exception translation to HTTP status codes

# Layer 2: Service Layer (packages/webui/services/)
# - Business logic orchestration
# - Transaction management
# - Cross-repository coordination

# Layer 3: Repository Layer (packages/shared/database/repositories/)
# - Database access abstraction
# - Query construction
# - ORM entity management
```

### Dependency Injection Pattern
FastAPI's dependency injection is used extensively:

```python
# packages/webui/services/factory.py
async def get_collection_service(db: AsyncSession = Depends(get_db)) -> CollectionService:
    """FastAPI dependency for CollectionService injection."""
    return create_collection_service(db)

# Usage in router
@router.post("/collections")
async def create_collection(
    request: CollectionCreate,
    service: CollectionService = Depends(get_collection_service),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    # Service layer handles business logic
    collection, operation = await service.create_collection(...)
```

### Middleware Stack
Middleware components are applied in specific order:

```python
# packages/webui/main.py - Middleware registration order matters!
# 1. Correlation Middleware (first - sets context for all subsequent middleware)
app.add_middleware(CorrelationMiddleware)

# 2. Rate Limit Middleware (sets user in request.state)
app.add_middleware(RateLimitMiddleware)

# 3. CORS Middleware (handles cross-origin requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Correlation-ID"],
)
```

## 3. Key Interfaces & Contracts

### API Endpoint Groups

#### Collections API (v2)
```python
# packages/webui/api/v2/collections.py
router = APIRouter(prefix="/api/v2/collections", tags=["collections-v2"])

# Primary endpoints:
POST   /api/v2/collections              # Create collection
GET    /api/v2/collections              # List collections (paginated)
GET    /api/v2/collections/{uuid}       # Get collection details
PUT    /api/v2/collections/{uuid}       # Update collection
DELETE /api/v2/collections/{uuid}       # Delete collection
POST   /api/v2/collections/{uuid}/add-source    # Add document source
POST   /api/v2/collections/{uuid}/reindex       # Trigger reindexing
```

#### Operations API
```python
# packages/webui/api/v2/operations.py
router = APIRouter(prefix="/api/v2/operations", tags=["operations-v2"])

# Primary endpoints:
GET    /api/v2/operations/{uuid}        # Get operation status
GET    /api/v2/operations               # List operations
POST   /api/v2/operations/{uuid}/cancel # Cancel operation
DELETE /api/v2/operations/{uuid}        # Delete operation record
```

#### Search API
```python
# packages/webui/api/v2/search.py
router = APIRouter(prefix="/api/v2/search", tags=["search-v2"])

# Primary endpoints:
POST   /api/v2/search                   # Perform semantic search
POST   /api/v2/search/batch             # Batch search operations
```

### Request/Response Schemas
```python
# packages/webui/api/schemas.py
class CollectionCreate(BaseModel):
    name: str
    description: Optional[str] = None
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    quantization: str = "float16"
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    chunking_strategy: Optional[str] = None
    chunking_config: Optional[Dict[str, Any]] = None
    is_public: bool = False
    metadata: Optional[Dict[str, Any]] = None

class CollectionResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    owner_id: int
    vector_store_name: str
    embedding_model: str
    quantization: str
    status: str
    status_message: Optional[str]
    document_count: int
    vector_count: int
    created_at: datetime
    updated_at: datetime
    initial_operation_id: Optional[str]  # UUID of initial operation
```

## 4. Data Flow & Dependencies

### Request Processing Flow
```
Client Request → CORS Middleware → Correlation Middleware → Rate Limit Middleware
    ↓
API Router (validation) → Dependency Injection (auth, services)
    ↓
Service Layer (business logic) → Repository Layer (database)
    ↓
Response Serialization → Middleware Chain (reverse) → Client Response
```

### Asynchronous Operation Flow
```python
# 1. API endpoint creates operation record
operation = await operation_repo.create(
    collection_id=collection.id,
    user_id=user_id,
    operation_type=OperationType.INDEX,
    config={...}
)

# 2. Commit transaction BEFORE dispatching
await db_session.commit()

# 3. Dispatch Celery task
celery_app.send_task(
    "webui.tasks.process_collection_operation",
    args=[operation.uuid],
    task_id=str(uuid.uuid4()),
)

# 4. WebSocket updates sent from worker via Redis
await ws_manager.send_update(
    operation_id=operation.uuid,
    update_type="status_update",
    data={"status": "processing"}
)
```

## 5. Critical Implementation Details

### Authentication System
```python
# packages/webui/auth.py
# JWT-based authentication with access and refresh tokens
def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(UTC) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.ALGORITHM)

# Dependency for route protection
async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security)
) -> dict[str, Any]:
    if settings.DISABLE_AUTH:  # Development mode bypass
        return {"id": 0, "username": "dev_user", ...}
    
    # Validate JWT token
    username = verify_token(token, "access")
    # Fetch user from database
    user = await user_repo.get_user_by_username(username)
    return user
```

### WebSocket Management with Redis Streams
```python
# packages/webui/websocket_manager.py
class RedisStreamWebSocketManager:
    """Distributed WebSocket state synchronization using Redis Streams."""
    
    async def send_update(self, operation_id: str, update_type: str, data: dict):
        """Send update to Redis Stream for distribution."""
        message = {
            "timestamp": datetime.now(UTC).isoformat(),
            "type": update_type,
            "data": data
        }
        
        stream_key = f"operation-progress:{operation_id}"
        await self.redis.xadd(
            stream_key,
            {"message": json.dumps(message)},
            maxlen=1000,  # Circular buffer
        )
        
        # Dynamic TTL based on operation status
        ttl = 86400 if data.get("status") == "processing" else 300
        await self.redis.expire(stream_key, ttl)
```

### Rate Limiting with Circuit Breaker
```python
# packages/webui/rate_limiter.py
def track_circuit_breaker_failure(key: str) -> None:
    """Implement circuit breaker pattern for rate limiting."""
    current_time = time.time()
    
    # Increment failure count
    circuit_breaker.failure_counts[key] = circuit_breaker.failure_counts.get(key, 0) + 1
    
    # Check if threshold reached
    if circuit_breaker.failure_counts[key] >= circuit_breaker.failure_threshold:
        # Block for timeout period
        circuit_breaker.blocked_until[key] = current_time + circuit_breaker.timeout_seconds
        logger.warning(f"Circuit breaker activated for {key}")
```

### Scalable WebSocket Architecture
```python
# packages/webui/websocket/scalable_manager.py
class ScalableWebSocketManager:
    """Horizontally scalable WebSocket manager supporting 10,000+ connections."""
    
    async def connect(self, websocket: WebSocket, operation_id: str, user_id: str):
        # Connection limits enforcement
        total_connections = sum(len(sockets) for sockets in self.connections.values())
        if total_connections >= self.max_total_connections:
            await websocket.close(code=1008, reason="Server connection limit exceeded")
            return
        
        # Redis Pub/Sub for cross-instance messaging
        await self.pubsub.subscribe(f"operation:{operation_id}")
        
        # Local connection tracking
        self.local_connections[f"{user_id}:{operation_id}"] = websocket
```

## 6. Security Considerations

### Authentication & Authorization
- **JWT Token Validation**: All protected endpoints require valid JWT tokens
- **User Context Injection**: User identity verified and injected into request context
- **Permission Checks**: Collection ownership verified at service layer
- **WebSocket Authentication**: Token validation for WebSocket connections

### Input Validation & Sanitization
```python
# Pydantic models enforce type safety and validation
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    
    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters long")
        if not all(c.isalnum() or c == "_" for c in v):
            raise ValueError("Username must contain only alphanumeric characters and underscores")
        return v
```

### CORS Configuration
```python
# Validate and restrict CORS origins
def _validate_cors_origins(origins: list[str]) -> list[str]:
    valid_origins = []
    for origin in origins:
        if origin in ["*", "null"]:
            if shared_settings.ENVIRONMENT == "production":
                logger.error(f"Rejecting insecure origin '{origin}' in production")
                continue
        # Validate URL format
        parsed = urlparse(origin)
        if parsed.scheme and parsed.netloc:
            valid_origins.append(origin)
    return valid_origins
```

### Rate Limiting & DDoS Protection
- Per-user rate limits with Redis backend
- Circuit breaker pattern for repeated violations
- Connection limits per user and globally
- Admin bypass token for maintenance operations

## 7. Testing Requirements

### Unit Tests
```python
# Test service layer business logic
async def test_create_collection_with_chunking_strategy():
    service = CollectionService(db_session, collection_repo, operation_repo, document_repo)
    
    collection, operation = await service.create_collection(
        user_id=1,
        name="Test Collection",
        config={
            "chunking_strategy": "semantic",
            "chunking_config": {"min_chunk_size": 100}
        }
    )
    
    assert collection["chunking_strategy"] == "semantic"
    assert operation["type"] == "INDEX"
```

### Integration Tests
```python
# Test API endpoints with dependencies
async def test_api_create_collection(client: TestClient, auth_headers: dict):
    response = client.post(
        "/api/v2/collections",
        json={
            "name": "Test Collection",
            "description": "Test description",
            "chunking_strategy": "semantic"
        },
        headers=auth_headers
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Collection"
    assert data["initial_operation_id"] is not None
```

### WebSocket Tests
```python
# Test WebSocket connection and message flow
async def test_operation_websocket_updates():
    async with websocket_connect(f"/ws/operations/{operation_id}") as websocket:
        # Send progress update
        await ws_manager.send_update(
            operation_id=operation_id,
            update_type="progress",
            data={"percentage": 50}
        )
        
        # Verify message received
        message = await websocket.receive_json()
        assert message["type"] == "progress"
        assert message["data"]["percentage"] == 50
```

## 8. Common Pitfalls & Best Practices

### ❌ Anti-Pattern: Direct Database Access in Routers
```python
# BAD: Business logic in router
@router.post("/collections")
async def create_collection(request: Request, db: AsyncSession = Depends(get_db)):
    new_collection = CollectionModel(**request.dict())
    db.add(new_collection)
    await db.commit()  # Business logic in router!
    return new_collection
```

### ✅ Best Practice: Service Layer Delegation
```python
# GOOD: Delegate to service layer
@router.post("/collections")
async def create_collection(
    request: CollectionCreate,
    service: CollectionService = Depends(get_collection_service)
):
    collection = await service.create_collection(request.dict())
    return collection
```

### ❌ Anti-Pattern: Dispatching Tasks Before Commit
```python
# BAD: Race condition risk
operation = await operation_repo.create(...)
celery_app.send_task("process_operation", args=[operation.uuid])
await db_session.commit()  # Task might run before data is committed!
```

### ✅ Best Practice: Commit Before Task Dispatch
```python
# GOOD: Ensure data is committed first
operation = await operation_repo.create(...)
await db_session.commit()  # Commit first
celery_app.send_task("process_operation", args=[operation.uuid])  # Then dispatch
```

### ❌ Anti-Pattern: Synchronous I/O in Async Functions
```python
# BAD: Blocking I/O in async context
async def process_file(path: str):
    with open(path, 'r') as f:  # Blocks event loop!
        content = f.read()
```

### ✅ Best Practice: Use Async I/O
```python
# GOOD: Non-blocking async I/O
async def process_file(path: str):
    async with aiofiles.open(path, 'r') as f:
        content = await f.read()
```

## 9. Configuration & Environment

### Required Environment Variables
```bash
# Core Configuration
WEBUI_PORT=9000
ENVIRONMENT=development|production
DISABLE_AUTH=false  # Set to true for development

# JWT Configuration
JWT_SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://localhost:9000

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Database Configuration
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/semantik

# Internal API Configuration
INTERNAL_API_KEY=change-me-in-production

# Rate Limiting
RATE_LIMIT_BYPASS_TOKEN=admin-token-here
DISABLE_RATE_LIMITING=false  # Set to true for testing

# Embedding Service
USE_MOCK_EMBEDDINGS=false
```

### Application Startup Configuration
```python
# packages/webui/main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup sequence
    configure_logging_with_correlation()
    await pg_connection_manager.initialize()
    await ws_manager.startup()
    await ensure_default_data()
    _configure_embedding_service()
    _configure_internal_api_key()
    await start_background_tasks()
    
    yield  # Application runs
    
    # Shutdown sequence
    await stop_background_tasks()
    await ws_manager.shutdown()
    await pg_connection_manager.close()
```

## 10. Integration Points

### Celery Worker Integration
```python
# Task dispatch from webui to worker
celery_app.send_task(
    "webui.tasks.process_collection_operation",
    args=[operation_uuid],
    task_id=str(uuid.uuid4()),
)

# Worker sends updates back via WebSocket manager
from packages.webui.websocket_manager import ws_manager
await ws_manager.send_update(
    operation_id=operation_uuid,
    update_type="chunking_progress",
    data={"progress": 75, "current_document": "doc.pdf"}
)
```

### VecPipe Service Integration
```python
# Direct HTTP calls to vecpipe for vector operations
async with httpx.AsyncClient() as client:
    response = await client.post(
        f"{settings.VECPIPE_URL}/api/embed",
        json={
            "texts": chunk_texts,
            "model": collection.embedding_model,
            "quantization": collection.quantization
        },
        timeout=30.0
    )
    embeddings = response.json()["embeddings"]
```

### PostgreSQL Database
- Async SQLAlchemy sessions via `AsyncSession`
- Repository pattern for database abstraction
- Alembic migrations for schema management
- Connection pooling with `pg_connection_manager`

### Qdrant Vector Database
```python
# packages/webui/utils/qdrant_manager.py
class QdrantManager:
    async def create_collection(self, name: str, dimension: int):
        """Create Qdrant collection for vector storage."""
        await self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE
            )
        )
```

### Redis Integration
- **Rate Limiting**: Distributed rate limit counters
- **WebSocket State**: Redis Streams for operation progress
- **Pub/Sub**: Cross-instance WebSocket messaging
- **Circuit Breaker**: Failure tracking and blocking
- **Session Management**: Optional session storage

### React Frontend
- Static file serving via `/static` mount
- SPA routing with catch-all handler
- WebSocket connections for real-time updates
- CORS configuration for API access

## File Structure Reference

```
packages/webui/
├── main.py                 # Application entry point and configuration
├── app.py                  # Backward compatibility shim
├── auth.py                 # JWT authentication system
├── rate_limiter.py         # Rate limiting with circuit breaker
├── websocket_manager.py    # Redis Streams WebSocket manager
├── dependencies.py         # FastAPI dependency providers
├── startup_tasks.py        # Application initialization tasks
├── api/
│   ├── v2/
│   │   ├── collections.py  # Collection management endpoints
│   │   ├── documents.py    # Document management endpoints
│   │   ├── operations.py   # Operation tracking endpoints
│   │   ├── search.py       # Semantic search endpoints
│   │   ├── chunking.py     # Chunking strategy endpoints
│   │   └── schemas.py      # Pydantic request/response models
│   └── schemas.py          # Shared schema definitions
├── services/
│   ├── factory.py          # Service instance factories
│   ├── collection_service.py    # Collection business logic
│   ├── operation_service.py     # Operation management
│   ├── search_service.py        # Search orchestration
│   ├── chunking_service.py      # Document chunking logic
│   └── redis_manager.py         # Redis connection management
├── middleware/
│   ├── correlation.py      # Request correlation tracking
│   └── rate_limit.py       # User identification for rate limiting
├── websocket/
│   └── scalable_manager.py # Horizontally scalable WebSocket manager
└── static/                 # React frontend build output
    ├── index.html
    └── assets/
```

## Critical Implementation Notes

1. **Transaction Boundaries**: Always commit database transactions before dispatching Celery tasks to avoid race conditions where workers try to access uncommitted data.

2. **WebSocket Authentication**: WebSocket connections cannot use standard HTTP authentication headers. Pass JWT tokens as query parameters or in the first message after connection.

3. **Rate Limit Headers**: Rate limit headers are NOT automatically added to responses due to FastAPI's dictionary-to-JSON conversion. Headers are only added to rate limit exceeded responses.

4. **Middleware Order**: The order of middleware registration is critical. CorrelationMiddleware must be first to establish context for all subsequent middleware.

5. **Service Layer Pattern**: All business logic MUST be in the service layer. API routers should only handle HTTP concerns and delegate to services.

6. **Redis Failover**: The application gracefully degrades when Redis is unavailable. Rate limiting falls back to in-memory, WebSockets work in local-only mode.

7. **Chunking Strategy Validation**: Chunking strategies are validated at the service layer, not just at the API layer, to ensure consistency across all entry points.

8. **Connection Limits**: WebSocket connections are limited both per-user and globally to prevent resource exhaustion attacks.

9. **Operation Progress Throttling**: WebSocket progress updates are throttled to prevent overwhelming clients with high-frequency updates.

10. **Static File Serving**: The React frontend is served from `/static` with a catch-all route for SPA routing. This must be mounted AFTER API routes to prevent interference.