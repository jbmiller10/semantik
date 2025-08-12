# Semantik Backend API Architecture Specification

## Executive Summary

Semantik is a self-hosted semantic search engine built with a microservices architecture using Python FastAPI. The system is currently undergoing a critical refactoring from a "job-centric" to a "collection-centric" architecture to improve scalability, maintainability, and user experience.

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [FastAPI Application Structure](#fastapi-application-structure)
3. [Service Layer Architecture](#service-layer-architecture)
4. [Repository Pattern Implementation](#repository-pattern-implementation)
5. [Celery Worker Architecture](#celery-worker-architecture)
6. [Integration Points](#integration-points)
7. [Configuration Management](#configuration-management)
8. [Security Architecture](#security-architecture)
9. [Error Handling & Monitoring](#error-handling--monitoring)
10. [Design Patterns & Best Practices](#design-patterns--best-practices)

## System Architecture Overview

### Technology Stack

- **Backend Framework**: Python 3.11+ with FastAPI
- **Database**: PostgreSQL (metadata) + Qdrant (vector storage)
- **Async Task Processing**: Celery with Redis broker
- **Caching & Pub/Sub**: Redis
- **WebSocket**: Scalable WebSocket manager with Redis Pub/Sub
- **Authentication**: JWT-based with refresh tokens
- **Containerization**: Docker & Docker Compose

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                      │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    WebUI Service (FastAPI)                   │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ API Routers │  │   Services   │  │ Repositories │       │
│  └─────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
        │                   │                    │
        ▼                   ▼                    ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Celery     │   │   VecPipe    │   │  PostgreSQL  │
│   Workers    │   │   Service    │   │   + Qdrant   │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │
        ▼                   ▼
┌─────────────────────────────────┐
│          Redis (Cache/Broker)     │
└───────────────────────────────────┘
```

## FastAPI Application Structure

### Main Application Setup (`packages/webui/main.py`)

The FastAPI application is initialized with comprehensive middleware stack and lifespan management:

```python
# Application Configuration
app = FastAPI(
    title="Document Embedding Web UI",
    version="1.1.0",
    lifespan=lifespan  # Manages startup/shutdown events
)
```

### Middleware Stack (Order Matters)

1. **CorrelationMiddleware** - Request tracing with correlation IDs
2. **RateLimitMiddleware** - User identification for rate limiting
3. **CORSMiddleware** - Cross-origin request handling

### API Router Organization

```
/api/
├── /auth           - Authentication endpoints
├── /metrics        - Prometheus metrics
├── /settings       - Application settings
├── /models         - Model management
├── /health         - Health checks
├── /internal       - Internal service APIs
└── /v2/
    ├── /chunking              - Document chunking
    ├── /collections           - Collection management
    ├── /directory-scan        - Directory scanning
    ├── /documents             - Document operations
    ├── /operations            - Async operations
    ├── /partition-monitoring  - DB partition monitoring
    ├── /search               - Search operations
    └── /system               - System utilities
```

### WebSocket Endpoints

```
/ws/operations/{operation_id}     - Operation progress updates
/ws/directory-scan/{scan_id}      - Directory scan progress
```

### Lifespan Management

The application uses an async context manager for proper resource initialization and cleanup:

**Startup Sequence:**
1. Configure logging with correlation support
2. Initialize PostgreSQL connection pool
3. Initialize WebSocket manager
4. Ensure default data exists
5. Configure global embedding service
6. Configure internal API key
7. Start background tasks

**Shutdown Sequence:**
1. Stop background tasks
2. Shutdown WebSocket manager
3. Close PostgreSQL connections

## Service Layer Architecture

### Design Principles

Services implement the **Domain-Driven Design (DDD)** pattern with clear separation of concerns:

- **No direct database access from API routers**
- **All business logic resides in services**
- **Services orchestrate repositories and external systems**
- **Transaction management at service level**

### Core Services

#### CollectionService (`packages/webui/services/collection_service.py`)

**Responsibilities:**
- Collection CRUD operations
- Source management (add/remove)
- Reindexing orchestration
- Permission validation
- Celery task dispatching

**Key Methods:**
```python
async def create_collection(
    user_id: int,
    name: str,
    description: str | None,
    config: dict[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any]]

async def add_source(
    collection_id: str,
    user_id: int,
    source_path: str,
    source_config: dict[str, Any] | None
) -> dict[str, Any]

async def reindex_collection(
    collection_id: str,
    user_id: int,
    config: dict[str, Any] | None
) -> dict[str, Any]
```

#### OperationService (`packages/webui/services/operation_service.py`)

**Responsibilities:**
- Operation status tracking
- Progress monitoring
- Operation cancellation
- Result retrieval

#### SearchService (`packages/webui/services/search_service.py`)

**Responsibilities:**
- Query processing
- Result aggregation
- Relevance scoring
- Cross-collection search

#### ChunkingService (`packages/webui/services/chunking_service.py`)

**Responsibilities:**
- Document chunking strategies
- Chunk validation
- Strategy selection
- Performance optimization

### Service Factory Pattern

Services are created using dependency injection through factory functions:

```python
def get_collection_service(
    db_session: AsyncSession = Depends(get_db),
    collection_repo: CollectionRepository = Depends(get_collection_repository),
    operation_repo: OperationRepository = Depends(get_operation_repository),
    document_repo: DocumentRepository = Depends(get_document_repository)
) -> CollectionService:
    return CollectionService(db_session, collection_repo, operation_repo, document_repo)
```

## Repository Pattern Implementation

### Base Repository (`packages/webui/repositories/postgres/base.py`)

All repositories inherit from a base class providing common functionality:

```python
class BaseRepository:
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def commit(self) -> None:
        await self.db.commit()
    
    async def rollback(self) -> None:
        await self.db.rollback()
```

### Repository Implementations

#### CollectionRepository
- CRUD operations for collections
- Permission-based queries
- Status management
- Atomic updates for reindexing

#### DocumentRepository
- Document metadata management
- Batch operations
- Source tracking
- Vector count aggregation

#### OperationRepository
- Operation lifecycle management
- Status updates
- Progress tracking
- Error recording

#### UserRepository & AuthRepository
- User management
- Authentication
- API key management
- Session handling

### Transaction Management

Repositories follow these patterns:
- **Read operations**: No explicit transaction required
- **Write operations**: Service layer manages transactions
- **Batch operations**: Use bulk insert/update for performance

## Celery Worker Architecture

### Configuration (`packages/webui/celery_app.py`)

```python
celery_app = Celery(
    "webui",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["webui.tasks"]
)

# Key configurations
task_soft_time_limit=3600      # 1 hour soft limit
task_time_limit=7200           # 2 hour hard limit
task_acks_late=True           # Late acknowledgment for reliability
worker_prefetch_multiplier=1   # No prefetching for long tasks
worker_max_tasks_per_child=100 # Restart after 100 tasks
```

### Task Architecture (`packages/webui/tasks.py`)

#### Main Task: `process_collection_operation`

Unified entry point for all collection operations:

```python
@celery_app.task(
    bind=True,
    soft_time_limit=OPERATION_SOFT_TIME_LIMIT,
    time_limit=OPERATION_HARD_TIME_LIMIT,
    acks_late=True,
    max_retries=DEFAULT_MAX_RETRIES,
    default_retry_delay=DEFAULT_RETRY_DELAY
)
async def process_collection_operation(self, operation_uuid: str):
    # Comprehensive operation processing with:
    # - Resource tracking
    # - Progress updates via Redis streams
    # - Atomic database updates
    # - Error recovery
    # - Metrics collection
```

### Background Tasks

#### Periodic Tasks (Beat Schedule)
1. **cleanup-old-results** - Daily cleanup of old Celery results
2. **refresh-collection-chunking-stats** - Hourly stats refresh
3. **monitor-partition-health** - 6-hour partition monitoring

#### Task Features
- **Progress Updates**: Real-time updates via Redis streams
- **Resource Tracking**: CPU, memory, duration monitoring
- **Error Handling**: Comprehensive error recovery with status updates
- **Metrics**: Prometheus metrics for monitoring

### Task Communication

Tasks communicate progress using Redis streams:

```python
class CeleryTaskWithOperationUpdates:
    async def send_update(self, update_type: str, data: dict):
        # Sends to operation-progress:{operation_id} stream
        message = {
            "timestamp": datetime.now(UTC).isoformat(),
            "type": update_type,
            "data": data
        }
        await redis_client.xadd(stream_key, {"message": json.dumps(message)})
```

## Integration Points

### WebUI ↔ VecPipe Communication

VecPipe provides a dedicated FastAPI service for document processing and search:

**Endpoints:**
- `POST /embed` - Batch embedding generation
- `POST /search` - Vector similarity search
- `POST /batch-search` - Multiple simultaneous searches
- `POST /upsert` - Vector storage operations
- `GET /health` - Service health check

**Communication Pattern:**
```python
async with httpx.AsyncClient() as client:
    response = await client.post(
        f"{VECPIPE_URL}/search",
        json=search_request,
        timeout=30.0
    )
```

### Redis Usage Patterns

#### 1. Celery Message Broker (DB 0)
- Task queue management
- Result backend
- Beat schedule coordination

#### 2. WebSocket Pub/Sub (DB 2)
- Real-time message broadcasting
- Cross-instance communication
- Connection registry with TTL

#### 3. Cache Layer (DB 1)
- Session storage
- Temporary data
- Rate limiting counters

### WebSocket Architecture

#### Scalable WebSocket Manager (`packages/webui/websocket/scalable_manager.py`)

**Features:**
- 10,000+ concurrent connections support
- <100ms message latency
- Horizontal scaling via Redis Pub/Sub
- Automatic dead connection cleanup
- User channel isolation

**Architecture:**
```python
class ScalableWebSocketManager:
    # Local connection tracking per instance
    local_connections: dict[str, WebSocket]
    
    # Redis Pub/Sub for cross-instance messaging
    async def broadcast_to_user(user_id: str, message: dict):
        await redis.publish(f"user:{user_id}", json.dumps(message))
    
    # Connection lifecycle management
    async def connect(websocket: WebSocket, user_id: str):
        # Register in Redis with TTL
        # Subscribe to user channel
        # Track locally
```

## Configuration Management

### Configuration Hierarchy

```
shared/config/
├── base.py         - Base configuration class
├── webui.py        - WebUI-specific settings
├── vecpipe.py      - VecPipe service settings
└── postgres.py     - Database configuration
```

### Environment-Based Configuration

```python
class WebuiConfig(BaseConfig):
    # JWT Configuration
    JWT_SECRET_KEY: str = "default-secret-key"  # Must override in production
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24
    
    # Service URLs
    WEBUI_URL: str = "http://localhost:8080"
    SEARCH_API_URL: str = "http://localhost:8000"
    
    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # CORS Configuration
    CORS_ORIGINS: str = "http://localhost:5173"
```

### Secret Management

**Development Mode:**
- Auto-generates JWT secret and saves to `.jwt_secret` file
- Generates internal API key if using default

**Production Mode:**
- Requires explicit JWT_SECRET_KEY environment variable
- Validates all security-critical configurations
- No default credentials allowed

## Security Architecture

### Authentication System (`packages/webui/auth.py`)

#### JWT Token Management
```python
# Dual-token system
ACCESS_TOKEN_EXPIRE = 24 hours
REFRESH_TOKEN_EXPIRE = 30 days

# Token generation with claims
def create_access_token(data: dict, expires_delta: timedelta | None):
    to_encode = data.copy()
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm="HS256")
```

#### Authentication Flow
1. User login with credentials
2. Password verification using bcrypt
3. Generate access + refresh tokens
4. Update last_login timestamp
5. Return token pair

### Authorization Patterns

#### Dependency-Based Authorization
```python
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict[str, Any]:
    # Token validation
    # User lookup
    # Permission check
    return user_dict
```

#### Collection-Level Permissions
```python
async def get_collection_for_user(
    collection_uuid: str,
    current_user: dict = Depends(get_current_user)
) -> Collection:
    # Ownership verification
    # Public collection check
    # Raise 403 if unauthorized
```

### Internal API Security

Protected endpoints for service-to-service communication:

```python
def verify_internal_api_key(
    x_internal_api_key: str | None = Header()
) -> None:
    if x_internal_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=401)
```

### Rate Limiting

Multi-layer rate limiting strategy:

1. **User-based limits** via Redis counters
2. **Endpoint-specific limits** with decorators
3. **Circuit breaker** for repeated violations
4. **IP-based fallback** for unauthenticated requests

## Error Handling & Monitoring

### Exception Hierarchy

```python
# Custom exceptions in shared/database/exceptions.py
EntityNotFoundError(404)
EntityAlreadyExistsError(409)
AccessDeniedError(403)
InvalidStateError(409)
ValidationError(400)
DimensionMismatchError(400)
```

### Error Handler Registration

```python
# Chunking-specific error handlers
register_chunking_exception_handlers(app)

# Rate limit handler
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
```

### Correlation ID Tracking

Every request gets a unique correlation ID for tracing:

```python
class CorrelationMiddleware(BaseHTTPMiddleware):
    async def dispatch(request: Request, call_next):
        correlation_id = request.headers.get("X-Correlation-ID") 
        or str(uuid.uuid4())
        correlation_id_var.set(correlation_id)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response
```

### Metrics Collection

#### Prometheus Metrics
```python
# API metrics
search_latency = Histogram("search_api_latency_seconds")
search_requests = Counter("search_api_requests_total")
search_errors = Counter("search_api_errors_total")

# Collection metrics
collection_operations = Counter("collection_operations_total")
collection_cpu_seconds = Counter("collection_cpu_seconds_total")
collection_memory_bytes = Gauge("collection_memory_usage_bytes")
```

#### Health Checks
```python
@router.get("/health")
async def health_check():
    # Database connectivity
    # Redis connectivity
    # Qdrant connectivity
    # Resource availability
    return {"status": "healthy", "checks": {...}}
```

## Design Patterns & Best Practices

### Architectural Patterns

#### 1. Domain-Driven Design (DDD)
- Clear bounded contexts (collections, operations, documents)
- Rich domain models with business logic
- Ubiquitous language (collection-centric terminology)

#### 2. Repository Pattern
- Abstracts data access logic
- Enables testing with mock repositories
- Consistent query interface

#### 3. Service Layer Pattern
- Orchestrates business operations
- Manages transactions
- Handles cross-cutting concerns

#### 4. Dependency Injection
- FastAPI's `Depends` for automatic injection
- Testable components
- Loose coupling

### Code Organization Principles

#### Separation of Concerns
```
API Router → Service → Repository → Database
     ↓          ↓           ↓
  Request   Business    Data Access
  Handling    Logic       Layer
```

#### Single Responsibility
- Routers: HTTP request/response handling only
- Services: Business logic and orchestration
- Repositories: Data persistence only
- Tasks: Async processing logic

### Anti-Patterns to Avoid

#### ❌ Direct Database Access from Routers
```python
# BAD
@router.post("/")
async def create_collection(db: AsyncSession = Depends(get_db)):
    collection = CollectionModel(...)
    db.add(collection)
    await db.commit()
```

#### ✅ Service Layer Delegation
```python
# GOOD
@router.post("/")
async def create_collection(
    service: CollectionService = Depends(get_collection_service)
):
    collection = await service.create_collection(...)
```

#### ❌ Mixing Sync/Async in Celery
```python
# BAD - causes event loop conflicts
def celery_task():
    asyncio.run(async_function())  # Never do this!
```

#### ✅ Proper Async Handling
```python
# GOOD - use sync Redis client in Celery
def celery_task():
    redis_client = redis_manager.sync_client
    redis_client.set("key", "value")
```

### Testing Strategy

#### Unit Tests
- Mock repositories for service tests
- Mock services for router tests
- Isolated component testing

#### Integration Tests
- Real database with test transactions
- End-to-end API testing
- WebSocket connection testing

#### Performance Tests
- Load testing with concurrent connections
- Memory leak detection
- Query optimization validation

### Deployment Considerations

#### Container Architecture
```yaml
services:
  webui:
    - FastAPI application
    - Static file serving
    - WebSocket handling
  
  worker:
    - Celery workers
    - Beat scheduler
    - Resource isolation
  
  vecpipe:
    - Embedding service
    - Search API
    - Model management
```

#### Scaling Strategy
- **Horizontal**: Multiple WebUI instances behind load balancer
- **Vertical**: Increase worker processes for CPU-bound tasks
- **Queue-based**: Separate queues for priority operations

#### Monitoring Requirements
- Prometheus metrics endpoint
- Structured logging with correlation IDs
- Health checks for all services
- Resource usage tracking

## Conclusion

The Semantik backend architecture demonstrates a well-structured, scalable microservices design with clear separation of concerns, comprehensive error handling, and robust security measures. The ongoing refactoring from job-centric to collection-centric architecture shows a commitment to improving the system's maintainability and user experience while maintaining backward compatibility where possible.

Key strengths include:
- Clean architectural boundaries
- Comprehensive async support
- Scalable WebSocket implementation
- Robust error handling and monitoring
- Security-first design

Areas for continuous improvement:
- Complete migration to collection-centric terminology
- Enhanced caching strategies
- Further optimization of vector operations
- Expanded test coverage