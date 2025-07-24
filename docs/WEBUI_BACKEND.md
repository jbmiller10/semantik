# WebUI Backend Documentation

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Main Application Structure](#main-application-structure)
3. [API Router Architecture](#api-router-architecture)
4. [Authentication System](#authentication-system)
5. [Database Layer](#database-layer)
6. [Schemas and Models](#schemas-and-models)
7. [Embedding Service Integration](#embedding-service-integration)
8. [WebSocket Implementation](#websocket-implementation)
9. [Rate Limiting](#rate-limiting)
10. [Utilities](#utilities)
11. [API Endpoints Reference](#api-endpoints-reference)

## Architecture Overview

The WebUI serves as a control plane for the Semantik semantic search engine. It provides:

- **User Interface**: React-based frontend for managing collections and searching
- **Collection Management**: Create, monitor, and manage document collections
- **Operation Tracking**: Monitor async operations (indexing, reindexing, etc.)
- **Search Proxy**: Routes search requests to the Semantik search API with multi-collection support
- **Authentication**: JWT-based user authentication with refresh tokens
- **Real-time Updates**: WebSocket connections for operation progress monitoring

### Key Architectural Principles

1. **Separation of Concerns**: WebUI acts as a control plane, never implementing core search or embedding logic
2. **Proxy Pattern**: All search functionality proxies to the Semantik search API
3. **Database Independence**: Semantik core engine never accesses the WebUI PostgreSQL database
4. **Scalability**: Designed to handle multiple concurrent operations and users
5. **Service Layer**: Business logic isolated in services, not in API routers

## Main Application Structure

### main.py

The entry point that creates and configures the FastAPI application:

```python
def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="Document Embedding Web UI",
        description="Create and search document embeddings",
        version="1.1.0"
    )
```

Key features:
- Rate limiting middleware via SlowAPI
- Router registration for all API endpoints
- WebSocket endpoints for real-time updates
- Static file serving for the React frontend

### app.py

A backward compatibility shim that imports the app instance from `main.py`, ensuring existing scripts using `webui.app:app` continue to work.

## API Router Architecture

The API is organized into modular routers, each handling specific functionality:

### api/__init__.py
Exports all router modules for easy import in the main application.

### api/auth.py
**Endpoints**: `/api/auth/*`

Handles user authentication:
- `POST /register` - Register new user
- `POST /login` - Login and receive JWT tokens
- `POST /refresh` - Refresh access token
- `POST /logout` - Logout and revoke refresh token
- `GET /me` - Get current user info

### api/v2/collections.py
**Endpoints**: `/api/v2/collections/*`

Manages collection lifecycle:
- `POST /` - Create new collection
- `GET /` - List all collections
- `GET /{collection_id}` - Get collection details
- `PUT /{collection_id}` - Update collection metadata
- `DELETE /{collection_id}` - Delete collection
- `POST /{collection_id}/sources` - Add source to collection
- `DELETE /{collection_id}/sources` - Remove source from collection
- `POST /{collection_id}/reindex` - Reindex collection
- `GET /{collection_id}/operations` - List collection operations
- `GET /{collection_id}/documents` - List collection documents

### api/v2/operations.py
**Endpoints**: `/api/v2/operations/*`

Manages async operations:
- `GET /{operation_id}` - Get operation details
- `DELETE /{operation_id}` - Cancel operation
- `GET /` - List all operations
- WebSocket: `/api/v2/operations/{operation_id}/ws` - Real-time progress

### api/v2/search.py
**Endpoints**: `/api/v2/*`

Proxies search requests to Semantik:
- `POST /search` - Multi-collection semantic search with optional reranking
- Supports searching across multiple collections simultaneously
- Validates user permissions for each collection

### api/files.py
**Endpoints**: `/api/*`

File system operations:
- `POST /scan-directory` - Scan directory for supported files

### api/metrics.py
**Endpoints**: `/api/*`

System monitoring:
- `GET /metrics` - Get Prometheus metrics

### api/models.py
**Endpoints**: `/api/*`

Model information:
- `GET /models` - List available embedding models
- `GET /model-info` - Get specific model details

### api/settings.py
**Endpoints**: `/api/*`

Configuration management:
- `GET /settings` - Get current settings
- `PUT /settings` - Update settings

### api/documents.py
**Endpoints**: `/api/*`

Document management:
- `GET /documents` - List documents
- `POST /documents/extract-text` - Extract text from document

### api/collection_metadata.py

Internal utilities for storing and retrieving collection metadata in Qdrant.

### api/root.py

Serves the React frontend and handles root-level routes.

## Authentication System

### JWT Implementation (auth.py)

The authentication system uses JWT tokens with refresh token support:

1. **Access Tokens**: Short-lived (configurable, default 30 minutes)
2. **Refresh Tokens**: Long-lived (30 days), stored hashed in database
3. **Password Hashing**: bcrypt with automatic salt generation

Key components:
- `UserCreate`: Pydantic model for registration validation
- `authenticate_user()`: Verifies credentials and updates last login
- `get_current_user()`: FastAPI dependency for protected endpoints

### Security Features

- Password minimum length: 8 characters
- Username validation: alphanumeric and underscores only
- Refresh token revocation on logout
- Automatic token refresh mechanism

## Database Layer

### Repository Pattern

The new architecture uses a repository pattern for clean data access:

- **CollectionRepository**: Manages collection CRUD operations
- **OperationRepository**: Tracks async operations
- **DocumentRepository**: Handles document metadata
- **UserRepository**: User management

### Core Tables (PostgreSQL)

#### Collections Table
```sql
CREATE TABLE collections (
    id VARCHAR PRIMARY KEY,                    -- UUID
    name VARCHAR UNIQUE NOT NULL,
    description TEXT,
    owner_id INTEGER NOT NULL,
    vector_store_name VARCHAR UNIQUE NOT NULL,
    embedding_model VARCHAR NOT NULL,
    quantization VARCHAR DEFAULT 'float16',
    chunk_size INTEGER DEFAULT 1000,
    chunk_overlap INTEGER DEFAULT 200,
    is_public BOOLEAN DEFAULT FALSE,
    metadata JSON,
    status VARCHAR,                            -- pending|ready|processing|error|degraded
    status_message TEXT,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ,
    FOREIGN KEY (owner_id) REFERENCES users(id)
)
```

#### Operations Table
```sql
CREATE TABLE operations (
    id SERIAL PRIMARY KEY,
    uuid VARCHAR UNIQUE NOT NULL,
    collection_id VARCHAR NOT NULL,
    user_id INTEGER NOT NULL,
    type VARCHAR NOT NULL,                     -- index|append|reindex|remove_source|delete
    status VARCHAR DEFAULT 'pending',          -- pending|processing|completed|failed|cancelled
    config JSON NOT NULL,
    progress JSON,
    error_message TEXT,
    created_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    FOREIGN KEY (collection_id) REFERENCES collections(id),
    FOREIGN KEY (user_id) REFERENCES users(id)
)
```

#### Documents Table
```sql
CREATE TABLE documents (
    id VARCHAR PRIMARY KEY,                    -- UUID
    collection_id VARCHAR NOT NULL,
    source_id INTEGER,
    file_path VARCHAR NOT NULL,
    file_name VARCHAR NOT NULL,
    file_size INTEGER NOT NULL,
    mime_type VARCHAR,
    content_hash VARCHAR NOT NULL,
    status VARCHAR DEFAULT 'pending',          -- pending|processing|completed|failed
    error_message TEXT,
    chunk_count INTEGER DEFAULT 0,
    metadata JSON,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ,
    FOREIGN KEY (collection_id) REFERENCES collections(id),
    FOREIGN KEY (source_id) REFERENCES sources(id)
)
```

#### Users Table
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT,
    hashed_password TEXT NOT NULL,
    is_active BOOLEAN DEFAULT 1,
    created_at TEXT NOT NULL,
    last_login TEXT
)
```

#### Refresh Tokens Table
```sql
CREATE TABLE refresh_tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    token_hash TEXT UNIQUE NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    is_revoked BOOLEAN DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(id)
)
```

### Service Layer

The application uses a service layer pattern for business logic:

- **CollectionService**: Collection lifecycle management
- **OperationService**: Operation tracking and management
- **DocumentService**: Document processing and deduplication
- **UserService**: User authentication and management

## Schemas and Models

### schemas.py

Defines shared Pydantic models:

```python
class FileInfo(BaseModel):
    path: str
    size: int
    modified: str
    extension: str
    hash: str | None = None
```

## Embedding Service Integration

### embedding_service.py

A unified service that bridges Semantik and WebUI for embedding generation:

#### Key Features

1. **Multi-model Support**: Handles both Sentence Transformers and Qwen3 models
2. **Quantization**: Supports float32, float16, and int8 quantization
3. **Adaptive Batch Sizing**: Automatically adjusts batch size on OOM errors
4. **Mock Mode**: For testing without loading actual models

#### Memory Management

- Lazy model loading and unloading
- Automatic garbage collection after processing
- Batch size reduction on OOM errors (minimum 4)
- Memory tracking and logging

#### Model Configuration

```python
QUANTIZED_MODEL_INFO = {
    "Qwen/Qwen3-Embedding-0.6B": {
        "dimension": 1024,
        "supports_quantization": True,
        "recommended_quantization": "float16",
        "memory_estimate": {
            "float32": 2400,
            "float16": 1200,
            "int8": 600
        }
    }
    # ... more models
}
```

## WebSocket Implementation

### Connection Manager

The `ConnectionManager` class handles WebSocket connections for real-time updates:

```python
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}
```

### WebSocket Endpoints

1. **Operation Progress**: `/api/v2/operations/{operation_id}/ws`
   - Real-time operation progress updates
   - Document processing notifications
   - Error messages and status changes

2. **Directory Scan**: `/ws/scan/{scan_id}`
   - Progress updates during directory scanning
   - File count and current path

### Message Types

Operation WebSocket messages:
- `operation_started`: Initial operation start with total documents
- `document_processing`: Current document being processed
- `document_completed`: Document successfully processed
- `operation_completed`: All documents processed
- `error`: Processing error occurred

## Rate Limiting

### rate_limiter.py

Uses SlowAPI for request rate limiting:

```python
limiter = Limiter(key_func=get_remote_address)
```

Applied globally to prevent abuse and ensure fair resource usage.

## Utilities

### utils/qdrant_manager.py

Singleton connection manager for Qdrant:

#### Features
- Connection pooling and reuse
- Automatic retry with exponential backoff
- Connection verification
- Thread-safe singleton pattern

#### Key Methods
- `get_client()`: Returns verified Qdrant client
- `create_collection()`: Creates collection with retry
- `verify_collection()`: Verifies collection exists

### utils/retry.py

Provides retry decorators for handling transient failures:

```python
@exponential_backoff_retry(
    max_retries=3,
    initial_delay=1.0,
    max_delay=8.0
)
def some_operation():
    # Operation that might fail transiently
```

## API Endpoints Reference

### Authentication Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/auth/register` | Register new user | No |
| POST | `/api/auth/login` | Login user | No |
| POST | `/api/auth/refresh` | Refresh access token | No |
| POST | `/api/auth/logout` | Logout user | Yes |
| GET | `/api/auth/me` | Get current user | Yes |

### Collection Management Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/v2/collections` | Create new collection | Yes |
| GET | `/api/v2/collections` | List all collections | Yes |
| GET | `/api/v2/collections/{id}` | Get collection details | Yes |
| PUT | `/api/v2/collections/{id}` | Update collection | Yes |
| DELETE | `/api/v2/collections/{id}` | Delete collection | Yes |
| POST | `/api/v2/collections/{id}/sources` | Add source | Yes |
| DELETE | `/api/v2/collections/{id}/sources` | Remove source | Yes |
| POST | `/api/v2/collections/{id}/reindex` | Reindex collection | Yes |
| GET | `/api/v2/collections/{id}/operations` | List operations | Yes |
| GET | `/api/v2/collections/{id}/documents` | List documents | Yes |

### Operation Management Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/api/v2/operations` | List all operations | Yes |
| GET | `/api/v2/operations/{id}` | Get operation details | Yes |
| DELETE | `/api/v2/operations/{id}` | Cancel operation | Yes |

### Search Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/v2/search` | Multi-collection search | Yes |

### File Management Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/scan-directory` | Scan directory | Yes |

### System Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/api/metrics` | Get metrics | Yes |
| GET | `/api/models` | List models | Yes |
| GET | `/api/model-info` | Get model info | Yes |
| GET | `/api/settings` | Get settings | Yes |
| PUT | `/api/settings` | Update settings | Yes |

### WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `/api/v2/operations/{operation_id}/ws` | Operation progress updates |
| `/ws/scan/{scan_id}` | Directory scan progress |

## Request Flow

### Search Request Flow

1. Frontend sends search request to `/api/v2/search` with collection UUIDs
2. WebUI validates authentication via JWT
3. WebUI verifies user access to specified collections
4. WebUI proxies request to Semantik search API with collection metadata
5. Semantik performs multi-collection search across Qdrant
6. Results are normalized and optionally reranked
7. WebUI enriches results with collection context
8. Frontend displays search results with collection indicators

### Collection Creation Flow

1. User configures collection (name, model, quantization)
2. Frontend sends POST to `/api/v2/collections`
3. WebUI creates collection record in PostgreSQL
4. WebUI creates Qdrant collection with deterministic naming
5. Collection marked as 'ready' status
6. User adds source via `/api/v2/collections/{id}/sources`
7. WebUI creates operation record and queues to Celery
8. Celery worker processes operation:
   - Scans source for supported documents
   - Creates document records with content hashing
   - Chunks documents and generates embeddings
   - Stores vectors in Qdrant
9. WebSocket updates progress in real-time
10. Operation marked complete or failed

## Error Handling

The WebUI implements comprehensive error handling:

1. **HTTP Exceptions**: Proper status codes and error messages
2. **Database Errors**: Transaction rollback and cleanup
3. **Qdrant Errors**: Retry logic with exponential backoff
4. **WebSocket Errors**: Graceful disconnection handling
5. **Authentication Errors**: Clear error messages for login failures

## Security Considerations

1. **Input Validation**: All user inputs validated via Pydantic
2. **Path Traversal Prevention**: File paths validated
3. **SQL Injection Prevention**: Parameterized queries
4. **JWT Security**: Secret key configuration required
5. **CORS Configuration**: Configurable allowed origins
6. **Rate Limiting**: Prevents API abuse

## Performance Optimizations

1. **Connection Pooling**: Reused Qdrant connections
2. **Async Processing**: Non-blocking I/O operations
3. **Batch Processing**: Files processed in configurable batches
4. **Memory Management**: Automatic cleanup and GC
5. **Adaptive Batch Sizing**: Prevents OOM errors
6. **Background Tasks**: Long operations don't block UI

## Monitoring and Metrics

The WebUI integrates with Prometheus for monitoring:

- Resource usage (CPU, memory, GPU)
- Job processing metrics
- Embedding generation performance
- API request counts and latencies
- Error rates and types

Metrics are exposed at `/api/metrics` when authentication is provided.