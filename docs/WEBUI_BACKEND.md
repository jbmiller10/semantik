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

- **User Interface**: React-based frontend for managing embedding jobs and searching
- **Job Management**: Create, monitor, and manage document embedding jobs
- **Search Proxy**: Routes search requests to the Semantik search API
- **Authentication**: JWT-based user authentication and authorization
- **Real-time Updates**: WebSocket connections for job progress monitoring

### Key Architectural Principles

1. **Separation of Concerns**: WebUI acts as a control plane, never implementing core search or embedding logic
2. **Proxy Pattern**: All search functionality proxies to the Semantik search API
3. **Database Independence**: Semantik core engine never accesses the WebUI SQLite database
4. **Scalability**: Designed to handle multiple concurrent jobs and users

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

### api/jobs.py
**Endpoints**: `/api/jobs/*`

Manages embedding job lifecycle:
- `GET /new-id` - Generate job ID for WebSocket connection
- `POST /` - Create new embedding job
- `GET /` - List all jobs
- `GET /{job_id}` - Get job details
- `POST /{job_id}/cancel` - Cancel running job
- `DELETE /{job_id}` - Delete job and collection
- `GET /collections-status` - Check Qdrant collection status
- `GET /{job_id}/collection-exists` - Verify specific collection

### api/search.py
**Endpoints**: `/api/*`

Proxies search requests to Semantik:
- `POST /search` - Unified search (vector/hybrid)
- `POST /hybrid_search` - Legacy hybrid search endpoint

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

### database.py

Centralized SQLite database management with the following tables:

#### Jobs Table
```sql
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    directory_path TEXT NOT NULL,
    model_name TEXT NOT NULL,
    chunk_size INTEGER,
    chunk_overlap INTEGER,
    batch_size INTEGER,
    vector_dim INTEGER,
    quantization TEXT,
    instruction TEXT,
    total_files INTEGER DEFAULT 0,
    processed_files INTEGER DEFAULT 0,
    failed_files INTEGER DEFAULT 0,
    current_file TEXT,
    start_time TEXT,
    error TEXT
)
```

#### Files Table
```sql
CREATE TABLE files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    path TEXT NOT NULL,
    size INTEGER NOT NULL,
    modified TEXT NOT NULL,
    extension TEXT NOT NULL,
    hash TEXT,
    doc_id TEXT,
    status TEXT DEFAULT 'pending',
    error TEXT,
    chunks_created INTEGER DEFAULT 0,
    vectors_created INTEGER DEFAULT 0,
    FOREIGN KEY (job_id) REFERENCES jobs(id)
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

### Key Functions

- `init_db()`: Initialize database with migration support
- `create_job()`, `update_job()`, `get_job()`: Job CRUD operations
- `add_files_to_job()`, `update_file_status()`: File tracking
- `create_user()`, `get_user()`: User management
- `save_refresh_token()`, `verify_refresh_token()`: Token management

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

1. **Job Progress**: `/ws/{job_id}`
   - Real-time job processing updates
   - File completion notifications
   - Error messages

2. **Directory Scan**: `/ws/scan/{scan_id}`
   - Progress updates during directory scanning
   - File count and current path

### Message Types

Job WebSocket messages:
- `job_started`: Initial job start with total files
- `file_processing`: Current file being processed
- `file_completed`: File successfully processed
- `job_completed`: All files processed
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

### Job Management Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/api/jobs/new-id` | Generate job ID | Yes |
| POST | `/api/jobs` | Create new job | Yes |
| GET | `/api/jobs` | List all jobs | Yes |
| GET | `/api/jobs/{job_id}` | Get job details | Yes |
| POST | `/api/jobs/{job_id}/cancel` | Cancel job | Yes |
| DELETE | `/api/jobs/{job_id}` | Delete job | Yes |
| GET | `/api/jobs/collections-status` | Check collections | Yes |

### Search Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/search` | Unified search | Yes |
| POST | `/api/hybrid_search` | Hybrid search | Yes |

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
| `/ws/{job_id}` | Job progress updates |
| `/ws/scan/{scan_id}` | Directory scan progress |

## Request Flow

### Search Request Flow

1. Frontend sends search request to `/api/search`
2. WebUI validates authentication via JWT
3. WebUI determines collection and model from job_id
4. WebUI proxies request to Semantik search API
5. Semantik processes search and returns results
6. WebUI transforms results for frontend format
7. Frontend displays search results

### Job Creation Flow

1. User selects directory and model configuration
2. Frontend requests new job ID via `/api/jobs/new-id`
3. Frontend establishes WebSocket connection
4. Frontend sends job creation request
5. WebUI scans directory for supported files
6. WebUI creates Qdrant collection
7. WebUI starts async job processing
8. Job processor sends progress via WebSocket
9. Files are processed in batches
10. Embeddings uploaded to Qdrant
11. Job marked complete or failed

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