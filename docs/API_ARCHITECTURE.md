# Semantik API Architecture Documentation

## Table of Contents
1. [API Architecture Overview](#api-architecture-overview)
2. [Search API (vecpipe/search_api.py)](#search-api-vecpipesearch_apipy)
3. [WebUI API Endpoints](#webui-api-endpoints)
4. [Request/Response Patterns](#requestresponse-patterns)
5. [Authentication & Authorization](#authentication--authorization)
6. [API Integration Patterns](#api-integration-patterns)
7. [Batch Operations](#batch-operations)
8. [API Testing](#api-testing)
9. [WebSocket Endpoints](#websocket-endpoints)
10. [Error Handling](#error-handling)

## API Architecture Overview

Semantik follows a clean three-package architecture with two main services:

1. **Vecpipe Service** (`vecpipe/search_api.py`) - Core search engine
   - Port: 8001 (default)
   - Pure REST API for vector similarity and hybrid search
   - Stateless service with Qdrant backend
   - Uses shared package for embeddings and text processing
   - No direct database access (uses webui API when needed)
   - Prometheus metrics on port 9091

2. **WebUI Service** (`webui/main.py`) - Control plane and user interface
   - Port: 8000 (default)
   - REST API + WebSocket support
   - User authentication and job management
   - Owns and manages the PostgreSQL database
   - Proxies search requests to Vecpipe API
   - Uses shared package for embeddings and database operations

### Key Design Principles

- **Clean Architecture**: Three packages - vecpipe (search), webui (control plane), shared (utilities)
- **No Circular Dependencies**: Both services depend on shared, but not on each other
- **Database Ownership**: WebUI exclusively owns the PostgreSQL database
- **RESTful Design**: Standard HTTP methods and status codes
- **Stateless Search**: All search state stored in Qdrant
- **JWT Authentication**: Secure token-based auth for WebUI
- **Real-time Updates**: WebSocket support for job progress
- **Metrics Integration**: Prometheus metrics for monitoring

## Vecpipe Search API (vecpipe/search_api.py)

The Vecpipe service is the core search engine, providing high-performance vector similarity and hybrid search capabilities. It operates independently and has no knowledge of users, authentication, or job management.

### Base URL
```
http://localhost:8001
```

### Endpoints

#### 1. Health Check
```http
GET /
```

**Response:**
```json
{
  "status": "healthy",
  "collection": {
    "name": "work_docs",
    "points_count": 1234,
    "vector_size": 1024
  },
  "embedding_mode": "real",
  "embedding_service": {
    "current_model": "Qwen/Qwen3-Embedding-0.6B",
    "quantization": "float32",
    "device": "cuda",
    "model_info": {...}
  }
}
```

#### 2. Vector Search (GET)
```http
GET /search?q={query}&k={num_results}&collection={collection_name}
```

**Query Parameters:**
- `q` (required): Search query text
- `k` (optional): Number of results (1-100, default: 10)
- `collection` (optional): Collection name (default: "work_docs")
- `search_type` (optional): Type of search (semantic, question, code, hybrid)
- `model_name` (optional): Override embedding model
- `quantization` (optional): Override quantization (float32, float16, int8)

**Example:**
```bash
curl "http://localhost:8001/search?q=machine%20learning&k=5&search_type=semantic"
```

**Response:**
```json
{
  "query": "machine learning",
  "results": [
    {
      "path": "/docs/ml_guide.pdf",
      "chunk_id": "chunk_123",
      "score": 0.89,
      "doc_id": "doc_456",
      "content": "Machine learning is...",
      "metadata": {
        "page_number": 15,
        "chunk_index": 3
      }
    }
  ],
  "num_results": 5,
  "search_type": "semantic",
  "model_used": "Qwen/Qwen3-Embedding-0.6B/float32",
  "embedding_time_ms": 45.2,
  "search_time_ms": 12.8
}
```

#### 3. Vector Search (POST)
```http
POST /search
Content-Type: application/json
```

**Request Body:**
```json
{
  "query": "machine learning algorithms",
  "k": 10,
  "search_type": "semantic",
  "model_name": "Qwen/Qwen3-Embedding-0.6B",
  "quantization": "float32",
  "filters": {
    "must": [
      {
        "key": "doc_type",
        "match": {
          "value": "pdf"
        }
      }
    ]
  },
  "include_content": true,
  "collection": "job_123"
}
```

**Response:** Same as GET endpoint

#### 4. Hybrid Search
```http
GET /hybrid_search?q={query}&k={num_results}&mode={mode}
```

**Query Parameters:**
- `q` (required): Search query
- `k` (optional): Number of results (default: 10)
- `collection` (optional): Collection name
- `mode` (optional): Hybrid mode - "filter" or "rerank" (default: "filter")
- `keyword_mode` (optional): Keyword matching - "any" or "all" (default: "any")
- `score_threshold` (optional): Minimum similarity score
- `model_name` (optional): Override embedding model
- `quantization` (optional): Override quantization

**Example:**
```bash
curl "http://localhost:8001/hybrid_search?q=python%20async%20programming&k=10&mode=rerank&keyword_mode=all"
```

**Response:**
```json
{
  "query": "python async programming",
  "results": [
    {
      "path": "/docs/python_guide.pdf",
      "chunk_id": "chunk_789",
      "score": 0.92,
      "doc_id": "doc_101",
      "matched_keywords": ["python", "async", "programming"],
      "keyword_score": 0.85,
      "combined_score": 0.89,
      "metadata": {...}
    }
  ],
  "num_results": 10,
  "keywords_extracted": ["python", "async", "programming"],
  "search_mode": "rerank"
}
```

#### 5. Batch Search
```http
POST /search/batch
Content-Type: application/json
```

**Request Body:**
```json
{
  "queries": [
    "machine learning",
    "deep learning",
    "neural networks"
  ],
  "k": 5,
  "search_type": "semantic",
  "model_name": "Qwen/Qwen3-Embedding-0.6B",
  "collection": "work_docs"
}
```

**Response:**
```json
{
  "responses": [
    {
      "query": "machine learning",
      "results": [...],
      "num_results": 5,
      "search_type": "semantic",
      "model_used": "Qwen/Qwen3-Embedding-0.6B/float32"
    },
    {
      "query": "deep learning",
      "results": [...],
      "num_results": 5
    }
  ],
  "total_time_ms": 156.3
}
```

#### 6. Keyword Search
```http
GET /keyword_search?q={keywords}&k={num_results}&mode={mode}
```

**Query Parameters:**
- `q` (required): Keywords to search (space-separated)
- `k` (optional): Number of results (default: 10)
- `collection` (optional): Collection name
- `mode` (optional): Keyword matching - "any" or "all" (default: "any")

**Response:** Similar to hybrid search but without vector scores

#### 7. Collection Info
```http
GET /collection/info
```

**Response:**
```json
{
  "name": "work_docs",
  "status": "ready",
  "points_count": 5432,
  "indexed_vectors_count": 5432,
  "vectors_count": 5432,
  "segments_count": 1,
  "config": {
    "params": {
      "vectors": {
        "size": 1024,
        "distance": "Cosine"
      }
    }
  }
}
```

#### 8. Model Management
```http
GET /models
```

**Response:**
```json
{
  "models": [
    {
      "name": "Qwen/Qwen3-Embedding-0.6B",
      "description": "Lightweight Chinese-English embedding model",
      "dimension": 1024,
      "supports_quantization": true,
      "recommended_quantization": "float32",
      "memory_estimate": {
        "float32": "2.4GB",
        "float16": "1.2GB",
        "int8": "0.6GB"
      },
      "is_qwen3": true
    }
  ],
  "current_model": "Qwen/Qwen3-Embedding-0.6B",
  "current_quantization": "float32"
}
```

```http
POST /models/load
Content-Type: application/json
```

**Request:**
```json
{
  "model_name": "Qwen/Qwen3-Embedding-0.6B",
  "quantization": "float16"
}
```

#### 9. Model Status
```http
GET /model/status
```

**Response:**
```json
{
  "loaded_models": {
    "Qwen/Qwen3-Embedding-0.6B/float32": {
      "last_used": "2024-01-15T10:30:00Z",
      "loaded_at": "2024-01-15T09:00:00Z",
      "memory_usage_mb": 2456.8
    }
  },
  "unload_after_seconds": 300,
  "device": "cuda",
  "total_memory_mb": 8192,
  "available_memory_mb": 5736
}
```

#### 10. Embedding Info
```http
GET /embedding/info
```

**Response:**
```json
{
  "mode": "real",
  "available": true,
  "current_model": "Qwen/Qwen3-Embedding-0.6B",
  "quantization": "float32",
  "device": "cuda",
  "default_model": "Qwen/Qwen3-Embedding-0.6B",
  "default_quantization": "float32",
  "model_details": {
    "embedding_dim": 1024,
    "max_seq_length": 8192,
    "memory_usage_mb": 2456.8
  }
}
```

## WebUI API Endpoints

The WebUI service provides user-facing APIs for authentication, job management, and search proxying.

### Base URL
```
http://localhost:8000
```

### Authentication Endpoints

#### 1. Register
```http
POST /api/auth/register
Content-Type: application/json
```

**Request:**
```json
{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "SecurePassword123!",
  "full_name": "John Doe"
}
```

**Response:**
```json
{
  "id": 1,
  "username": "john_doe",
  "email": "john@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "created_at": "2024-01-15T10:00:00Z",
  "last_login": null
}
```

#### 2. Login
```http
POST /api/auth/login
Content-Type: application/json
```

**Request:**
```json
{
  "username": "john_doe",
  "password": "SecurePassword123!"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

#### 3. Refresh Token
```http
POST /api/auth/refresh
Content-Type: application/json
```

**Request:**
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response:** Same as login endpoint

#### 4. Logout
```http
POST /api/auth/logout
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request:**
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response:**
```json
{
  "message": "Logged out successfully"
}
```

#### 5. Get Current User
```http
GET /api/auth/me
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "id": 1,
  "username": "john_doe",
  "email": "john@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "created_at": "2024-01-15T10:00:00Z",
  "last_login": "2024-01-15T10:05:00Z"
}
```

### Job Management Endpoints

#### 1. Create Job
```http
POST /api/jobs
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request:**
```json
{
  "name": "Technical Documentation",
  "description": "Embedding technical docs",
  "directory_path": "/path/to/docs",
  "model_name": "Qwen/Qwen3-Embedding-0.6B",
  "chunk_size": 600,
  "chunk_overlap": 200,
  "batch_size": 96,
  "vector_dim": 1024,
  "quantization": "float32",
  "instruction": "Represent this technical document for searching:",
  "job_id": "optional-pre-generated-id"
}
```

**Response:**
```json
{
  "id": "job_123",
  "name": "Technical Documentation",
  "status": "created",
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:00:00Z",
  "total_files": 42,
  "processed_files": 0,
  "failed_files": 0,
  "model_name": "Qwen/Qwen3-Embedding-0.6B",
  "directory_path": "/path/to/docs",
  "quantization": "float32",
  "batch_size": 96,
  "chunk_size": 600,
  "chunk_overlap": 200
}
```

#### 2. List Jobs
```http
GET /api/jobs
Authorization: Bearer {access_token}
```

**Response:**
```json
[
  {
    "id": "job_123",
    "name": "Technical Documentation",
    "status": "completed",
    "created_at": "2024-01-15T10:00:00Z",
    "updated_at": "2024-01-15T10:30:00Z",
    "total_files": 42,
    "processed_files": 42,
    "failed_files": 0,
    "model_name": "Qwen/Qwen3-Embedding-0.6B",
    "directory_path": "/path/to/docs"
  }
]
```

#### 3. Get Job Details
```http
GET /api/jobs/{job_id}
Authorization: Bearer {access_token}
```

**Response:** Same structure as single job in list

#### 4. Cancel Job
```http
POST /api/jobs/{job_id}/cancel
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "message": "Job cancellation requested"
}
```

#### 5. Delete Job
```http
DELETE /api/jobs/{job_id}
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "message": "Job deleted successfully"
}
```

#### 6. Get New Job ID
```http
GET /api/jobs/new-id
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### 7. Check Collection Exists
```http
GET /api/jobs/{job_id}/collection-exists
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "exists": true,
  "collection_name": "job_123",
  "point_count": 5432
}
```

#### 8. Check All Collections Status
```http
GET /api/jobs/collections-status
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "job_123": {
    "exists": true,
    "point_count": 5432,
    "status": "completed"
  },
  "job_456": {
    "exists": false,
    "point_count": 0,
    "status": "failed"
  }
}
```

### File Operations Endpoints

#### 1. Scan Directory
```http
POST /api/scan-directory
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request:**
```json
{
  "path": "/path/to/documents",
  "recursive": true
}
```

**Response:**
```json
{
  "files": [
    {
      "path": "/path/to/documents/doc1.pdf",
      "size": 1048576,
      "modified": "2024-01-15T09:00:00Z",
      "extension": ".pdf"
    }
  ],
  "count": 42
}
```

### Search Endpoints

#### 1. Unified Search (Proxies to Search API)
```http
POST /api/search
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request:**
```json
{
  "query": "machine learning",
  "collection": "job_123",
  "job_id": "123",
  "top_k": 10,
  "score_threshold": 0.5,
  "search_type": "hybrid",
  "hybrid_alpha": 0.7,
  "hybrid_mode": "rerank",
  "keyword_mode": "any"
}
```

**Response:**
```json
{
  "query": "machine learning",
  "results": [
    {
      "doc_id": "doc_456",
      "chunk_id": "chunk_789",
      "score": 0.89,
      "content": "Machine learning is...",
      "file_path": "/docs/ml_guide.pdf",
      "file_name": "ml_guide.pdf",
      "chunk_index": 3,
      "total_chunks": 15,
      "job_id": "123",
      "matched_keywords": ["machine", "learning"],
      "keyword_score": 0.85,
      "combined_score": 0.87
    }
  ],
  "collection": "job_123",
  "num_results": 10,
  "search_type": "hybrid",
  "keywords_extracted": ["machine", "learning"],
  "search_mode": "rerank"
}
```

#### 2. Hybrid Search (Legacy)
```http
POST /api/hybrid_search
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request:**
```json
{
  "query": "python async",
  "k": 10,
  "job_id": "123",
  "mode": "filter",
  "keyword_mode": "all",
  "score_threshold": 0.5
}
```

**Response:** Similar to Search API hybrid search response

### Document Serving Endpoints

#### 1. Get Document
```http
GET /api/documents/{job_id}/{doc_id}
Authorization: Bearer {access_token}
Range: bytes=0-1023 (optional)
```

**Response:**
- Binary file content with appropriate Content-Type
- Supports range requests for partial content (HTTP 206)
- Special handling for PPTX files (converts to Markdown)

**Response Headers:**
```http
Content-Type: application/pdf
Content-Length: 1048576
Accept-Ranges: bytes
Content-Range: bytes 0-1023/1048576
Content-Disposition: inline; filename="document.pdf"
Cache-Control: private, max-age=3600
ETag: "1705315200-1048576"
Last-Modified: Mon, 15 Jan 2024 10:00:00 GMT
```

#### 2. Get Document Info
```http
GET /api/documents/{job_id}/{doc_id}/info
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "doc_id": "doc_456",
  "filename": "ml_guide.pdf",
  "path": "/docs/ml_guide.pdf",
  "size": 1048576,
  "extension": ".pdf",
  "modified": "2024-01-15T09:00:00Z",
  "supported": true
}
```

#### 3. Get Temporary Image
```http
GET /api/documents/temp-images/{session_id}/{filename}
```

**Note:** Used for serving images extracted from PPTX conversions

### Model Management Endpoints

#### 1. List Available Models
```http
GET /api/models
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "models": {
    "Qwen/Qwen3-Embedding-0.6B": {
      "description": "Lightweight Chinese-English embedding model",
      "dim": 1024,
      "supports_quantization": true,
      "recommended_quantization": "float32"
    }
  },
  "current_device": "cuda",
  "using_real_embeddings": true
}
```

### Metrics Endpoints

#### 1. Get Metrics
```http
GET /api/metrics
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "available": true,
  "metrics_port": 9092,
  "data": "# HELP python_gc_objects_collected_total...\n# TYPE python_gc_objects_collected_total counter\n..."
}
```

### Settings Endpoints

#### 1. Reset Database
```http
POST /api/settings/reset-database
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "status": "success",
  "message": "Database reset successfully"
}
```

#### 2. Get Database Stats
```http
GET /api/settings/stats
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "job_count": 15,
  "file_count": 342,
  "database_size_mb": 12.5,
  "parquet_files_count": 5,
  "parquet_size_mb": 156.8
}
```

### Internal API Endpoints

The WebUI service exposes internal API endpoints for the vecpipe maintenance service. These endpoints do not require authentication and are designed for service-to-service communication.

#### 1. List All Jobs (Internal)
```http
GET /internal/api/jobs
```

**Response:**
```json
{
  "jobs": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "name": "Technical Documentation",
      "status": "completed",
      "created_at": "2024-01-15T10:00:00Z",
      "updated_at": "2024-01-15T10:30:00Z",
      "model_name": "Qwen/Qwen3-Embedding-0.6B",
      "directory_path": "/docs",
      "total_files": 50,
      "processed_files": 50,
      "failed_files": 0,
      "user_id": 1
    }
  ]
}
```

#### 2. Get Job Files (Internal)
```http
GET /internal/api/jobs/{job_id}/files
```

**Response:**
```json
{
  "files": [
    {
      "id": 1,
      "job_id": "123e4567-e89b-12d3-a456-426614174000",
      "path": "/docs/guide.pdf",
      "size": 1048576,
      "modified": "2024-01-15T09:00:00Z",
      "extension": ".pdf",
      "hash": "abc123...",
      "doc_id": "def456",
      "status": "completed",
      "chunks_created": 15,
      "vectors_created": 15
    }
  ]
}
```

#### 3. Delete Collection Data (Internal)
```http
DELETE /internal/api/jobs/{job_id}/collection
```

**Response:**
```json
{
  "status": "success",
  "message": "Collection data deleted",
  "points_deleted": 500
}
```

**Note:** These internal endpoints are intended for use by the vecpipe maintenance service only. They bypass authentication to allow the maintenance service to perform cleanup operations without requiring user credentials.

## Request/Response Patterns

### Standard Response Format

All successful responses follow this pattern:
```json
{
  "data": {...},     // For single items
  "results": [...],  // For lists
  "message": "...",  // Optional success message
  "metadata": {      // Optional metadata
    "total": 100,
    "page": 1,
    "per_page": 20
  }
}
```

### Error Response Format

All error responses follow RFC 7807 Problem Details:
```json
{
  "detail": "Detailed error message",
  "status": 400,
  "title": "Bad Request",
  "type": "about:blank",
  "instance": "/api/jobs/invalid-id"
}
```

### Common HTTP Status Codes

- **200 OK**: Successful GET, PUT
- **201 Created**: Successful POST creating resource
- **204 No Content**: Successful DELETE
- **206 Partial Content**: Range request response
- **400 Bad Request**: Invalid request data
- **401 Unauthorized**: Missing or invalid authentication
- **403 Forbidden**: Authenticated but not authorized
- **404 Not Found**: Resource not found
- **409 Conflict**: Resource already exists
- **413 Payload Too Large**: File too large
- **415 Unsupported Media Type**: Unsupported file type
- **422 Unprocessable Entity**: Validation errors
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server error
- **502 Bad Gateway**: Downstream service error
- **503 Service Unavailable**: Service temporarily down

### Pagination

While not currently implemented, the API structure supports pagination:
```json
{
  "results": [...],
  "pagination": {
    "total": 1000,
    "page": 1,
    "per_page": 20,
    "pages": 50,
    "has_next": true,
    "has_prev": false
  }
}
```

### Filtering

Search endpoints support Qdrant filter syntax:
```json
{
  "filters": {
    "must": [
      {
        "key": "metadata.doc_type",
        "match": {
          "value": "pdf"
        }
      }
    ],
    "should": [
      {
        "key": "metadata.author",
        "match": {
          "value": "John Doe"
        }
      }
    ]
  }
}
```

## Authentication & Authorization

### JWT Token Flow

1. **Registration**: Create user account
2. **Login**: Exchange credentials for tokens
3. **Access**: Include access token in Authorization header
4. **Refresh**: Use refresh token to get new access token
5. **Logout**: Revoke refresh token

### Token Structure

**Access Token (JWT)**:
- Algorithm: HS256
- Expiration: 30 minutes
- Claims:
  ```json
  {
    "sub": "username",
    "exp": 1705318800,
    "iat": 1705317000
  }
  ```

**Refresh Token**:
- Random secure token
- Expiration: 30 days
- Stored hashed in database

### Protected Endpoints

All endpoints except the following require authentication:
- `GET /` (root)
- `GET /login`
- `GET /api/auth/register`
- `GET /api/auth/login`
- `GET /api/documents/temp-images/*` (session-based)

### Authorization Header

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## API Integration Patterns

### WebUI to Vecpipe API Proxy

The WebUI service acts as a control plane and proxy for search requests:

1. **Client** → WebUI: Authenticated request with job context
2. **WebUI** → Vecpipe API: Transform and forward request
3. **Vecpipe** → Qdrant: Execute vector search using shared embedding service
4. **Response flows back** with transformed format

**Key Points**:
- WebUI handles all authentication and authorization
- Vecpipe focuses purely on search functionality
- Both services use the shared package for common operations

### Service Communication

```mermaid
graph TD
    Client[Client] --> WebUI[WebUI Service<br/>Control Plane]
    WebUI --> Vecpipe[Vecpipe Service<br/>Search Engine]
    WebUI --> PostgreSQL[(PostgreSQL DB<br/>Owned by WebUI)]
    Vecpipe --> Qdrant[(Qdrant<br/>Vector DB)]
    
    WebUI -.->|uses| Shared[Shared Package]
    Vecpipe -.->|uses| Shared
    
    style WebUI fill:#f9f,stroke:#333,stroke-width:2px
    style Vecpipe fill:#9ff,stroke:#333,stroke-width:2px
    style Shared fill:#ff9,stroke:#333,stroke-width:2px
```

### Retry Strategy

For inter-service communication:
- Max retries: 3
- Backoff: Exponential (1s, 2s, 4s)
- Timeout: 30 seconds per request

### Collection Metadata Synchronization

When creating a job:
1. WebUI creates job in PostgreSQL via JobRepository
2. WebUI processes documents using shared.text_processing
3. WebUI generates embeddings using shared.embedding
4. WebUI stores vectors in Qdrant
5. Vecpipe reads collection metadata to determine correct model for search

## Batch Operations

### Batch Search

The Search API supports efficient batch searching:

**Advantages**:
- Parallel embedding generation
- Reduced overhead for multiple queries
- Single HTTP request for multiple searches

**Implementation**:
```python
# Parallel embedding generation
embedding_tasks = [
    generate_embedding_async(query, model, quant, instruction) 
    for query in queries
]
embeddings = await asyncio.gather(*embedding_tasks)

# Parallel search execution
search_tasks = [
    search_qdrant(host, port, collection, vector, k)
    for vector in embeddings
]
results = await asyncio.gather(*search_tasks)
```

### Batch Document Processing

Job processing handles documents in batches:
- Chunk batching: Process multiple chunks together
- Upload batching: Upload to Qdrant in batches of 100
- Memory management: Force garbage collection between files

## API Testing

### Test Suite Location

```
apps/webui-react/tests/api_test_suite.py
```

### Test Categories

1. **Health Checks**: API availability
2. **Authentication**: Login, registration, token refresh
3. **Job Management**: Create, status, cancel, delete
4. **Search**: Vector, hybrid, batch search
5. **WebSocket**: Real-time updates
6. **Error Handling**: Invalid requests, edge cases

### Running Tests

```bash
python api_test_suite.py --base-url http://localhost:8000 --auth-token <token>
```

### Example Test Request

```python
async def test_vector_search():
    async with aiohttp.ClientSession() as session:
        payload = {
            "query": "machine learning",
            "top_k": 10,
            "search_type": "vector",
            "job_id": "test_job"
        }
        
        async with session.post(
            f"{base_url}/api/search",
            json=payload,
            headers={"Authorization": f"Bearer {token}"}
        ) as response:
            assert response.status == 200
            data = await response.json()
            assert len(data["results"]) <= 10
```

## WebSocket Endpoints

### Job Progress WebSocket

```
ws://localhost:8000/ws/{job_id}
```

**Connection Flow**:
1. Generate job_id via `/api/jobs/new-id`
2. Connect WebSocket with job_id
3. Create job with same job_id
4. Receive real-time updates

**Message Types**:
```json
// Job started
{
  "type": "job_started",
  "total_files": 42
}

// File processing
{
  "type": "file_processing",
  "current_file": "/path/to/file.pdf",
  "processed_files": 10,
  "total_files": 42,
  "status": "Processing"
}

// File completed
{
  "type": "file_completed",
  "processed_files": 11,
  "total_files": 42
}

// Job completed
{
  "type": "job_completed",
  "message": "Job completed successfully"
}

// Error
{
  "type": "error",
  "message": "Failed to process file: ..."
}
```

### Directory Scan WebSocket

```
ws://localhost:8000/ws/scan/{scan_id}
```

**Client Messages**:
```json
// Start scan
{
  "action": "scan",
  "path": "/path/to/directory",
  "recursive": true
}

// Cancel scan
{
  "action": "cancel"
}
```

**Server Messages**:
```json
// Scan started
{
  "type": "started",
  "path": "/path/to/directory"
}

// Progress update
{
  "type": "progress",
  "scanned": 100,
  "total": 500,
  "current_path": "/path/to/current/file"
}

// Scan completed
{
  "type": "completed",
  "files": [...],
  "count": 42
}

// Error
{
  "type": "error",
  "error": "Permission denied"
}
```

## Error Handling

### Rate Limiting

Using slowapi for rate limiting:
- Login: 5 requests/minute
- Search: 30 requests/minute  
- Document access: 10 requests/minute
- General API: 100 requests/minute

**Rate Limit Response**:
```json
{
  "detail": "Rate limit exceeded: 30 per 1 minute",
  "status": 429
}
```

### Validation Errors

Using Pydantic for request validation:
```json
{
  "detail": [
    {
      "loc": ["body", "chunk_size"],
      "msg": "ensure this value is greater than or equal to 100",
      "type": "value_error.number.not_ge"
    }
  ],
  "status": 422
}
```

### Service Errors

When downstream services fail:
```json
{
  "detail": "Search service unavailable",
  "status": 503
}
```

### Security Errors

For authentication/authorization failures:
```json
{
  "detail": "Could not validate credentials",
  "status": 401
}
```

## Performance Considerations

### Caching

- **Model caching**: Models kept in memory for 5 minutes
- **Document caching**: 1-hour cache headers for documents
- **Image caching**: Temporary images cached for 1 hour

### Timeouts

- **Search requests**: 30 seconds
- **Embedding generation**: 5 minutes per file
- **WebSocket idle**: No timeout (kept alive)
- **File extraction**: 5 minutes per file

### Resource Limits

- **Max search results**: 100 per query
- **Max batch queries**: Unlimited (memory permitting)
- **Max file size**: 500 MB for preview
- **Max chunk size**: 50,000 tokens
- **Upload batch size**: 100 points per batch

### Metrics

Both services expose Prometheus metrics:
- **Search API**: Port 9091
- **WebUI**: Port 9092

Key metrics:
- `search_api_latency_seconds`: Search request latency
- `search_api_requests_total`: Total requests by endpoint
- `embedding_generation_duration_seconds`: Embedding time
- `job_processing_duration_seconds`: Job completion time
- `active_websocket_connections`: Current WebSocket connections

## API Versioning

Currently, the API uses URL-based versioning:
- Search API: `/v1/` prefix planned
- WebUI API: `/api/` prefix (v1 implied)

Future versions will maintain backward compatibility or provide migration guides.

## OpenAPI/Swagger Documentation

Both services provide OpenAPI documentation:
- Search API: `http://localhost:8001/docs`
- WebUI: `http://localhost:8000/docs`

The interactive documentation allows testing endpoints directly in the browser.