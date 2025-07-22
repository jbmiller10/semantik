# API Reference

## Overview

Semantik provides two main API services:
- **WebUI API** (Port 8080) - User-facing API with authentication, collection management, and search
- **Search API** (Port 8001) - Core search engine API for vector similarity search

All APIs follow RESTful principles with JSON request/response bodies.

## WebUI API

### Base URL
```
http://localhost:8080
```

### Authentication
All API endpoints except authentication endpoints require JWT authentication. Include the token in the Authorization header:
```
Authorization: Bearer {token}
```

### V2 API Endpoints

The v2 API provides a modern collection-based architecture for document management and search.

#### Authentication Endpoints

##### Register New User
```http
POST /api/auth/register
Content-Type: application/json

{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "SecurePassword123!",
  "full_name": "John Doe"
}
```

**Response (201):**
```json
{
  "id": 1,
  "username": "john_doe",
  "email": "john@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "created_at": "2024-01-15T10:00:00Z"
}
```

##### Login
```http
POST /api/auth/login
Content-Type: application/json

{
  "username": "john_doe",
  "password": "SecurePassword123!"
}
```

**Response (200):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

##### Refresh Token
```http
POST /api/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response (200):** Same as login endpoint

##### Logout
```http
POST /api/auth/logout
Authorization: Bearer {token}
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response (200):**
```json
{
  "message": "Logged out successfully"
}
```

##### Get Current User
```http
GET /api/auth/me
Authorization: Bearer {token}
```

**Response (200):**
```json
{
  "id": 1,
  "username": "john_doe",
  "email": "john@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "created_at": "2024-01-15T10:00:00Z",
  "last_login": "2024-01-15T14:30:00Z"
}
```

#### Collection Management Endpoints

##### Create Collection
```http
POST /api/v2/collections
Authorization: Bearer {token}
Content-Type: application/json

{
  "name": "Technical Documentation",
  "description": "Company technical documentation and guides",
  "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
  "quantization": "float16",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "is_public": false
}
```

**Response (201):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Technical Documentation",
  "description": "Company technical documentation and guides",
  "owner_id": 1,
  "vector_store_name": "coll_550e8400_qwen06b_f16",
  "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
  "quantization": "float16",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "is_public": false,
  "status": "pending",
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:00:00Z",
  "document_count": 0,
  "total_chunks": 0,
  "total_size_bytes": 0
}
```

##### List Collections
```http
GET /api/v2/collections?page=1&per_page=20&search=docs&sort_by=created_at&sort_order=desc
Authorization: Bearer {token}
```

**Query Parameters:**
- `page` (optional): Page number (default: 1)
- `per_page` (optional): Items per page (default: 20, max: 100)
- `search` (optional): Search collections by name
- `sort_by` (optional): Sort field (name, created_at, updated_at, document_count)
- `sort_order` (optional): Sort order (asc, desc)

**Response (200):**
```json
{
  "collections": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "Technical Documentation",
      "description": "Company technical documentation and guides",
      "owner_id": 1,
      "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
      "status": "ready",
      "document_count": 150,
      "total_chunks": 3420,
      "total_size_bytes": 45678900,
      "created_at": "2024-01-15T10:00:00Z",
      "updated_at": "2024-01-15T14:30:00Z"
    }
  ],
  "total": 5,
  "page": 1,
  "per_page": 20,
  "pages": 1
}
```

##### Get Collection Details
```http
GET /api/v2/collections/{collection_id}
Authorization: Bearer {token}
```

**Response (200):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Technical Documentation",
  "description": "Company technical documentation and guides",
  "owner_id": 1,
  "vector_store_name": "coll_550e8400_qwen06b_f16",
  "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
  "quantization": "float16",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "is_public": false,
  "status": "ready",
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T14:30:00Z",
  "document_count": 150,
  "total_chunks": 3420,
  "total_size_bytes": 45678900,
  "sources": [
    {
      "id": 1,
      "source_path": "/docs/technical",
      "source_type": "directory",
      "document_count": 150,
      "size_bytes": 45678900,
      "last_indexed_at": "2024-01-15T14:30:00Z"
    }
  ],
  "recent_operations": [
    {
      "id": 1,
      "uuid": "op_123e4567-e89b-12d3-a456-426614174000",
      "type": "index",
      "status": "completed",
      "created_at": "2024-01-15T10:00:00Z",
      "completed_at": "2024-01-15T10:30:00Z"
    }
  ]
}
```

##### Update Collection
```http
PATCH /api/v2/collections/{collection_id}
Authorization: Bearer {token}
Content-Type: application/json

{
  "name": "Technical Documentation v2",
  "description": "Updated company technical documentation",
  "is_public": true
}
```

**Response (200):** Same structure as get collection details

##### Delete Collection
```http
DELETE /api/v2/collections/{collection_id}
Authorization: Bearer {token}
```

**Response (204):** No content

##### Add Source to Collection
```http
POST /api/v2/collections/{collection_id}/sources
Authorization: Bearer {token}
Content-Type: application/json

{
  "source_type": "directory",
  "source_path": "/docs/api",
  "filters": {
    "extensions": [".md", ".txt", ".pdf"],
    "ignore_patterns": ["**/node_modules/**", "**/.git/**"]
  },
  "config": {
    "recursive": true,
    "follow_symlinks": false
  }
}
```

**Response (202):**
```json
{
  "operation_id": "op_123e4567-e89b-12d3-a456-426614174000",
  "operation_type": "append",
  "status": "pending",
  "message": "Source addition operation started"
}
```

##### List Collection Documents
```http
GET /api/v2/collections/{collection_id}/documents?page=1&per_page=50&status=completed
Authorization: Bearer {token}
```

**Query Parameters:**
- `page` (optional): Page number (default: 1)
- `per_page` (optional): Items per page (default: 50, max: 100)
- `status` (optional): Filter by status (pending, processing, completed, failed)
- `source_id` (optional): Filter by source ID

**Response (200):**
```json
{
  "documents": [
    {
      "id": "doc_123e4567-e89b-12d3-a456-426614174000",
      "collection_id": "550e8400-e29b-41d4-a716-446655440000",
      "file_path": "/docs/api/endpoints.md",
      "file_name": "endpoints.md",
      "file_size": 15420,
      "mime_type": "text/markdown",
      "content_hash": "sha256:abcd...",
      "status": "completed",
      "chunk_count": 28,
      "created_at": "2024-01-15T10:15:00Z",
      "updated_at": "2024-01-15T10:16:00Z"
    }
  ],
  "total": 150,
  "page": 1,
  "per_page": 50,
  "pages": 3
}
```

##### Reindex Collection
```http
POST /api/v2/collections/{collection_id}/reindex
Authorization: Bearer {token}
Content-Type: application/json

{
  "config": {
    "force": false,
    "only_failed": false
  }
}
```

**Response (202):**
```json
{
  "operation_id": "op_456e7890-e89b-12d3-a456-426614174001",
  "operation_type": "reindex",
  "status": "pending",
  "message": "Reindex operation started"
}
```

#### Operation Management Endpoints

##### Get Operation Details
```http
GET /api/v2/operations/{operation_uuid}
Authorization: Bearer {token}
```

**Response (200):**
```json
{
  "id": 1,
  "uuid": "op_123e4567-e89b-12d3-a456-426614174000",
  "collection_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": 1,
  "type": "index",
  "status": "processing",
  "task_id": "celery_task_12345",
  "config": {
    "source_path": "/docs/technical",
    "recursive": true
  },
  "created_at": "2024-01-15T10:00:00Z",
  "started_at": "2024-01-15T10:01:00Z",
  "completed_at": null,
  "error_message": null,
  "progress": {
    "total_files": 150,
    "processed_files": 45,
    "failed_files": 2,
    "current_file": "api/endpoints.md",
    "percentage": 30.0
  }
}
```

##### List Operations
```http
GET /api/v2/operations?collection_id={collection_id}&status=processing&type=index
Authorization: Bearer {token}
```

**Query Parameters:**
- `collection_id` (optional): Filter by collection UUID
- `status` (optional): Filter by status (pending, processing, completed, failed, cancelled)
- `type` (optional): Filter by type (index, append, reindex, remove_source, delete)
- `page` (optional): Page number (default: 1)
- `per_page` (optional): Items per page (default: 20)

**Response (200):**
```json
{
  "operations": [
    {
      "id": 1,
      "uuid": "op_123e4567-e89b-12d3-a456-426614174000",
      "collection_id": "550e8400-e29b-41d4-a716-446655440000",
      "type": "index",
      "status": "processing",
      "created_at": "2024-01-15T10:00:00Z",
      "progress_percentage": 30.0
    }
  ],
  "total": 5,
  "page": 1,
  "per_page": 20,
  "pages": 1
}
```

##### Cancel Operation
```http
POST /api/v2/operations/{operation_uuid}/cancel
Authorization: Bearer {token}
```

**Response (200):**
```json
{
  "message": "Operation cancellation requested",
  "operation_id": "op_123e4567-e89b-12d3-a456-426614174000",
  "status": "cancelled"
}
```

#### Search Endpoints

##### Multi-Collection Search
```http
POST /api/v2/search
Authorization: Bearer {token}
Content-Type: application/json

{
  "collection_uuids": [
    "550e8400-e89b-41d4-a716-446655440000",
    "660f9511-f29c-52e5-b827-557755551111"
  ],
  "query": "How to implement authentication?",
  "k": 20,
  "search_type": "semantic",
  "use_reranker": true,
  "score_threshold": 0.5,
  "metadata_filter": {
    "mime_type": "text/markdown"
  },
  "include_content": true,
  "hybrid_alpha": 0.7,
  "hybrid_mode": "rerank",
  "keyword_mode": "any"
}
```

**Parameters:**
- `collection_uuids` (required): List of collection UUIDs to search (1-10)
- `query` (required): Search query text
- `k` (optional): Number of results (default: 10, max: 100)
- `search_type` (optional): Type of search (semantic, question, code, hybrid)
- `use_reranker` (optional): Enable cross-encoder reranking (default: true)
- `rerank_model` (optional): Override reranker model
- `score_threshold` (optional): Minimum score threshold (0.0-1.0)
- `metadata_filter` (optional): Filter results by metadata
- `include_content` (optional): Include chunk content (default: true)
- `hybrid_alpha` (optional): Weight for hybrid search (0.0-1.0, default: 0.7)
- `hybrid_mode` (optional): Hybrid mode (filter, rerank, default: rerank)
- `keyword_mode` (optional): Keyword matching (any, all, default: any)

**Response (200):**
```json
{
  "query": "How to implement authentication?",
  "results": [
    {
      "document_id": "doc_123e4567-e89b-12d3-a456-426614174000",
      "chunk_id": "chunk_456",
      "score": 0.95,
      "original_score": 0.85,
      "reranked_score": 0.95,
      "text": "To implement authentication, you can use JWT tokens...",
      "metadata": {
        "page": 1,
        "section": "Authentication"
      },
      "file_name": "auth_guide.md",
      "file_path": "/docs/auth_guide.md",
      "collection_id": "550e8400-e89b-41d4-a716-446655440000",
      "collection_name": "Technical Documentation",
      "embedding_model": "Qwen/Qwen3-Embedding-0.6B"
    }
  ],
  "total_results": 15,
  "collections_searched": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "Technical Documentation",
      "embedding_model": "Qwen/Qwen3-Embedding-0.6B"
    }
  ],
  "search_type": "semantic",
  "reranking_used": true,
  "reranker_model": "Qwen/Qwen3-Reranker-0.6B",
  "search_time_ms": 245.5
}
```

#### Document Access Endpoints

##### Get Document Content
```http
GET /api/v2/documents/{document_id}
Authorization: Bearer {token}
Range: bytes=0-1023 (optional)
```

**Response:**
- Binary file content with appropriate Content-Type
- Supports range requests for partial content (HTTP 206)

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

##### Get Document Metadata
```http
GET /api/v2/documents/{document_id}/metadata
Authorization: Bearer {token}
```

**Response (200):**
```json
{
  "id": "doc_123e4567-e89b-12d3-a456-426614174000",
  "collection_id": "550e8400-e29b-41d4-a716-446655440000",
  "file_path": "/docs/api/endpoints.md",
  "file_name": "endpoints.md",
  "file_size": 15420,
  "mime_type": "text/markdown",
  "content_hash": "sha256:abcd...",
  "status": "completed",
  "chunk_count": 28,
  "created_at": "2024-01-15T10:15:00Z",
  "updated_at": "2024-01-15T10:16:00Z",
  "collection_name": "Technical Documentation",
  "owner_id": 1
}
```

#### Directory Scanning Endpoints

##### Scan Directory
```http
POST /api/v2/directory-scan
Authorization: Bearer {token}
Content-Type: application/json

{
  "path": "/docs/technical",
  "recursive": true,
  "follow_symlinks": false,
  "filters": {
    "extensions": [".md", ".txt", ".pdf"],
    "ignore_patterns": ["**/node_modules/**", "**/.git/**"],
    "min_size": 100,
    "max_size": 104857600
  }
}
```

**Response (200):**
```json
{
  "scan_id": "scan_123e4567",
  "path": "/docs/technical",
  "files": [
    {
      "path": "/docs/technical/api/endpoints.md",
      "name": "endpoints.md",
      "size": 15420,
      "mime_type": "text/markdown",
      "modified": "2024-01-15T09:00:00Z"
    }
  ],
  "summary": {
    "total_files": 150,
    "total_size_bytes": 45678900,
    "by_extension": {
      ".md": 120,
      ".txt": 20,
      ".pdf": 10
    }
  },
  "errors": []
}
```

### WebSocket Endpoints

#### Operation Progress
```
WS /api/v2/operations/{operation_uuid}/ws
```

Real-time operation progress updates via WebSocket.

**Connection Example (JavaScript):**
```javascript
const ws = new WebSocket(`ws://localhost:8080/api/v2/operations/${operationId}/ws`);

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    
    switch(message.type) {
        case 'progress':
            console.log(`Progress: ${message.percentage}%`);
            break;
        case 'file_processed':
            console.log(`Processed: ${message.file_path}`);
            break;
        case 'error':
            console.error(`Error: ${message.message}`);
            break;
        case 'completed':
            console.log('Operation completed!');
            ws.close();
            break;
    }
};
```

**Message Types:**

1. **Progress Update:**
```json
{
  "type": "progress",
  "percentage": 45.5,
  "processed_files": 68,
  "total_files": 150,
  "current_file": "api/authentication.md"
}
```

2. **File Processed:**
```json
{
  "type": "file_processed",
  "file_path": "/docs/api/authentication.md",
  "chunks_created": 15,
  "status": "completed"
}
```

3. **Error:**
```json
{
  "type": "error",
  "message": "Failed to process file",
  "file_path": "/docs/corrupted.pdf",
  "error_code": "PARSE_ERROR"
}
```

4. **Completed:**
```json
{
  "type": "completed",
  "total_files": 150,
  "processed_files": 148,
  "failed_files": 2,
  "total_chunks": 3420,
  "duration_seconds": 125.5
}
```

### System Endpoints

#### Health Check
```http
GET /api/health
```

**Response (200):**
```json
{
  "status": "healthy",
  "service": "webui",
  "version": "2.0.0",
  "database": "connected",
  "search_api": "connected",
  "qdrant": "connected"
}
```

#### Metrics
```http
GET /api/metrics
Authorization: Bearer {token}
```

**Response:** Prometheus-formatted metrics

#### Settings

##### Get Database Statistics
```http
GET /api/settings/stats
Authorization: Bearer {token}
```

**Response (200):**
```json
{
  "total_collections": 15,
  "total_documents": 1250,
  "total_operations": 45,
  "total_users": 3,
  "database_size_mb": 125.4,
  "vector_storage_size_mb": 2048.5,
  "oldest_collection": "2024-01-01T00:00:00Z",
  "newest_collection": "2024-01-15T10:30:00Z"
}
```

## Search API

The Search API provides the core vector search functionality.

### Base URL
```
http://localhost:8001
```

### Authentication
The Search API does not require authentication. Authentication is handled by the WebUI layer.

### Endpoints

#### Health Check
```http
GET /
```

**Response:**
```json
{
  "status": "healthy",
  "service": "Document Embedding Search API",
  "version": "2.0.0",
  "embedding_service": {
    "status": "ready",
    "mock_mode": false,
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "quantization": "float16"
  },
  "qdrant": {
    "connected": true,
    "host": "localhost",
    "port": 6333
  }
}
```

#### Search
```http
GET /search?q={query}&k={num_results}&collection={collection_name}
```

```http
POST /search
Content-Type: application/json

{
  "query": "string",
  "k": 10,
  "collection": "string",
  "search_type": "semantic|question|code|hybrid",
  "model_name": "string",
  "quantization": "float32|float16|int8",
  "include_content": true,
  "filters": {},
  "use_reranker": false,
  "rerank_model": "string",
  "rerank_quantization": "float32|float16|int8"
}
```

Performs vector similarity search with optional cross-encoder reranking.

**Parameters:**
- `query` (required): Search query text
- `k` (optional): Number of results to return (default: 10, max: 100)
- `collection` (optional): Collection to search in (default: from config)
- `search_type` (optional): Type of search optimization
- `model_name` (optional): Override default embedding model
- `quantization` (optional): Override default quantization
- `include_content` (optional): Include full text content in results
- `filters` (optional): Filter results by metadata
- `use_reranker` (optional): Enable cross-encoder reranking (default: false)
- `rerank_model` (optional): Override default reranker model
- `rerank_quantization` (optional): Override reranker quantization

**Response:**
```json
{
  "query": "machine learning",
  "results": [
    {
      "path": "/docs/ml_guide.pdf",
      "chunk_id": "chunk_123",
      "score": 0.95,
      "doc_id": "doc_456",
      "content": "Machine learning is...",
      "metadata": {}
    }
  ],
  "num_results": 10,
  "search_type": "semantic",
  "model_used": "Qwen/Qwen3-Embedding-0.6B/float16",
  "embedding_time_ms": 12.3,
  "search_time_ms": 45.2,
  "reranking_used": false,
  "reranker_model": null,
  "reranking_time_ms": null
}
```

#### Batch Search
```http
POST /search/batch
Content-Type: application/json

{
  "queries": ["query1", "query2", "query3"],
  "k": 10,
  "collection": "string",
  "search_type": "semantic",
  "model_name": "string",
  "quantization": "float16"
}
```

Process multiple search queries in a single request.

**Response:**
```json
{
  "responses": [
    {
      "query": "query1",
      "results": [...],
      "num_results": 10,
      "search_type": "semantic",
      "model_used": "Qwen/Qwen3-Embedding-0.6B/float16",
      "embedding_time_ms": 10.3,
      "search_time_ms": 37.8
    }
  ],
  "total_time_ms": 145.7
}
```

#### Hybrid Search
```http
GET /hybrid_search?q={query}&k={num_results}&mode={mode}&keyword_mode={keyword_mode}
```

Combines vector similarity with keyword matching for improved search results.

**Parameters:**
- `q` (required): Search query
- `k` (optional): Number of results (default: 10)
- `mode` (optional): "filter" or "rerank" (default: "filter")
- `keyword_mode` (optional): "any" or "all" (default: "any")
- `collection` (optional): Collection to search
- `score_threshold` (optional): Minimum similarity score threshold

**Response:**
```json
{
  "query": "machine learning algorithms",
  "results": [
    {
      "path": "/docs/ml/intro.pdf",
      "chunk_id": "chunk_567",
      "score": 0.88,
      "doc_id": "doc_xyz789",
      "matched_keywords": ["machine", "learning"],
      "keyword_score": 0.75,
      "combined_score": 0.815,
      "metadata": {}
    }
  ],
  "num_results": 3,
  "keywords_extracted": ["machine", "learning", "algorithms"],
  "search_mode": "filter"
}
```

#### Model Management

##### List Available Models
```http
GET /models
```

**Response:**
```json
{
  "models": [
    {
      "name": "Qwen/Qwen3-Embedding-0.6B",
      "dimensions": 1024,
      "description": "Lightweight Chinese-English embedding model"
    },
    {
      "name": "BAAI/bge-large-en-v1.5",
      "dimensions": 1024,
      "description": "High quality general purpose model"
    }
  ],
  "current_model": "Qwen/Qwen3-Embedding-0.6B"
}
```

##### Load Model
```http
POST /models/load
Content-Type: application/json

{
  "model_name": "Qwen/Qwen3-Embedding-4B",
  "quantization": "float16"
}
```

**Success Response (200):**
```json
{
  "message": "Model loaded successfully",
  "model_name": "Qwen/Qwen3-Embedding-4B",
  "quantization": "float16",
  "load_time_seconds": 12.5
}
```

##### Model Status
```http
GET /model/status
```

**Response:**
```json
{
  "embeddings": {
    "Qwen/Qwen3-Embedding-0.6B/float16": {
      "last_used": "2024-01-15T10:30:00Z",
      "device": "cuda",
      "param_count": 635000000,
      "estimated_size_gb": 1.18
    }
  },
  "rerankers": {
    "Qwen/Qwen3-Reranker-0.6B/float16": {
      "last_used": "2024-01-15T10:35:00Z",
      "device": "cuda",
      "param_count": 671000000,
      "estimated_size_gb": 1.25
    }
  },
  "total_models": 2,
  "inactivity_timeout": 300
}
```

## Error Responses

All endpoints return standard HTTP status codes:

- `200 OK` - Success
- `201 Created` - Resource created successfully
- `202 Accepted` - Async operation started
- `204 No Content` - Success with no response body
- `206 Partial Content` - Partial file content (range requests)
- `400 Bad Request` - Invalid parameters
- `401 Unauthorized` - Missing or invalid authentication
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `409 Conflict` - Resource already exists
- `413 Payload Too Large` - Request body too large
- `415 Unsupported Media Type` - Unsupported file type
- `422 Unprocessable Entity` - Validation error
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error
- `502 Bad Gateway` - Upstream service error
- `503 Service Unavailable` - Service temporarily unavailable
- `507 Insufficient Storage` - Insufficient GPU memory

### Error Response Format

All error responses follow a consistent format:
```json
{
  "detail": "Detailed error message",
  "error_code": "UNIQUE_ERROR_CODE",
  "status_code": 400,
  "context": {
    "field": "additional context"
  }
}
```

### Common Error Examples

**Validation Error (422):**
```json
{
  "detail": "Validation failed",
  "error_code": "VALIDATION_ERROR",
  "status_code": 422,
  "context": {
    "chunk_size": "must be between 100 and 50000",
    "chunk_overlap": "must be less than chunk_size"
  }
}
```

**Authentication Error (401):**
```json
{
  "detail": "Could not validate credentials",
  "error_code": "AUTH_INVALID_CREDENTIALS",
  "status_code": 401
}
```

**Resource Not Found (404):**
```json
{
  "detail": "Collection not found",
  "error_code": "COLLECTION_NOT_FOUND",
  "status_code": 404,
  "context": {
    "collection_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

**Rate Limit Exceeded (429):**
```json
{
  "detail": "Rate limit exceeded",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "status_code": 429,
  "context": {
    "limit": 30,
    "window": "1 minute",
    "retry_after": 45
  }
}
```

## Rate Limiting

The WebUI API implements rate limiting to ensure fair usage:

| Endpoint Category | Rate Limit | Window |
|------------------|------------|---------|
| Authentication | 5 requests | 1 minute |
| Search | 30 requests | 1 minute |
| Collection Management | 20 requests | 1 minute |
| Document Access | 10 requests | 1 minute |
| General API | 100 requests | 1 minute |

### Rate Limit Headers

All API responses include rate limit information:

```http
X-RateLimit-Limit: 30
X-RateLimit-Remaining: 27
X-RateLimit-Reset: 1705318860
X-RateLimit-Reset-After: 45
```

## Authentication Flow

### JWT Token Flow

1. **Register**: Create a new user account
2. **Login**: Exchange credentials for access and refresh tokens
3. **Access**: Include access token in Authorization header
4. **Refresh**: Use refresh token to get new access token when expired
5. **Logout**: Revoke refresh token

### Token Details

**Access Token**:
- Algorithm: HS256
- Expiration: 30 minutes
- Claims: `sub` (username), `exp`, `iat`

**Refresh Token**:
- Random secure token
- Expiration: 30 days
- Stored hashed in database

## Best Practices

### API Usage

1. **Batch Operations**: Use batch endpoints when processing multiple items
2. **Pagination**: Always paginate large result sets
3. **Error Handling**: Implement retry logic with exponential backoff
4. **Rate Limiting**: Monitor rate limit headers and implement client-side limiting
5. **WebSocket**: Use WebSocket for real-time progress updates on long operations

### Security

1. **Token Storage**: Store tokens securely (not in localStorage for web apps)
2. **HTTPS**: Always use HTTPS in production
3. **Input Validation**: Validate all inputs client-side before sending
4. **Error Messages**: Don't expose sensitive information in error responses

### Performance

1. **Caching**: Cache search results when appropriate
2. **Compression**: Enable gzip compression for API responses
3. **Connection Pooling**: Reuse HTTP connections
4. **Async Operations**: Use WebSocket for long-running operations

## SDK Support

While no official SDK is provided, the API is designed to work with any HTTP client library:

### Python Example
```python
import httpx

class SemantikClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}
        self.client = httpx.AsyncClient()
    
    async def search(self, collection_ids: list[str], query: str):
        response = await self.client.post(
            f"{self.base_url}/api/v2/search",
            headers=self.headers,
            json={
                "collection_uuids": collection_ids,
                "query": query,
                "k": 20,
                "use_reranker": True
            }
        )
        response.raise_for_status()
        return response.json()
```

### JavaScript Example
```javascript
class SemantikClient {
    constructor(baseUrl, token) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        };
    }
    
    async search(collectionIds, query) {
        const response = await fetch(`${this.baseUrl}/api/v2/search`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                collection_uuids: collectionIds,
                query: query,
                k: 20,
                use_reranker: true
            })
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        return response.json();
    }
}
```

## OpenAPI Documentation

Both services provide interactive OpenAPI documentation:
- WebUI API: `http://localhost:8080/docs`
- Search API: `http://localhost:8001/docs`

The OpenAPI spec can be accessed at:
- WebUI API: `http://localhost:8080/openapi.json`
- Search API: `http://localhost:8001/openapi.json`