# Semantik API Architecture Documentation

## Table of Contents
1. [API Architecture Overview](#api-architecture-overview)
2. [Search API (vecpipe/search_api.py)](#search-api-vecpipesearch_apipy)
3. [WebUI API v2 Endpoints](#webui-api-v2-endpoints)
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
   - Port: 8000 (default; `SEARCH_API_PORT`)
   - Pure REST API for vector similarity and hybrid search
   - Stateless service with Qdrant backend
   - Uses shared package for embeddings and text processing
   - No direct database access (uses webui API when needed)
   - Prometheus metrics on port 9091

2. **WebUI Service** (`webui/main.py`) - Control plane and user interface
   - Port: 8080 (default)
   - REST API v2 + WebSocket support
   - User authentication and collection management
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
- **Real-time Updates**: WebSocket support for operation progress
- **Metrics Integration**: Prometheus metrics for monitoring
- **Collection-Centric**: All operations are organized around collections, not jobs
- **Async Operations**: Long-running tasks are handled through operations that can be tracked

## Vecpipe Search API (vecpipe/search_api.py)

The Vecpipe service is the core search engine, providing high-performance vector similarity and hybrid search capabilities. It operates independently and has no knowledge of users, authentication, or collection management.

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```http
GET /
```

**Response (example):**
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
    "provider": "local",
    "model_info": { ... },
    "is_mock_mode": false
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
curl "http://localhost:8000/search?q=machine%20learning&k=5&search_type=semantic"
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
  "collection": "coll_550e8400_qwen06b_f16"
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
- `mode` (optional): Hybrid mode - `"filter"` or `"weighted"` (default: `"filter"`)
- `keyword_mode` (optional): Keyword matching - "any" or "all" (default: "any")
- `score_threshold` (optional): Minimum similarity score
- `model_name` (optional): Override embedding model
- `quantization` (optional): Override quantization

**Example:**
```bash
curl "http://localhost:8000/hybrid_search?q=python%20async%20programming&k=10&mode=weighted&keyword_mode=all"
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
  "search_mode": "weighted"
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
  "collection": "coll_550e8400_qwen06b_f16"
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
  "name": "coll_550e8400_qwen06b_f16",
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

## WebUI API v2 Endpoints

The WebUI service provides user-facing APIs for authentication, collection management, and search proxying. The v2 API introduces a collection-centric architecture replacing the legacy job-based system.

### Base URL
```
http://localhost:8080
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

### Collection Management Endpoints

#### 1. Create Collection
```http
POST /api/v2/collections
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request:**
```json
{
  "name": "Technical Documentation",
  "description": "Company technical documentation",
  "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
  "quantization": "float16",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "is_public": false
}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Technical Documentation",
  "description": "Company technical documentation",
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

#### 2. List Collections
```http
GET /api/v2/collections
Authorization: Bearer {access_token}
```

**Query Parameters:**
- `page` (optional): Page number (default: 1)
- `per_page` (optional): Items per page (default: 20)
- `search` (optional): Search by name
- `sort_by` (optional): Sort field
- `sort_order` (optional): Sort order (asc/desc)

**Response:**
```json
{
  "collections": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "Technical Documentation",
      "description": "Company technical documentation",
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

#### 3. Get Collection Details
```http
GET /api/v2/collections/{collection_id}
Authorization: Bearer {access_token}
```

**Response:** Detailed collection information including sources and recent operations

#### 4. Update Collection
```http
PUT /api/v2/collections/{collection_uuid}
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request:**
```json
{
  "name": "Technical Documentation v2",
  "description": "Updated documentation",
  "is_public": true
}
```

#### 5. Delete Collection
```http
DELETE /api/v2/collections/{collection_uuid}
Authorization: Bearer {access_token}
```

**Response:** 204 No Content

#### 6. Add Source to Collection
```http
POST /api/v2/collections/{collection_uuid}/sources
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request:**
```json
{
  "source_type": "directory",
  "source_config": {
    "path": "/docs/api",
    "recursive": true,
    "follow_symlinks": false
  },
  "config": {
    "filters": {
      "extensions": [".md", ".txt", ".pdf"],
      "ignore_patterns": ["**/node_modules/**"]
    }
  }
}
```

**Response:**
```json
{
  "id": "op_123e4567-e89b-12d3-a456-426614174000",
  "collection_id": "550e8400-e29b-41d4-a716-446655440000",
  "type": "append",
  "status": "pending",
  "config": {
    "source_id": 1,
    "source_type": "directory",
    "source_config": {
      "path": "/docs/technical",
      "recursive": true
    },
    "source_path": "/docs/technical",
    "additional_config": {
      "filters": {
        "extensions": [".md", ".txt", ".pdf"],
        "ignore_patterns": ["**/node_modules/**"]
      }
    }
  },
  "created_at": "2024-01-15T10:00:00Z",
  "started_at": null,
  "completed_at": null,
  "error_message": null
}
```

#### 7. Remove Source from Collection
```http
DELETE /api/v2/collections/{collection_uuid}/sources?source_path=/docs/api
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "id": "op_789e0123-e89b-12d3-a456-426614174002",
  "collection_id": "550e8400-e29b-41d4-a716-446655440000",
  "type": "remove_source",
  "status": "pending",
  "config": {
    "source_id": 1,
    "source_path": "/docs/api"
  },
  "created_at": "2024-01-15T11:00:00Z",
  "started_at": null,
  "completed_at": null,
  "error_message": null
}
```

#### 8. List Collection Documents
```http
GET /api/v2/collections/{collection_uuid}/documents
Authorization: Bearer {access_token}
```

**Query Parameters:**
- `page` (optional): Page number
- `per_page` (optional): Items per page
- `status` (optional): Filter by status
- `source_id` (optional): Filter by source

**Response:**
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

#### 9. Reindex Collection
```http
POST /api/v2/collections/{collection_uuid}/reindex
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request:**
```json
{
  "config": {
    "force": false,
    "only_failed": false
  }
}
```

**Response:**
```json
{
  "id": "op_456e7890-e89b-12d3-a456-426614174001",
  "collection_id": "550e8400-e29b-41d4-a716-446655440000",
  "type": "reindex",
  "status": "pending",
  "config": {},
  "created_at": "2024-01-15T10:00:00Z",
  "started_at": null,
  "completed_at": null,
  "error_message": null
}
```

#### 10. List Collection Operations
```http
GET /api/v2/collections/{collection_uuid}/operations?status=processing&type=index&page=1&per_page=50
Authorization: Bearer {access_token}
```

**Query Parameters:**
- `status` (optional): Filter by status (pending, processing, completed, failed, cancelled)
- `type` (optional): Filter by operation type (index, append, reindex, remove_source, delete)
- `page` (optional): Page number (default: 1)
- `per_page` (optional): Items per page (default: 50, max: 100)

**Response:**
```json
[
  {
    "id": "op_123e4567-e89b-12d3-a456-426614174000",
    "collection_id": "550e8400-e29b-41d4-a716-446655440000",
    "type": "index",
    "status": "completed",
    "config": {
      "source_path": "/docs/technical",
      "recursive": true
    },
    "created_at": "2024-01-15T10:00:00Z",
    "started_at": "2024-01-15T10:01:00Z",
    "completed_at": "2024-01-15T10:30:00Z",
    "error_message": null
  }
]
```

### Operation Management Endpoints

Operations represent asynchronous tasks performed on collections (indexing, reindexing, etc.).

#### 1. Get Operation Details
```http
GET /api/v2/operations/{operation_uuid}
Authorization: Bearer {access_token}
```

**Response:**
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

#### 2. List Operations
```http
GET /api/v2/operations
Authorization: Bearer {access_token}
```

**Query Parameters:**
- `status` (optional): Filter by status - comma-separated for multiple values
- `operation_type` (optional): Filter by operation type
- `page` (optional): Page number
- `per_page` (optional): Items per page

#### 3. Cancel Operation
```http
DELETE /api/v2/operations/{operation_uuid}
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "id": "op_123e4567-e89b-12d3-a456-426614174000",
  "collection_id": "550e8400-e29b-41d4-a716-446655440000",
  "type": "index",
  "status": "cancelled",
  "config": {},
  "error_message": "Cancelled by user",
  "created_at": "2024-01-15T10:00:00Z",
  "started_at": "2024-01-15T10:01:00Z",
  "completed_at": "2024-01-15T10:05:00Z"
}
```

### Search Endpoints

#### 1. Multi-Collection Search
```http
POST /api/v2/search
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request:**
```json
{
  "collection_uuids": [
    "550e8400-e29b-41d4-a716-446655440000",
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

**Response:**
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
      "collection_id": "550e8400-e29b-41d4-a716-446655440000",
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

#### 2. Single Collection Search
```http
POST /api/v2/search/single
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request:**
```json
{
  "collection_id": "550e8400-e29b-41d4-a716-446655440000",
  "query": "How to implement authentication?",
  "k": 10,
  "search_type": "semantic",
  "use_reranker": false,
  "score_threshold": 0.7,
  "metadata_filter": {
    "mime_type": "text/markdown"
  },
  "include_content": true
}
```

**Response:** Same format as multi-collection search

**Note:** This endpoint is optimized for single collection searches and has higher rate limits (60/minute vs 30/minute) than the multi-collection endpoint.

### Document Access Endpoints

#### 1. Get Document Content
```http
GET /api/v2/collections/{collection_uuid}/documents/{document_uuid}/content
Authorization: Bearer {access_token}
```

**Response:**
- Binary file content with appropriate Content-Type
- Does not support range requests in the current implementation
- Enforces strict access control - user must have access to the collection
- Document must belong to the specified collection

**Note:** Document metadata is included when listing documents through `/api/v2/collections/{collection_uuid}/documents`. There is no separate metadata endpoint in the v2 API.

### Directory Scanning Endpoints

#### 1. Scan Directory
```http
POST /api/v2/directory-scan
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request:**
```json
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

**Response:**
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

### Internal API Endpoints

The WebUI service exposes internal API endpoints for the vecpipe maintenance service. These endpoints do not require authentication and are designed for service-to-service communication.

#### 1. List All Collections (Internal)
```http
GET /internal/api/collections
```

**Note:** Internal endpoints are for service-to-service communication only.

**Response:**
```json
{
  "collections": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "Technical Documentation",
      "vector_store_name": "coll_550e8400_qwen06b_f16",
      "status": "ready",
      "owner_id": 1,
      "document_count": 150,
      "total_chunks": 3420
    }
  ]
}
```

#### 2. Get Collection Documents (Internal)
```http
GET /internal/api/collections/{collection_id}/documents
```

**Response:**
```json
{
  "documents": [
    {
      "id": "doc_123e4567-e89b-12d3-a456-426614174000",
      "file_path": "/docs/guide.pdf",
      "content_hash": "sha256:abc123...",
      "status": "completed",
      "chunk_count": 15
    }
  ]
}
```

#### 3. Delete Vector Store (Internal)
```http
DELETE /internal/api/collections/{collection_id}/vector-store
```

**Response:**
```json
{
  "status": "success",
  "message": "Vector store deleted",
  "points_deleted": 3420
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
  "instance": "/api/v2/collections/invalid-id"
}
```

### Common HTTP Status Codes

- **200 OK**: Successful GET, PUT, PATCH
- **201 Created**: Successful POST creating resource
- **202 Accepted**: Async operation started
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

API endpoints support pagination:
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
- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /internal/api/*` (internal endpoints)

### Authorization Header

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## API Integration Patterns

### WebUI to Vecpipe API Proxy

The WebUI service acts as a control plane and proxy for search requests:

1. **Client** → WebUI: Authenticated request with collection context
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

When creating a collection:
1. WebUI creates collection in PostgreSQL via CollectionRepository
2. WebUI initializes Qdrant collection with unique name
3. Collection status updated to 'ready'
4. Vecpipe reads collection by vector_store_name for search

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

Operation processing handles documents in batches:
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
3. **Collection Management**: Create, update, delete, list
4. **Operation Management**: Create, monitor, cancel
5. **Search**: Vector, hybrid, batch search
6. **WebSocket**: Real-time updates
7. **Error Handling**: Invalid requests, edge cases

### Running Tests

```bash
python api_test_suite.py --base-url http://localhost:8080 --auth-token <token>
```

### Example Test Request

```python
async def test_multi_collection_search():
    async with aiohttp.ClientSession() as session:
        payload = {
            "collection_uuids": [
                "550e8400-e29b-41d4-a716-446655440000"
            ],
            "query": "machine learning",
            "k": 10,
            "use_reranker": True
        }
        
        async with session.post(
            f"{base_url}/api/v2/search",
            json=payload,
            headers={"Authorization": f"Bearer {token}"}
        ) as response:
            assert response.status == 200
            data = await response.json()
            assert len(data["results"]) <= 10
```

## WebSocket Endpoints

WebSockets are mounted at the app level and authenticate via `?token=<jwt_token>` (see `packages/webui/main.py`).

- **Global operations stream**: `ws://localhost:8080/ws/operations?token={jwt_token}`
- **Operation progress**: `ws://localhost:8080/ws/operations/{operation_id}?token={jwt_token}`
- **Directory scan progress**: `ws://localhost:8080/ws/directory-scan/{scan_id}?token={jwt_token}`

Operations are started via REST and emit task‑specific update events. Directory scans are started with `POST /api/v2/directory-scan/preview`.

Full message schemas and event types are documented in `docs/WEBSOCKET_API.md`.

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
- **Search results**: Client-side caching recommended

### Timeouts

- **Search requests**: 30 seconds
- **Embedding generation**: 5 minutes per operation
- **WebSocket idle**: No timeout (kept alive)
- **Document extraction**: 5 minutes per file

### Resource Limits

- **Max search results**: 100 per query
- **Max batch queries**: Unlimited (memory permitting)
- **Max file size**: 500 MB for processing
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
- `operation_processing_duration_seconds`: Operation completion time
- `active_websocket_connections`: Current WebSocket connections

## API Versioning

Currently, the API uses URL-based versioning:
- WebUI API: `/api/v2/` prefix for new endpoints
- Search API: No versioning (stable interface)

Future versions will maintain backward compatibility or provide migration guides.

## OpenAPI/Swagger Documentation

Both services provide OpenAPI documentation:
- Search API: `http://localhost:8000/docs`
- WebUI: `http://localhost:8080/docs`

The interactive documentation allows testing endpoints directly in the browser.
