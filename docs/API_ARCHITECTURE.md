# API Architecture

Two services: **Vecpipe** (search engine, port 8000) and **WebUI** (control plane, port 8080).

Note: In Docker, vecpipe is internal-only by default. Use `http://vecpipe:8000` from inside the Docker network, or
publish port 8000 if you need host access.

## Services

**Vecpipe** - Pure search engine
- Vector similarity and hybrid search
- Stateless, uses Qdrant for storage
- No database access
- Prometheus metrics on 9091

**WebUI** - Control plane
- REST API v2 + WebSockets
- Auth, collection management, search proxy
- Owns PostgreSQL database
- Metrics on 9092

## Design Principles

- Three-package architecture: vecpipe, webui, shared
- No circular dependencies
- RESTful with standard HTTP codes
- JWT auth for WebUI
- Collection-centric (not job-based)
- Async operations with WebSocket progress

## Vecpipe Search API

Core search engine. Requires the internal API key for protected endpoints; no user management.

**Base URL**: `http://vecpipe:8000` (internal Docker network)

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

## WebUI API v2

User-facing API for auth, collections, and search. Collection-centric architecture.

**Base URL**: `http://localhost:8080`

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

This endpoint starts an `append` operation and (re)uses a `collection_sources` record under the hood. Use the “Manage Sources” endpoints below to update sync settings and store encrypted credentials.

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

**Supported `source_type` values (built-ins):**
- `directory` (local filesystem directory)
- `git` (remote Git repository)
- `imap` (IMAP mailbox)

**Example (Git repo, public or auth configured separately):**
```json
{
  "source_type": "git",
  "source_config": {
    "repo_url": "https://github.com/org/repo.git",
    "ref": "main",
    "auth_method": "none",
    "include_globs": ["docs/**", "*.md"],
    "exclude_globs": ["node_modules/**"]
  }
}
```

**Example (IMAP mailbox; credentials configured separately):**
```json
{
  "source_type": "imap",
  "source_config": {
    "host": "imap.gmail.com",
    "port": 993,
    "use_ssl": true,
    "username": "user@example.com",
    "mailboxes": ["INBOX"]
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

**Connector credentials (secrets):**
- Passwords/tokens/SSH keys are encrypted and stored in the database (never returned in API responses).
- Set `CONNECTOR_SECRETS_KEY` in your environment (see `.env.docker.example` and `docs/CONFIGURATION.md`).
- After the source exists, update secrets via `PATCH /api/v2/collections/{collection_id}/sources/{source_id}` and then trigger a run via `POST /api/v2/collections/{collection_id}/sources/{source_id}/run`.

#### Manage Sources (recommended for scheduling + secrets)

List sources (to get `source_id` for updates/runs):
```http
GET /api/v2/collections/{collection_id}/sources?offset=0&limit=50
Authorization: Bearer {access_token}
```

Update a source’s sync settings and/or encrypted secrets:
```http
PATCH /api/v2/collections/{collection_id}/sources/{source_id}
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "sync_mode": "continuous",
  "interval_minutes": 60,
  "secrets": {
    "password": "app-password"
  }
}
```

Trigger a run immediately (creates an `append` operation for that source):
```http
POST /api/v2/collections/{collection_id}/sources/{source_id}/run
Authorization: Bearer {access_token}
```

Pause/resume continuous sync:
```http
POST /api/v2/collections/{collection_id}/sources/{source_id}/pause
POST /api/v2/collections/{collection_id}/sources/{source_id}/resume
Authorization: Bearer {access_token}
```

#### 7. Remove Source from Collection
**Preferred:** delete by `source_id` (removes documents/vectors, then deletes the source record):
```http
DELETE /api/v2/collections/{collection_id}/sources/{source_id}
Authorization: Bearer {access_token}
```

**Response (200):**
```json
{
  "id": 123,
  "uuid": "op_789e0123-e89b-12d3-a456-426614174002",
  "type": "remove_source",
  "status": "pending"
}
```

**Legacy (still supported):** delete by `source_path`:
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
  "hybrid_mode": "weighted",
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
- For non-file sources (e.g., Git/IMAP), content may be served from database-backed document artifacts.

**Note:** Document metadata is included when listing documents through `/api/v2/collections/{collection_uuid}/documents`. There is no separate metadata endpoint in the v2 API.

### Connector Catalog Endpoints

These endpoints expose the connector catalog used by the UI and provide lightweight “preview” checks for external connectors.

#### 1. List Connectors
```http
GET /api/v2/connectors
Authorization: Bearer {access_token}
```

#### 2. Get Connector Definition
```http
GET /api/v2/connectors/{connector_type}
Authorization: Bearer {access_token}
```

#### 3. Preview Git Connection
```http
POST /api/v2/connectors/preview/git
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "repo_url": "https://github.com/org/repo.git",
  "ref": "main",
  "auth_method": "https_token",
  "token": "ghp_..."
}
```

#### 4. Preview IMAP Connection
```http
POST /api/v2/connectors/preview/imap
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "host": "imap.gmail.com",
  "port": 993,
  "use_ssl": true,
  "username": "user@example.com",
  "password": "app-password"
}
```

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

**Success**: `data` for single items, `results` for lists, optional `metadata` for pagination.

**Errors**: RFC 7807 Problem Details with `detail`, `status`, `title`, `type`, `instance`.

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

## Authentication

JWT tokens: access (30min, HS256) + refresh (30 days, hashed in DB).

Flow: Register → Login → Access (with token) → Refresh → Logout

**Protected**: Everything except `/`, `/login`, `/api/auth/register`, `/api/auth/login`, `/internal/api/*`

**Header**: `Authorization: Bearer <token>`

## Service Integration

Client → WebUI (auth) → Vecpipe (search) → Qdrant

WebUI owns PostgreSQL. Both services use shared package.

**Retry**: 3 attempts, exponential backoff (1s, 2s, 4s), 30s timeout

## Batch Operations

**Batch search**: Parallel embeddings + parallel Qdrant queries = fast multi-query searches.

**Document processing**: Process chunks in batches, upload to Qdrant 100 at a time, GC between files.

## Testing

Test suite: `apps/webui-react/tests/api_test_suite.py`

Categories: Health, auth, collections, operations, search, WebSocket, errors

Run: `python api_test_suite.py --base-url http://localhost:8080 --auth-token <token>`

## WebSocket Endpoints

WebSockets are mounted at the app level and authenticate via `?token=<jwt_token>` (see `packages/webui/main.py`).

- **Global operations stream**: `ws://localhost:8080/ws/operations?token={jwt_token}`
- **Operation progress**: `ws://localhost:8080/ws/operations/{operation_id}?token={jwt_token}`
- **Directory scan progress**: `ws://localhost:8080/ws/directory-scan/{scan_id}?token={jwt_token}`

Operations are started via REST and emit task‑specific update events. Directory scans are started with `POST /api/v2/directory-scan/preview`.

Full message schemas and event types are documented in `docs/WEBSOCKET_API.md`.

## Error Handling

**Rate limits** (slowapi): Login 5/min, Search 30/min, Docs 10/min, General 100/min

**Validation**: Pydantic errors with field locations and messages (422)

**Service errors**: 503 when downstream fails, 401 for auth failures

## Performance

**Caching**: Models (5min), documents (1hr), search results (client-side)

**Timeouts**: Search 30s, embeddings 5min, WebSocket no timeout, extraction 5min/file

**Limits**: 100 results/query, 500MB file size, 50k token chunks, 100 point batches

**Metrics** (Prometheus): Search API on 9091, WebUI on 9092

## Versioning

WebUI: `/api/v2/` prefix. Search API: No versioning (stable).

## Docs

- Search API: `http://localhost:8000/docs`
- WebUI: `http://localhost:8080/docs`
