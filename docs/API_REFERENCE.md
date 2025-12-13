# API Reference

Two services: **WebUI** (8080) for auth/collections/search, **Search API** (8000) for vector search.

RESTful with JSON.

## WebUI API

**Base**: `http://localhost:8080`

**Auth**: JWT required except for auth endpoints. Header: `Authorization: Bearer {token}`

### V2 API

Collection-based architecture for docs and search.

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
      "source_config": {
        "path": "/docs/technical",
        "recursive": true
      },
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
PUT /api/v2/collections/{collection_uuid}
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
```

This endpoint starts an `append` operation and (re)uses a `collection_sources` record under the hood. Use the “Manage Sources” endpoints below to update sync settings and store encrypted credentials.

**Request (preferred flexible format):**
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
      "ignore_patterns": ["**/node_modules/**", "**/.git/**"]
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
    "mailboxes": ["INBOX"],
    "since_days": 30,
    "max_messages": 1000
  }
}
```

**Request (legacy, still supported):**
```json
{
  "source_path": "/docs/api"
}
```

**Response (202):**
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
      "path": "/docs/api",
      "recursive": true,
      "follow_symlinks": false
    },
    "source_path": "/docs/api",
    "additional_config": {
      "filters": {
        "extensions": [".md", ".txt", ".pdf"],
        "ignore_patterns": ["**/node_modules/**", "**/.git/**"]
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

##### Manage Sources (recommended for scheduling + secrets)

List sources (to get `source_id` for updates/runs):
```http
GET /api/v2/collections/{collection_id}/sources?offset=0&limit=50
Authorization: Bearer {token}
```

Update a source’s sync settings and/or encrypted secrets:
```http
PATCH /api/v2/collections/{collection_id}/sources/{source_id}
Authorization: Bearer {token}
Content-Type: application/json

{
  "sync_mode": "continuous",
  "interval_minutes": 60,
  "secrets": {
    "token": "ghp_...",
    "ssh_key": "",
    "ssh_passphrase": ""
  }
}
```

Trigger a run immediately (creates an `append` operation for that source):
```http
POST /api/v2/collections/{collection_id}/sources/{source_id}/run
Authorization: Bearer {token}
```

Pause/resume continuous sync:
```http
POST /api/v2/collections/{collection_id}/sources/{source_id}/pause
POST /api/v2/collections/{collection_id}/sources/{source_id}/resume
Authorization: Bearer {token}
```
##### Remove Source from Collection
**Preferred:** delete by `source_id` (removes documents/vectors, then deletes the source record):
```http
DELETE /api/v2/collections/{collection_id}/sources/{source_id}
Authorization: Bearer {token}
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
Authorization: Bearer {token}
```

**Response (202):**
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

##### List Collection Documents
```http
GET /api/v2/collections/{collection_uuid}/documents?page=1&per_page=50&status=completed
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
POST /api/v2/collections/{collection_uuid}/reindex
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
  "id": "op_456e7890-e89b-12d3-a456-426614174001",
  "collection_id": "550e8400-e29b-41d4-a716-446655440000",
  "type": "reindex",
  "status": "pending",
  "config": {},
  "created_at": "2024-01-15T12:00:00Z",
  "started_at": null,
  "completed_at": null,
  "error_message": null
}
```

##### List Collection Operations
```http
GET /api/v2/collections/{collection_uuid}/operations?status=processing&type=index&page=1&per_page=50
Authorization: Bearer {token}
```

**Query Parameters:**
- `status` (optional): Filter by status (pending, processing, completed, failed, cancelled)
- `type` (optional): Filter by operation type (index, append, reindex, remove_source, delete)
- `page` (optional): Page number (default: 1)
- `per_page` (optional): Items per page (default: 50, max: 100)

**Response (200):**
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
GET /api/v2/operations?status=processing,pending&operation_type=index&page=1&per_page=50
Authorization: Bearer {token}
```

**Query Parameters:**
- `status` (optional): Filter by status - comma-separated for multiple values (pending, processing, completed, failed, cancelled)
- `operation_type` (optional): Filter by type (index, append, reindex, remove_source, delete)
- `page` (optional): Page number (default: 1)
- `per_page` (optional): Items per page (default: 50, max: 100)

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
DELETE /api/v2/operations/{operation_uuid}
Authorization: Bearer {token}
```

**Response (200):**
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
  "hybrid_mode": "weighted",
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
- `hybrid_mode` (optional): Hybrid mode (`filter`, `weighted`, default: `weighted`)
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

##### Single Collection Search
```http
POST /api/v2/search/single
Authorization: Bearer {token}
Content-Type: application/json

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

**Parameters:**
- `collection_id` (required): Collection UUID to search
- `query` (required): Search query text
- `k` (optional): Number of results (default: 10, max: 100)
- `search_type` (optional): Type of search (semantic, question, code, hybrid)
- `use_reranker` (optional): Enable cross-encoder reranking (default: false)
- `score_threshold` (optional): Minimum score threshold (0.0-1.0)
- `metadata_filter` (optional): Filter results by metadata
- `include_content` (optional): Include chunk content (default: true)

**Response (200):** Same format as multi-collection search

**Note:** This endpoint is optimized for single collection searches and has higher rate limits than the multi-collection endpoint.

#### Chunking API

Multiple strategies, real-time processing, quality analysis. See [CHUNKING_API.md](/docs/api/CHUNKING_API.md) for full docs.

**Key endpoints**: strategies, preview (10/min), compare (5/min), collection processing, metrics, quality scores, configs

**Progress**: WebSocket at `ws://localhost:8080/ws/operations/{operation_id}?token=<jwt_token>`

#### Document Access Endpoints

##### Get Document Content
```http
GET /api/v2/collections/{collection_uuid}/documents/{document_uuid}/content
Authorization: Bearer {token}
```

**Response:**
- Binary file content with appropriate Content-Type
- Does not support range requests in the current implementation
- Enforces strict access control - user must have access to the collection
- Document must belong to the specified collection
- For non-file sources (e.g., Git/IMAP), content may be served from database-backed document artifacts.

**Response Headers:**
```http
Content-Type: application/pdf
Content-Disposition: inline; filename="document.pdf"
Cache-Control: private, max-age=3600
```

**Error Responses:**
- `404 Not Found`: Document not found or file not on disk
- `403 Forbidden`: Document doesn't belong to collection or access denied
- `500 Internal Server Error`: File access error

**Note:** Document metadata is included when listing documents through `/api/v2/collections/{collection_uuid}/documents`. There is no separate metadata endpoint in the v2 API.

#### Connector Catalog Endpoints

These endpoints expose the connector catalog used by the UI and provide lightweight “preview” checks for external connectors.

##### List Connectors
```http
GET /api/v2/connectors
Authorization: Bearer {token}
```

##### Get Connector Definition
```http
GET /api/v2/connectors/{connector_type}
Authorization: Bearer {token}
```

##### Preview Git Connection
```http
POST /api/v2/connectors/preview/git
Authorization: Bearer {token}
Content-Type: application/json

{
  "repo_url": "https://github.com/org/repo.git",
  "ref": "main",
  "auth_method": "https_token",
  "token": "ghp_...",
  "include_globs": ["docs/**"],
  "exclude_globs": ["node_modules/**"]
}
```

##### Preview IMAP Connection
```http
POST /api/v2/connectors/preview/imap
Authorization: Bearer {token}
Content-Type: application/json

{
  "host": "imap.gmail.com",
  "port": 993,
  "use_ssl": true,
  "username": "user@example.com",
  "password": "app-password"
}
```

#### Directory Scanning Endpoints

##### Scan Directory
```http
POST /api/v2/directory-scan/preview
Authorization: Bearer {token}
Content-Type: application/json

{
  "path": "/docs/technical",
  "scan_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "recursive": true,
  "include_patterns": ["*.md", "*.txt", "*.pdf"],
  "exclude_patterns": ["**/node_modules/**", "**/.git/**"]
}
```

**Response (200):**
```json
{
  "scan_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "path": "/docs/technical",
  "files": [
    {
      "file_path": "/docs/technical/api/endpoints.md",
      "file_name": "endpoints.md",
      "file_size": 15420,
      "mime_type": "text/markdown",
      "content_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "modified_at": "2024-01-15T09:00:00Z"
    }
  ],
  "total_files": 1,
  "total_size": 15420,
  "warnings": []
}
```

### WebSocket Endpoints

WebSockets are mounted at the app level and authenticate via `?token=<jwt_token>`.

- **Global operations stream**: `ws://localhost:8080/ws/operations?token={jwt_token}`
- **Operation progress**: `ws://localhost:8080/ws/operations/{operation_id}?token={jwt_token}`
- **Directory scan progress**: `ws://localhost:8080/ws/directory-scan/{scan_id}?token={jwt_token}`

Operations are started via REST; directory scans via `POST /api/v2/directory-scan/preview`. See `docs/WEBSOCKET_API.md` for message schemas and event types.

### System Endpoints

#### Health Check
```http
GET /api/health
```

**Response (200):**
```json
{
  "status": "healthy"
}
```

For dependency checks, see `GET /api/health/readyz` and `GET /api/health/search-api`.

#### Metrics
```http
GET /metrics
```

**Response:** Prometheus-formatted metrics

**Note:** Metrics endpoints are typically exposed on separate ports:
- WebUI metrics: Port 9092
- Search API metrics: Port 9091

**Note:** Statistics endpoints are not currently implemented in the v2 API. Use the collection and operation list endpoints with appropriate counting logic for statistics.

## Search API

Core vector search. No auth required (WebUI handles it).

**Base**: `http://localhost:8000`

### Endpoints

#### Health Check
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
    "provider": "local",
    "model_info": { ... },
    "is_mock_mode": false
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

## Errors

Standard HTTP codes: 200 (OK), 201 (created), 202 (async started), 204 (no content), 400 (bad request), 401 (unauthorized), 403 (forbidden), 404 (not found), 409 (conflict), 422 (validation), 429 (rate limit), 500/502/503 (server errors), 507 (GPU memory)

Format: `{"detail": "message", "error_code": "CODE", "status_code": 400, "context": {...}}`

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

## Rate Limits

Auth 5/min, Search 30/min, Collections 20/min, Docs 10/min, General 100/min

Headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`, `X-RateLimit-Reset-After`

## Auth Flow

Register → Login (get tokens) → Access (with header) → Refresh (when expired) → Logout (revoke)

**Access token**: HS256, 30min, claims: `sub`, `exp`, `iat`
**Refresh token**: Random, 30 days, hashed in DB

## Best Practices

**Usage**: Batch ops, paginate, retry with backoff, monitor rate limits, use WebSocket for progress

**Security**: Secure token storage (not localStorage), HTTPS in prod, validate inputs, don't leak info in errors

**Performance**: Cache results, gzip, connection pooling, WebSocket for long ops

## SDK

No official SDK. Use any HTTP client:

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

## OpenAPI Docs

Interactive docs:
- WebUI: `http://localhost:8080/docs`
- Search: `http://localhost:8000/docs`

Specs:
- WebUI: `http://localhost:8080/openapi.json`
- Search: `http://localhost:8000/openapi.json`
