# API Reference

## Search API

The Search API provides programmatic access to the vector search functionality.

### Base URL
```
http://localhost:8000
```

### Authentication
The Search API does not require authentication. Authentication is handled by the Web UI layer.

### Endpoints

#### Health Check
```http
GET /
```

Returns detailed API status and configuration information.

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
- `use_reranker` (optional): Enable cross-encoder reranking for improved accuracy (default: false)
- `rerank_model` (optional): Override default reranker model (auto-selected based on embedding model)
- `rerank_quantization` (optional): Override reranker quantization (default: matches embedding quantization)

**Note on Reranking:** When `use_reranker` is enabled, the system automatically retrieves more candidates (5x the requested `k` value, between 20-200) and uses a cross-encoder model to re-score them, returning only the top `k` results. This improves search accuracy at the cost of slightly increased latency.

**Response:**
```json
{
  "results": [
    {
      "id": "doc_chunk_0001",
      "score": 0.95,
      "metadata": {
        "doc_id": "document_hash",
        "path": "/path/to/document.pdf",
        "content": "Full text content..."
      }
    }
  ],
  "query": "original query",
  "k": 10,
  "search_time_ms": 45.2,
  "embedding_time_ms": 12.3,
  "reranking_used": false,
  "reranker_model": null,
  "reranking_time_ms": null
}
```

**Response fields with reranking enabled:**
- `reranking_used`: Whether reranking was applied
- `reranker_model`: The model used for reranking (e.g., "Qwen/Qwen3-Reranker-0.6B")
- `reranking_time_ms`: Time spent on reranking in milliseconds

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
      "model_used": "BAAI/bge-large-en-v1.5/float16",
      "embedding_time_ms": 12.5,
      "search_time_ms": 32.7
    },
    {
      "query": "query2",
      "results": [...],
      "num_results": 10,
      "search_type": "semantic", 
      "model_used": "BAAI/bge-large-en-v1.5/float16",
      "embedding_time_ms": 10.3,
      "search_time_ms": 37.8
    },
    {
      "query": "query3",
      "results": [...],
      "num_results": 10,
      "search_type": "semantic",
      "model_used": "BAAI/bge-large-en-v1.5/float16",
      "embedding_time_ms": 11.2,
      "search_time_ms": 41.2
    }
  ],
  "total_time_ms": 145.7
}
```

#### Hybrid Search
```http
GET /hybrid_search?q={query}&k={num_results}&mode={mode}&keyword_mode={keyword_mode}
```

Combines vector similarity with keyword matching.

**Parameters:**
- `q` (required): Search query
- `k` (optional): Number of results (default: 10)
- `mode` (optional): "filter" or "rerank" (default: "filter")
- `keyword_mode` (optional): "any" or "all" (default: "any")
- `collection` (optional): Collection to search
- `score_threshold` (optional): Minimum score threshold

**Response:**
```json
{
  "results": [...],
  "query": "original query",
  "mode": "filter",
  "keywords": ["extracted", "keywords"],
  "search_time_ms": 52.3
}
```

#### Keyword Search
```http
GET /keyword_search?q={query}&k={num_results}&mode={mode}
```

Text-only search without vector similarity.

**Parameters:**
- `q` (required): Search query
- `k` (optional): Number of results (default: 10)
- `mode` (optional): "any" or "all" (default: "any")
- `collection` (optional): Collection to search

#### Collection Info
```http
GET /collection/info?name={collection_name}
```

Get statistics about a Qdrant collection.

**Response:**
```json
{
  "collection": "work_docs",
  "vectors_count": 15420,
  "points_count": 15420,
  "segments_count": 2,
  "config": {
    "vector_size": 1024,
    "distance": "Cosine"
  }
}
```

#### Available Models
```http
GET /models
```

List available embedding models.

**Response:**
```json
{
  "models": [
    {
      "name": "BAAI/bge-large-en-v1.5",
      "dimensions": 1024,
      "description": "High quality general purpose"
    },
    {
      "name": "Qwen/Qwen3-Embedding-0.6B",
      "dimensions": 1024,
      "description": "Fast, good quality"
    }
  ],
  "current_model": "Qwen/Qwen3-Embedding-0.6B"
}
```

#### Load Model
```http
POST /models/load
Content-Type: application/json

{
  "model_name": "Qwen/Qwen3-Embedding-4B",
  "quantization": "float16"
}
```

Dynamically load a different embedding model.

#### Embedding Info
```http
GET /embedding/info
```

Get current embedding configuration.

**Response:**
```json
{
  "model_name": "Qwen/Qwen3-Embedding-0.6B",
  "quantization": "float16",
  "mock_mode": false,
  "device": "cuda",
  "max_batch_size": 96
}
```

#### Model Status
```http
GET /model/status
```

Get detailed model manager status.

**Response:**
```json
{
  "loaded_models": {
    "Qwen/Qwen3-Embedding-0.6B": {
      "quantization": "float16",
      "device": "cuda",
      "last_used": "2024-01-15T10:30:00Z",
      "memory_usage_mb": 1200
    }
  },
  "total_memory_mb": 1200,
  "device": "cuda"
}
```

## Web UI API

The Web UI provides additional endpoints with authentication.

### Base URL
```
http://localhost:8080
```

### Authentication
All API endpoints require JWT authentication. Include the token in the Authorization header:
```
Authorization: Bearer {token}
```

### Endpoints

#### Scan Directory
```http
POST /api/scan-directory
Content-Type: application/json

{
  "path": "/path/to/documents"
}
```

Scan a directory for documents.

**Response:**
```json
{
  "files": [
    {
      "path": "/path/to/document.pdf",
      "size": 1048576,
      "type": "pdf"
    }
  ],
  "total_files": 42,
  "total_size": 104857600
}
```

#### Create Job
```http
POST /api/jobs
Content-Type: application/json

{
  "name": "Job Name",
  "directories": ["/path/to/docs"],
  "model_name": "Qwen/Qwen3-Embedding-0.6B",
  "quantization": "float16",
  "instruction": "Represent this document for retrieval:"
}
```

Create a new embedding job.

**Response:**
```json
{
  "job_id": "job_abc123",
  "status": "created",
  "created_at": "2024-01-15T10:00:00Z"
}
```

#### List Jobs
```http
GET /api/jobs
```

List all embedding jobs.

**Response:**
```json
{
  "jobs": [
    {
      "id": "job_abc123",
      "name": "Job Name",
      "status": "completed",
      "progress": 100,
      "created_at": "2024-01-15T10:00:00Z",
      "completed_at": "2024-01-15T11:30:00Z"
    }
  ]
}
```

#### Get Job Details
```http
GET /api/jobs/{job_id}
```

Get detailed information about a specific job.

#### Search (Web UI)
```http
POST /api/search
Content-Type: application/json

{
  "query": "search text",
  "top_k": 10,
  "collection": "job_abc123",
  "search_type": "vector|hybrid",
  "use_reranker": false,
  "rerank_model": "string",
  "rerank_quantization": "float32|float16|int8",
  "hybrid_alpha": 0.7,
  "hybrid_mode": "rerank|filter",
  "keyword_mode": "any|all"
}
```

Search within a specific collection. This endpoint proxies to the Search API.

**Parameters:**
- `query` (required): Search query text
- `top_k` (optional): Number of results to return (default: 10)
- `collection` (optional): Collection to search in (e.g., "job_123")
- `search_type` (optional): "vector" or "hybrid" (default: "vector")
- `use_reranker` (optional): Enable cross-encoder reranking (default: false)
- `rerank_model` (optional): Override reranker model selection
- `rerank_quantization` (optional): Override reranker quantization
- `hybrid_alpha` (optional): Weight for hybrid search (0.0-1.0, default: 0.7)
- `hybrid_mode` (optional): "rerank" or "filter" for hybrid search (default: "rerank")
- `keyword_mode` (optional): "any" or "all" for keyword matching (default: "any")

#### Hybrid Search (Web UI)
```http
POST /api/hybrid_search
Content-Type: application/json

{
  "query": "search text",
  "k": 10,
  "job_id": "job_abc123",
  "mode": "filter",
  "keyword_mode": "any"
}
```

Hybrid search within a specific job's embeddings.

#### Available Models (Web UI)
```http
GET /api/models
```

Get list of available embedding models.

### Collections Management

#### List Collections
```http
GET /api/collections/
```

Get all unique collection names with aggregated statistics.

**Response:**
```json
{
  "collections": [
    {
      "name": "Technical Documentation",
      "total_files": 156,
      "total_chunks": 3420,
      "created_at": "2024-01-15T10:30:00Z",
      "root_job_id": 1
    }
  ]
}
```

#### Get Collection Details
```http
GET /api/collections/{collection_name}
```

Get detailed information about a specific collection.

**Response:**
```json
{
  "name": "Technical Documentation",
  "jobs": [
    {
      "id": 1,
      "mode": "create",
      "source_path": "/docs/api",
      "files_found": 45,
      "status": "completed"
    }
  ],
  "total_files": 156,
  "total_chunks": 3420,
  "duplicates_found": 3
}
```

#### Get Collection Files
```http
GET /api/collections/{collection_name}/files?page={page}&per_page={per_page}
```

Get paginated list of files in a collection.

**Parameters:**
- `page` (optional): Page number (default: 1)
- `per_page` (optional): Items per page (default: 50)

#### Rename Collection
```http
PUT /api/collections/{collection_name}/rename
Content-Type: application/json

{
  "new_name": "API Documentation v2"
}
```

Rename a collection.

#### Delete Collection
```http
DELETE /api/collections/{collection_name}
```

Delete a collection and all associated data.

#### Add to Collection
```http
POST /api/jobs/add-to-collection
Content-Type: application/json

{
  "collection_name": "Technical Documentation",
  "source_type": "file|directory|github|url",
  "source_path": "/path/to/new/docs",
  "filters": {
    "extensions": [".md", ".txt"],
    "ignore_patterns": ["**/node_modules/**"]
  }
}
```

Add new data to an existing collection.

#### Preload Model
```http
POST /api/preload_model
Content-Type: application/json

{
  "model_name": "BAAI/bge-large-en-v1.5",
  "quantization": "float16"
}
```

Preload an embedding model to prevent timeout issues during search.

### WebSocket Endpoints

#### Job Progress
```
WS /ws/{job_id}
```

Real-time job progress updates via WebSocket.

**Messages:**
```json
{
  "type": "progress",
  "data": {
    "progress": 45,
    "current_file": "document.pdf",
    "files_processed": 15,
    "total_files": 33
  }
}
```

## Error Responses

All endpoints return standard HTTP status codes:

- `200 OK` - Success
- `400 Bad Request` - Invalid parameters
- `401 Unauthorized` - Missing or invalid authentication
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error
- `507 Insufficient Storage` - Insufficient GPU memory for operation

Error response format:
```json
{
  "detail": "Error message",
  "status_code": 400
}
```

**Insufficient Memory Error (507):**
When GPU memory is insufficient for reranking, the API returns:
```json
{
  "detail": {
    "error": "insufficient_memory",
    "message": "Not enough GPU memory to load reranker model Qwen/Qwen3-Reranker-8B with quantization float16",
    "suggestion": "Try using a smaller model (e.g., Qwen/Qwen3-Reranker-0.6B) or different quantization (e.g., int8)"
  },
  "status_code": 507
}
```

## Rate Limiting

Currently, there are no rate limits implemented. For production use, consider implementing rate limiting based on your requirements.

## CORS

Both APIs support CORS for browser-based access. The Web UI is configured to allow requests from common development origins.