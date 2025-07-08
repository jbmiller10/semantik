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

**Example Requests:**

1. **Simple GET Search:**
```bash
curl "http://localhost:8000/search?q=machine%20learning%20algorithms&k=5"
```

2. **POST Search with Reranking:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do neural networks learn?",
    "k": 10,
    "search_type": "question",
    "use_reranker": true,
    "include_content": true
  }'
```

3. **Search with Metadata Filters:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "API documentation",
    "k": 20,
    "collection": "technical_docs",
    "filters": {
      "must": [
        {
          "key": "metadata.doc_type",
          "match": {"value": "api"}
        },
        {
          "key": "metadata.version",
          "range": {
            "gte": "2.0",
            "lte": "3.0"
          }
        }
      ]
    }
  }'
```

4. **Code Search with Custom Model:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "async function implementation",
    "k": 15,
    "search_type": "code",
    "model_name": "Qwen/Qwen3-Embedding-4B",
    "quantization": "int8",
    "use_reranker": true,
    "rerank_model": "Qwen/Qwen3-Reranker-4B",
    "rerank_quantization": "int8"
  }'
```

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

**Cross-Encoder Reranking:**

When `use_reranker` is enabled, the system uses a two-stage retrieval process:

1. **Candidate Retrieval**: The system retrieves more candidates than requested (5x the `k` value, bounded between 20-200 results) using vector similarity search.
2. **Reranking**: A cross-encoder model re-scores each candidate by considering the query and document together, providing more accurate relevance scores.

**Automatic Model Selection**: If `rerank_model` is not specified, the system automatically selects an appropriate reranker based on the embedding model:
- `BAAI/bge-*` models → `BAAI/bge-reranker-v2-m3`
- `sentence-transformers/*` models → `cross-encoder/ms-marco-MiniLM-L-12-v2`
- `Qwen/Qwen3-Embedding-*` models → Matching Qwen3-Reranker size (0.6B, 4B, or 8B)
- Default fallback → `Qwen/Qwen3-Reranker-0.6B`

**Quantization**: If `rerank_quantization` is not specified, it defaults to match the embedding model's quantization for consistency.

**Benefits**: Reranking significantly improves search accuracy, especially for:
- Complex queries requiring semantic understanding
- Questions seeking specific answers
- Searches where exact relevance matters more than speed

**Response:**
```json
{
  "query": "original query",
  "results": [
    {
      "path": "/path/to/document.pdf",
      "chunk_id": "chunk_0001",
      "score": 0.95,
      "doc_id": "document_hash",
      "content": "Full text content if include_content=true...",
      "metadata": {}
    }
  ],
  "num_results": 1,
  "search_type": "semantic",
  "model_used": "Qwen/Qwen3-Embedding-0.6B/float16",
  "embedding_time_ms": 12.3,
  "search_time_ms": 45.2,
  "reranking_used": false,
  "reranker_model": null,
  "reranking_time_ms": null
}
```

**Response fields:**
- `query`: The original search query
- `results`: Array of search results
  - `path`: Path to the source document
  - `chunk_id`: Unique identifier for the text chunk
  - `score`: Similarity score (0-1)
  - `doc_id`: Document identifier (optional)
  - `content`: Full text content (only included if `include_content=true` or `use_reranker=true`)
  - `metadata`: Additional metadata (optional)
- `num_results`: Number of results returned
- `search_type`: Type of search performed
- `model_used`: Embedding model used (e.g., "Qwen/Qwen3-Embedding-0.6B/float16")
- `embedding_time_ms`: Time to generate query embedding
- `search_time_ms`: Time to search vector database
- `reranking_used`: Whether cross-encoder reranking was applied
- `reranker_model`: The model used for reranking (e.g., "Qwen/Qwen3-Reranker-0.6B/float16")
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

**Example Request:**
```bash
curl -X POST http://localhost:8000/search/batch \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "What is machine learning?",
      "How to implement async/await in Python?",
      "Database indexing best practices"
    ],
    "k": 5,
    "collection": "knowledge_base",
    "search_type": "question",
    "use_reranker": true
  }'
```

**Response:**
```json
{
  "responses": [
    {
      "query": "query1",
      "results": [
        {
          "path": "/docs/api/overview.md",
          "chunk_id": "chunk_001",
          "score": 0.92,
          "doc_id": "doc_abc123",
          "content": null,
          "metadata": null
        }
      ],
      "num_results": 10,
      "search_type": "semantic",
      "model_used": "BAAI/bge-large-en-v1.5/float16",
      "embedding_time_ms": null,
      "search_time_ms": null,
      "reranking_used": null,
      "reranker_model": null,
      "reranking_time_ms": null
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

Combines vector similarity with keyword matching for improved search results.

**Example Requests:**

1. **Filter Mode with Any Keywords:**
```bash
curl "http://localhost:8000/hybrid_search?q=python%20async%20programming&k=10&mode=filter&keyword_mode=any"
```

2. **Rerank Mode with All Keywords Required:**
```bash
curl "http://localhost:8000/hybrid_search?q=docker%20kubernetes%20deployment&k=20&mode=rerank&keyword_mode=all&score_threshold=0.7"
```

**Example Response:**
```json
{
  "query": "docker kubernetes deployment",
  "results": [
    {
      "path": "/docs/devops/k8s-docker-guide.md",
      "chunk_id": "chunk_k8s_001",
      "score": 0.91,
      "doc_id": "doc_devops_k8s",
      "matched_keywords": ["docker", "kubernetes", "deployment"],
      "keyword_score": 1.0,
      "combined_score": 0.955,
      "content": "Deploying Docker containers to Kubernetes involves...",
      "metadata": {
        "category": "devops",
        "tags": ["docker", "k8s", "containers"]
      }
    }
  ],
  "num_results": 2,
  "keywords_extracted": ["docker", "kubernetes", "deployment"],
  "search_mode": "rerank",
  "candidates_retrieved": 100,
  "candidates_after_keyword_filter": 15
}
```

**Parameters:**
- `q` (required): Search query
- `k` (optional): Number of results (default: 10)
- `mode` (optional): "filter" or "rerank" (default: "filter")
  - `filter`: Use Qdrant filters to match keywords
  - `rerank`: Retrieve candidates and rerank based on keyword matches
- `keyword_mode` (optional): "any" or "all" (default: "any")
  - `any`: Match documents containing any of the keywords
  - `all`: Match documents containing all keywords
- `collection` (optional): Collection to search
- `score_threshold` (optional): Minimum similarity score threshold
- `model_name` (optional): Override embedding model
- `quantization` (optional): Override quantization

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

**Response fields:**
- `matched_keywords`: Keywords from the query found in this result
- `keyword_score`: Score based on keyword matches (0-1)
- `combined_score`: Combined vector and keyword score (when mode="rerank")

#### Keyword Search
```http
GET /keyword_search?q={query}&k={num_results}&mode={mode}
```

Text-only search without vector similarity.

**Example Request:**
```bash
curl "http://localhost:8000/keyword_search?q=python%20decorators%20metaclass&k=10&mode=all"
```

**Parameters:**
- `q` (required): Search query
- `k` (optional): Number of results (default: 10)
- `mode` (optional): "any" or "all" (default: "any")
- `collection` (optional): Collection to search

#### Collection Info
```http
GET /collection/info?name={collection_name}
```

Get information about the default Qdrant collection. Note: The collection name parameter in the path is not currently used.

**Example Request:**
```bash
curl "http://localhost:8000/collection/info?name=work_docs"
```

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

**Example Request:**
```bash
curl http://localhost:8000/models
```

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

**Example Request:**
```bash
curl -X POST http://localhost:8000/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Qwen/Qwen3-Embedding-4B",
    "quantization": "float16"
  }'
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

#### Embedding Info
```http
GET /embedding/info
```

Get current embedding configuration.

**Example Request:**
```bash
curl http://localhost:8000/embedding/info
```

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

Get detailed model manager status including loaded embedding and reranker models.

**Example Request:**
```bash
curl http://localhost:8000/model/status
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

**Note**: Models are automatically unloaded after the configured inactivity timeout (default: 300 seconds) to free GPU memory.

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

#### Authentication Flow Examples

**1. Register New User:**
```bash
curl -X POST http://localhost:8080/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "email": "john@example.com",
    "password": "SecurePassword123!",
    "full_name": "John Doe"
  }'
```

**2. Login and Get Token:**
```bash
curl -X POST http://localhost:8080/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "password": "SecurePassword123!"
  }'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "dGhpcyBpcyBhIHJlZnJlc2ggdG9rZW4gZXhhbXBsZQ==",
  "token_type": "bearer"
}
```

**3. Use Token in Requests:**
```bash
# Save token
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# Use in API calls
curl -X GET http://localhost:8080/api/jobs \
  -H "Authorization: Bearer $TOKEN"
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

**Example Request:**
```bash
curl -X POST http://localhost:8080/api/scan-directory \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/mnt/shared/documents",
    "recursive": true
  }'
```

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

**Example Request:**
```bash
curl -X POST http://localhost:8080/api/jobs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Technical Documentation v2.0",
    "description": "Embedding all technical docs for Q4 2024",
    "directory_path": "/mnt/shared/docs/technical",
    "model_name": "Qwen/Qwen3-Embedding-4B",
    "chunk_size": 800,
    "chunk_overlap": 200,
    "batch_size": 64,
    "vector_dim": 2560,
    "quantization": "float16",
    "instruction": "Represent this technical document for searching:",
    "job_id": "tech_docs_v2_2024q4"
  }'
```

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

**Example Request:**
```bash
curl -X GET http://localhost:8080/api/jobs \
  -H "Authorization: Bearer $TOKEN"
```

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

**Example Request:**
```bash
curl -X GET http://localhost:8080/api/jobs/tech_docs_v2_2024q4 \
  -H "Authorization: Bearer $TOKEN"
```

**Response:**
```json
{
  "id": "tech_docs_v2_2024q4",
  "name": "Technical Documentation v2.0",
  "status": "processing",
  "progress": 67.5,
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:45:00Z",
  "total_files": 156,
  "processed_files": 105,
  "failed_files": 2,
  "model_name": "Qwen/Qwen3-Embedding-4B",
  "directory_path": "/mnt/shared/docs/technical",
  "error_details": [
    {"file": "/docs/corrupted.pdf", "error": "PDF parsing failed"},
    {"file": "/docs/empty.txt", "error": "Empty file"}
  ]
}
```

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

**Example Request:**
```bash
curl -X POST http://localhost:8080/api/search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "kubernetes deployment strategies",
    "top_k": 10,
    "collection": "tech_docs_v2_2024q4",
    "search_type": "hybrid",
    "use_reranker": true,
    "hybrid_alpha": 0.7,
    "hybrid_mode": "rerank",
    "keyword_mode": "any"
  }'
```

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

**Example Request:**
```bash
curl -X POST http://localhost:8080/api/hybrid_search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "async programming patterns",
    "k": 15,
    "job_id": "tech_docs_v2_2024q4",
    "mode": "rerank",
    "keyword_mode": "any",
    "score_threshold": 0.6
  }'
```

#### Available Models (Web UI)
```http
GET /api/models
```

Get list of available embedding models.

**Example Request:**
```bash
curl -X GET http://localhost:8080/api/models \
  -H "Authorization: Bearer $TOKEN"
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
    },
    "Qwen/Qwen3-Embedding-4B": {
      "description": "Large high-quality embedding model",
      "dim": 2560,
      "supports_quantization": true,
      "recommended_quantization": "float16"
    }
  },
  "current_device": "cuda",
  "using_real_embeddings": true
}
```

### Collections Management

#### List Collections
```http
GET /api/collections/
```

Get all unique collection names with aggregated statistics.

**Example Request:**
```bash
curl -X GET http://localhost:8080/api/collections/ \
  -H "Authorization: Bearer $TOKEN"
```

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

**Example Request:**
```bash
curl -X GET http://localhost:8080/api/collections/Technical%20Documentation \
  -H "Authorization: Bearer $TOKEN"
```

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

**Connection Example (JavaScript):**
```javascript
// Connect to job progress WebSocket
const jobId = 'tech_docs_v2_2024q4';
const ws = new WebSocket(`ws://localhost:8080/ws/${jobId}`);

ws.onopen = () => {
    console.log('Connected to job progress');
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    console.log('Progress update:', message);
    
    switch(message.type) {
        case 'job_started':
            console.log(`Job started with ${message.total_files} files`);
            break;
        case 'file_processing':
            console.log(`Processing: ${message.current_file}`);
            break;
        case 'file_completed':
            console.log(`Progress: ${message.processed_files}/${message.total_files}`);
            break;
        case 'job_completed':
            console.log('Job completed successfully!');
            ws.close();
            break;
        case 'error':
            console.error('Job error:', message.message);
            break;
    }
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onclose = (event) => {
    console.log(`Connection closed: ${event.code} - ${event.reason}`);
};
```

**Message Types:**

1. **Job Started:**
```json
{
  "type": "job_started",
  "total_files": 156,
  "job_id": "tech_docs_v2_2024q4",
  "timestamp": "2024-01-15T10:00:05Z"
}
```

2. **File Processing:**
```json
{
  "type": "file_processing",
  "current_file": "/mnt/shared/docs/technical/api/rest-api-guide.pdf",
  "processed_files": 42,
  "total_files": 156,
  "status": "Extracting chunks",
  "file_size_mb": 2.4,
  "timestamp": "2024-01-15T10:15:23Z"
}
```

3. **File Completed:**
```json
{
  "type": "file_completed",
  "processed_files": 43,
  "total_files": 156,
  "chunks_created": 127,
  "processing_time_ms": 3456,
  "timestamp": "2024-01-15T10:15:27Z"
}
```

4. **Progress Update:**
```json
{
  "type": "progress",
  "data": {
    "progress": 27.56,
    "processed_files": 43,
    "total_files": 156,
    "failed_files": 2,
    "total_chunks": 5234,
    "estimated_time_remaining_seconds": 4320,
    "current_file": "architecture-overview.md",
    "processing_rate": "2.5 files/minute"
  },
  "timestamp": "2024-01-15T10:15:30Z"
}
```

5. **Error Message:**
```json
{
  "type": "error",
  "message": "Failed to process file: /docs/corrupted.pdf - PDF parsing error",
  "file": "/docs/corrupted.pdf",
  "error_code": "PDF_PARSE_ERROR",
  "recoverable": true,
  "timestamp": "2024-01-15T10:16:45Z"
}
```

6. **Job Completed:**
```json
{
  "type": "job_completed",
  "message": "Job completed successfully",
  "total_files": 156,
  "processed_files": 154,
  "failed_files": 2,
  "total_chunks": 12567,
  "total_time_seconds": 7625,
  "collection_name": "tech_docs_v2_2024q4",
  "timestamp": "2024-01-15T12:07:30Z"
}
```

**Connection Best Practices:**

1. **Reconnection Logic:**
```javascript
class JobProgressClient {
    constructor(jobId) {
        this.jobId = jobId;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 3000;
        this.connect();
    }

    connect() {
        this.ws = new WebSocket(`ws://localhost:8080/ws/${this.jobId}`);
        this.setupEventHandlers();
    }

    setupEventHandlers() {
        this.ws.onopen = () => {
            console.log('Connected');
            this.reconnectAttempts = 0;
        };

        this.ws.onclose = (event) => {
            if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
                console.log(`Reconnecting in ${this.reconnectDelay}ms...`);
                setTimeout(() => {
                    this.reconnectAttempts++;
                    this.connect();
                }, this.reconnectDelay);
            }
        };

        // ... other handlers
    }
}
```

## Error Responses

All endpoints return standard HTTP status codes:

- `200 OK` - Success
- `201 Created` - Resource created successfully
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
- `507 Insufficient Storage` - Insufficient GPU memory for operation

### Error Response Examples

**1. Validation Error (422):**
```json
{
  "detail": [
    {
      "loc": ["body", "chunk_size"],
      "msg": "ensure this value is greater than or equal to 100",
      "type": "value_error.number.not_ge",
      "ctx": {"limit_value": 100}
    },
    {
      "loc": ["body", "chunk_overlap"],
      "msg": "chunk_overlap must be less than chunk_size",
      "type": "value_error"
    }
  ],
  "status_code": 422
}
```

**2. Authentication Error (401):**
```json
{
  "detail": "Could not validate credentials",
  "status_code": 401,
  "headers": {
    "WWW-Authenticate": "Bearer"
  }
}
```

**3. Resource Not Found (404):**
```json
{
  "detail": "Job with ID 'non_existent_job' not found",
  "status_code": 404,
  "resource_type": "job",
  "resource_id": "non_existent_job"
}
```

**4. Insufficient GPU Memory (507):**
```json
{
  "detail": {
    "error": "insufficient_memory",
    "message": "Not enough GPU memory to load reranker model Qwen/Qwen3-Reranker-8B with quantization float16",
    "required_memory_gb": 15.2,
    "available_memory_gb": 7.8,
    "suggestion": "Try using a smaller model (e.g., Qwen/Qwen3-Reranker-0.6B) or different quantization (e.g., int8)",
    "alternatives": [
      {"model": "Qwen/Qwen3-Reranker-0.6B", "quantization": "float16", "memory_gb": 1.2},
      {"model": "Qwen/Qwen3-Reranker-4B", "quantization": "int8", "memory_gb": 4.0},
      {"model": "cross-encoder/ms-marco-MiniLM-L-12-v2", "quantization": "float32", "memory_gb": 1.5}
    ]
  },
  "status_code": 507
}
```

**5. Service Unavailable (503):**
```json
{
  "detail": {
    "error": "service_unavailable",
    "message": "Search API service is temporarily unavailable",
    "service": "search_api",
    "retry_after_seconds": 30,
    "maintenance_mode": false
  },
  "status_code": 503
}
```

**6. Rate Limit Exceeded (429):**
```json
{
  "detail": "Rate limit exceeded: 30 per 1 minute",
  "status_code": 429,
  "retry_after": 45
}
```

**7. File Processing Error (500):**
```json
{
  "detail": {
    "error": "processing_error",
    "message": "Failed to extract text from PDF file",
    "file": "/docs/corrupted_scan.pdf",
    "error_type": "PDFParsingError",
    "traceback": "File \"extract_chunks.py\", line 234...\nPDFParsingError: Unable to decode PDF structure"
  },
  "status_code": 500
}
```

## Rate Limiting

The WebUI API implements rate limiting to prevent abuse and ensure fair usage. The Search API currently has no rate limits.

### Rate Limit Configuration

| Endpoint Category | Rate Limit | Window | Notes |
|------------------|------------|--------|---------|
| Authentication (`/api/auth/*`) | 5 requests | 1 minute | Prevents brute force attacks |
| Search (`/api/search`, `/api/hybrid_search`) | 30 requests | 1 minute | Per user |
| Job Management (`/api/jobs/*`) | 20 requests | 1 minute | Per user |
| Document Access (`/api/documents/*`) | 10 requests | 1 minute | Prevents abuse |
| General API | 100 requests | 1 minute | Default for other endpoints |

### Rate Limit Headers

All API responses include rate limit information in headers:

```http
X-RateLimit-Limit: 30
X-RateLimit-Remaining: 27
X-RateLimit-Reset: 1705318860
X-RateLimit-Reset-After: 45
```

- `X-RateLimit-Limit`: Maximum requests allowed in the window
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Reset`: Unix timestamp when the window resets
- `X-RateLimit-Reset-After`: Seconds until the window resets

### Rate Limit Exceeded Response

When rate limit is exceeded:

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 45
X-RateLimit-Limit: 30
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1705318860
Content-Type: application/json

{
  "detail": "Rate limit exceeded: 30 per 1 minute",
  "status_code": 429,
  "retry_after": 45
}
```

### Implementing Rate Limit Handling

**Python Example with Retry Logic:**
```python
import time
import requests
from typing import Optional

class RateLimitedClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}
        self.session = requests.Session()
    
    def request_with_retry(
        self, 
        method: str, 
        endpoint: str, 
        max_retries: int = 3,
        **kwargs
    ) -> Optional[requests.Response]:
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(max_retries):
            response = self.session.request(
                method, url, headers=self.headers, **kwargs
            )
            
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                print(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue
            
            # Check remaining requests
            remaining = response.headers.get('X-RateLimit-Remaining')
            if remaining and int(remaining) < 5:
                print(f"Warning: Only {remaining} requests remaining")
            
            return response
        
        raise Exception(f"Max retries ({max_retries}) exceeded")

# Usage
client = RateLimitedClient("http://localhost:8080", "your_token")
response = client.request_with_retry(
    "POST", 
    "/api/search", 
    json={"query": "test", "top_k": 10}
)
```

**JavaScript Example with Exponential Backoff:**
```javascript
class ApiClient {
    constructor(baseUrl, token) {
        this.baseUrl = baseUrl;
        this.token = token;
        this.defaultHeaders = {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        };
    }

    async requestWithRetry(method, endpoint, options = {}, maxRetries = 3) {
        let lastError;
        
        for (let attempt = 0; attempt < maxRetries; attempt++) {
            try {
                const response = await fetch(`${this.baseUrl}${endpoint}`, {
                    method,
                    headers: { ...this.defaultHeaders, ...options.headers },
                    ...options
                });
                
                // Check rate limit headers
                const remaining = response.headers.get('X-RateLimit-Remaining');
                if (remaining && parseInt(remaining) < 5) {
                    console.warn(`Low rate limit: ${remaining} requests remaining`);
                }
                
                if (response.status === 429) {
                    const retryAfter = parseInt(
                        response.headers.get('Retry-After') || '60'
                    );
                    console.log(`Rate limited. Retrying after ${retryAfter}s...`);
                    await this.sleep(retryAfter * 1000);
                    continue;
                }
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                return response;
                
            } catch (error) {
                lastError = error;
                if (attempt < maxRetries - 1) {
                    // Exponential backoff
                    const delay = Math.min(1000 * Math.pow(2, attempt), 10000);
                    console.log(`Retrying in ${delay}ms...`);
                    await this.sleep(delay);
                }
            }
        }
        
        throw lastError;
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Usage
const client = new ApiClient('http://localhost:8080', 'your_token');
const response = await client.requestWithRetry('POST', '/api/search', {
    body: JSON.stringify({ query: 'test', top_k: 10 })
});
```

### Rate Limiting Best Practices

1. **Implement Client-Side Rate Limiting**: Don't wait for 429 errors
2. **Use Batch Endpoints**: Reduce request count with batch operations
3. **Cache Responses**: Store results to avoid repeated requests
4. **Monitor Rate Limit Headers**: Proactively slow down when limits approach
5. **Implement Backoff**: Use exponential backoff for retries
6. **Queue Requests**: Implement a request queue with rate control

## CORS

Both APIs support CORS for browser-based access. The Web UI is configured to allow requests from common development origins.