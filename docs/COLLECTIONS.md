# Collections Management System

## Overview

The Collections Management System in Semantik provides a powerful abstraction for organizing and managing document repositories. Collections are the primary unit of organization, allowing users to create isolated document sets with specific embedding models and search configurations.

## Architecture

### Core Concepts

1. **Collection**: A logical grouping of documents with shared configuration (embedding model, chunk settings)
2. **Operation**: An asynchronous task performed on a collection (indexing, reindexing, etc.)
3. **Document**: An individual file within a collection
4. **Source**: A data source (directory, file, URL) that provides documents to a collection

### Database Schema

Collections are implemented using dedicated tables in PostgreSQL:

```sql
-- Collections table - primary organizational unit
collections (
    id VARCHAR PRIMARY KEY,                    -- UUID
    name VARCHAR UNIQUE NOT NULL,              -- User-friendly name
    description TEXT,                          -- Optional description
    owner_id INTEGER NOT NULL,                 -- Owner user ID
    vector_store_name VARCHAR UNIQUE NOT NULL, -- Qdrant collection name
    embedding_model VARCHAR NOT NULL,          -- Model for embeddings
    quantization VARCHAR DEFAULT 'float16',    -- float32|float16|int8
    chunk_size INTEGER DEFAULT 1000,           -- Chunk size in tokens
    chunk_overlap INTEGER DEFAULT 200,         -- Overlap between chunks
    is_public BOOLEAN DEFAULT FALSE,           -- Public visibility
    status VARCHAR,                            -- pending|ready|processing|error|degraded
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
)

-- Operations table - async tasks on collections
operations (
    id SERIAL PRIMARY KEY,
    uuid VARCHAR UNIQUE NOT NULL,              -- External reference
    collection_id VARCHAR NOT NULL,            -- Target collection
    user_id INTEGER NOT NULL,                  -- Initiating user
    type VARCHAR NOT NULL,                     -- index|append|reindex|remove_source|delete
    status VARCHAR DEFAULT 'pending',          -- pending|processing|completed|failed|cancelled
    config JSON NOT NULL,                      -- Operation configuration
    created_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
)

-- Documents table - files in collections
documents (
    id VARCHAR PRIMARY KEY,                    -- UUID
    collection_id VARCHAR NOT NULL,            -- Parent collection
    source_id INTEGER,                         -- Source reference
    file_path VARCHAR NOT NULL,                -- Full path
    file_name VARCHAR NOT NULL,                -- Name only
    file_size INTEGER NOT NULL,                -- Size in bytes
    mime_type VARCHAR,                         -- MIME type
    content_hash VARCHAR NOT NULL,             -- SHA256 for deduplication
    status VARCHAR DEFAULT 'pending',          -- pending|processing|completed|failed
    chunk_count INTEGER DEFAULT 0,             -- Number of chunks
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
)
```

### Collection Lifecycle

1. **Creation**: User creates collection with configuration
2. **Initialization**: Qdrant vector collection created
3. **Indexing**: Documents added via operations
4. **Ready**: Collection available for search
5. **Maintenance**: Reindexing, source updates
6. **Deletion**: Complete cleanup of all data

## API Endpoints

### Create Collection
```http
POST /api/v2/collections
Content-Type: application/json
Authorization: Bearer {token}

{
  "name": "Technical Documentation",
  "description": "Company technical docs",
  "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
  "quantization": "float16",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "is_public": false
}
```

Creates a new collection with specified configuration. The collection starts in 'pending' status and transitions to 'ready' once the vector store is initialized.

### List Collections
```http
GET /api/v2/collections?page=1&per_page=20&search=tech
Authorization: Bearer {token}
```

Returns paginated list of collections accessible to the user:

```json
{
  "collections": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "Technical Documentation",
      "description": "Company technical docs",
      "owner_id": 1,
      "status": "ready",
      "document_count": 156,
      "total_chunks": 3420,
      "total_size_bytes": 45678900,
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 5,
  "page": 1,
  "per_page": 20
}
```

### Get Collection Details
```http
GET /api/v2/collections/{collection_id}
Authorization: Bearer {token}
```

Returns comprehensive collection information including sources and recent operations:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Technical Documentation",
  "description": "Company technical docs",
  "owner_id": 1,
  "vector_store_name": "coll_550e8400_qwen06b_f16",
  "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
  "quantization": "float16",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "is_public": false,
  "status": "ready",
  "document_count": 156,
  "total_chunks": 3420,
  "total_size_bytes": 45678900,
  "sources": [
    {
      "id": 1,
      "source_path": "/docs/technical",
      "source_type": "directory",
      "document_count": 156,
      "size_bytes": 45678900,
      "last_indexed_at": "2024-01-15T14:30:00Z"
    }
  ],
  "recent_operations": [
    {
      "uuid": "op_123e4567-e89b-12d3-a456-426614174000",
      "type": "index",
      "status": "completed",
      "created_at": "2024-01-15T10:00:00Z",
      "completed_at": "2024-01-15T10:30:00Z"
    }
  ]
}
```

### Update Collection
```http
PATCH /api/v2/collections/{collection_id}
Content-Type: application/json
Authorization: Bearer {token}

{
  "name": "Technical Documentation v2",
  "description": "Updated technical documentation",
  "is_public": true
}
```

Updates collection metadata. Note that embedding model and chunk settings cannot be changed after creation.

### Delete Collection
```http
DELETE /api/v2/collections/{collection_id}
Authorization: Bearer {token}
```

Permanently removes the collection and all associated data including documents and vectors.

### Add Source to Collection
```http
POST /api/v2/collections/{collection_id}/sources
Content-Type: application/json
Authorization: Bearer {token}

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

Initiates an operation to add documents from a source to the collection.

### List Collection Documents
```http
GET /api/v2/collections/{collection_id}/documents?page=1&per_page=50&status=completed
Authorization: Bearer {token}
```

Returns paginated list of documents in the collection with filtering options.

### Reindex Collection
```http
POST /api/v2/collections/{collection_id}/reindex
Content-Type: application/json
Authorization: Bearer {token}

{
  "config": {
    "force": false,
    "only_failed": true
  }
}
```

Creates a reindexing operation to update document embeddings.

## Features

### Duplicate Detection

The system automatically detects duplicate documents using content hashing:

1. **Content Hash Calculation**: SHA-256 hash of document content
2. **Cross-Collection Detection**: Duplicates detected across all collections
3. **Efficient Storage**: Duplicate documents reference same content
4. **User Notification**: Duplicate count reported in operations

### Multi-Model Support

Collections support various embedding models with different characteristics:

- **Qwen/Qwen3-Embedding-0.6B**: Lightweight, fast, 1024 dimensions
- **Qwen/Qwen3-Embedding-4B**: High quality, 2560 dimensions
- **BAAI/bge-large-en-v1.5**: General purpose, 1024 dimensions
- **Custom models**: Any HuggingFace compatible model

### Quantization Options

Optimize memory usage and performance with quantization:

- **float32**: Full precision (best quality)
- **float16**: Half precision (balanced)
- **int8**: 8-bit integers (smallest size)

### Access Control

Fine-grained permissions system:

- **Owner**: Full control over collection
- **Public**: Read-only access for all users
- **Shared**: Specific user permissions (coming soon)

## Operations

### Operation Types

1. **index**: Initial document indexing from source
2. **append**: Add documents to existing collection
3. **reindex**: Re-process all documents
4. **remove_source**: Remove documents from specific source
5. **delete**: Delete entire collection

### Operation Lifecycle

```
pending → processing → completed
                    ↘ failed
                    ↘ cancelled
```

### Real-time Progress

Operations support WebSocket connections for real-time progress:

```javascript
const ws = new WebSocket(`ws://localhost:8080/api/v2/operations/${operationId}/ws`);

ws.onmessage = (event) => {
  const progress = JSON.parse(event.data);
  console.log(`Progress: ${progress.percentage}%`);
};
```

## Search Integration

### Multi-Collection Search

Search across multiple collections simultaneously:

```http
POST /api/v2/search
Content-Type: application/json
Authorization: Bearer {token}

{
  "collection_uuids": [
    "550e8400-e29b-41d4-a716-446655440000",
    "660f9511-f29c-52e5-b827-557755551111"
  ],
  "query": "machine learning algorithms",
  "k": 20,
  "use_reranker": true
}
```

### Collection-Specific Features

- **Model Awareness**: Results indicate which embedding model was used
- **Score Normalization**: Scores normalized across different models
- **Metadata Enrichment**: Results include collection context

## Best Practices

### Collection Organization

1. **Logical Grouping**: Create collections by topic or purpose
2. **Model Selection**: Choose models based on use case:
   - Small models for speed
   - Large models for accuracy
   - Domain-specific models for specialized content
3. **Size Management**: Keep collections under 1M chunks for optimal performance

### Performance Optimization

1. **Batch Operations**: Add multiple sources in single operation
2. **Incremental Updates**: Use append operations for new content
3. **Smart Reindexing**: Only reindex failed or changed documents
4. **Quantization**: Use int8 for large collections with GPU constraints

### Data Management

1. **Regular Maintenance**: Periodically check for orphaned documents
2. **Source Tracking**: Maintain clear source paths for updates
3. **Version Control**: Use collection names with versions
4. **Backup Strategy**: Regular exports of collection metadata

## Implementation Details

### Vector Store Naming

Qdrant collections use a deterministic naming scheme:
```
{collection_uuid}_{model_short}_{quantization}
```

Example: `550e8400-e29b-41d4-a716-446655440000_qwen06b_f16`

### Chunk Processing

Documents are processed into chunks for embedding:

1. **Text Extraction**: Format-specific parsers
2. **Chunking**: Split by token count with overlap
3. **Metadata Preservation**: Maintain document context
4. **Embedding Generation**: Batch processing for efficiency

### Status Management

Collection status reflects overall health:

- **pending**: Being created
- **ready**: Available for use
- **processing**: Operation in progress
- **error**: Critical error occurred
- **degraded**: Partial functionality

## Error Handling

### Common Errors

1. **Duplicate Name**: Collection names must be unique
2. **Invalid Model**: Specified model not available
3. **Insufficient Resources**: Not enough GPU memory
4. **Permission Denied**: User lacks required permissions

### Recovery Procedures

1. **Failed Operations**: Retry with force flag
2. **Corrupted Vectors**: Reindex collection
3. **Missing Documents**: Re-add source
4. **Status Errors**: Check operation logs

## Future Enhancements

1. **Collection Sharing**: Fine-grained access control
2. **Collection Templates**: Pre-configured settings
3. **Auto-Scaling**: Dynamic resource allocation
4. **Collection Cloning**: Duplicate with new settings
5. **Incremental Indexing**: Only process changed files
6. **Collection Versioning**: Track changes over time
7. **Export/Import**: Backup and migration tools
8. **Collection Analytics**: Usage statistics and insights

## Migration from Legacy System

For users migrating from the old job-based system:

1. **Jobs → Collections**: Each job becomes a collection
2. **Job IDs → Collection UUIDs**: New identifier format
3. **Files → Documents**: Terminology update
4. **Processing → Operations**: Async task management
5. **Search Updates**: Use collection UUIDs instead of job IDs

### Migration Script Example

```python
# Migrate legacy job to new collection
async def migrate_job_to_collection(job_id: str):
    # Get legacy job data
    job = await get_legacy_job(job_id)
    
    # Create new collection
    collection = await create_collection({
        "name": job["name"],
        "description": job["description"],
        "embedding_model": job["model_name"],
        "quantization": job["quantization"] or "float16",
        "chunk_size": job["chunk_size"],
        "chunk_overlap": job["chunk_overlap"]
    })
    
    # Create indexing operation
    operation = await create_operation({
        "collection_id": collection["id"],
        "type": "index",
        "config": {
            "source_path": job["directory_path"],
            "source_type": "directory"
        }
    })
    
    return collection, operation
```