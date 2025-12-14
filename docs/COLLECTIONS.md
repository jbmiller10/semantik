# Collections Architecture & Technical Reference

## Overview

Technical architecture and implementation of Semantik's collection system. For user docs, see [Collection Management Guide](./COLLECTION_MANAGEMENT.md).

Collections are self-contained document repositories with dedicated vector stores and consistent embedding configs.

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

Creates a collection with the given config. Status starts as 'pending', transitions to 'ready' when the vector store is initialized.

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
      "source_config": {
        "path": "/docs/technical",
        "recursive": true
      },
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

Updates metadata only. Embedding model and chunk settings are immutable after creation.

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

**Request (legacy, still supported):**
```json
{
  "source_path": "/docs/api"
}
```

Initiates an operation to add documents from a source to the collection.

### Manage Sources (sync, secrets, runs)

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
    "token": "ghp_..."
  }
}
```

Trigger a run immediately:
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

Delete a source (removes documents/vectors, then deletes the source record):
```http
DELETE /api/v2/collections/{collection_id}/sources/{source_id}
Authorization: Bearer {token}
```

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

### Ingestion DTO & Hashing Contract

- All connectors emit `shared.dtos.ingestion.IngestedDocument` with `content`, `unique_id`, `source_type`, `metadata`, `content_hash`, and optional `file_path`
- Content hashes via `shared.utils.hashing.compute_content_hash` (SHA-256, 64-char hex)
- `DocumentRegistryService` deduplicates on `(collection_id, content_hash)` and optionally `(collection_id, uri)`

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

- **Owner**: Full control
- **Public**: Read-only for all users
- **Shared**: Coming soon

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
const token = localStorage.getItem('authToken');
const ws = new WebSocket(`ws://localhost:8080/ws/operations/${operationId}?token=${token}`);

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.data?.progress != null) {
    console.log(`Progress: ${msg.data.progress}%`);
  }
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

- Fine-grained access control
- Collection templates
- Auto-scaling
- Collection cloning
- Incremental indexing
- Collection versioning
- Export/import tools
- Usage analytics

## Practical Examples

### Example 1: Creating a Technical Documentation Collection

```python
import httpx
import asyncio

async def setup_tech_docs_collection():
    """Set up a collection for technical documentation."""
    client = httpx.AsyncClient()
    base_url = "http://localhost:8080"
    headers = {"Authorization": "Bearer YOUR_TOKEN"}
    
    # Step 1: Create the collection
    collection_data = {
        "name": "Technical Documentation",
        "description": "API docs, architecture guides, and tutorials",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "quantization": "float16",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "is_public": false
    }
    
    response = await client.post(
        f"{base_url}/api/v2/collections",
        json=collection_data,
        headers=headers
    )
    collection = response.json()
    print(f"Created collection: {collection['id']}")
    
    # Step 2: Add multiple source directories
    sources = [
        {
            "path": "/docs/api",
            "extensions": [".md", ".rst"],
            "description": "API documentation"
        },
        {
            "path": "/docs/guides",
            "extensions": [".md", ".pdf"],
            "description": "User guides and tutorials"
        },
        {
            "path": "/docs/architecture",
            "extensions": [".md", ".png", ".svg"],
            "description": "Architecture diagrams and docs"
        }
    ]
    
    operations = []
    for source in sources:
        response = await client.post(
            f"{base_url}/api/v2/collections/{collection['id']}/sources",
            json={
                "source_type": "directory",
                "source_path": source["path"],
                "filters": {
                    "extensions": source["extensions"],
                    "ignore_patterns": ["**/drafts/**", "**/.git/**"]
                },
                "config": {
                    "recursive": true,
                    "follow_symlinks": false
                }
            },
            headers=headers
        )
        operation = response.json()
        operations.append(operation)
        print(f"Started indexing {source['description']}: {operation['id']}")
    
    # Step 3: Monitor all operations
    for op in operations:
        await monitor_operation_completion(client, base_url, headers, op['id'])
    
    print("Technical documentation collection ready!")
    return collection

async def monitor_operation_completion(client, base_url, headers, operation_id):
    """Monitor an operation until completion."""
    while True:
        response = await client.get(
            f"{base_url}/api/v2/operations/{operation_id}",
            headers=headers
        )
        data = response.json()
        
        if data['status'] == 'completed':
            print(f"Operation {operation_id} completed successfully")
            break
        elif data['status'] == 'failed':
            print(f"Operation {operation_id} failed: {data.get('error_message')}")
            break
        
        if 'progress' in data:
            print(f"Progress: {data['progress']['percentage']:.1f}%")
        
        await asyncio.sleep(2)
```

### Example 2: Managing a Research Paper Collection

```javascript
// Frontend example using React and Zustand
class ResearchPaperManager {
    constructor(apiClient) {
        this.api = apiClient;
    }
    
    async createResearchCollection() {
        // Create a specialized collection for academic papers
        const collection = await this.api.createCollection({
            name: "ML Research Papers 2024",
            description: "Machine learning research papers and preprints",
            embedding_model: "Qwen/Qwen3-Embedding-4B", // Higher quality for technical content
            quantization: "float16",
            chunk_size: 1500, // Larger chunks for academic content
            chunk_overlap: 300,
            is_public: false
        });
        
        return collection;
    }
    
    async addPapersBatch(collectionId, paperPaths) {
        // Add papers with progress tracking
        const operations = [];
        
        for (const path of paperPaths) {
            const operation = await this.api.addSource(collectionId, {
                source_type: "file",
                source_path: path,
                config: {
                    extract_metadata: true, // Extract paper metadata
                    ocr_enabled: true      // For scanned PDFs
                }
            });
            operations.push(operation);
        }
        
        // Monitor all operations
        return this.monitorBatchOperations(operations);
    }
    
    async monitorBatchOperations(operations) {
        const results = await Promise.all(
            operations.map(op => this.monitorOperation(op.id))
        );
        
        const summary = {
            total: operations.length,
            successful: results.filter(r => r.status === 'completed').length,
            failed: results.filter(r => r.status === 'failed').length
        };
        
        return summary;
    }
    
    async searchPapers(collectionId, query, filters = {}) {
        // Semantic search with filters
        const results = await this.api.search({
            collection_ids: [collectionId],
            query: query,
            k: 20,
            filters: {
                file_type: "pdf",
                ...filters
            },
            use_reranker: true // Better relevance for academic content
        });
        
        return results;
    }
}

// Usage in React component
function ResearchDashboard() {
    const [collection, setCollection] = useState(null);
    const [searchResults, setSearchResults] = useState([]);
    const manager = new ResearchPaperManager(apiClient);
    
    const handleCreateCollection = async () => {
        try {
            const newCollection = await manager.createResearchCollection();
            setCollection(newCollection);
            toast.success("Research collection created!");
        } catch (error) {
            toast.error(`Failed to create collection: ${error.message}`);
        }
    };
    
    const handleSearch = async (query) => {
        if (!collection) return;
        
        try {
            const results = await manager.searchPapers(collection.id, query);
            setSearchResults(results);
        } catch (error) {
            toast.error(`Search failed: ${error.message}`);
        }
    };
    
    return (
        <div className="research-dashboard">
            {/* UI components */}
        </div>
    );
}
```

### Example 3: Collection Health Monitoring

```python
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any

class CollectionHealthMonitor:
    """Monitor and maintain collection health."""
    
    def __init__(self, api_client):
        self.api = api_client
        
    async def check_collection_health(self, collection_id: str) -> Dict[str, Any]:
        """Comprehensive health check for a collection."""
        collection = await self.api.get_collection(collection_id)
        operations = await self.api.get_collection_operations(
            collection_id, 
            limit=10
        )
        
        health_report = {
            "collection_id": collection_id,
            "status": collection["status"],
            "checks": {},
            "recommendations": []
        }
        
        # Check 1: Collection status
        health_report["checks"]["status"] = {
            "passed": collection["status"] == "ready",
            "message": f"Collection status: {collection['status']}"
        }
        
        # Check 2: Document processing
        failed_docs = await self.api.get_documents(
            collection_id,
            status="failed"
        )
        health_report["checks"]["failed_documents"] = {
            "passed": len(failed_docs) == 0,
            "count": len(failed_docs),
            "message": f"{len(failed_docs)} failed documents"
        }
        
        # Check 3: Recent operations
        failed_ops = [op for op in operations if op["status"] == "failed"]
        health_report["checks"]["recent_operations"] = {
            "passed": len(failed_ops) == 0,
            "failed_count": len(failed_ops),
            "message": f"{len(failed_ops)} failed operations in recent history"
        }
        
        # Check 4: Storage efficiency
        total_size = collection.get("total_size_bytes", 0)
        doc_count = collection.get("document_count", 0)
        avg_size = total_size / doc_count if doc_count > 0 else 0
        
        health_report["checks"]["storage_efficiency"] = {
            "passed": avg_size < 10_000_000,  # 10MB average
            "average_document_size": avg_size,
            "message": f"Average document size: {avg_size / 1_000_000:.2f}MB"
        }
        
        # Generate recommendations
        if failed_docs:
            health_report["recommendations"].append(
                "Run reindex operation with only_failed=true to retry failed documents"
            )
        
        if failed_ops:
            health_report["recommendations"].append(
                "Review failed operation logs to identify and fix issues"
            )
        
        if collection["status"] != "ready":
            health_report["recommendations"].append(
                "Wait for current operations to complete or investigate stuck operations"
            )
        
        # Overall health score
        passed_checks = sum(1 for check in health_report["checks"].values() 
                          if check["passed"])
        total_checks = len(health_report["checks"])
        health_report["health_score"] = (passed_checks / total_checks) * 100
        
        return health_report
    
    async def auto_maintenance(self, collection_id: str):
        """Perform automatic maintenance tasks."""
        health = await self.check_collection_health(collection_id)
        
        if health["health_score"] < 80:
            print(f"Collection {collection_id} needs maintenance")
            
            # Auto-fix failed documents
            if not health["checks"]["failed_documents"]["passed"]:
                print("Reindexing failed documents...")
                operation = await self.api.reindex_collection(
                    collection_id,
                    config={"only_failed": True}
                )
                await self.wait_for_operation(operation["id"])
            
            # Clean up old operations
            await self.cleanup_old_operations(collection_id)
    
    async def monitor_collections(self, collection_ids: List[str]):
        """Monitor multiple collections continuously."""
        while True:
            for collection_id in collection_ids:
                try:
                    health = await self.check_collection_health(collection_id)
                    
                    if health["health_score"] < 90:
                        print(f"Collection {collection_id}: Health score {health['health_score']:.1f}%")
                        for rec in health["recommendations"]:
                            print(f"  - {rec}")
                    
                    # Perform maintenance if needed
                    if health["health_score"] < 80:
                        await self.auto_maintenance(collection_id)
                        
                except Exception as e:
                    print(f"Error monitoring {collection_id}: {e}")
            
            # Wait before next check
            await asyncio.sleep(300)  # Check every 5 minutes

# Usage
async def main():
    monitor = CollectionHealthMonitor(api_client)
    
    # One-time health check
    health = await monitor.check_collection_health("collection-123")
    print(f"Health Score: {health['health_score']:.1f}%")
    
    # Continuous monitoring
    collections = ["collection-123", "collection-456"]
    await monitor.monitor_collections(collections)
```

## Integration Patterns

### Pattern 1: Multi-Tenant Collections

```python
class MultiTenantCollectionManager:
    """Manage collections for multiple tenants/organizations."""
    
    async def create_tenant_collection(self, tenant_id: str, config: dict):
        """Create a collection with tenant isolation."""
        collection_name = f"{tenant_id}_{config['name']}"
        
        collection = await self.api.create_collection({
            "name": collection_name,
            "description": f"{config['description']} (Tenant: {tenant_id})",
            "embedding_model": config.get("model", "Qwen/Qwen3-Embedding-0.6B"),
            "metadata": {
                "tenant_id": tenant_id,
                "created_by": config.get("created_by"),
                "department": config.get("department")
            },
            **config
        })
        
        # Set up tenant-specific permissions
        await self.setup_tenant_permissions(collection["id"], tenant_id)
        
        return collection
```

### Pattern 2: Collection Templates

```javascript
const collectionTemplates = {
    legal: {
        name: "Legal Documents",
        embedding_model: "Qwen/Qwen3-Embedding-4B",
        chunk_size: 2000,  // Larger for legal documents
        chunk_overlap: 400,
        filters: {
            extensions: [".pdf", ".docx"],
            ignore_patterns: ["**/drafts/**"]
        }
    },
    
    codebase: {
        name: "Source Code",
        embedding_model: "Qwen/Qwen3-Embedding-0.6B",
        chunk_size: 500,   // Smaller for code
        chunk_overlap: 100,
        filters: {
            extensions: [".py", ".js", ".ts", ".java"],
            ignore_patterns: ["**/node_modules/**", "**/venv/**"]
        }
    },
    
    knowledge_base: {
        name: "Knowledge Base",
        embedding_model: "Qwen/Qwen3-Embedding-0.6B",
        chunk_size: 1000,
        chunk_overlap: 200,
        filters: {
            extensions: [".md", ".txt", ".pdf", ".html"],
            ignore_patterns: ["**/archive/**"]
        }
    }
};

async function createFromTemplate(templateName, customizations = {}) {
    const template = collectionTemplates[templateName];
    if (!template) {
        throw new Error(`Unknown template: ${templateName}`);
    }
    
    const collectionConfig = {
        ...template,
        ...customizations,
        name: customizations.name || `${template.name} - ${new Date().toISOString()}`
    };
    
    return await api.createCollection(collectionConfig);
}
```

## Performance Tuning Guide

### Optimizing for Different Use Cases

#### High-Volume Document Processing
```python
# Configuration for processing large document sets
high_volume_config = {
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",  # Faster model
    "quantization": "int8",                           # Reduce memory usage
    "chunk_size": 1500,                               # Larger chunks = fewer vectors
    "chunk_overlap": 100,                             # Minimal overlap
    "batch_size": 64,                                 # Large batch processing
    "parallel_workers": 4                             # Multiple workers
}
```

#### High-Accuracy Search
```python
# Configuration for maximum search accuracy
high_accuracy_config = {
    "embedding_model": "Qwen/Qwen3-Embedding-4B",    # Best model
    "quantization": "float32",                        # Full precision
    "chunk_size": 750,                                # Smaller chunks
    "chunk_overlap": 250,                             # More overlap
    "use_reranker": true,                             # Enable reranking
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-12-v2"
}
```

## Migration from Legacy System

The collection-centric architecture replaces the previous job-based system. Key differences:

1. **Terminology Changes**:
   - Jobs → Collections (persistent entities)
   - Job execution → Operations (temporary tasks)
   - Files → Documents
   - Processing → Operations (index, append, reindex)

2. **Conceptual Shift**:
   - Collections are long-lived, first-class entities
   - Operations are temporary tasks that modify collections
   - Focus on collection lifecycle rather than job execution

3. **API Migration**:
   - Old: `POST /api/jobs` → New: `POST /api/v2/collections` + operation
   - Old: `GET /api/jobs/{id}` → New: `GET /api/v2/collections/{id}`
   - Old: Search by job → New: Search by collection

### Automated Migration Tool

```python
class LegacyMigrator:
    """Migrate from job-based to collection-based system."""
    
    async def migrate_all_jobs(self):
        """Migrate all legacy jobs to collections."""
        legacy_jobs = await self.get_all_legacy_jobs()
        migration_report = {
            "total": len(legacy_jobs),
            "successful": 0,
            "failed": 0,
            "mappings": {}
        }
        
        for job in legacy_jobs:
            try:
                collection = await self.migrate_single_job(job)
                migration_report["successful"] += 1
                migration_report["mappings"][job["id"]] = collection["id"]
            except Exception as e:
                migration_report["failed"] += 1
                print(f"Failed to migrate job {job['id']}: {e}")
        
        return migration_report
    
    async def migrate_single_job(self, job: dict) -> dict:
        """Migrate a single job to a collection."""
        # Create collection from job metadata
        collection = await self.api.create_collection({
            "name": job["name"],
            "description": f"Migrated from job {job['id']}",
            "embedding_model": job["model_name"],
            "quantization": job.get("quantization", "float16"),
            "chunk_size": job.get("chunk_size", 1000),
            "chunk_overlap": job.get("chunk_overlap", 200),
            "metadata": {
                "legacy_job_id": job["id"],
                "migrated_at": datetime.utcnow().isoformat()
            }
        })
        
        # Create index operation for job's documents
        if job.get("directory_path"):
            operation = await self.api.add_source(collection["id"], {
                "source_type": "directory",
                "source_path": job["directory_path"],
                "config": {
                    "recursive": True
                }
            })
            
            # Wait for completion
            await self.wait_for_operation(operation["id"])
        
        return collection
```
