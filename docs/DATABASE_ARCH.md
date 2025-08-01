# Database Architecture Documentation

## Overview

The Semantik system uses a hybrid database architecture combining:
- **PostgreSQL** for relational data (collections, operations, documents, users, authentication)
- **Qdrant** for vector storage and similarity search

This architecture separates transactional/metadata storage from high-performance vector operations, following the architectural principle of using the right tool for each job.

**Key Architectural Decisions**:
- The PostgreSQL database is owned and managed exclusively by the **webui package**
- All database operations use the **repository pattern** implemented in `packages/shared/database/`
- The **vecpipe package** has no direct database access - it must use webui API endpoints
- The **shared package** provides the repository interfaces and implementations used by webui
- The system has migrated from a **job-centric** to a **collection-centric** architecture for better organization and scalability

## Migration from Job-Centric to Collection-Centric Architecture

### Background

The system originally used a **job-centric** approach where each indexing task was tracked as a "job" with documents associated with specific job runs. This created several limitations:
- Documents were tied to specific job executions rather than logical collections
- Difficult to manage document lifecycle across multiple indexing runs
- No clear organization of related documents
- Complex tracking of incremental updates

### Collection-Centric Benefits

The new **collection-centric** architecture provides:
1. **Logical Organization**: Documents belong to collections, not job runs
2. **Persistent Identity**: Collections maintain identity across operations
3. **Multi-Model Support**: Each collection can use different embedding models
4. **Better State Management**: Collection status independent of operation status
5. **Cleaner Relationships**: Clear ownership and permission models

### Migration Details

The migration replaced the old "jobs" concept with two new concepts:
- **Collections**: Persistent containers for related documents
- **Operations**: Temporary tasks performed on collections (index, append, reindex, etc.)

Key changes:
- `jobs` table → `operations` table
- Documents now reference `collection_id` instead of job IDs
- New `collection_sources` table tracks data origins
- Operation lifecycle is separate from collection lifecycle

## Database Distribution Strategy

### PostgreSQL (Relational Data)
- **Purpose**: Stores structured metadata, operation tracking, user management, authentication, collections
- **Connection**: Configured via `DATABASE_URL` or individual connection parameters
- **Why PostgreSQL**: Production-ready, supports concurrent operations, advanced features, scalable
- **Data Types**: User accounts, collection metadata, document processing status, authentication tokens, operation tracking

### Qdrant (Vector Database)
- **Purpose**: Stores document embeddings and enables similarity search
- **Location**: Configured via `QDRANT_HOST:QDRANT_PORT` (default: localhost:6333)
- **Why Qdrant**: High-performance vector search, supports various distance metrics, scalable
- **Data Types**: Document chunks as vectors with metadata payloads

## PostgreSQL Database Schema

### Configuration
- **Connection**: Via `DATABASE_URL` environment variable or individual parameters
- **Ownership**: Exclusively owned by webui service
- **Access Pattern**: Repository pattern via `packages/shared/database/`
- **Migration Tool**: Alembic for schema versioning and migrations
- **Connection Pool**: SQLAlchemy connection pooling for performance
- **Database Name**: Configurable, defaults to `semantik`

### Core Table Structures

#### 1. Collections Table
Represents logical groupings of documents with shared configuration.

```sql
CREATE TABLE collections (
    id VARCHAR PRIMARY KEY,                    -- UUID for collection identification
    name VARCHAR UNIQUE NOT NULL,              -- Human-readable collection name
    description TEXT,                          -- Optional collection description
    owner_id INTEGER NOT NULL,                 -- Foreign key to users table
    vector_store_name VARCHAR UNIQUE NOT NULL, -- Qdrant collection name
    embedding_model VARCHAR NOT NULL,          -- Model used for embeddings
    quantization VARCHAR NOT NULL DEFAULT 'float16', -- float32|float16|int8
    chunk_size INTEGER NOT NULL DEFAULT 1000,  -- Token size for chunks
    chunk_overlap INTEGER NOT NULL DEFAULT 200,-- Token overlap between chunks
    is_public BOOLEAN NOT NULL DEFAULT FALSE,  -- Public visibility flag
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    meta JSON,                                  -- Additional metadata
    
    -- Status and metrics fields
    status ENUM('pending','ready','processing','error','degraded') NOT NULL DEFAULT 'pending',
    status_message TEXT,                        -- Human-readable status details
    qdrant_collections JSON,                    -- List of Qdrant collection names
    qdrant_staging JSON,                        -- Staging collections during reindex
    document_count INTEGER NOT NULL DEFAULT 0,  -- Total documents in collection
    vector_count INTEGER NOT NULL DEFAULT 0,    -- Total vectors in Qdrant
    total_size_bytes INTEGER NOT NULL DEFAULT 0,-- Total storage used
    
    FOREIGN KEY (owner_id) REFERENCES users(id)
);

-- Indexes for performance
CREATE INDEX idx_collections_owner_id ON collections(owner_id);
CREATE INDEX idx_collections_name ON collections(name);
CREATE INDEX idx_collections_is_public ON collections(is_public);
CREATE INDEX idx_collections_status ON collections(status);
```

**Key Points**:
- UUID primary keys for external reference
- Each collection has its own Qdrant collection with unique name
- Supports multiple embedding models and quantization strategies
- Status tracking with detailed metrics for monitoring
- Staging support for zero-downtime reindexing
- Document and vector counts for quick statistics

#### 2. Documents Table
Tracks individual documents within collections.

```sql
CREATE TABLE documents (
    id VARCHAR PRIMARY KEY,                     -- UUID for document identification
    collection_id VARCHAR NOT NULL,             -- Foreign key to collections
    source_id INTEGER,                          -- Foreign key to collection_sources
    file_path VARCHAR NOT NULL,                 -- Full file path
    file_name VARCHAR NOT NULL,                 -- File name only
    file_size INTEGER NOT NULL,                 -- File size in bytes
    mime_type VARCHAR,                          -- MIME type if detected
    content_hash VARCHAR NOT NULL,              -- SHA256 hash for deduplication
    status ENUM('pending','processing','completed','failed','deleted') NOT NULL DEFAULT 'pending',
    error_message TEXT,                         -- Error details if failed
    chunk_count INTEGER NOT NULL DEFAULT 0,     -- Number of chunks created
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    meta JSON,                                  -- Additional metadata
    FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
    FOREIGN KEY (source_id) REFERENCES collection_sources(id)
);

-- Indexes for performance
CREATE INDEX idx_documents_collection_id ON documents(collection_id);
CREATE INDEX idx_documents_source_id ON documents(source_id);
CREATE INDEX idx_documents_content_hash ON documents(content_hash);
CREATE INDEX idx_documents_status ON documents(status);
-- Composite unique index for duplicate prevention within collections
CREATE UNIQUE INDEX idx_documents_collection_content_hash ON documents(collection_id, content_hash);
```

**Key Points**:
- Content hash enables duplicate detection across collections
- Unique constraint on (collection_id, content_hash) prevents duplicates within a collection
- Supports soft delete via 'deleted' status for maintaining referential integrity
- Tracks processing status and chunk creation
- Cascade delete ensures cleanup when collection is removed

#### 3. Operations Table
Manages asynchronous operations on collections (formerly the "jobs" table).

```sql
CREATE TABLE operations (
    id SERIAL PRIMARY KEY,                      -- Auto-increment ID
    uuid VARCHAR UNIQUE NOT NULL,               -- UUID for external reference
    collection_id VARCHAR NOT NULL,             -- Target collection
    user_id INTEGER NOT NULL,                   -- Initiating user
    type ENUM('index','append','reindex','remove_source','delete') NOT NULL,
    status ENUM('pending','processing','completed','failed','cancelled') NOT NULL DEFAULT 'pending',
    task_id VARCHAR,                            -- Celery task ID
    config JSON NOT NULL,                       -- Operation configuration
    error_message TEXT,                         -- Error details if failed
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,                     -- When processing began
    completed_at TIMESTAMPTZ,                   -- When operation finished
    meta JSON,                                  -- Additional metadata
    FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Indexes for performance
CREATE INDEX idx_operations_collection_id ON operations(collection_id);
CREATE INDEX idx_operations_user_id ON operations(user_id);
CREATE INDEX idx_operations_type ON operations(type);
CREATE INDEX idx_operations_status ON operations(status);
CREATE INDEX idx_operations_created_at ON operations(created_at);
```

**Key Points**:
- Replaces the old "jobs" table in the job-centric architecture
- Tracks all async operations with full lifecycle (pending → processing → completed/failed)
- Stores operation configuration for reproducibility and debugging
- Links to Celery tasks for distributed processing
- Operations are ephemeral - they track tasks, while collections persist

#### 4. Collection Sources Table
Tracks data sources for collections.

```sql
CREATE TABLE collection_sources (
    id SERIAL PRIMARY KEY,                      -- Auto-increment ID
    collection_id VARCHAR NOT NULL,             -- Parent collection
    source_path VARCHAR NOT NULL,               -- Path or URL
    source_type VARCHAR NOT NULL DEFAULT 'directory', -- directory|file|url|github
    document_count INTEGER NOT NULL DEFAULT 0,  -- Documents from this source
    size_bytes INTEGER NOT NULL DEFAULT 0,      -- Total size
    last_indexed_at TIMESTAMPTZ,                -- Last indexing time
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    meta JSON,                                  -- Source-specific metadata
    FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX idx_collection_sources_collection_id ON collection_sources(collection_id);
```

#### 5. Collection Permissions Table
Fine-grained access control for collections.

```sql
CREATE TABLE collection_permissions (
    id SERIAL PRIMARY KEY,
    collection_id VARCHAR NOT NULL,
    user_id INTEGER,                            -- Either user_id or api_key_id
    api_key_id VARCHAR,
    permission VARCHAR NOT NULL,                -- read|write|admin
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (api_key_id) REFERENCES api_keys(id) ON DELETE CASCADE,
    CHECK ((user_id IS NOT NULL AND api_key_id IS NULL) OR 
           (user_id IS NULL AND api_key_id IS NOT NULL))
);

-- Indexes
CREATE INDEX idx_collection_permissions_collection_id ON collection_permissions(collection_id);
CREATE INDEX idx_collection_permissions_user_id ON collection_permissions(user_id);
```

#### 6. Supporting Tables

**Users Table** - User authentication and profiles:
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR UNIQUE NOT NULL,
    email VARCHAR UNIQUE NOT NULL,
    full_name VARCHAR,
    hashed_password VARCHAR NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login TIMESTAMPTZ,
    preferences JSON
);
```

**Refresh Tokens Table** - JWT refresh token management:
```sql
CREATE TABLE refresh_tokens (
    id SERIAL PRIMARY KEY,                      -- Auto-increment ID
    user_id INTEGER NOT NULL,
    token_hash VARCHAR UNIQUE NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_revoked BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX idx_refresh_tokens_token_hash ON refresh_tokens(token_hash);
```

**API Keys Table** - API key authentication:
```sql
CREATE TABLE api_keys (
    id VARCHAR PRIMARY KEY,
    user_id INTEGER NOT NULL,
    name VARCHAR NOT NULL,
    key_hash VARCHAR UNIQUE NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    last_used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

**Collection Audit Log** - Tracks all collection actions:
```sql
CREATE TABLE collection_audit_log (
    id SERIAL PRIMARY KEY,
    collection_id VARCHAR NOT NULL,
    operation_id INTEGER,
    user_id INTEGER,
    action VARCHAR NOT NULL,
    details JSON,
    ip_address VARCHAR,
    user_agent VARCHAR,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
    FOREIGN KEY (operation_id) REFERENCES operations(id),
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### Database Relationships

```
users (1) ──────< (N) collections (owner)
      │
      ├──────< (N) operations
      ├──────< (N) refresh_tokens
      ├──────< (N) api_keys
      └──────< (N) collection_permissions

collections (1) ──────< (N) documents
            │
            ├──────< (N) operations
            ├──────< (N) collection_sources
            ├──────< (N) collection_permissions
            ├──────< (N) collection_audit_log
            └──────< (1) collection_resource_limits

collection_sources (1) ──────< (N) documents

operations (1) ──────< (N) collection_audit_log
           └──────< (N) operation_metrics

api_keys (1) ──────< (N) collection_permissions
```

### Naming Conventions

The database follows consistent naming conventions for maintainability:

1. **Tables**: Lowercase with underscores (snake_case)
   - Plural for entities: `users`, `collections`, `documents`
   - Singular for relationships: `collection_permission`

2. **Columns**: Lowercase with underscores
   - Foreign keys: `{table}_id` (e.g., `collection_id`, `user_id`)
   - Timestamps: `{action}_at` (e.g., `created_at`, `updated_at`)
   - Booleans: `is_{state}` (e.g., `is_public`, `is_active`)

3. **Indexes**: Descriptive prefixes
   - `idx_{table}_{column}` for single column indexes
   - `idx_{table}_{col1}_{col2}` for composite indexes
   - `uq_{table}_{columns}` for unique constraints

4. **Constraints**: Descriptive names
   - `fk_{table}_{column}_{ref_table}` for foreign keys
   - `check_{table}_{description}` for check constraints

5. **Enums**: PascalCase for types, UPPER_CASE for values
   - Types: `DocumentStatus`, `OperationType`
   - Values: `PENDING`, `COMPLETED`, `FAILED`

### Operation Lifecycle States

Operations follow a strict state machine for reliability:

```
┌─────────┐      ┌────────────┐      ┌───────────┐
│ PENDING │ ───► │ PROCESSING │ ───► │ COMPLETED │
└─────────┘      └────────────┘      └───────────┘
                        │                    
                        ├──────────► ┌─────────┐
                        │            │ FAILED  │
                        │            └─────────┘
                        │
                        └──────────► ┌───────────┐
                                    │ CANCELLED │
                                    └───────────┘
```

**State Transitions**:
- `PENDING`: Initial state when operation is created
- `PROCESSING`: Worker has started the operation
- `COMPLETED`: Operation finished successfully
- `FAILED`: Operation encountered an error
- `CANCELLED`: Operation was cancelled by user

**Operation Types**:
- `INDEX`: Initial indexing of a collection
- `APPEND`: Add new documents to existing collection
- `REINDEX`: Complete re-indexing with new settings
- `REMOVE_SOURCE`: Remove documents from a specific source
- `DELETE`: Delete the entire collection

### Soft Delete Implementation

The system implements soft deletes at the document level for data safety:

1. **Document Soft Delete**:
   - Documents are marked with status='deleted' instead of being removed
   - Preserves referential integrity and audit trail
   - Deleted documents are excluded from searches
   - Vectors are removed from Qdrant but metadata retained

2. **Benefits**:
   - Recovery possible if deletion was accidental
   - Maintains historical record for compliance
   - Prevents orphaned references in audit logs
   - Enables "undelete" functionality

3. **Hard Delete Scenarios**:
   - Collections use hard delete (CASCADE) for clean removal
   - User deletion cascades to owned resources
   - Expired tokens are hard deleted

### Collection Source Management

Collection sources track the origin of documents:

1. **Source Types**:
   - `directory`: Local filesystem directory
   - `file`: Single file upload
   - `url`: Web URL or API endpoint
   - `github`: GitHub repository

2. **Source Tracking**:
   - Each source maintains document count and size
   - Last indexed timestamp for incremental updates
   - Source metadata (credentials, options)

3. **Benefits**:
   - Enables incremental indexing
   - Tracks data lineage
   - Supports source-specific removal
   - Facilitates re-indexing from source

### Transaction Patterns

The system uses specific transaction patterns for data integrity:

1. **Collection Creation Pattern**:
   ```python
   async with db.begin():
       # 1. Create collection record
       collection = await create_collection(...)
       
       # 2. Create resource limits
       await create_resource_limits(collection.id)
       
       # 3. Initialize Qdrant collection
       await init_vector_store(collection)
       
       # 4. Update collection status
       await update_status(collection.id, "ready")
   ```

2. **Document Processing Pattern**:
   ```python
   async with db.begin():
       # 1. Update document status
       await update_document_status(doc_id, "processing")
       
       # 2. Process and create chunks
       chunks = await process_document(doc)
       
       # 3. Update chunk count
       await update_chunk_count(doc_id, len(chunks))
       
       # 4. Update collection metrics
       await increment_collection_vectors(collection_id, len(chunks))
   ```

3. **Operation Completion Pattern**:
   ```python
   async with db.begin():
       # 1. Update operation status
       await update_operation_status(op_id, "completed")
       
       # 2. Record metrics
       await record_operation_metrics(op_id, metrics)
       
       # 3. Update collection statistics
       await refresh_collection_stats(collection_id)
       
       # 4. Create audit log entry
       await create_audit_log(collection_id, op_id, "completed")
   ```

### Migration Strategy

The system uses Alembic for database migrations:
1. Version control for schema changes
2. Automatic migration generation from model changes
3. Rollback capability for failed migrations
4. Migration history tracking
5. Support for enum alterations (with PostgreSQL limitations)

Example migration workflow:
```bash
# Generate new migration
alembic revision --autogenerate -m "Add collection status field"

# Apply migrations
alembic upgrade head

# Rollback if needed
alembic downgrade -1
```

**Migration Best Practices**:
- Always review auto-generated migrations
- Test migrations on a copy of production data
- Include both upgrade and downgrade paths
- Use batch operations for large data migrations
- Consider zero-downtime deployment strategies

## Qdrant Vector Database

### Collection Structure

Each PostgreSQL collection has a corresponding Qdrant collection for vector storage.

#### Collection Naming Convention
```
{collection_uuid}_{embedding_model_short}_{quantization}
```
Example: `550e8400-e29b-41d4-a716-446655440000_qwen06b_f16`

**Vector Configuration**:
```python
VectorParams(
    size=vector_dim,           # Varies by model (384-2560)
    distance=Distance.COSINE   # Cosine similarity
)
```

**Point Structure**:
```python
PointStruct(
    id=str,                    # Document chunk UUID
    vector=List[float],        # Embedding vector
    payload={
        "document_id": str,    # Reference to documents table
        "collection_id": str,  # Reference to collections table
        "chunk_index": int,    # Position in document
        "content": str,        # Text content
        "metadata": {          # Additional metadata
            "source_path": str,
            "mime_type": str,
            "created_at": str,
            ...
        }
    }
)
```

### Indexing Strategy

Qdrant configuration for optimal performance:
- **indexing_threshold**: 20000 (switches from flat to HNSW index)
- **memmap_threshold**: 50000 (switches to disk-based storage)
- **ef_construction**: 512 (HNSW parameter for index quality)
- **m**: 16 (HNSW parameter for connectivity)

### Performance Optimization

1. **Batch Operations**: Upload vectors in batches (default: 100)
2. **Parallel Processing**: Multiple workers for document processing
3. **Connection Pooling**: Reusable connections to Qdrant
4. **Async Operations**: Non-blocking search and updates

## Design Decisions and Rationale

### Why Collection-Centric Architecture?

1. **Logical Organization**:
   - Documents naturally belong to collections, not job runs
   - Users think in terms of "my technical docs" not "indexing job #123"
   - Enables better permission management at the collection level

2. **Multi-Model Support**:
   - Each collection can use different embedding models
   - Allows experimentation without affecting other data
   - Future-proof for model upgrades

3. **Incremental Updates**:
   - Append operations add to existing collections
   - Remove operations can target specific sources
   - Reindex operations maintain service availability

4. **Better State Management**:
   - Collection status reflects overall health
   - Operation status tracks individual tasks
   - Clear separation of concerns

### Document-Collection Relationships

1. **One-to-Many with CASCADE**:
   - Documents belong to exactly one collection
   - Deleting a collection removes all documents
   - Ensures no orphaned documents

2. **Content Hash Strategy**:
   - SHA256 hash prevents duplicates within collection
   - Allows same file in different collections
   - Enables deduplication at query time

3. **Source Tracking**:
   - Optional source_id links to collection_sources
   - Enables source-based operations
   - Maintains data lineage

### Why Soft Deletes for Documents?

1. **Data Safety**:
   - Accidental deletions are recoverable
   - Audit trail maintained
   - Compliance requirements

2. **Performance**:
   - Avoids expensive CASCADE operations
   - Vectors removed from Qdrant immediately
   - Metadata retained for history

3. **Flexibility**:
   - Enables "trash bin" functionality
   - Supports undo operations
   - Allows for data retention policies

## Data Models (SQLAlchemy/Pydantic)

### Collection Model
```python
class Collection(Base):
    __tablename__ = "collections"
    
    id: str  # UUID
    name: str
    description: Optional[str]
    owner_id: int
    vector_store_name: str
    embedding_model: str
    quantization: str = "float16"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    is_public: bool = False
    status: CollectionStatus
    status_message: Optional[str]
    qdrant_collections: Optional[List[str]]
    qdrant_staging: Optional[List[str]]
    document_count: int = 0
    vector_count: int = 0
    total_size_bytes: int = 0
    created_at: datetime
    updated_at: datetime
    meta: Optional[dict]
    
    # Relationships
    owner: User
    documents: List[Document]
    operations: List[Operation]
    permissions: List[CollectionPermission]
    sources: List[CollectionSource]
    resource_limits: Optional[CollectionResourceLimits]
```

### Operation Model
```python
class Operation(Base):
    __tablename__ = "operations"
    
    id: int
    uuid: str
    collection_id: str
    user_id: int
    type: OperationType
    status: OperationStatus
    task_id: Optional[str]
    config: dict
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    meta: Optional[dict]
    
    # Relationships
    collection: Collection
    user: User
    audit_logs: List[CollectionAuditLog]
    metrics: List[OperationMetrics]
```

## Database Operations

### Repository Pattern Implementation

The system uses a clean repository pattern for all database operations:

**Repository Structure** (in `packages/shared/database/`):

**Repository Interfaces**:
- `CollectionRepository`: Abstract interface for collection operations
- `DocumentRepository`: Abstract interface for document operations
- `OperationRepository`: Abstract interface for operation management
- `UserRepository`: Abstract interface for user management
- `AuthTokenRepository`: Abstract interface for authentication tokens

**Usage Example**:
```python
from shared.database import create_collection_repository

# Repository handles connection lifecycle
async with create_collection_repository() as repo:
    # Create collection
    collection = await repo.create_collection({
        "name": "Technical Docs",
        "description": "Company technical documentation",
        "owner_id": user_id,
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "quantization": "float16"
    })
    
    # Update status
    await repo.update_collection_status(
        collection.id, 
        CollectionStatus.READY
    )
    
    # Query collections
    user_collections = await repo.list_collections(
        owner_id=user_id,
        is_public=False
    )
```

### Connection Management

**PostgreSQL**:
- Async connection pooling via SQLAlchemy
- Configurable pool size and overflow settings
- Connection string: `postgresql+asyncpg://user:password@host:port/database`
- Automatic retry on connection failures

**Qdrant**:
- Singleton async client with connection pooling
- Health check before operations
- Automatic reconnection on failure

### Transaction Patterns

**Collection Creation Flow**:
1. Create collection record with status='pending'
2. Initialize Qdrant collection
3. Update collection status to 'ready'
4. Log creation in audit table

**Document Processing Flow**:
1. Create operation record
2. Add documents with status='pending'
3. Process documents in batches
4. Update document status and chunk counts
5. Complete operation with metrics

**Atomic Operations**:
- PostgreSQL transactions ensure ACID compliance
- Document status updates are atomic
- Operation metrics computed from document aggregations
- Proper rollback on errors

### Error Handling

**PostgreSQL Errors**:
- Unique constraint violations (duplicate collections)
- Foreign key violations (invalid references)
- Connection pool exhaustion handling
- Deadlock detection and retry

**Qdrant Errors**:
- Collection already exists
- Invalid vector dimensions
- Connection failures
- Insufficient memory

## Data Flow

### Document Indexing Pipeline

```
1. User creates collection
   └─> WebUI creates collection via CollectionRepository
   
2. User initiates index operation
   └─> WebUI creates operation via OperationRepository
   └─> Celery worker picks up task
   
3. Worker scans for documents
   └─> Creates document records via DocumentRepository
   
4. Extract and chunk text (using shared.text_processing)
   └─> Update document status via DocumentRepository
   
5. Generate embeddings (using shared.embedding)
   └─> Batch upload to Qdrant
   └─> Update chunk counts via DocumentRepository
   
6. Complete operation
   └─> Update operation status via OperationRepository
   └─> Vectors available for search
```

### Search Flow

```
1. User queries via WebUI
   └─> WebUI validates permissions
   
2. WebUI proxies to Vecpipe Search API
   └─> Includes collection UUIDs
   
3. Vecpipe generates query embedding
   └─> Searches relevant Qdrant collections
   
4. Results enriched with metadata
   └─> Document info from PostgreSQL
   └─> Formatted for UI display
```

## Performance Considerations

### Query Optimization

**PostgreSQL Optimization Strategies**:

1. **Index Strategy**:
   ```sql
   -- B-tree indexes for equality and range queries
   CREATE INDEX idx_operations_created_at ON operations(created_at);
   
   -- Partial indexes for filtered queries
   CREATE INDEX idx_operations_pending ON operations(status) 
   WHERE status = 'pending';
   
   -- Composite indexes for multi-column queries
   CREATE INDEX idx_documents_collection_status ON documents(collection_id, status);
   
   -- GIN indexes for JSON queries
   CREATE INDEX idx_collections_meta ON collections USING GIN(meta);
   ```

2. **Query Patterns**:
   ```sql
   -- Efficient pagination with cursor
   SELECT * FROM documents 
   WHERE collection_id = ? AND id > ? 
   ORDER BY id 
   LIMIT 100;
   
   -- Use EXISTS for existence checks
   SELECT EXISTS(
       SELECT 1 FROM documents 
       WHERE collection_id = ? AND status = 'completed'
   );
   
   -- Aggregate with window functions
   SELECT collection_id, 
          COUNT(*) OVER (PARTITION BY collection_id) as total_docs
   FROM documents;
   ```

3. **Common Optimizations**:
   - Use prepared statements to avoid query parsing overhead
   - Batch INSERT operations for bulk document creation
   - Use COPY for large data imports
   - Implement connection pooling with appropriate limits
   - Regular VACUUM and ANALYZE for statistics

**Qdrant Optimization Strategies**:

1. **Vector Search Optimization**:
   - Pre-filter by metadata to reduce search space
   - Use appropriate HNSW parameters (ef, m)
   - Enable mmap for large collections
   - Optimize vector dimensions and quantization

2. **Batch Operations**:
   ```python
   # Batch vector upload
   points = [
       PointStruct(id=str(i), vector=vec, payload=meta)
       for i, (vec, meta) in enumerate(vectors)
   ]
   await client.upsert(collection_name, points, batch_size=100)
   ```

3. **Payload Indexing**:
   ```python
   # Create payload index for fast filtering
   await client.create_payload_index(
       collection_name,
       field_name="document_id",
       field_type="keyword"
   )
   ```

### Query Performance Monitoring

1. **PostgreSQL Monitoring**:
   ```sql
   -- Slow query analysis
   SELECT query, mean_time, calls 
   FROM pg_stat_statements 
   ORDER BY mean_time DESC 
   LIMIT 10;
   
   -- Index usage statistics
   SELECT schemaname, tablename, indexname, idx_scan
   FROM pg_stat_user_indexes
   ORDER BY idx_scan;
   
   -- Table bloat analysis
   SELECT schemaname, tablename, 
          pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
   FROM pg_stat_user_tables
   ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
   ```

2. **Connection Pool Monitoring**:
   - Track active vs idle connections
   - Monitor connection wait times
   - Adjust pool size based on load

3. **Query Plan Analysis**:
   ```sql
   EXPLAIN (ANALYZE, BUFFERS) 
   SELECT * FROM documents 
   WHERE collection_id = ? AND status = 'completed';
   ```

### Scaling Strategies

**Horizontal Scaling**:
- Read replicas for PostgreSQL with streaming replication
- Qdrant cluster mode for distributed search
- Multiple Celery workers for parallel processing
- Load balancing across service instances

**Vertical Scaling**:
- Increased connection pool sizes (carefully tuned)
- Larger Qdrant cache settings for frequently accessed vectors
- More worker processes for CPU-bound operations
- SSD storage for improved I/O performance

**Caching Strategy**:
- Redis for frequently accessed metadata
- Application-level caching for user permissions
- Query result caching with TTL
- Qdrant's built-in caching mechanisms

## Backup and Recovery

### PostgreSQL Backup

**Automated Backup**:
```bash
# Daily backup with rotation
pg_dump $DATABASE_URL | gzip > backup/semantik_$(date +%Y%m%d).sql.gz

# Keep last 30 days
find backup/ -name "semantik_*.sql.gz" -mtime +30 -delete
```

**Point-in-Time Recovery**:
- WAL archiving enabled
- Continuous archiving to object storage
- Recovery to any point in time

### Qdrant Backup

**Collection Snapshots**:
```python
# Create snapshot
await client.create_snapshot(collection_name=collection.vector_store_name)

# Download snapshot
snapshot_info = await client.list_snapshots(collection_name)
await client.download_snapshot(collection_name, snapshot_name)
```

### Disaster Recovery

1. **PostgreSQL Recovery**:
   - Restore from latest backup
   - Apply WAL logs to target time
   - Verify data integrity

2. **Qdrant Recovery**:
   - Restore collection snapshots
   - Or rebuild from documents if needed
   - Verify vector counts match documents

## Security Considerations

### Authentication & Authorization
- Password hashing with Argon2
- JWT tokens with short expiration
- API keys for programmatic access
- Row-level security via permissions

### Data Protection
- Input validation at all layers
- SQL injection prevention via ORM
- Path traversal protection
- Rate limiting on all endpoints

### Audit Trail
- All collection modifications logged
- User actions tracked with IP
- Sensitive operations require confirmation
- Audit logs retained for compliance

## Future Enhancements

### Planned Improvements

1. **Multi-tenancy**: 
   - Database-level isolation using PostgreSQL schemas
   - Tenant-specific connection pools
   - Cross-tenant query prevention

2. **Sharding**: 
   - Partition large collections by date or hash
   - Distributed query coordination
   - Automatic shard rebalancing

3. **Full-text Search**: 
   - PostgreSQL FTS integration for metadata
   - Hybrid search combining vectors and keywords
   - Language-specific stemming and tokenization

4. **Vector Versioning**: 
   - Track embedding model changes
   - Support multiple vector representations
   - Gradual migration between models

5. **Incremental Indexing**: 
   - File modification detection
   - Delta processing for large documents
   - Change data capture (CDC) integration

6. **Collection Templates**: 
   - Predefined configurations for common use cases
   - Template marketplace
   - Custom template creation

7. **Advanced Features**:
   - Time-based document expiration
   - Geographic distribution of collections
   - Real-time collaborative indexing
   - Webhook notifications for operations

### Technical Debt and Improvements

1. **Schema Refinements**:
   - Convert string timestamps to proper DateTime types
   - Standardize UUID handling across tables
   - Add missing foreign key constraints

2. **Performance Enhancements**:
   - Implement query result caching
   - Add database connection pooling metrics
   - Optimize large JSON column storage

3. **Monitoring and Observability**:
   - Built-in query performance tracking
   - Automated index recommendations
   - Resource usage dashboards

### Migration Path

1. **Data Migration Tools**: 
   - Automated scripts for bulk operations
   - Progress tracking for large migrations
   - Rollback capabilities

2. **Version Management**: 
   - Semantic versioning for schema changes
   - Compatibility matrix maintenance
   - Automated migration testing

3. **Zero-downtime Updates**: 
   - Blue-green deployment support
   - Online schema changes
   - Gradual rollout strategies

4. **Backward Compatibility**:
   - API versioning for schema changes
   - Deprecation warnings
   - Migration guides and tooling

## Conclusion

The Semantik database architecture provides a robust foundation for semantic search with:
- Clear separation between metadata (PostgreSQL) and vectors (Qdrant)
- Collection-centric design for better organization
- Comprehensive audit and permission systems
- Scalable architecture supporting growth
- Safe migration paths and data integrity guarantees

The transition from job-centric to collection-centric architecture has created a more intuitive and maintainable system that better aligns with user mental models and supports future enhancements.