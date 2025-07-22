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
    status VARCHAR NOT NULL DEFAULT 'pending',  -- pending|ready|processing|error|degraded
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    meta JSON,                                  -- Additional metadata
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
- Status tracking for collection health monitoring

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
    status VARCHAR DEFAULT 'pending',           -- pending|processing|completed|failed
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
```

**Key Points**:
- Content hash enables duplicate detection across collections
- Tracks processing status and chunk creation
- Cascade delete ensures cleanup when collection is removed

#### 3. Operations Table
Manages asynchronous operations on collections.

```sql
CREATE TABLE operations (
    id SERIAL PRIMARY KEY,                      -- Auto-increment ID
    uuid VARCHAR UNIQUE NOT NULL,               -- UUID for external reference
    collection_id VARCHAR NOT NULL,             -- Target collection
    user_id INTEGER NOT NULL,                   -- Initiating user
    type VARCHAR NOT NULL,                      -- index|append|reindex|remove_source|delete
    status VARCHAR NOT NULL DEFAULT 'pending',  -- pending|processing|completed|failed|cancelled
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
- Tracks all async operations with full lifecycle
- Stores operation configuration for reproducibility
- Links to Celery tasks for distributed processing

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
    id VARCHAR PRIMARY KEY,
    user_id INTEGER NOT NULL,
    token_hash VARCHAR UNIQUE NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_revoked BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
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
            └──────< (N) collection_audit_log

collection_sources (1) ──────< (N) documents

operations (1) ──────< (N) collection_audit_log
           └──────< (N) operation_metrics
```

### Migration Strategy

The system uses Alembic for database migrations:
1. Version control for schema changes
2. Automatic migration generation from model changes
3. Rollback capability for failed migrations
4. Migration history tracking

Example migration workflow:
```bash
# Generate new migration
alembic revision --autogenerate -m "Add collection status field"

# Apply migrations
alembic upgrade head

# Rollback if needed
alembic downgrade -1
```

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
    created_at: datetime
    updated_at: datetime
    meta: Optional[dict]
    
    # Relationships
    owner: User
    documents: List[Document]
    operations: List[Operation]
    permissions: List[CollectionPermission]
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

**PostgreSQL**:
- B-tree indexes on UUIDs and foreign keys
- Partial indexes for status queries
- Composite indexes for complex filters
- Query plan analysis and optimization

**Qdrant**:
- Pre-filtering with metadata before vector search
- Payload indexing for fast filtering
- Quantization for memory efficiency
- Batch search for multiple collections

### Scaling Strategies

**Horizontal Scaling**:
- Read replicas for PostgreSQL
- Qdrant cluster mode for distributed search
- Multiple Celery workers for parallel processing

**Vertical Scaling**:
- Increased connection pool sizes
- Larger Qdrant cache settings
- More worker processes

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
1. **Multi-tenancy**: Database-level isolation
2. **Sharding**: Partition large collections
3. **Full-text Search**: PostgreSQL FTS integration
4. **Vector Versioning**: Track embedding model changes
5. **Incremental Indexing**: Only process changed documents
6. **Collection Templates**: Predefined configurations
7. **Backup Automation**: Managed backup service
8. **Performance Analytics**: Query performance tracking

### Migration Path
1. **Data Migration Tools**: Scripts for bulk operations
2. **Version Management**: Track schema versions
3. **Zero-downtime Updates**: Blue-green deployments
4. **Rollback Procedures**: Safe version downgrades