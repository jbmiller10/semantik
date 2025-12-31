# Database Architecture

## Overview

Semantik uses a hybrid database architecture:
- **PostgreSQL** for relational data (collections, operations, documents, users, auth)
- **Qdrant** for vector storage and similarity search

**Key decisions**:
- PostgreSQL is owned exclusively by the webui package
- All database operations use the repository pattern (`packages/shared/database/`)
- Vecpipe has no direct database access - it uses webui API endpoints
- The system uses a collection-centric architecture (migrated from job-centric)

## Job-Centric to Collection-Centric Migration

The system originally tracked each indexing task as a "job" with documents tied to specific job runs. This sucked because:
- Documents couldn't outlive the job that created them
- Managing document lifecycle across multiple runs was a nightmare
- No logical organization of related documents

The new collection-centric model:
- Documents belong to collections, not jobs
- Collections persist across operations
- Each collection can use different embedding models
- Clear ownership and permissions

The migration:
- `jobs` table became `operations` table
- Documents reference `collection_id` instead of job IDs
- New `collection_sources` table tracks data origins
- Operations are ephemeral, collections are persistent

## Database Distribution

### PostgreSQL
Stores all relational data: users, collections, documents, operations, auth tokens.
- Connection via `DATABASE_URL`
- Owned exclusively by webui service

### Qdrant
Stores vector embeddings for similarity search.
- Configured via `QDRANT_HOST:QDRANT_PORT` (default: localhost:6333)
- Document chunks as vectors with metadata payloads

## PostgreSQL Schema

**Configuration:**
- Connection via `DATABASE_URL`
- Alembic for migrations
- SQLAlchemy connection pooling

### Core Table Structures

#### Collections
Logical groupings of documents with shared configuration.

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

**Key points:**
- UUID primary keys
- Each collection has its own Qdrant collection
- Supports multiple embedding models and quantization
- Staging support for zero-downtime reindexing

#### Documents
Individual documents within collections.

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
    uri VARCHAR,                                -- Logical identifier (URL, path, message ID, etc.)
    source_metadata JSON,                       -- Connector-specific metadata (headers, timestamps, etc.)
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
-- Ensure logical identifiers are unique per collection when present
CREATE UNIQUE INDEX idx_documents_collection_uri_unique
    ON documents(collection_id, uri)
    WHERE uri IS NOT NULL;
```

**Key points:**
- Content hash (SHA256) for deduplication
- Unique constraint on (collection_id, content_hash)
- `uri` for connector-agnostic logical identifiers
- Soft delete via 'deleted' status
- Cascade delete when collection is removed

#### Document Artifacts
Database-backed document content for non-file sources (Git/IMAP/web), used by the document content endpoint.

```sql
CREATE TABLE document_artifacts (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR NOT NULL,              -- FK to documents (cascade delete)
    collection_id VARCHAR NOT NULL,            -- FK to collections (cascade delete)
    artifact_kind VARCHAR(20) NOT NULL DEFAULT 'primary', -- primary|preview|thumbnail
    mime_type VARCHAR(255) NOT NULL,
    charset VARCHAR(50),
    content_text TEXT,
    content_bytes BYTEA,
    content_hash VARCHAR(64) NOT NULL,
    size_bytes INTEGER NOT NULL,
    is_truncated BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
    CONSTRAINT uq_document_artifact_kind UNIQUE (document_id, artifact_kind),
    CONSTRAINT ck_content_present CHECK (content_text IS NOT NULL OR content_bytes IS NOT NULL),
    CONSTRAINT ck_artifact_kind_values CHECK (artifact_kind IN ('primary', 'preview', 'thumbnail'))
);

CREATE INDEX ix_document_artifacts_document ON document_artifacts(document_id);
CREATE INDEX ix_document_artifacts_collection ON document_artifacts(collection_id);
```

**Key points:**
- Enables serving document content when `documents.file_path` is not present/meaningful (non-file sources).
- The content endpoint prefers artifacts and falls back to filesystem serving for local directory sources.

#### Operations
Asynchronous operations on collections (formerly "jobs").

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

**Key points:**
- Tracks full lifecycle: pending → processing → completed/failed
- Stores config for reproducibility
- Links to Celery tasks
- Operations are ephemeral; collections persist

#### Collection Sources
Data sources for collections.

```sql
CREATE TABLE collection_sources (
    id SERIAL PRIMARY KEY,                      -- Auto-increment ID
    collection_id VARCHAR NOT NULL,             -- Parent collection
    source_path VARCHAR NOT NULL,               -- Path or URL
    source_type VARCHAR NOT NULL,               -- directory|web|slack|github|...
    source_config JSON,                         -- Connector-specific configuration (e.g. {"path": "/data/docs"})
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
-- Ensure a source path is only registered once per collection
ALTER TABLE collection_sources
    ADD CONSTRAINT uq_collection_source_path UNIQUE (collection_id, source_path);
```

#### Connector Secrets
Encrypted connector credentials associated with `collection_sources` (Git tokens, IMAP passwords, SSH keys).

```sql
CREATE TABLE connector_secrets (
    id SERIAL PRIMARY KEY,
    collection_source_id INTEGER NOT NULL,     -- FK to collection_sources (cascade delete)
    secret_type VARCHAR(50) NOT NULL,          -- password|token|ssh_key|ssh_passphrase
    ciphertext BYTEA NOT NULL,                 -- Fernet-encrypted payload
    key_id VARCHAR(64) NOT NULL,               -- Which CONNECTOR_SECRETS_KEY encrypted this row
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (collection_source_id) REFERENCES collection_sources(id) ON DELETE CASCADE
);

CREATE INDEX ix_connector_secrets_source ON connector_secrets(collection_source_id);
ALTER TABLE connector_secrets
    ADD CONSTRAINT uq_source_secret_type UNIQUE (collection_source_id, secret_type);
```

**Key points:**
- Secrets are encrypted at rest (Fernet) using `CONNECTOR_SECRETS_KEY`.
- Secrets are never returned via API responses; only presence indicators are exposed (e.g., `has_token`).

#### Collection Permissions
Access control for collections.

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

#### Supporting Tables

**Users** - Authentication and profiles:
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

**Refresh Tokens** - JWT token management:
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

**API Keys** - Programmatic authentication:
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

**Collection Audit Log** - Action tracking:
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

- **Tables**: snake_case, plural for entities (`users`, `collections`)
- **Columns**: snake_case
  - Foreign keys: `{table}_id`
  - Timestamps: `{action}_at`
  - Booleans: `is_{state}`
- **Indexes**: `idx_{table}_{column}` or `uq_{table}_{columns}`
- **Enums**: PascalCase types, UPPER_CASE values

### Operation Lifecycle

```
PENDING → PROCESSING → COMPLETED
                    ↘ FAILED
                    ↘ CANCELLED
```

**States:**
- `PENDING`: Created, waiting for worker
- `PROCESSING`: Worker executing
- `COMPLETED`: Success
- `FAILED`: Error occurred
- `CANCELLED`: User cancelled

**Types:**
- `INDEX`: Initial indexing
- `APPEND`: Add documents
- `REINDEX`: Rebuild with new settings
- `REMOVE_SOURCE`: Remove documents from a source
- `DELETE`: Delete collection

### Soft Delete

Documents use soft delete (status='deleted') for safety:
- Preserves audit trail
- Vectors removed from Qdrant but metadata retained
- Enables recovery/undelete

Collections use hard delete (CASCADE) for clean removal.

### Collection Sources

Track document origins for incremental indexing and data lineage.

**Source types:** `directory`, `file`, `url`, `github`

Each source tracks document count, size, and last indexed timestamp.

### Transaction Patterns

All operations use transactions for ACID compliance. Key patterns:

- **Collection creation**: Create record → Initialize Qdrant → Update status
- **Document processing**: Update status → Process → Update counts
- **Operation completion**: Update status → Record metrics → Audit log

### Migrations

Alembic for schema versioning with auto-generation and rollback.

```bash
alembic revision --autogenerate -m "Description"
alembic upgrade head
alembic downgrade -1  # if needed
```

Always review auto-generated migrations before applying.

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

### Configuration

- **indexing_threshold**: 20000 (flat → HNSW)
- **memmap_threshold**: 50000 (memory → disk)
- **ef_construction**: 512, **m**: 16 (HNSW params)
- Batch uploads (100 vectors), async operations

## Design Rationale

**Why collection-centric?**
- Users think "my docs", not "job #123"
- Each collection can use different models
- Incremental updates (append/remove) without rebuilding
- Clear permission boundaries

**Why soft delete for documents?**
- Recovery from mistakes
- Audit trail for compliance
- Vectors removed immediately, metadata retained

## Data Models

See SQLAlchemy models in `packages/shared/database/models.py` for implementation details.

## Database Operations

### Repository Pattern

All database access uses repositories in `packages/shared/database/`:
- `CollectionRepository`
- `DocumentRepository`
- `OperationRepository`
- `UserRepository`
- `AuthTokenRepository`

Repositories handle connection lifecycle and provide clean interfaces for data access.

### Connection Management

- **PostgreSQL**: Async pooling via SQLAlchemy, automatic retry
- **Qdrant**: Singleton async client with health checks

All operations use transactions for ACID guarantees.

## Data Flow

**Indexing:**
1. User creates collection (WebUI → CollectionRepository)
2. User starts operation (WebUI → OperationRepository → Celery)
3. Worker scans files via connector
4. Documents registered with deduplication (DocumentRepository)
5. Text extraction and chunking
6. Embedding generation and batch upload to Qdrant
7. Operation completed

**Search:**
1. User queries WebUI (permission check)
2. WebUI proxies to Vecpipe
3. Vecpipe generates embedding and searches Qdrant
4. Results enriched with PostgreSQL metadata

## Performance

**PostgreSQL:**
- B-tree indexes on common queries
- Partial indexes for filtered queries
- Cursor-based pagination
- Connection pooling
- Regular VACUUM and ANALYZE

**Qdrant:**
- Batch vector uploads (100 per batch)
- Payload indexes for filtering
- HNSW params tuned for speed/accuracy tradeoff
- Mmap for large collections

**Scaling:**
- PostgreSQL read replicas
- Qdrant cluster mode
- Multiple Celery workers
- Redis caching for metadata

## Backup and Recovery

**PostgreSQL:**
```bash
pg_dump $DATABASE_URL | gzip > backup/semantik_$(date +%Y%m%d).sql.gz
```

**Qdrant:**
```python
await client.create_snapshot(collection_name)
```

Qdrant collections can be rebuilt from source documents if snapshots are unavailable.

## Security

- **Auth**: bcrypt password hashing, JWT tokens, API keys
- **Data**: Input validation, ORM prevents SQL injection
- **Audit**: All collection modifications logged with user/IP tracking

## Future Ideas

- Multi-tenancy with schema-level isolation
- Sharding for large collections
- Hybrid search (vectors + PostgreSQL FTS)
- Vector versioning for model migrations
- Incremental indexing with change detection
- Collection templates
- Webhook notifications
