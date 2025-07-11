# Database Architecture Documentation

## Overview

The Document Embedding System uses a hybrid database architecture combining:
- **SQLite** for relational data (jobs, files, users, authentication)
- **Qdrant** for vector storage and similarity search

This architecture separates transactional/metadata storage from high-performance vector operations, following the architectural principle of using the right tool for each job.

**Important**: The SQLite database is owned and managed exclusively by the webui service. All database operations from vecpipe must go through the webui API endpoints.

## Database Distribution Strategy

### SQLite (Relational Data)
- **Purpose**: Stores structured metadata, job tracking, user management, authentication
- **Location**: `data/webui.db` (primary), with backups at `data/webui.db.backup`
- **Why SQLite**: Lightweight, serverless, perfect for single-instance deployments
- **Data Types**: User accounts, job metadata, file processing status, authentication tokens

### Qdrant (Vector Database)
- **Purpose**: Stores document embeddings and enables similarity search
- **Location**: Configured via `QDRANT_HOST:QDRANT_PORT` (default: localhost:6333)
- **Why Qdrant**: High-performance vector search, supports various distance metrics, scalable
- **Data Types**: Document chunks as vectors with metadata payloads

## SQLite Database Schema

### Location and Configuration
- **Primary Database**: `data/webui.db`
- **Ownership**: Exclusively owned by webui service
- **Access Pattern**: Repository pattern via `packages/shared/database/`
- **Legacy Access**: Deprecated wrappers in `packages/shared/database/legacy_wrappers.py`
- **Backup Strategy**: Manual backups to `data/webui.db.backup`
- **Connection Management**: Handled by `packages/webui/database.py`
- **Initialization**: Auto-creates tables on first run via `init_db()`

### Table Structures

#### 1. Jobs Table
Tracks embedding job lifecycle and configuration.

```sql
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,                    -- UUID for job identification
    name TEXT NOT NULL,                     -- Human-readable job name
    description TEXT,                       -- Optional job description
    status TEXT NOT NULL,                   -- created|scanning|processing|completed|failed|cancelled
    created_at TEXT NOT NULL,               -- ISO timestamp
    updated_at TEXT NOT NULL,               -- ISO timestamp
    directory_path TEXT NOT NULL,           -- Source directory for files
    model_name TEXT NOT NULL,               -- Embedding model identifier
    chunk_size INTEGER,                     -- Token size for chunks
    chunk_overlap INTEGER,                  -- Token overlap between chunks
    batch_size INTEGER,                     -- Batch size for embedding
    vector_dim INTEGER,                     -- Dimension of output vectors
    quantization TEXT DEFAULT 'float32',    -- Vector quantization type
    instruction TEXT,                       -- Optional instruction prompt
    total_files INTEGER DEFAULT 0,          -- Total files to process
    processed_files INTEGER DEFAULT 0,      -- Successfully processed files
    failed_files INTEGER DEFAULT 0,         -- Failed file count
    current_file TEXT,                      -- Currently processing file
    start_time TEXT,                        -- Job start timestamp
    error TEXT,                             -- Error message if failed
    user_id TEXT                            -- Future: link to user (not enforced)
);
```

**Key Points**:
- Status transitions: created → scanning → processing → completed/failed/cancelled
- Supports multiple embedding models with different vector dimensions
- Tracks detailed progress for real-time UI updates

#### 2. Files Table
Tracks individual file processing within jobs.

```sql
CREATE TABLE files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,   -- Auto-increment ID
    job_id TEXT NOT NULL,                   -- Foreign key to jobs table
    path TEXT NOT NULL,                     -- Full file path
    size INTEGER NOT NULL,                  -- File size in bytes
    modified TEXT NOT NULL,                 -- File modification timestamp
    extension TEXT NOT NULL,                -- File extension (.pdf, .docx, etc.)
    hash TEXT,                              -- SHA256 hash for change detection
    doc_id TEXT,                            -- MD5 hash of path (16 chars)
    status TEXT DEFAULT 'pending',          -- pending|processing|completed|failed
    error TEXT,                             -- Error message if failed
    chunks_created INTEGER DEFAULT 0,       -- Number of chunks created
    vectors_created INTEGER DEFAULT 0,      -- Number of vectors created
    FOREIGN KEY (job_id) REFERENCES jobs(id)
);

-- Indexes for performance
CREATE INDEX idx_files_job_id ON files(job_id);
CREATE INDEX idx_files_status ON files(status);
CREATE INDEX idx_files_doc_id ON files(doc_id);
```

**Key Points**:
- doc_id is MD5(file_path)[:16] for consistent identification
- Tracks granular progress (chunks and vectors separately)
- Hash field enables change detection for incremental updates

#### 3. Users Table
Manages user accounts for authentication.

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,   -- User ID
    username TEXT UNIQUE NOT NULL,          -- Unique username
    email TEXT UNIQUE NOT NULL,             -- Unique email
    full_name TEXT,                         -- Optional full name
    hashed_password TEXT NOT NULL,          -- BCrypt hashed password
    is_active BOOLEAN DEFAULT 1,            -- Account active status
    created_at TEXT NOT NULL,               -- Registration timestamp
    last_login TEXT                         -- Last login timestamp
);

-- Indexes for auth lookups
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
```

#### 4. Refresh Tokens Table
Manages JWT refresh tokens for persistent sessions.

```sql
CREATE TABLE refresh_tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,   -- Token ID
    user_id INTEGER NOT NULL,               -- Foreign key to users
    token_hash TEXT UNIQUE NOT NULL,        -- BCrypt hash of token
    expires_at TEXT NOT NULL,               -- Token expiration timestamp
    created_at TEXT NOT NULL,               -- Token creation timestamp
    is_revoked BOOLEAN DEFAULT 0,           -- Revocation status
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Index for token lookup
CREATE INDEX idx_refresh_tokens_hash ON refresh_tokens(token_hash);
```

### Database Relationships

```
users (1) ──────< (N) refresh_tokens
        │
        └───────< (N) jobs (via user_id)
                        │
                        ├───────< (N) files
                        │
                        └───────< (N) jobs (via parent_job_id)
```

### Migration Strategy

The system uses an inline migration approach in `database.py`:
1. On startup, checks for table existence
2. Uses PRAGMA table_info to detect missing columns
3. Applies ALTER TABLE commands to add new columns
4. Maintains backward compatibility by using defaults

Example migrations applied:
- Added `vector_dim`, `quantization`, `instruction` to jobs table
- Added `doc_id` to files table
- Added `start_time` to jobs table

## Qdrant Vector Database

### Collection Structure

#### Primary Collection: `work_docs`
Stores document embeddings with rich metadata.

**Vector Configuration**:
```python
VectorParams(
    size=vector_dim,           # Varies by model (384-1024)
    distance=Distance.COSINE   # Cosine similarity
)
```

**Point Structure**:
```python
PointStruct(
    id=str,                    # Unique identifier (UUID or doc_id_chunk)
    vector=List[float],        # Embedding vector
    payload={
        "path": str,           # Source file path
        "doc_id": str,         # Document identifier
        "chunk_id": str,       # Chunk identifier
        "content": str,        # Text content
        "metadata": {          # Additional metadata
            "job_id": str,
            "model": str,
            "created_at": str,
            ...
        }
    }
)
```

#### Metadata Collection: `_collection_metadata`
Stores metadata about other collections.

**Purpose**: Track which embedding model was used for each collection
**Structure**: Small vectors (size=4) with collection metadata in payload

### Indexing Strategy

Qdrant uses HNSW (Hierarchical Navigable Small World) index by default:
- **indexing_threshold**: 20000 (switches from flat to HNSW index)
- **memmap_threshold**: 0 (keeps vectors in memory for speed)

### Performance Optimization

1. **Batch Operations**: Upload vectors in batches of 4000
2. **Connection Pooling**: Singleton QdrantConnectionManager
3. **Retry Logic**: Exponential backoff for resilience
4. **Async Operations**: Uses AsyncQdrantClient for search

## Data Models (Pydantic Schemas)

### FileInfo Model
```python
class FileInfo(BaseModel):
    path: str
    size: int
    modified: str
    extension: str
    hash: str | None = None
```

### Job Status Model
```python
class JobStatus(BaseModel):
    id: str
    name: str
    status: str
    created_at: str
    updated_at: str
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    current_file: str | None = None
    error: str | None = None
    model_name: str
    directory_path: str
    quantization: str | None = None
    batch_size: int | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
```

## Database Operations

### Repository Pattern Implementation

The system now uses a repository pattern for all database operations:

**Repository Classes** (in `packages/shared/database/`):
- `JobRepository`: Manages job-related operations
- `FileRepository`: Manages file-related operations
- `UserRepository`: Manages user accounts and authentication
- `CollectionRepository`: Manages Qdrant collection metadata

**Usage Example**:
```python
from shared.database import create_job_repository

with create_job_repository() as repo:
    job = repo.create(directory_path="/docs", user_id="user123")
    repo.update_status(job.id, "processing")
```

### Connection Management

**SQLite**:
- Direct connection per operation (no pooling needed)
- Thread-safe with proper transaction handling
- Connection string: `sqlite:///data/webui.db`
- Access only through repository pattern

**Qdrant**:
- Singleton connection manager with retry logic
- Connection verification before operations
- Exponential backoff: 1s, 2s, 4s, 8s (max 3 retries)

### Transaction Patterns

**Job Creation Flow**:
1. Create job record with status='created'
2. Scan directory and add files
3. Update job total_files count
4. Process files individually
5. Update file status and job progress
6. Final job status update

**Atomic Operations**:
- All SQLite operations use implicit transactions
- File status updates are atomic
- Job statistics computed from file aggregations

### Error Handling

**SQLite Errors**:
- Unique constraint violations (duplicate users/jobs)
- Foreign key violations (orphaned files)
- Graceful handling with meaningful error messages

**Repository Pattern Benefits**:
- Centralized error handling
- Consistent transaction management
- Type-safe operations with proper models
- Easy to mock for testing

**Qdrant Errors**:
- Connection failures trigger retry mechanism
- Collection not found errors handled gracefully
- Batch upload failures don't lose entire job

## Data Flow

### Document Processing Pipeline

```
1. User uploads directory
   └─> WebUI creates job via JobRepository
   
2. Scan directory for files
   └─> WebUI creates file records via FileRepository
   
3. Extract and chunk text (using shared.text_processing)
   └─> WebUI updates file status via FileRepository
   
4. Generate embeddings (using shared.embedding)
   └─> Batch upload to Qdrant
   └─> WebUI updates vectors_created via FileRepository
   
5. Complete job
   └─> WebUI updates job status via JobRepository
   └─> Vectors available for search (Qdrant)
```

### Search Flow

```
1. User enters query in WebUI
   └─> WebUI proxies to Vecpipe Search API
   
2. Vecpipe generates query embedding
   └─> Uses shared.embedding service
   └─> Searches Qdrant for similar vectors
   
3. Vecpipe returns results to WebUI
   └─> WebUI enriches with file metadata (from SQLite)
   └─> Format for UI display
```

## Performance Considerations

### Query Optimization

**SQLite**:
- Indexes on foreign keys and frequently queried fields
- Compound queries use indexed fields first
- PRAGMA optimizations for read-heavy workload

**Qdrant**:
- Pre-filtering with metadata before vector search
- Hybrid search combines vector + text matching
- Pagination support for large result sets

### Batch Operations

**File Processing**:
- Process files in parallel (ThreadPoolExecutor)
- Batch embedding generation (configurable size)
- Batch vector uploads (4000 points/batch)

**Database Writes**:
- Bulk insert for initial file records
- Batch status updates during processing
- Transaction batching for related updates

## Backup and Recovery

### SQLite Backup

**Manual Backup**:
```bash
cp data/webui.db data/webui.db.backup
```

**Automated Backup** (not implemented):
- Could use SQLite backup API
- Schedule regular snapshots
- Rotate old backups

### Qdrant Backup

**Collection Snapshot**:
```python
client.create_snapshot(collection_name="work_docs")
```

**Recovery Process**:
1. Restore SQLite from backup
2. Recreate Qdrant collections
3. Re-ingest vectors from parquet files

### Data Integrity

**Consistency Checks**:
- Job total_files matches COUNT(*) from files
- All completed files have vectors in Qdrant
- No orphaned vectors without file records

**Recovery Tools**:
- `reset_database()`: Clean slate restart
- Cleanup service removes orphaned data
- Parquet files preserve original embeddings

## Security Considerations

### Authentication
- Passwords hashed with BCrypt (cost factor 12)
- JWT tokens with configurable expiration
- Refresh token rotation for security

### Data Protection
- File paths validated to prevent traversal
- SQL injection prevented via parameterized queries
- User input sanitized before storage

### Access Control
- Authentication required for all operations
- Job isolation (future: user-specific collections)
- Rate limiting on search endpoints

## Future Enhancements

### Planned Improvements
1. **Multi-tenancy**: User-specific collections in Qdrant
2. **Audit Logging**: Track all database operations
3. **Incremental Updates**: Only process changed files
4. **Collection Management**: UI for managing vector collections
5. **Advanced Search**: Faceted search with metadata filters
6. **Backup Automation**: Scheduled backups with retention
7. **Performance Monitoring**: Query performance tracking
8. **Data Versioning**: Track embedding model versions
9. **Migration to PostgreSQL**: For better concurrency and scaling
10. **Remove Legacy Wrappers**: Complete migration from legacy database functions

### Scalability Path
1. **SQLite → PostgreSQL**: For concurrent write scaling
2. **Qdrant Clustering**: For distributed vector search
3. **Read Replicas**: For search performance
4. **Caching Layer**: Redis for frequent queries