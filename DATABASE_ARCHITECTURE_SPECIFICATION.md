# Semantik Database Architecture Specification

## Executive Summary

Semantik employs a sophisticated dual-database architecture combining PostgreSQL for relational metadata and Qdrant for vector storage. The system is designed for scalability, performance, and maintainability through strategic use of table partitioning, connection pooling, and blue-green deployment patterns.

## 1. Database Systems Overview

### 1.1 PostgreSQL (Metadata & Relational Data)
- **Version Requirements**: PostgreSQL 11+ (12+ recommended for GENERATED columns)
- **Primary Use**: User management, collections, documents, operations, audit logs, and chunked text storage
- **Connection Strategy**: Async SQLAlchemy with connection pooling
- **Partitioning**: LIST partitioning on chunks table (100 partitions)

### 1.2 Qdrant (Vector Database)
- **Primary Use**: Semantic vector storage and similarity search
- **Collections**: One Qdrant collection per Semantik collection
- **Blue-Green Strategy**: Staging collections for zero-downtime reindexing
- **Distance Metric**: Cosine similarity (default)

## 2. PostgreSQL Schema Architecture

### 2.1 Core Tables

#### Users Table
```sql
users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR UNIQUE NOT NULL (indexed),
    email VARCHAR UNIQUE NOT NULL (indexed),
    full_name VARCHAR,
    hashed_password VARCHAR NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ
)
```

#### Collections Table
```sql
collections (
    id VARCHAR PRIMARY KEY (UUID),
    name VARCHAR UNIQUE NOT NULL (indexed),
    description TEXT,
    owner_id INTEGER REFERENCES users(id) NOT NULL (indexed),
    vector_store_name VARCHAR UNIQUE NOT NULL,
    embedding_model VARCHAR NOT NULL,
    quantization VARCHAR DEFAULT 'float16',
    chunk_size INTEGER DEFAULT 1000,
    chunk_overlap INTEGER DEFAULT 200,
    chunking_strategy VARCHAR,
    chunking_config JSONB,
    is_public BOOLEAN DEFAULT FALSE (indexed),
    status ENUM(CollectionStatus) DEFAULT 'PENDING' (indexed),
    status_message TEXT,
    qdrant_collections JSONB,
    qdrant_staging JSONB,
    document_count INTEGER DEFAULT 0,
    vector_count INTEGER DEFAULT 0,
    total_size_bytes INTEGER DEFAULT 0,
    default_chunking_config_id INTEGER REFERENCES chunking_configs(id),
    chunks_total_count INTEGER DEFAULT 0,
    chunking_completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    meta JSONB
)
```

#### Documents Table
```sql
documents (
    id VARCHAR PRIMARY KEY (UUID),
    collection_id VARCHAR REFERENCES collections(id) ON DELETE CASCADE (indexed),
    source_id INTEGER REFERENCES collection_sources(id),
    file_path VARCHAR NOT NULL,
    file_name VARCHAR NOT NULL,
    file_size INTEGER NOT NULL,
    mime_type VARCHAR,
    content_hash VARCHAR NOT NULL (indexed),
    status ENUM(DocumentStatus) DEFAULT 'PENDING' (indexed),
    error_message TEXT,
    chunk_count INTEGER DEFAULT 0,
    chunking_config_id INTEGER REFERENCES chunking_configs(id),
    chunks_count INTEGER DEFAULT 0,
    chunking_started_at TIMESTAMPTZ,
    chunking_completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    meta JSONB
)

Composite Indexes:
- UNIQUE(collection_id, content_hash)
- INDEX(collection_id, chunking_completed_at)
```

#### Chunks Table (Partitioned)
```sql
chunks (
    id BIGSERIAL,
    collection_id VARCHAR NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    partition_key INTEGER NOT NULL (computed via trigger/generated column),
    document_id VARCHAR REFERENCES documents(id) ON DELETE CASCADE,
    chunking_config_id INTEGER REFERENCES chunking_configs(id),
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    start_offset INTEGER,
    end_offset INTEGER,
    token_count INTEGER,
    embedding_vector_id VARCHAR,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, collection_id, partition_key)
) PARTITION BY LIST (partition_key);

-- 100 partitions created: chunks_part_00 through chunks_part_99
-- partition_key = abs(hashtext(collection_id)) % 100
```

#### Operations Table
```sql
operations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid VARCHAR UNIQUE NOT NULL,
    collection_id VARCHAR REFERENCES collections(id) ON DELETE CASCADE (indexed),
    user_id INTEGER REFERENCES users(id) NOT NULL (indexed),
    type ENUM(OperationType) NOT NULL (indexed),
    status ENUM(OperationStatus) DEFAULT 'PENDING' (indexed),
    task_id VARCHAR,
    config JSONB NOT NULL,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW() (indexed),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    meta JSONB
)
```

### 2.2 Supporting Tables

- **api_keys**: API key management with hashed keys
- **refresh_tokens**: JWT refresh token storage
- **collection_permissions**: Fine-grained access control
- **collection_sources**: Track data sources for collections
- **collection_audit_log**: Audit trail for collection operations
- **collection_resource_limits**: Per-collection quotas
- **operation_metrics**: Performance metrics for operations
- **chunking_strategies**: Available chunking strategies
- **chunking_configs**: Deduplicated chunking configurations

### 2.3 Partitioning Strategy

#### Chunk Table Partitioning
- **Method**: LIST partitioning on `partition_key` column
- **Partition Count**: 100 partitions (chunks_part_00 to chunks_part_99)
- **Key Computation**: `abs(hashtext(collection_id)) % 100`
- **Implementation**: 
  - PostgreSQL < 12: Trigger-based computation
  - PostgreSQL 12+: GENERATED column (recommended)

#### Partition Distribution
- Even distribution achieved through PostgreSQL's hashtext() function
- Each collection's chunks go to a single partition
- Enables efficient partition pruning for collection-specific queries

#### Per-Partition Indexes
Each partition has:
- `idx_chunks_part_XX_collection` on collection_id
- `idx_chunks_part_XX_created` on created_at
- `idx_chunks_part_XX_chunk_index` on (collection_id, chunk_index)
- `idx_chunks_part_XX_document` on document_id (conditional, WHERE NOT NULL)

### 2.4 Monitoring Views

#### partition_health
Monitors partition health with metrics:
- Row count and size per partition
- Deviation from average
- Dead tuple ratio
- Vacuum status
- Hot/Cold partition identification

#### partition_distribution
Analyzes chunk distribution:
- Partitions in use
- Average/min/max chunks per partition
- Skew ratio calculation
- Rebalancing recommendations

#### collection_chunking_stats (Materialized)
Aggregated statistics per collection:
- Document and chunk counts
- Average tokens per chunk
- Partition assignment
- Refresh via `refresh_collection_chunking_stats()`

## 3. Alembic Migration Strategy

### 3.1 Migration History
1. `005a8fe3aedc`: Initial unified schema
2. `52db15bd2686`: Add chunking tables with partitioning
3. `6596eda04faa`: Fix chunk table schema issues
4. `ae558c9e183f`: Implement 100 LIST partitions
5. `8547ff31e80c`: Safe partition migration with data preservation
6. `8f67aa430c5d`: Add partition monitoring views
7. `db003`: Replace trigger with GENERATED column (PostgreSQL 12+)
8. `db004`: Add chunking performance indexes

### 3.2 Migration Patterns

#### Safe Partition Migration
```python
# 1. Clean up dependencies
cleanup_chunks_dependencies(conn)

# 2. Drop old structure
conn.execute(text("DROP TABLE IF EXISTS chunks CASCADE"))

# 3. Create new partitioned table
CREATE TABLE chunks (...) PARTITION BY LIST (partition_key)

# 4. Create partitions with indexes
FOR i IN 0..99:
    CREATE TABLE chunks_part_XX PARTITION OF chunks FOR VALUES IN (i)
    CREATE INDEX on each partition

# 5. Create monitoring infrastructure
CREATE VIEW partition_health AS ...
CREATE MATERIALIZED VIEW collection_chunking_stats AS ...
```

#### Data Migration Strategy
- Use `INSERT INTO ... SELECT` for data preservation
- Group by partition key for efficiency
- Validate partition key computation post-migration

## 4. Qdrant Integration

### 4.1 Collection Management

#### Naming Convention
```python
vector_store_name = f"col_{collection_id.replace('-', '_')}"
staging_name = f"staging_{base_name}_{timestamp}"
```

#### Collection Configuration
```python
VectorParams(
    size=vector_dimension,  # Model-specific
    distance=Distance.COSINE,
    quantization="float16"  # Configurable per collection
)

OptimizerConfig(
    indexing_threshold=20000,
    memmap_threshold=0
)
```

### 4.2 Blue-Green Deployment

#### Reindexing Workflow
1. Create staging collection with timestamp
2. Index new data into staging
3. Validate staging collection
4. Atomic swap: staging → production
5. Clean up old collection

#### Orphaned Collection Cleanup
- Identify collections not in active set
- Check staging collection age (>24 hours)
- Safe deletion with vector count logging

### 4.3 Search Integration

#### Query Flow
1. PostgreSQL: Validate collection access
2. PostgreSQL: Get collection metadata
3. Qdrant: Vector similarity search
4. PostgreSQL: Enrich with document metadata
5. Return aggregated results

#### Multi-Collection Search
- Parallel search across collections
- Result aggregation and re-ranking
- Timeout handling with retry logic

## 5. Database Access Patterns

### 5.1 Connection Pooling

#### PostgreSQL Configuration
```python
DB_POOL_SIZE = 20
DB_MAX_OVERFLOW = 40
DB_POOL_TIMEOUT = 30
DB_POOL_RECYCLE = 3600
DB_POOL_PRE_PING = True
```

#### Async Session Management
```python
async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False
)
```

### 5.2 Repository Pattern

#### Base Repository Features
- Partition-aware operations
- Bulk insert optimization
- Automatic validation
- Retry logic for transient failures

#### Partition-Aware Operations
```python
# Always include partition key in queries
query = select(Chunk).where(
    and_(
        Chunk.collection_id == collection_id,  # Partition key
        Chunk.document_id == document_id
    )
)

# Group bulk operations by partition
chunks_by_collection = group_by_partition_key(chunks)
for collection_id, chunks in chunks_by_collection.items():
    bulk_insert_mappings(Chunk, chunks)
```

### 5.3 Query Optimization

#### Partition Pruning
- Always include collection_id in chunk queries
- Use ChunkPartitionHelper for query construction
- Monitor partition access patterns

#### Index Usage
- Composite indexes for common query patterns
- Partial indexes for NULL columns
- BRIN indexes for time-series data (created_at)

## 6. Transaction Management

### 6.1 Transaction Boundaries

#### Service Layer Transactions
```python
async with session.begin():
    # All repository operations within transaction
    collection = await collection_repo.create(...)
    operation = await operation_repo.create(...)
    # Automatic commit on success, rollback on exception
```

### 6.2 Consistency Guarantees

#### PostgreSQL → Qdrant Sync
1. Update PostgreSQL metadata (transactional)
2. Update Qdrant vectors (eventual consistency)
3. Update PostgreSQL with Qdrant references
4. Handle partial failures with status tracking

## 7. Performance Considerations

### 7.1 Chunk Table Performance

#### Partition Benefits
- Query performance: ~100x improvement for collection-specific queries
- Maintenance: VACUUM/ANALYZE per partition
- Parallel query execution across partitions

#### Bulk Operation Guidelines
- Batch size limit: 10,000 items
- Group by collection_id for partition efficiency
- Use COPY for large imports

### 7.2 Query Performance

#### Common Optimizations
```sql
-- Efficient collection chunk count
SELECT COUNT(*) FROM chunks 
WHERE collection_id = ? -- Partition pruning

-- Avoid cross-partition joins
-- Bad: JOIN without partition key
-- Good: Include collection_id in all joins
```

### 7.3 Monitoring & Maintenance

#### Health Checks
```python
# Partition health monitoring
SELECT * FROM partition_health 
WHERE partition_status != 'NORMAL';

# Skew detection
SELECT * FROM analyze_partition_skew();

# Performance metrics
SELECT * FROM get_partition_statistics(?);
```

#### Maintenance Tasks
- Daily: Refresh materialized views
- Weekly: Analyze partition distribution
- Monthly: Clean orphaned Qdrant collections
- Quarterly: Review partition rebalancing needs

## 8. Security & Validation

### 8.1 Input Validation

#### UUID Validation
- Format: UUID v4 only
- Case normalization to lowercase
- Regex pattern matching

#### Partition Key Validation
- Collection ID format validation
- Partition key range check (0-99)
- Batch size limits (10,000 items)

### 8.2 SQL Injection Prevention
- Parameterized queries throughout
- SQLAlchemy ORM for query construction
- Input sanitization for dynamic queries

## 9. Disaster Recovery

### 9.1 Backup Strategy

#### PostgreSQL
- Daily full backups
- Continuous WAL archiving
- Point-in-time recovery capability

#### Qdrant
- Collection snapshots
- Export/import for migration
- Staging collections as backup during reindex

### 9.2 Recovery Procedures

#### Partition Corruption Recovery
1. Identify affected partition
2. Create temporary table from partition
3. Rebuild partition with indexes
4. Validate data integrity
5. Re-enable partition in main table

## 10. Future Considerations

### 10.1 Scaling Options

#### Horizontal Partitioning
- Increase to 256 partitions if needed
- Consider range partitioning for time-series
- Evaluate partition-wise joins

#### Qdrant Clustering
- Multi-node Qdrant deployment
- Sharding by collection
- Read replicas for search scaling

### 10.2 Migration to PostgreSQL 12+

Benefits of upgrading:
- GENERATED columns for partition_key
- Improved partition pruning
- Better parallel query execution
- Reduced trigger overhead (2-3ms per insert)

Migration path:
1. Run `detect_implementation()` health check
2. Execute migration db003 if PostgreSQL 12+
3. Verify partition key computation
4. Monitor performance improvements

## Appendix A: Critical SQL Queries

### A.1 Partition Key Computation
```sql
-- Trigger function (PostgreSQL < 12)
CREATE FUNCTION compute_partition_key() RETURNS TRIGGER AS $$
BEGIN
    NEW.partition_key := abs(hashtext(NEW.collection_id::text)) % 100;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- GENERATED column (PostgreSQL 12+)
ALTER TABLE chunks 
ALTER COLUMN partition_key 
SET GENERATED ALWAYS AS (abs(hashtext(collection_id::text)) % 100) STORED;
```

### A.2 Partition Distribution Analysis
```sql
WITH partition_stats AS (
    SELECT 
        partition_key,
        COUNT(*) as chunk_count,
        COUNT(DISTINCT collection_id) as collections
    FROM chunks
    GROUP BY partition_key
)
SELECT 
    partition_key,
    chunk_count,
    collections,
    ROUND(chunk_count::NUMERIC / AVG(chunk_count) OVER () * 100, 2) as relative_size
FROM partition_stats
ORDER BY chunk_count DESC;
```

## Appendix B: Connection Strings

### B.1 PostgreSQL
```
# Async (runtime)
postgresql+asyncpg://user:pass@host:5432/semantik

# Sync (migrations)
postgresql+psycopg2://user:pass@host:5432/semantik

# Connection parameters
?server_settings=jit:off&statement_timeout:30000
```

### B.2 Qdrant
```
# Local development
http://localhost:6333

# Production
http://qdrant:6333
```

## Appendix C: Environment Variables

```bash
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=semantik
POSTGRES_USER=semantik
POSTGRES_PASSWORD=password

# Connection Pool
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600

# Query Settings
DB_ECHO=false
DB_QUERY_TIMEOUT=30

# Partitioning
CHUNK_PARTITION_COUNT=100
```

---

This specification provides a complete blueprint for recreating the Semantik database architecture. All design decisions, patterns, and implementation details are documented to ensure maintainability and facilitate onboarding of new team members.