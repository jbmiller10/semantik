# Database Layer Architecture

> **Location:** `packages/shared/database/`

## Overview

The database layer provides PostgreSQL-backed persistence for all Semantik metadata using SQLAlchemy 2.0 with async support. It implements the Repository pattern for data access abstraction.

## Core Models

### User
```python
class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str | None]
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime]
    last_login: Mapped[datetime | None]
```

### Collection
```python
class Collection(Base):
    __tablename__ = "collections"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None]
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    status: Mapped[CollectionStatus]
    vector_store_name: Mapped[str]  # Qdrant collection name
    embedding_model: Mapped[str]
    embedding_dimension: Mapped[int]
    sync_mode: Mapped[str]  # "one_time" | "continuous"
    sync_interval_minutes: Mapped[int | None]
    is_public: Mapped[bool] = mapped_column(default=False)
    meta: Mapped[dict | None]  # JSON metadata
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
```

### Document
```python
class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    collection_id: Mapped[str] = mapped_column(ForeignKey("collections.id"))
    file_name: Mapped[str]
    file_path: Mapped[str]
    file_size: Mapped[int]
    content_hash: Mapped[str]  # SHA-256 for deduplication
    source_path: Mapped[str | None]  # Connector source
    status: Mapped[DocumentStatus]
    chunk_count: Mapped[int]
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
```

### Chunk (Partitioned)
```python
class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[str] = mapped_column(primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    collection_id: Mapped[str]  # Denormalized for partition pruning
    content: Mapped[str]
    chunk_index: Mapped[int]
    partition_key: Mapped[int]  # Computed: abs(hashtext(collection_id)) % 100
    metadata: Mapped[dict | None]
```

### Operation
```python
class Operation(Base):
    __tablename__ = "operations"
    uuid: Mapped[str] = mapped_column(primary_key=True)
    type: Mapped[OperationType]  # INDEX, APPEND, REINDEX, DELETE, REMOVE_SOURCE
    status: Mapped[OperationStatus]  # PENDING, PROCESSING, COMPLETED, FAILED, CANCELLED
    collection_id: Mapped[str] = mapped_column(ForeignKey("collections.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    progress: Mapped[int] = mapped_column(default=0)  # 0-100
    error_message: Mapped[str | None]
    config: Mapped[dict | None]  # JSON operation config
    created_at: Mapped[datetime]
    started_at: Mapped[datetime | None]
    completed_at: Mapped[datetime | None]
```

## Partitioning Strategy

The `chunks` table uses **100 LIST partitions** for horizontal scaling:

```sql
-- Partition function
CREATE OR REPLACE FUNCTION compute_chunk_partition_key(collection_id TEXT)
RETURNS INTEGER AS $$
BEGIN
    RETURN abs(hashtext(collection_id)) % 100;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Trigger for auto-computing partition_key
CREATE TRIGGER chunk_partition_trigger
BEFORE INSERT ON chunks
FOR EACH ROW
EXECUTE FUNCTION set_chunk_partition_key();
```

**Critical Rule:** Always include `collection_id` in chunk queries for partition pruning:

```python
# GOOD - Uses partition pruning
await session.execute(
    select(Chunk)
    .where(Chunk.collection_id == collection_id)
    .where(Chunk.document_id == document_id)
)

# BAD - Full table scan across all partitions
await session.execute(
    select(Chunk).where(Chunk.document_id == document_id)
)
```

## Repository Pattern

All database access goes through repository classes:

### CollectionRepository
```python
class CollectionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, collection_id: str, user_id: int) -> Collection | None:
        """Get collection by ID with owner check."""
        result = await self.session.execute(
            select(Collection)
            .where(Collection.id == collection_id)
            .where(or_(Collection.owner_id == user_id, Collection.is_public == True))
        )
        return result.scalar_one_or_none()

    async def list_for_user(
        self,
        user_id: int,
        status: CollectionStatus | None = None,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[list[Collection], int]:
        """List collections accessible to user with pagination."""
        query = select(Collection).where(
            or_(Collection.owner_id == user_id, Collection.is_public == True)
        )
        if status:
            query = query.where(Collection.status == status)

        count = await self.session.scalar(
            select(func.count()).select_from(query.subquery())
        )
        result = await self.session.execute(
            query.order_by(Collection.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all()), count
```

### ChunkRepository
```python
class ChunkRepository:
    async def get_by_document(
        self,
        document_id: str,
        collection_id: str  # Required for partition pruning
    ) -> list[Chunk]:
        result = await self.session.execute(
            select(Chunk)
            .where(Chunk.collection_id == collection_id)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        )
        return list(result.scalars().all())
```

## Connection Pooling

```python
# Configuration via environment
DB_POOL_SIZE = 20          # Base pool connections
DB_MAX_OVERFLOW = 40       # Additional overflow connections
DB_POOL_TIMEOUT = 30       # Wait timeout (seconds)
DB_POOL_RECYCLE = 3600     # Recycle connections hourly
DB_POOL_PRE_PING = True    # Health check before use

# AsyncEngine creation
engine = create_async_engine(
    database_url,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_timeout=settings.db_pool_timeout,
    pool_recycle=settings.db_pool_recycle,
    pool_pre_ping=settings.db_pool_pre_ping,
)
```

## Migrations (Alembic)

```bash
# Generate migration
uv run alembic revision --autogenerate -m "description"

# Apply migrations
uv run alembic upgrade head

# Rollback
uv run alembic downgrade -1
```

**Migration Safety:**
- Use `require_destructive_flag()` for DROP operations
- Create backups via `create_table_backup()` before destructive changes
- Verify backups with `verify_backup()`

## Extension Points

### Adding a New Model
1. Define in `models.py`
2. Create repository in `repositories/`
3. Generate migration
4. Add factory fixture for testing

### Adding Repository Method
1. Implement in repository class
2. Always include appropriate WHERE clauses for security/partitioning
3. Add unit test with `db_session` fixture
