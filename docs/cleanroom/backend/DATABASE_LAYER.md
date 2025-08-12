# DATABASE_LAYER - Data Persistence Component

## 1. Component Overview

The DATABASE_LAYER is the data persistence component of the Semantik application, providing a robust, scalable, and secure data storage solution using PostgreSQL as the primary database system. This component handles all data persistence operations including user management, collection metadata, document tracking, operation history, and the partitioned storage of document chunks for semantic search.

### Key Responsibilities
- **Data Persistence**: Store and retrieve all application data using PostgreSQL
- **Schema Management**: Define and maintain database schema through SQLAlchemy models
- **Repository Pattern**: Provide clean data access abstractions via repository interfaces
- **Partition Management**: Handle efficient storage of millions of chunks using table partitioning
- **Transaction Management**: Ensure data consistency with proper transaction boundaries
- **Connection Pooling**: Manage database connections efficiently for concurrent access
- **Migration System**: Track and apply schema changes through Alembic migrations

### Technology Stack
- **Database**: PostgreSQL 12+ (16+ recommended for GENERATED columns)
- **ORM**: SQLAlchemy 2.0+ with async support
- **Migration Tool**: Alembic
- **Async Driver**: asyncpg for PostgreSQL
- **Connection Pooling**: SQLAlchemy connection pool with configurable settings
- **Partitioning**: PostgreSQL native LIST partitioning with 100 partitions

## 2. Architecture & Design Patterns

### Repository Pattern
The database layer implements the Repository pattern to provide a clean abstraction between the business logic and data access layers:

```python
# Abstract repository interface
class BaseRepository(ABC, Generic[T]):
    async def get(self, id: str) -> T | None
    async def list(self, **filters: Any) -> list[T]
    async def create(self, entity: T) -> T
    async def update(self, id: str, updates: dict[str, Any]) -> T | None
    async def delete(self, id: str) -> bool

# Concrete implementation
class CollectionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, name: str, owner_id: int, ...) -> Collection:
        # Implementation with proper validation and error handling
```

### Unit of Work Pattern
Database sessions represent units of work with automatic transaction management:

```python
async with pg_connection_manager.get_session() as session:
    try:
        # Perform operations
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
```

### Partitioning Strategy
The chunks table uses LIST partitioning with 100 partitions for scalability:

```sql
CREATE TABLE chunks (
    id BIGSERIAL,
    collection_id VARCHAR NOT NULL,
    partition_key INTEGER NOT NULL,
    ...
    PRIMARY KEY (id, collection_id, partition_key)
) PARTITION BY LIST (partition_key);

-- Partition key computed via trigger
CREATE FUNCTION compute_partition_key() RETURNS TRIGGER AS $$
BEGIN
    NEW.partition_key := abs(hashtext(NEW.collection_id::text)) % 100;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

### Migration Architecture
Alembic manages schema evolution with versioned migrations:

```
alembic/
├── versions/
│   ├── 005a8fe3aedc_initial_schema.py      # Initial schema
│   ├── 52db15bd2686_add_chunking_tables.py # Chunking tables
│   ├── ae558c9e183f_implement_partitions.py # 100 partitions
│   └── db003_replace_trigger.py            # Performance optimization
└── env.py                                   # Migration environment
```

## 3. Key Interfaces & Contracts

### Model Schemas

#### User Model
```python
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    full_name = Column(String)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now())
    last_login = Column(DateTime(timezone=True))
    
    # Relationships with cascade delete
    collections = relationship("Collection", back_populates="owner", cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
```

#### Collection Model
```python
class Collection(Base):
    __tablename__ = "collections"
    
    id = Column(String, primary_key=True)  # UUID as string
    name = Column(String, unique=True, nullable=False, index=True)
    description = Column(Text)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    vector_store_name = Column(String, unique=True, nullable=False)  # Qdrant collection
    embedding_model = Column(String, nullable=False)
    quantization = Column(String, nullable=False, default="float16")
    chunk_size = Column(Integer, nullable=False, default=1000)
    chunk_overlap = Column(Integer, nullable=False, default=200)
    chunking_strategy = Column(String, nullable=True)
    chunking_config = Column(JSON, nullable=True)
    is_public = Column(Boolean, nullable=False, default=False, index=True)
    status = Column(Enum(CollectionStatus), nullable=False, default=CollectionStatus.PENDING, index=True)
    status_message = Column(Text)
    document_count = Column(Integer, nullable=False, default=0)
    vector_count = Column(Integer, nullable=False, default=0)
    total_size_bytes = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    meta = Column(JSON)
```

#### Document Model
```python
class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True)  # UUID as string
    collection_id = Column(String, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True)
    source_id = Column(Integer, ForeignKey("collection_sources.id"), nullable=True, index=True)
    file_path = Column(String, nullable=False)
    file_name = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String)
    content_hash = Column(String, nullable=False, index=True)  # SHA-256 for deduplication
    status = Column(Enum(DocumentStatus), nullable=False, default=DocumentStatus.PENDING, index=True)
    error_message = Column(Text)
    chunk_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Composite unique index for deduplication
    __table_args__ = (
        Index("ix_documents_collection_content_hash", "collection_id", "content_hash", unique=True),
    )
```

#### Chunk Model (Partitioned)
```python
class Chunk(Base):
    __tablename__ = "chunks"
    
    # Composite primary key for partitioning support
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    collection_id = Column(String, ForeignKey("collections.id", ondelete="CASCADE"), primary_key=True, nullable=False)
    partition_key = Column(Integer, primary_key=True, nullable=False, server_default="0")  # Computed via trigger
    
    # Data columns
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    start_offset = Column(Integer)
    end_offset = Column(Integer)
    token_count = Column(Integer)
    embedding_vector_id = Column(String)  # Reference to Qdrant
    metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Partitioned indexes
    __table_args__ = (
        Index("idx_chunks_part_collection", "collection_id"),
        Index("idx_chunks_part_document", "document_id"),
        Index("idx_chunks_part_chunk_index", "collection_id", "chunk_index"),
    )
```

#### Operation Model
```python
class Operation(Base):
    __tablename__ = "operations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String, unique=True, nullable=False)  # External reference
    collection_id = Column(String, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    type = Column(Enum(OperationType), nullable=False, index=True)
    status = Column(Enum(OperationStatus), nullable=False, default=OperationStatus.PENDING, index=True)
    task_id = Column(String)  # Celery task ID
    config = Column(JSON, nullable=False)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    meta = Column(JSON)
```

### Repository Interfaces

#### CollectionRepository
```python
class CollectionRepository:
    async def create(name: str, owner_id: int, ...) -> Collection
    async def get_by_uuid(collection_uuid: str) -> Collection | None
    async def get_by_name(name: str) -> Collection | None
    async def get_by_uuid_with_permission_check(collection_uuid: str, user_id: int) -> Collection
    async def list_for_user(user_id: int, offset: int = 0, limit: int = 50) -> tuple[list[Collection], int]
    async def update_status(collection_uuid: str, status: CollectionStatus, message: str | None) -> Collection
    async def update_stats(collection_uuid: str, document_count: int | None, ...) -> Collection
    async def rename(collection_uuid: str, new_name: str, user_id: int) -> Collection
    async def delete(collection_uuid: str, user_id: int) -> None
    async def update(collection_uuid: str, updates: dict[str, Any]) -> Collection
```

#### ChunkRepository (Partition-Aware)
```python
class ChunkRepository(PartitionAwareMixin):
    async def create_chunk(chunk_data: dict[str, Any]) -> Chunk
    async def create_chunks_bulk(chunks_data: list[dict[str, Any]]) -> int
    async def get_chunk_by_id(chunk_id: int, collection_id: str, partition_key: int | None) -> Chunk | None
    async def get_chunks_by_document(document_id: str, collection_id: str, limit: int | None) -> list[Chunk]
    async def get_chunks_by_collection(collection_id: str, limit: int | None) -> list[Chunk]
    async def update_chunk_embeddings(chunk_updates: list[dict[str, str]]) -> int
    async def delete_chunks_by_document(document_id: str, collection_id: str) -> int
    async def delete_chunks_by_collection(collection_id: str) -> int
    async def get_chunk_statistics(collection_id: str) -> dict[str, Any]
    async def get_chunks_without_embeddings(collection_id: str, limit: int) -> list[Chunk]
```

## 4. Data Flow & Dependencies

### Session Lifecycle
```python
# 1. Connection Manager Initialization
pg_connection_manager = PostgresConnectionManager(postgres_config)
await pg_connection_manager.initialize()

# 2. Session Creation (Dependency Injection)
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with pg_connection_manager.get_session() as session:
        yield session

# 3. Repository Creation
@router.post("/collections")
async def create_collection(
    request: CreateCollectionRequest,
    db: AsyncSession = Depends(get_db)
):
    repo = CollectionRepository(db)
    return await repo.create(...)

# 4. Automatic Cleanup
# Session is automatically committed/rolled back and closed
```

### Transaction Boundaries
```python
# Service Layer - Transaction per Business Operation
class CollectionService:
    async def create_collection_with_source(self, data: dict) -> Collection:
        async with self.session.begin():  # Transaction starts
            # Create collection
            collection = await self.collection_repo.create(...)
            
            # Create source
            source = await self.source_repo.create(...)
            
            # Create initial operation
            operation = await self.operation_repo.create(...)
            
            # All changes committed atomically
        return collection  # Transaction ends
```

### Partition-Aware Query Optimization
```python
# GOOD: Includes partition key for pruning
chunks = await session.execute(
    select(Chunk).where(
        and_(
            Chunk.collection_id == collection_id,  # Partition key
            Chunk.document_id == document_id
        )
    )
)

# BAD: Scans all 100 partitions
chunks = await session.execute(
    select(Chunk).where(Chunk.document_id == document_id)
)
```

## 5. Critical Implementation Details

### Partitioning Strategy

#### Partition Key Computation
```sql
-- Trigger function for automatic partition key computation
CREATE OR REPLACE FUNCTION compute_partition_key()
RETURNS TRIGGER AS $$
BEGIN
    -- Use PostgreSQL's hashtext() for even distribution
    NEW.partition_key := abs(hashtext(NEW.collection_id::text)) % 100;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Applied via trigger before INSERT
CREATE TRIGGER set_partition_key
BEFORE INSERT ON chunks
FOR EACH ROW
EXECUTE FUNCTION compute_partition_key();
```

#### Partition Creation
```sql
-- Create 100 LIST partitions
DO $$
DECLARE
    i INTEGER;
BEGIN
    FOR i IN 0..99 LOOP
        EXECUTE format('
            CREATE TABLE chunks_p%s PARTITION OF chunks
            FOR VALUES IN (%s)
        ', i, i);
        
        -- Create per-partition indexes
        EXECUTE format('
            CREATE INDEX idx_chunks_p%s_collection 
            ON chunks_p%s(collection_id)
        ', i, i);
    END LOOP;
END $$;
```

### Index Strategy

#### Primary Indexes
```sql
-- Users table
CREATE UNIQUE INDEX ix_users_username ON users(username);
CREATE UNIQUE INDEX ix_users_email ON users(email);

-- Collections table
CREATE UNIQUE INDEX ix_collections_name ON collections(name);
CREATE INDEX ix_collections_owner_id ON collections(owner_id);
CREATE INDEX ix_collections_status ON collections(status);
CREATE INDEX ix_collections_is_public ON collections(is_public);

-- Documents table
CREATE INDEX ix_documents_collection_id ON documents(collection_id);
CREATE INDEX ix_documents_content_hash ON documents(content_hash);
CREATE UNIQUE INDEX ix_documents_collection_content_hash ON documents(collection_id, content_hash);

-- Chunks table (per-partition)
CREATE INDEX idx_chunks_part_collection ON chunks(collection_id);
CREATE INDEX idx_chunks_part_document ON chunks(document_id);
CREATE INDEX idx_chunks_part_chunk_index ON chunks(collection_id, chunk_index);
```

### Constraints & Referential Integrity

#### Foreign Key Constraints with CASCADE
```sql
-- Documents cascade delete with collection
ALTER TABLE documents 
ADD CONSTRAINT fk_documents_collection 
FOREIGN KEY (collection_id) 
REFERENCES collections(id) 
ON DELETE CASCADE;

-- Chunks cascade delete with both collection and document
ALTER TABLE chunks 
ADD CONSTRAINT fk_chunks_collection 
FOREIGN KEY (collection_id) 
REFERENCES collections(id) 
ON DELETE CASCADE;

ALTER TABLE chunks 
ADD CONSTRAINT fk_chunks_document 
FOREIGN KEY (document_id) 
REFERENCES documents(id) 
ON DELETE CASCADE;
```

#### Check Constraints
```sql
-- Ensure permission is either for user OR api_key, not both
ALTER TABLE collection_permissions
ADD CONSTRAINT check_user_or_api_key CHECK (
    (user_id IS NOT NULL AND api_key_id IS NULL) OR 
    (user_id IS NULL AND api_key_id IS NOT NULL)
);
```

### Connection Pooling Configuration

```python
class PostgresConfig:
    # Connection pool settings
    DB_POOL_SIZE: int = 20           # Base pool size
    DB_MAX_OVERFLOW: int = 40        # Additional connections when needed
    DB_POOL_TIMEOUT: int = 30        # Seconds to wait for connection
    DB_POOL_RECYCLE: int = 3600      # Recycle connections after 1 hour
    DB_POOL_PRE_PING: bool = True    # Test connections before use
    
    # Query settings
    DB_QUERY_TIMEOUT: int = 30       # Query timeout in seconds
    
    # PostgreSQL-specific optimizations
    connect_args = {
        "server_settings": {
            "application_name": "semantik",
            "jit": "off",  # Disable JIT for predictable performance
            "statement_timeout": "30000",  # 30 seconds
            "lock_timeout": "5000",  # 5 seconds
            "idle_in_transaction_session_timeout": "60000",  # 60 seconds
        }
    }
```

## 6. Security Considerations

### SQL Injection Prevention

#### Parameterized Queries
```python
# GOOD: Using SQLAlchemy ORM with parameterized queries
async def get_user_by_username(db: AsyncSession, username: str) -> User | None:
    stmt = select(User).where(User.username == username)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()

# GOOD: Using bound parameters for raw SQL
async def search_documents(pattern: str):
    stmt = text("SELECT * FROM documents WHERE file_name LIKE :pattern")
    result = await db.execute(stmt, {"pattern": f"%{pattern}%"})
```

#### Input Validation
```python
class PartitionValidation:
    UUID_PATTERN = re.compile(
        r"^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$", 
        re.IGNORECASE
    )
    
    @classmethod
    def validate_uuid(cls, value: Any, field_name: str = "id") -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")
        if not cls.UUID_PATTERN.match(value):
            raise ValueError(f"{field_name} must be a valid UUID v4")
        return value.lower()
    
    @classmethod
    def sanitize_string(cls, value: str, max_length: int = 255) -> str:
        # Truncate if too long
        if len(value) > max_length:
            value = value[:max_length]
        # Remove null bytes
        return value.replace("\x00", "")
```

### Password Security

```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Hash password before storage
hashed_password = pwd_context.hash(plain_password)

# Verify password
is_valid = pwd_context.verify(plain_password, hashed_password)
```

### Access Control

#### Repository-Level Permission Checks
```python
async def get_by_uuid_with_permission_check(
    self, collection_uuid: str, user_id: int
) -> Collection:
    collection = await self.get_by_uuid(collection_uuid)
    
    if not collection:
        raise EntityNotFoundError("collection", collection_uuid)
    
    # Check ownership or public access
    if collection.owner_id != user_id and not collection.is_public:
        # Check CollectionPermission table for shared access
        permission = await self.check_permission(collection_uuid, user_id)
        if not permission:
            raise AccessDeniedError(str(user_id), "collection", collection_uuid)
    
    return collection
```

### Connection Security

```python
# Use SSL for database connections in production
DATABASE_URL = "postgresql+asyncpg://user:pass@host:5432/db?ssl=require"

# Connection string should never be logged with credentials
parsed = urlparse(DATABASE_URL)
safe_url = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}/{parsed.path}"
logger.info(f"Connecting to database: {safe_url}")
```

## 7. Testing Requirements

### Unit Tests

#### Repository Tests
```python
@pytest.mark.asyncio
async def test_collection_repository_create():
    async with test_session() as session:
        repo = CollectionRepository(session)
        
        collection = await repo.create(
            name="test-collection",
            owner_id=1,
            embedding_model="test-model"
        )
        
        assert collection.id is not None
        assert collection.name == "test-collection"
        assert collection.status == CollectionStatus.PENDING
```

#### Partition Tests
```python
@pytest.mark.asyncio
async def test_chunk_partition_distribution():
    """Verify chunks are evenly distributed across partitions"""
    async with test_session() as session:
        repo = ChunkRepository(session)
        
        # Create chunks for multiple collections
        collections = [str(uuid4()) for _ in range(10)]
        for collection_id in collections:
            chunks_data = [
                {
                    "collection_id": collection_id,
                    "chunk_index": i,
                    "content": f"Test chunk {i}"
                }
                for i in range(100)
            ]
            await repo.create_chunks_bulk(chunks_data)
        
        # Verify distribution
        stats = await repo.get_partition_statistics(collections[0])
        assert stats["chunk_count"] == 100
```

### Integration Tests

#### Transaction Tests
```python
@pytest.mark.asyncio
async def test_transaction_rollback_on_error():
    async with test_session() as session:
        async with session.begin():
            collection = Collection(id=str(uuid4()), name="test")
            session.add(collection)
            
            # Force an error
            with pytest.raises(IntegrityError):
                duplicate = Collection(id=str(uuid4()), name="test")
                session.add(duplicate)
                await session.flush()
        
        # Verify rollback
        result = await session.execute(
            select(Collection).where(Collection.name == "test")
        )
        assert result.scalar_one_or_none() is None
```

#### Migration Tests
```python
def test_migration_up_and_down():
    """Test migration can be applied and rolled back"""
    alembic_cfg = Config("alembic.ini")
    
    # Upgrade to latest
    command.upgrade(alembic_cfg, "head")
    
    # Verify schema
    with engine.connect() as conn:
        tables = inspect(conn).get_table_names()
        assert "chunks" in tables
        assert "chunks_p0" in tables  # First partition
    
    # Downgrade
    command.downgrade(alembic_cfg, "-1")
```

### Performance Tests

#### Partition Pruning Tests
```python
@pytest.mark.asyncio
async def test_partition_pruning_performance():
    """Verify queries use partition pruning"""
    async with test_session() as session:
        # Explain plan should show partition pruning
        result = await session.execute(
            text("""
                EXPLAIN (ANALYZE, BUFFERS) 
                SELECT * FROM chunks 
                WHERE collection_id = :collection_id
            """),
            {"collection_id": "test-uuid"}
        )
        
        plan = result.fetchall()
        # Verify only one partition is scanned
        assert "chunks_p" in str(plan)
        assert "Partitions Selected: 1" in str(plan)
```

## 8. Common Pitfalls & Best Practices

### Pitfall: Missing Partition Key in Queries
```python
# BAD: Scans all 100 partitions
chunks = await session.query(Chunk).filter(
    Chunk.document_id == document_id
).all()

# GOOD: Uses partition pruning
chunks = await session.query(Chunk).filter(
    Chunk.collection_id == collection_id,  # Always include partition key
    Chunk.document_id == document_id
).all()
```

### Pitfall: N+1 Query Problem
```python
# BAD: N+1 queries
collections = await session.query(Collection).all()
for collection in collections:
    documents = await session.query(Document).filter(
        Document.collection_id == collection.id
    ).all()

# GOOD: Eager loading
collections = await session.query(Collection).options(
    selectinload(Collection.documents)
).all()
```

### Pitfall: Unbounded Result Sets
```python
# BAD: Could return millions of rows
chunks = await session.query(Chunk).all()

# GOOD: Always paginate
chunks = await session.query(Chunk).limit(1000).offset(0).all()
```

### Pitfall: Long-Running Transactions
```python
# BAD: Transaction held during slow operation
async with session.begin():
    collection = await create_collection(...)
    await slow_external_api_call()  # Transaction held!
    await session.commit()

# GOOD: Minimize transaction scope
collection = await create_collection(...)
await session.commit()
await slow_external_api_call()  # Outside transaction
```

### Best Practice: Bulk Operations
```python
# Partition-aware bulk insert
class ChunkRepository:
    async def create_chunks_bulk(self, chunks_data: list[dict[str, Any]]) -> int:
        # Group by partition key for efficiency
        chunks_by_collection = {}
        for chunk in chunks_data:
            collection_id = chunk["collection_id"]
            chunks_by_collection.setdefault(collection_id, []).append(chunk)
        
        # Insert per partition
        for collection_id, chunks in chunks_by_collection.items():
            await self.session.run_sync(
                lambda sync_session: sync_session.bulk_insert_mappings(
                    Chunk, chunks
                )
            )
```

## 9. Configuration & Environment

### Environment Variables
```bash
# PostgreSQL Connection
DATABASE_URL=postgresql://user:password@localhost:5432/semantik
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=semantik
POSTGRES_USER=semantik
POSTGRES_PASSWORD=secure_password

# Connection Pool Settings
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600
DB_POOL_PRE_PING=true

# Query Settings
DB_QUERY_TIMEOUT=30
DB_ECHO=false
DB_ECHO_POOL=false

# Retry Settings
DB_RETRY_LIMIT=3
DB_RETRY_INTERVAL=0.5

# Partitioning
CHUNK_PARTITION_COUNT=100
```

### Connection String Formats
```python
# Async driver for application
postgresql+asyncpg://user:pass@host:5432/db

# Sync driver for migrations
postgresql+psycopg2://user:pass@host:5432/db

# With SSL
postgresql+asyncpg://user:pass@host:5432/db?ssl=require
```

### Pool Configuration Guidelines
```python
# Development
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10

# Production (based on expected concurrency)
DB_POOL_SIZE=20  # Base connections
DB_MAX_OVERFLOW=40  # Burst capacity
# Total possible connections = 20 + 40 = 60

# High-load production
DB_POOL_SIZE=50
DB_MAX_OVERFLOW=100
# Consider PostgreSQL max_connections setting
```

## 10. Integration Points

### Service Layer Integration
```python
# services/collection_service.py
class CollectionService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.collection_repo = CollectionRepository(session)
        self.document_repo = DocumentRepository(session)
        self.operation_repo = OperationRepository(session)
    
    async def create_collection_with_operation(self, data: dict) -> Collection:
        async with self.session.begin():
            # Repository calls within transaction
            collection = await self.collection_repo.create(...)
            operation = await self.operation_repo.create(...)
            return collection
```

### API Layer Integration
```python
# api/collections.py
@router.post("/collections")
async def create_collection(
    request: CreateCollectionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    service = CollectionService(db)
    collection = await service.create_collection_with_operation({
        "name": request.name,
        "owner_id": current_user.id,
        ...
    })
    return CollectionResponse.from_orm(collection)
```

### Worker Integration
```python
# worker/tasks/indexing.py
@celery_app.task
def index_collection(collection_id: str):
    # Workers use sync session
    with get_sync_session() as session:
        repo = CollectionRepository(session)
        collection = repo.get_by_uuid(collection_id)
        
        # Update status
        repo.update_status(
            collection_id, 
            CollectionStatus.PROCESSING
        )
```

### Vector Store Integration
```python
# Chunks reference Qdrant vectors
class ChunkRepository:
    async def update_chunk_embeddings(
        self, 
        chunk_updates: list[dict[str, str]]
    ) -> int:
        """Update chunks with Qdrant vector IDs"""
        for update in chunk_updates:
            await self.session.execute(
                update(Chunk)
                .where(Chunk.id == update["chunk_id"])
                .values(embedding_vector_id=update["vector_id"])
            )
```

## 11. Monitoring & Health Checks

### Database Health Check
```python
async def check_postgres_connection() -> bool:
    """Health check endpoint for database connectivity"""
    try:
        async with pg_connection_manager.get_session() as session:
            result = await session.execute(text("SELECT 1"))
            return bool(result.scalar() == 1)
    except Exception as e:
        logger.error(f"PostgreSQL health check failed: {e}")
        return False
```

### Partition Health Monitoring
```python
class PartitionImplementationDetector:
    @staticmethod
    async def generate_health_report(session: AsyncSession) -> str:
        """Generate comprehensive partition health report"""
        # Check implementation method (trigger vs GENERATED)
        impl = await detect_implementation(session)
        
        # Verify partition keys are correct
        verification = await verify_partition_keys(session)
        
        # Get distribution metrics
        metrics = await get_performance_metrics(session)
        
        return format_health_report(impl, verification, metrics)
```

### Query Performance Monitoring
```sql
-- Monitor slow queries
SELECT 
    query,
    mean_exec_time,
    calls,
    total_exec_time
FROM pg_stat_statements
WHERE mean_exec_time > 1000  -- Queries slower than 1 second
ORDER BY mean_exec_time DESC;

-- Monitor partition distribution
SELECT 
    partition_key,
    COUNT(*) as chunk_count
FROM chunks
GROUP BY partition_key
ORDER BY chunk_count DESC;
```

## 12. Backup & Recovery Considerations

### Backup Strategy
```bash
# Full backup with pg_dump
pg_dump -h localhost -U semantik -d semantik -F custom -f backup.dump

# Backup specific tables
pg_dump -h localhost -U semantik -d semantik -t collections -t documents -F custom -f data.dump

# Continuous archiving with WAL
archive_mode = on
archive_command = 'cp %p /backup/wal/%f'
```

### Recovery Procedures
```bash
# Restore from dump
pg_restore -h localhost -U semantik -d semantik -F custom backup.dump

# Point-in-time recovery
recovery_target_time = '2024-01-01 12:00:00'
```

### Data Integrity Checks
```sql
-- Verify referential integrity
SELECT 
    'documents without collection' as issue,
    COUNT(*) as count
FROM documents d
LEFT JOIN collections c ON d.collection_id = c.id
WHERE c.id IS NULL;

-- Verify partition key consistency
SELECT 
    collection_id,
    partition_key,
    abs(hashtext(collection_id::text)) % 100 as expected_key,
    CASE 
        WHEN partition_key = abs(hashtext(collection_id::text)) % 100 
        THEN 'OK' 
        ELSE 'MISMATCH' 
    END as status
FROM chunks
WHERE partition_key != abs(hashtext(collection_id::text)) % 100;
```

## 13. Status Enums & State Transitions

### Collection Status
```python
class CollectionStatus(str, enum.Enum):
    PENDING = "pending"      # Initial state, awaiting first operation
    READY = "ready"          # Has vectors, ready for search
    PROCESSING = "processing" # Operation in progress
    ERROR = "error"          # Operation failed
    DEGRADED = "degraded"    # Partially functional

# Valid transitions
COLLECTION_STATUS_TRANSITIONS = {
    CollectionStatus.PENDING: [CollectionStatus.PROCESSING, CollectionStatus.ERROR],
    CollectionStatus.PROCESSING: [CollectionStatus.READY, CollectionStatus.ERROR, CollectionStatus.DEGRADED],
    CollectionStatus.READY: [CollectionStatus.PROCESSING, CollectionStatus.DEGRADED],
    CollectionStatus.ERROR: [CollectionStatus.PROCESSING],
    CollectionStatus.DEGRADED: [CollectionStatus.PROCESSING, CollectionStatus.READY],
}
```

### Document Status
```python
class DocumentStatus(str, enum.Enum):
    PENDING = "pending"      # Awaiting processing
    PROCESSING = "processing" # Being chunked/embedded
    COMPLETED = "completed"  # Successfully processed
    FAILED = "failed"        # Processing failed
    DELETED = "deleted"      # Soft deleted

# Processing pipeline
PENDING -> PROCESSING -> COMPLETED
                      -> FAILED
```

### Operation Status
```python
class OperationStatus(str, enum.Enum):
    PENDING = "pending"      # Queued
    PROCESSING = "processing" # Being executed
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"        # Execution failed
    CANCELLED = "cancelled"  # User cancelled

class OperationType(str, enum.Enum):
    INDEX = "index"          # Initial indexing
    APPEND = "append"        # Add new documents
    REINDEX = "reindex"      # Full re-indexing
    REMOVE_SOURCE = "remove_source" # Remove source documents
    DELETE = "delete"        # Delete collection
```

## 14. Migration History & Patterns

### Key Migrations

1. **005a8fe3aedc** - Initial unified schema
   - Creates base tables (users, collections, documents, operations)
   - Establishes foreign key relationships
   - Sets up initial indexes

2. **52db15bd2686** - Add chunking tables with partitioning
   - Creates chunking_strategies and chunking_configs tables
   - Initial chunks table setup

3. **ae558c9e183f** - Implement 100 direct LIST partitions
   - Drops old chunks table
   - Creates partitioned chunks table with 100 partitions
   - Adds partition key trigger

4. **db003_replace_trigger** - Performance optimization (PostgreSQL 12+)
   - Replaces trigger with GENERATED column for partition_key
   - Reduces INSERT overhead by 2-3ms per row

### Migration Best Practices
```python
# Always include both upgrade and downgrade
def upgrade() -> None:
    # Create new structures
    op.create_table(...)
    op.create_index(...)

def downgrade() -> None:
    # Reverse operations in opposite order
    op.drop_index(...)
    op.drop_table(...)

# Handle data migrations carefully
def upgrade() -> None:
    # Add nullable column first
    op.add_column('table', sa.Column('new_col', sa.String(), nullable=True))
    
    # Backfill data
    op.execute("UPDATE table SET new_col = 'default'")
    
    # Then make non-nullable
    op.alter_column('table', 'new_col', nullable=False)
```

## 15. Future Considerations

### Planned Improvements
1. **Table Partitioning**: Consider time-based partitioning for audit_logs
2. **Read Replicas**: Add support for read replica load balancing
3. **Connection Pooling**: Implement PgBouncer for better connection management
4. **Caching Layer**: Add Redis caching for frequently accessed data
5. **Full-Text Search**: Integrate PostgreSQL FTS for metadata search

### Scalability Considerations
- Current partition design supports ~100M chunks efficiently
- Consider increasing partition count to 1000 for billion-scale
- Implement archival strategy for old operations and audit logs
- Add table statistics and VACUUM optimization

### PostgreSQL Version Requirements
- **Minimum**: PostgreSQL 12 (for basic partitioning)
- **Recommended**: PostgreSQL 16+ (for GENERATED columns and better partition performance)
- **Future**: PostgreSQL 17 will include improved partition-wise joins

## Appendix: Quick Reference

### Common Queries

```sql
-- Check partition distribution
SELECT 
    'chunks_p' || partition_key as partition,
    COUNT(*) as row_count,
    pg_size_pretty(pg_relation_size('chunks_p' || partition_key)) as size
FROM chunks
GROUP BY partition_key
ORDER BY COUNT(*) DESC;

-- Find duplicate documents
SELECT 
    collection_id,
    content_hash,
    COUNT(*) as duplicate_count
FROM documents
GROUP BY collection_id, content_hash
HAVING COUNT(*) > 1;

-- Active operations by user
SELECT 
    u.username,
    o.type,
    o.status,
    o.created_at
FROM operations o
JOIN users u ON o.user_id = u.id
WHERE o.status IN ('pending', 'processing')
ORDER BY o.created_at DESC;
```

### Troubleshooting Commands

```bash
# Check database connections
SELECT pid, usename, application_name, state, query_start 
FROM pg_stat_activity;

# Kill long-running queries
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'active' 
  AND query_start < now() - interval '1 hour';

# Analyze table statistics
ANALYZE chunks;
VACUUM ANALYZE collections;

# Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### Performance Tuning

```sql
-- PostgreSQL configuration (postgresql.conf)
shared_buffers = 256MB          # 25% of RAM
effective_cache_size = 1GB      # 50-75% of RAM
maintenance_work_mem = 64MB
work_mem = 4MB
max_connections = 200
random_page_cost = 1.1          # For SSD storage

-- Create missing indexes
CREATE INDEX CONCURRENTLY idx_operations_status_created 
ON operations(status, created_at DESC) 
WHERE status IN ('pending', 'processing');

-- Partition maintenance
ALTER TABLE chunks_p0 SET (autovacuum_vacuum_scale_factor = 0.01);
ALTER TABLE chunks_p0 SET (autovacuum_analyze_scale_factor = 0.01);
```

---

This documentation provides a comprehensive reference for the DATABASE_LAYER component. It should be regularly updated as the schema evolves and new patterns are established.