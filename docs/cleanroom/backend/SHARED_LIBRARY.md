# SHARED_LIBRARY Component Documentation

## 1. Component Overview

The SHARED_LIBRARY (`packages/shared/`) is the foundational code library for the Semantik application, providing core functionality shared across all services (webui, vecpipe, worker). It implements critical infrastructure including database models, repository patterns, chunking strategies, embedding services, and configuration management.

### Purpose and Scope

- **Centralized Data Models**: Single source of truth for all database schemas and domain entities
- **Repository Pattern Implementation**: Consistent data access layer with transaction support
- **Document Processing**: Chunking strategies and text processing pipelines
- **Vector Operations**: Embedding service abstractions and Qdrant management
- **Cross-Service Contracts**: Shared DTOs and API contracts for consistency
- **Infrastructure Utilities**: Configuration, metrics, error handling, and validation

### Key Design Principles

- **Domain-Driven Design (DDD)**: Clear separation between domain logic and infrastructure
- **Repository Pattern**: Abstract data access behind clean interfaces
- **Dependency Injection**: Services receive dependencies through constructors
- **Type Safety**: Comprehensive type hints and Pydantic models
- **Async-First**: All I/O operations use async/await patterns
- **Transaction Management**: Proper UnitOfWork pattern for data consistency

## 2. Architecture & Design Patterns

### Repository Pattern Implementation

The shared library implements a comprehensive repository pattern with these layers:

```
Application Layer (Use Cases)
    ↓
Domain Layer (Entities, Value Objects, Services)
    ↓
Repository Interfaces (Abstract Contracts)
    ↓
Repository Implementations (SQLAlchemy, Qdrant)
    ↓
Database Layer (PostgreSQL, Qdrant)
```

### Domain-Driven Design Structure

```
chunking/
├── application/        # Use cases and DTOs
│   ├── dto/           # Request/Response objects
│   ├── interfaces/    # Service contracts
│   └── use_cases/     # Business operations
├── domain/            # Pure business logic
│   ├── entities/      # Domain entities
│   ├── services/      # Domain services
│   └── value_objects/ # Immutable value objects
└── infrastructure/    # Technical implementations
    ├── repositories/  # Data persistence
    └── streaming/     # Memory-efficient processing
```

### Key Patterns Used

1. **Unit of Work Pattern**: Transaction management across repositories
2. **Factory Pattern**: Strategy creation and service instantiation
3. **Singleton Pattern**: Embedding service management
4. **Mixin Pattern**: Shared functionality for partitioned tables
5. **Context Manager Pattern**: Resource management and timing

## 3. Key Interfaces & Contracts

### Repository Interfaces

```python
# Base repository contract (implied through implementation)
class BaseRepository:
    async def create(self, entity: T) -> T
    async def get_by_id(self, id: str) -> T | None
    async def update(self, id: str, updates: dict) -> T
    async def delete(self, id: str) -> None
    async def list(self, filters: dict) -> list[T]
```

### Service Contracts

```python
# packages/shared/chunking/application/interfaces/services.py

class ChunkingStrategyFactory(Protocol):
    """Factory for creating chunking strategies."""
    def create_strategy(
        self, strategy_type: str, config: dict
    ) -> ChunkingStrategy

class DocumentService(Protocol):
    """Service for document operations."""
    async def load(self, file_path: str) -> Document
    async def extract_text(self, document: Document) -> str

class UnitOfWork(Protocol):
    """Transaction management."""
    async def __aenter__(self) -> 'UnitOfWork'
    async def __aexit__(self, *args) -> None
    async def commit(self) -> None
    async def rollback(self) -> None
```

### Error Contracts

```python
# packages/shared/contracts/errors.py

class ErrorResponse(BaseModel):
    error: str           # Error type/category
    message: str         # Human-readable message
    details: dict | None # Additional context
    status_code: int     # HTTP status code
```

## 4. Data Flow & Dependencies

### Service Dependencies

```
┌─────────────┐     ┌─────────────┐     ┌────────────┐
│   WebUI     │────▶│   Shared    │◀────│  VecPipe   │
└─────────────┘     │   Library   │     └────────────┘
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Worker    │
                    └─────────────┘
```

### Data Flow for Document Processing

1. **Document Ingestion** → WebUI receives file
2. **Operation Creation** → Creates operation record via repository
3. **Task Queuing** → Worker picks up processing task
4. **Document Loading** → DocumentService extracts text
5. **Chunking Strategy** → Factory creates appropriate strategy
6. **Chunk Generation** → Strategy processes document into chunks
7. **Persistence** → ChunkRepository saves to PostgreSQL
8. **Embedding** → EmbeddingService generates vectors
9. **Vector Storage** → QdrantManager stores in Qdrant
10. **Status Update** → OperationRepository updates completion

## 5. Critical Implementation Details

### Database Models (`database/models.py`)

#### Core Models

- **User**: Authentication and authorization
- **Collection**: Document organization with vector store mapping
- **Document**: File metadata and processing status
- **Chunk**: Partitioned table for scalable chunk storage
- **Operation**: Async task tracking with status management
- **ChunkingStrategy**: Available chunking algorithms
- **ChunkingConfig**: Deduplicated configuration storage

#### Partitioned Table Implementation

The `chunks` table uses PostgreSQL LIST partitioning:

```python
class Chunk(Base):
    """Partitioned by LIST(partition_key) with 100 partitions."""
    __tablename__ = "chunks"
    
    # Composite primary key for partitioning
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    collection_id = Column(String, primary_key=True, nullable=False)
    partition_key = Column(Integer, primary_key=True, nullable=False)
    
    # Partition key computed via trigger: abs(hashtext(collection_id)) % 100
```

### Chunking Strategies (`chunking/domain/services/chunking_strategies/`)

#### Available Strategies

1. **Character Chunking**: Simple character-based splitting
2. **Recursive Chunking**: Hierarchical text splitting with separators
3. **Semantic Chunking**: Groups text by semantic similarity
4. **Markdown Chunking**: Structure-aware markdown processing
5. **Hierarchical Chunking**: Multi-level document hierarchy
6. **Hybrid Chunking**: Combines multiple strategies

#### Strategy Base Class

```python
class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(
        self, content: str, config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None
    ) -> list[Chunk]
    
    @abstractmethod
    def validate_content(self, content: str) -> tuple[bool, str | None]
    
    @abstractmethod
    def estimate_chunks(self, content_length: int, config: ChunkConfig) -> int
```

### Embedding Service (`embedding/service.py`)

Singleton service for managing embeddings:

```python
# Global service instance with async initialization
_embedding_service: BaseEmbeddingService | None = None

async def get_embedding_service(config: VecpipeConfig | None = None) -> BaseEmbeddingService:
    """Get or create singleton embedding service."""
    
async def embed_texts(texts: list[str], model_name: str, batch_size: int = 32) -> NDArray[np.float32]:
    """Convenience function for batch embedding."""
```

### Repository Pattern (`database/repositories/`)

#### CollectionRepository Example

```python
class CollectionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, name: str, owner_id: int, **kwargs) -> Collection:
        """Create with validation and UUID generation."""
        
    async def get_by_uuid_with_permission_check(
        self, collection_uuid: str, user_id: int
    ) -> Collection:
        """Get with access control validation."""
        
    async def update(self, collection_uuid: str, updates: dict) -> Collection:
        """Atomic update with field validation."""
```

### Partition Utilities (`database/partition_utils.py`)

```python
class PartitionAwareMixin:
    """Efficient operations on partitioned tables."""
    
    async def bulk_insert_partitioned(
        self, session: AsyncSession, model_class: type[T],
        items: Sequence[dict], partition_key_field: str = "collection_id"
    ) -> None:
        """Groups by partition key for efficiency."""

class ChunkPartitionHelper:
    """Chunk-specific partition operations."""
    
    @staticmethod
    def create_chunk_query_with_partition(
        collection_id: str, additional_filters: list | None = None
    ) -> Select:
        """Creates query with partition pruning."""
```

## 6. Security Considerations

### Input Validation

All user inputs are validated at multiple levels:

1. **DTO Validation**: Pydantic models with strict typing
2. **Repository Validation**: Business rule enforcement
3. **Database Constraints**: Foreign keys, unique constraints
4. **Partition Key Validation**: UUID format verification

```python
class PartitionValidation:
    UUID_PATTERN = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$")
    MAX_BATCH_SIZE = 10000
    MAX_STRING_LENGTH = 1000000
    
    @classmethod
    def validate_uuid(cls, value: Any, field_name: str = "id") -> str:
        """Strict UUID v4 validation."""
```

### SQL Injection Prevention

All database queries use parameterized statements:

```python
# GOOD: Parameterized query
stmt = select(User).where(User.username == username)
result = await db.execute(stmt)

# NEVER: String concatenation
query = f"SELECT * FROM users WHERE username = '{username}'"  # VULNERABLE
```

### Access Control

Repository methods enforce access control:

```python
async def get_by_uuid_with_permission_check(
    self, collection_uuid: str, user_id: int
) -> Collection:
    collection = await self.get_by_uuid(collection_uuid)
    if collection.owner_id != user_id and not collection.is_public:
        raise AccessDeniedError(str(user_id), "collection", collection_uuid)
    return collection
```

### Error Handling

Comprehensive exception hierarchy prevents information leakage:

```python
class RepositoryError(Exception): """Base exception"""
class EntityNotFoundError(RepositoryError): """Resource not found"""
class AccessDeniedError(RepositoryError): """Authorization failure"""
class ValidationError(RepositoryError): """Input validation failure"""
class DatabaseOperationError(RepositoryError): """Database failure"""
```

## 7. Testing Requirements

### Unit Tests Required

1. **Repository Tests**: Each repository method
2. **Chunking Strategy Tests**: All strategies with edge cases
3. **Validation Tests**: Input validation and sanitization
4. **Partition Tests**: Partition key computation and routing
5. **Service Tests**: Embedding, Qdrant management

### Integration Tests Required

1. **Database Transactions**: UnitOfWork pattern
2. **Partition Operations**: Bulk inserts and queries
3. **Chunking Pipeline**: End-to-end document processing
4. **Permission System**: Access control validation
5. **Error Recovery**: Retry logic and rollback

### Test Utilities

```python
# packages/shared/utils/testing_utils.py
# Provides test fixtures and mock objects
```

## 8. Common Pitfalls & Best Practices

### Pitfall: Missing Partition Key in Queries

```python
# BAD: Scans all 100 partitions
chunks = session.query(Chunk).filter(Chunk.document_id == doc_id).all()

# GOOD: Enables partition pruning
chunks = session.query(Chunk).filter(
    Chunk.collection_id == collection_id,  # Partition key first!
    Chunk.document_id == doc_id
).all()
```

### Pitfall: Synchronous Operations in Async Context

```python
# BAD: Blocks event loop
def get_embedding_sync():
    return embedding_model.embed(text)  # Blocking I/O

# GOOD: Proper async
async def get_embedding_async():
    return await embedding_service.embed_single(text)
```

### Pitfall: Transaction Scope Issues

```python
# BAD: Multiple transactions
async def process():
    collection = await repo.create(...)  # Transaction 1
    document = await repo.create(...)    # Transaction 2 - can fail!

# GOOD: Single transaction
async def process():
    async with unit_of_work:
        collection = await unit_of_work.collections.create(...)
        document = await unit_of_work.documents.create(...)
        await unit_of_work.commit()  # Atomic
```

### Best Practice: Batch Operations

```python
# Group by partition for efficiency
chunks_by_collection = {}
for chunk in chunks_to_insert:
    chunks_by_collection.setdefault(chunk.collection_id, []).append(chunk)

for collection_id, chunks in chunks_by_collection.items():
    await session.bulk_insert_mappings(Chunk, chunks)
```

### Best Practice: Error Context

```python
try:
    result = await repository.operation()
except EntityNotFoundError as e:
    # Preserve error context
    raise HTTPException(
        status_code=404,
        detail=f"{e.entity_type} not found: {e.entity_id}"
    )
```

## 9. Configuration & Environment

### Configuration Classes (`config/`)

```python
class BaseConfig(BaseSettings):
    """Base configuration for all services."""
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent.resolve()
    ENVIRONMENT: str = "development"
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    
    @property
    def data_dir(self) -> Path:
        """Dynamic path resolution for Docker/local."""
        docker_data = Path("/app/data")
        return docker_data if docker_data.exists() else self.DATA_DIR
```

### Environment Variables

```bash
# Core settings
ENVIRONMENT=development|production
DATABASE_URL=postgresql+asyncpg://user:pass@host/db
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Feature flags
USE_LOCAL_EMBEDDINGS=true|false
ENABLE_METRICS=true|false
TESTING=true|false  # Test mode
```

### Service-Specific Configs

- **VecpipeConfig**: Embedding and vector processing settings
- **WebuiConfig**: API and authentication settings
- **PostgresConfig**: Database connection pooling

## 10. Integration Points

### Used By All Services

#### WebUI Service
- User authentication via User model
- Collection management via CollectionRepository
- Operation tracking via OperationRepository
- Error contracts for API responses

#### VecPipe Service
- Document processing via chunking strategies
- Embedding generation via EmbeddingService
- Vector storage via QdrantManager
- Chunk persistence via ChunkRepository

#### Worker Service
- Async task processing via Operation model
- Document chunking via ProcessDocumentUseCase
- Progress tracking via repositories
- Metrics recording via MetricsService

### External Dependencies

1. **PostgreSQL**: Primary data storage with partitioning
2. **Qdrant**: Vector database for semantic search
3. **Redis**: Task queue and caching (via Celery)
4. **HuggingFace**: Embedding models (optional)
5. **OpenAI**: Alternative embedding provider

### API Contracts

All services share common contracts:
- Error response format (`contracts/errors.py`)
- Search contracts (`contracts/search.py`)
- Validation patterns
- Metric definitions

## Critical Files Reference

### Database Layer
- `/packages/shared/database/models.py` - All SQLAlchemy models
- `/packages/shared/database/repositories/` - Repository implementations
- `/packages/shared/database/partition_utils.py` - Partition helpers
- `/packages/shared/database/exceptions.py` - Error hierarchy

### Chunking Domain
- `/packages/shared/chunking/domain/services/chunking_strategies/` - All strategies
- `/packages/shared/chunking/application/use_cases/process_document.py` - Main processing
- `/packages/shared/chunking/domain/value_objects/chunk_config.py` - Configuration

### Infrastructure
- `/packages/shared/embedding/service.py` - Embedding singleton
- `/packages/shared/managers/qdrant_manager.py` - Vector DB management
- `/packages/shared/config/base.py` - Base configuration
- `/packages/shared/metrics/collection_metrics.py` - Prometheus metrics

## Implementation Notes

### Partition Key Implementation
- PostgreSQL < 12: Uses trigger-based computation
- PostgreSQL >= 12: Can use GENERATED columns (more efficient)
- Automatic detection via `PartitionImplementationDetector`

### Memory Management
- Streaming strategies for large documents
- Memory pool allocation for chunking
- Batch processing with configurable sizes
- Checkpoint/resume for long operations

### Performance Optimizations
- Partition pruning for chunk queries
- Bulk insert grouping by partition
- Connection pooling for database
- Embedding batch processing
- Async I/O throughout

## Security Checklist

- ✅ All inputs validated with Pydantic
- ✅ Parameterized SQL queries only
- ✅ Access control in repositories
- ✅ UUID validation for identifiers
- ✅ String length limits enforced
- ✅ Null byte sanitization
- ✅ Transaction atomicity maintained
- ✅ Error messages don't leak internals
- ✅ Comprehensive exception handling
- ✅ Rate limiting support via metrics

## Maintenance Guidelines

1. **Adding New Models**: Update models.py, create repository, add tests
2. **New Chunking Strategy**: Extend base class, register in factory, test
3. **Database Migrations**: Use Alembic, test rollback scenarios
4. **Configuration Changes**: Update BaseConfig, document env vars
5. **Error Handling**: Use existing exception hierarchy, preserve context
6. **Performance Issues**: Check partition usage, batch operations, async patterns
7. **Security Updates**: Validate inputs, use parameterized queries, check access

---

*This documentation represents the current state of the SHARED_LIBRARY component and should be updated as the codebase evolves. All code examples are taken directly from the implementation.*