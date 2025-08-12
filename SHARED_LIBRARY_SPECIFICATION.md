# Semantik Shared Library Specification

## Overview

The shared package (`packages/shared/`) serves as the central library for common utilities, models, and services used across all Semantik microservices (webui, vecpipe, worker). This document provides a comprehensive specification of all shared code, documenting its purpose, patterns, and usage.

## 1. Package Structure

```
packages/shared/
├── chunking/           # Domain-driven text chunking implementation
├── config/            # Configuration management for all services
├── contracts/         # API contracts and shared DTOs
├── database/          # Database models, repositories, and utilities
├── embedding/         # Embedding service and model management
├── managers/          # External service managers (Qdrant)
├── metrics/           # Prometheus metrics and monitoring
├── text_processing/   # Document extraction and text processing
└── utils/            # Common utilities and helpers
```

## 2. Database Module (`shared/database`)

### 2.1 SQLAlchemy Models (`models.py`)

The database schema is defined using SQLAlchemy's declarative mapping with the following key characteristics:

#### Core Models:
- **User**: Authentication and user management
- **Collection**: Document collection organization with vector store integration
- **Document**: Individual documents within collections
- **Chunk**: Partitioned storage for document chunks (100 partitions by collection_id)
- **Operation**: Async task tracking for collection operations
- **ApiKey**: Programmatic access tokens
- **CollectionPermission**: Fine-grained access control
- **ChunkingStrategy/ChunkingConfig**: Deduplicated chunking configurations

#### Key Design Patterns:
1. **Timezone-aware DateTime fields**: All timestamps use `DateTime(timezone=True)`
2. **Partitioned tables**: Chunks table uses LIST partitioning for scalability
3. **Enum types**: Strong typing for status fields (DocumentStatus, OperationStatus, etc.)
4. **JSON columns**: Flexible metadata storage
5. **Composite indexes**: Optimized for common query patterns

#### Relationships:
- Cascade delete patterns for data integrity
- Bidirectional relationships with proper back_populates
- Foreign key constraints with appropriate ondelete behaviors

### 2.2 Repository Pattern (`repositories/`)

Abstract base classes define repository interfaces:

#### Base Repository Classes:
- **BaseRepository<T>**: Generic CRUD operations
- **UserRepository**: User-specific operations
- **CollectionRepository**: Collection management
- **AuthRepository**: Authentication operations
- **ApiKeyRepository**: API key management

#### Concrete Implementations:
- **ChunkRepository**: Partition-aware chunk operations
- **DocumentRepository**: Document lifecycle management
- **OperationRepository**: Async operation tracking
- **CollectionRepository**: Collection metadata and status

### 2.3 Database Utilities

#### Exception Hierarchy (`exceptions.py`):
```python
RepositoryError (base)
├── EntityNotFoundError
├── EntityAlreadyExistsError
├── InvalidUserIdError
├── AccessDeniedError
├── ValidationError
├── DatabaseOperationError
├── TransactionError
├── ConcurrencyError
├── InvalidStateError
└── DimensionMismatchError
```

#### Partition Utilities (`partition_utils.py`):
- **PartitionValidation**: UUID validation, batch size limits
- **PartitionQueryBuilder**: Efficient partition-aware query construction
- **PartitionBulkOperations**: Optimized bulk inserts/updates

Key Features:
- Automatic partition key computation via PostgreSQL trigger
- Partition pruning optimization for queries
- Batch operations grouped by partition key

### 2.4 Factory Pattern (`factory.py`)

Repository factory functions for dependency injection:
- `create_user_repository(session)`
- `create_auth_repository(session)`
- `create_api_key_repository(session)`
- `create_operation_repository(session)`
- `create_document_repository(session)`
- `create_collection_repository(session)`
- `create_chunk_repository(session)`

## 3. Configuration Management (`shared/config`)

### 3.1 Configuration Hierarchy

```python
BaseConfig (base settings)
├── VecpipeConfig (vecpipe-specific)
├── WebuiConfig (webui-specific)
└── PostgresConfig (database connection)
```

### 3.2 Key Configuration Classes

#### BaseConfig:
- Environment detection (development/production)
- Qdrant connection settings
- Data and log directory management
- Docker-aware path resolution

#### VecpipeConfig:
- Text extraction settings (chunk_size, overlap)
- Model configurations
- Processing parameters
- Redis configuration for task queuing

#### WebuiConfig:
- JWT authentication settings
- API configuration
- Frontend serving paths
- Security settings (CORS, rate limiting)

#### PostgresConfig:
- Connection pooling settings
- Async SQLAlchemy configuration
- Migration support with Alembic

### 3.3 Environment Variable Support

All configurations use Pydantic's `BaseSettings` with:
- `.env` file support
- Environment variable override
- Type validation and coercion
- Default value management

## 4. Embedding Module (`shared/embedding`)

### 4.1 Core Components

#### Service Classes:
- **BaseEmbeddingService**: Abstract interface for embedding providers
- **DenseEmbeddingService**: Dense vector embedding implementation
- **ManagedEmbeddingService**: Context-managed embedding service

#### Management:
- **AdaptiveBatchSizeManager**: Dynamic batch sizing for optimal performance
- **ModelConfig**: Model configuration and metadata
- **EmbeddingServiceProtocol**: Type protocol for embedding services

### 4.2 Model Management (`models.py`)

Predefined configurations for popular models:
- OpenAI models (text-embedding-ada-002, etc.)
- Sentence Transformers (all-MiniLM-L6-v2, etc.)
- Multilingual models
- Quantized model variants

### 4.3 Validation Utilities (`validation.py`)

- Dimension compatibility checking
- Collection-model compatibility validation
- Automatic dimension adjustment
- Model-specific validation rules

### 4.4 Context Managers (`context.py`)

```python
# Temporary model switching
async with temporary_embedding_service(model_name):
    embeddings = await embed_texts(texts)

# Managed service lifecycle
async with embedding_service_context() as service:
    embeddings = await service.embed(texts)
```

## 5. Text Processing Module (`shared/text_processing`)

### 5.1 Document Extraction (`extraction.py`)

Supported formats:
- PDF (with OCR support)
- Office documents (DOCX, XLSX, PPTX)
- Plain text and code files
- HTML and Markdown
- Images (with OCR)

Features:
- Automatic format detection
- Metadata extraction
- Error recovery and fallback strategies

### 5.2 Chunking Strategies (`chunking.py`)

**TokenChunker**: Token-based text splitting
- Configurable chunk size and overlap
- Token counting with multiple tokenizer support
- Sentence boundary preservation

### 5.3 Strategy Pattern Implementation

```python
strategies/
├── base.py          # Abstract strategy interface
├── pdf_strategy.py  # PDF-specific extraction
├── office_strategy.py # MS Office extraction
└── ocr_strategy.py  # OCR for images
```

## 6. Chunking Module (`shared/chunking`)

### 6.1 Domain-Driven Design Architecture

Following DDD principles with clear separation:

```
domain/           # Pure business logic
├── entities/     # Domain entities
├── services/     # Domain services
└── value_objects/ # Value objects

application/      # Use cases and DTOs
├── use_cases/    # Application services
├── dto/          # Data transfer objects
└── interfaces/   # Port interfaces

infrastructure/   # External integrations
├── repositories/ # Data persistence
└── streaming/    # Streaming implementations
```

### 6.2 Chunking Strategies

Available strategies:
- **FixedSizeChunker**: Fixed character/token chunks
- **SentenceChunker**: Sentence-aware splitting
- **SemanticChunker**: Semantic similarity-based
- **RecursiveChunker**: Hierarchical splitting
- **MarkdownChunker**: Markdown-aware splitting
- **CodeChunker**: Programming language-aware

### 6.3 Streaming Support

Memory-efficient processing for large files:
- Bounded memory usage
- Async generator patterns
- Progress tracking
- Error recovery

## 7. Contracts Module (`shared/contracts`)

### 7.1 Search Contracts (`search.py`)

**Request Models:**
- `SearchRequest`: Unified search parameters
- `BatchSearchRequest`: Batch search operations
- `HybridSearchRequest`: Hybrid search configuration

**Response Models:**
- `SearchResult`: Individual result with metadata
- `SearchResponse`: Complete search response
- `HybridSearchResult`: Hybrid search results

**Key Features:**
- Backward compatibility (k/top_k aliasing)
- Search type validation
- Reranking support
- Hybrid search parameters

### 7.2 Error Contracts (`errors.py`)

Standardized error responses:
- `ValidationErrorResponse`: Input validation errors
- `AuthenticationErrorResponse`: Auth failures
- `AuthorizationErrorResponse`: Permission denied
- `NotFoundErrorResponse`: Resource not found
- `InsufficientResourcesErrorResponse`: Resource limits
- `ServiceUnavailableError`: Service down
- `RateLimitError`: Rate limiting

## 8. Managers Module (`shared/managers`)

### 8.1 QdrantManager (`qdrant_manager.py`)

Vector database management:
- Collection lifecycle (create, delete, update)
- Point operations (insert, search, delete)
- Batch operations with retry logic
- Connection pooling
- Error handling and recovery

**Key Methods:**
```python
async def create_collection(name, dimension, quantization)
async def insert_points(collection, points, batch_size)
async def search(collection, vector, limit, filters)
async def delete_collection(name)
```

## 9. Metrics Module (`shared/metrics`)

### 9.1 Prometheus Metrics (`prometheus.py`)

**System Metrics:**
- CPU, memory, GPU utilization
- Disk I/O statistics
- Network throughput

**Application Metrics:**
- Operations (created, completed, failed)
- Processing durations (extraction, chunking, embedding)
- Queue lengths and processing lag
- File processing statistics
- Error rates and types

**Qdrant Metrics:**
- Point counts per collection
- Upload errors and retries
- Query performance

### 9.2 Metrics Collection (`collection_metrics.py`)

Automated collection patterns:
- Context managers for timing
- Decorator-based instrumentation
- Async-aware collectors

### 9.3 Helper Functions

```python
# Timing context
async with TimingContext(operation_duration, labels):
    await process_operation()

# Record helpers
record_operation_completed(operation_type, duration)
record_file_processed(file_type, size)
update_queue_length(queue_name, length)
```

## 10. Utilities Module (`shared/utils`)

### 10.1 Testing Utilities (`testing_utils.py`)

Environment detection and mocking support:
- `is_testing()`: Detect test environment
- `is_redis_mock_allowed()`: Redis mock configuration
- `validate_redis_client()`: Client validation with test support

## 11. Cross-Service Communication Patterns

### 11.1 Service Discovery

Services communicate via:
- Internal API keys for authentication
- Predefined service URLs from configuration
- Health check endpoints for availability

### 11.2 Message Formats

Standardized using Pydantic models:
- Request/response contracts in `shared/contracts`
- Consistent error handling
- Version compatibility through optional fields

### 11.3 Event Schemas

Operation events follow consistent patterns:
```python
{
    "operation_id": "uuid",
    "type": "INDEX|APPEND|REINDEX",
    "status": "PENDING|PROCESSING|COMPLETED|FAILED",
    "collection_id": "uuid",
    "config": {...},
    "meta": {...}
}
```

## 12. Type Definitions and Validation

### 12.1 Custom Types

Common type aliases:
```python
UserId = str  # UUID string
CollectionId = str  # UUID string
DocumentId = str  # UUID string
ChunkId = int  # BigInteger
OperationId = str  # UUID string
```

### 12.2 Pydantic Models

All API contracts use Pydantic for:
- Automatic validation
- Type coercion
- JSON Schema generation
- OpenAPI documentation

### 12.3 Validation Patterns

Common validators:
- UUID format validation
- Path existence checks
- Dimension compatibility
- Configuration validation

## 13. Security Patterns

### 13.1 Input Validation

All user inputs validated through:
- Pydantic model validation
- SQL injection prevention via parameterized queries
- Path traversal prevention
- Size limits on all text fields

### 13.2 Authentication

Shared authentication utilities:
- JWT token validation
- API key hashing and verification
- Permission checking helpers

## 14. Performance Optimizations

### 14.1 Database Optimizations

- Partition-aware queries for chunks table
- Composite indexes for common queries
- Batch operations with optimal sizing
- Connection pooling configuration

### 14.2 Caching Strategies

- Redis-based caching for embeddings
- In-memory caching for model configs
- Query result caching with TTL

### 14.3 Async Patterns

- Proper async/await usage throughout
- Concurrent operations with asyncio.gather
- Streaming for large data processing

## 15. Error Handling and Recovery

### 15.1 Retry Logic

Exponential backoff for:
- Database connections
- Qdrant operations
- External API calls

### 15.2 Graceful Degradation

- Fallback strategies for extraction
- Partial success handling
- Error accumulation and reporting

### 15.3 Logging

Structured logging with:
- Consistent log levels
- Contextual information
- Error tracking integration

## Usage Guidelines

### For Service Developers

1. **Always use repository pattern** for database access
2. **Import from shared package** for common functionality
3. **Follow established patterns** for consistency
4. **Use provided validators** for input validation
5. **Leverage metrics helpers** for monitoring

### For Frontend Integration

1. **Use contract models** for API communication
2. **Handle standardized errors** appropriately
3. **Respect rate limits** and resource constraints
4. **Use provided search contracts** for queries

### For Testing

1. **Use testing utilities** for environment detection
2. **Mock at repository level** for database tests
3. **Use factory functions** for dependency injection
4. **Validate against contracts** for API tests

## Version Compatibility

The shared package maintains backward compatibility through:
- Optional fields in Pydantic models
- Deprecation warnings for changed APIs
- Version-specific behavior flags
- Migration utilities for schema changes