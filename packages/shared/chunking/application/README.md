# Chunking Application Layer

## Overview

This application layer implements the business use cases for the chunking system following clean architecture principles. It orchestrates interactions between the domain layer and infrastructure services without containing any business logic or infrastructure code itself.

## Structure

```
application/
├── use_cases/          # Business operation orchestrators
│   ├── preview_chunking.py      # Generate preview of chunking results
│   ├── process_document.py      # Full document processing with persistence
│   ├── compare_strategies.py    # Compare multiple chunking strategies
│   ├── get_operation_status.py  # Query operation status and progress
│   └── cancel_operation.py      # Cancel in-progress operations
├── dto/                # Data Transfer Objects
│   ├── requests.py     # Input DTOs with validation
│   └── responses.py    # Output DTOs with serialization
├── interfaces/         # Contracts for infrastructure
│   ├── repositories.py # Repository interfaces (data persistence)
│   └── services.py     # Service interfaces (external services)
└── tests/             # Example tests with mocked dependencies
```

## Key Design Patterns

### 1. Use Case Pattern
Each use case:
- Has a single public `execute()` method
- Represents one complete business operation
- Manages transaction boundaries
- Handles error mapping and notifications
- Uses dependency injection for all infrastructure needs

### 2. Repository Pattern
Repository interfaces define contracts for data persistence:
- `ChunkRepository` - Manages chunk storage
- `ChunkingOperationRepository` - Manages operation lifecycle
- `CheckpointRepository` - Handles operation checkpointing
- `DocumentRepository` - Document metadata management

### 3. Unit of Work Pattern
The `UnitOfWork` interface ensures transactional consistency:
- Manages database transactions
- Provides access to repositories within transaction
- Ensures all-or-nothing persistence

### 4. DTO Pattern
Clear separation between internal and external representations:
- Request DTOs include validation logic
- Response DTOs provide serialization methods
- No domain entities exposed outside application layer

## Use Cases

### 1. PreviewChunkingUseCase
**Purpose**: Generate quick preview without full processing
- Loads first 10KB of document
- Applies selected strategy
- Returns first 5 chunks
- Estimates total chunks for full document

### 2. ProcessDocumentUseCase
**Purpose**: Full document processing with persistence
- Processes entire document
- Saves chunks to database
- Supports checkpointing for resumability
- Manages transaction boundaries
- Emits progress events

### 3. CompareStrategiesUseCase
**Purpose**: Help users choose optimal chunking strategy
- Runs multiple strategies on same content
- Collects performance metrics
- Provides comparison analysis
- Recommends best strategy

### 4. GetOperationStatusUseCase
**Purpose**: Query operation progress and results
- Returns current status and progress
- Optionally includes chunk details
- Provides performance metrics
- Shows error details for failed operations

### 5. CancelOperationUseCase
**Purpose**: Cancel in-progress operations
- Validates cancellation is allowed
- Updates operation status
- Optionally cleans up created chunks
- Removes checkpoints
- Ensures transactional consistency

## Dependency Injection

All use cases receive dependencies through constructor injection:

```python
use_case = ProcessDocumentUseCase(
    unit_of_work=uow_implementation,
    document_service=doc_service_impl,
    strategy_factory=strategy_factory_impl,
    notification_service=notification_impl,
    metrics_service=metrics_impl  # Optional
)
```

## Transaction Management

Transaction boundaries are managed at the use case level:

```python
async with self.unit_of_work:
    try:
        # All database operations within transaction
        operation = await self._create_operation(request)
        chunks = await self._process_document(operation)
        await self._save_chunks(chunks)
        
        # Commit on success
        await self.unit_of_work.commit()
        
        # Send events after commit
        await self._send_completion_event(operation)
        
    except Exception as e:
        # Automatic rollback
        await self.unit_of_work.rollback()
        raise
```

## Error Handling

Three levels of error handling:

1. **Domain Errors**: Validation and business rule violations
2. **Infrastructure Errors**: Database, file system, network failures
3. **Unexpected Errors**: Caught and logged with context

## Testing

All use cases are fully testable with mocked dependencies:

```python
# Mock all dependencies
mock_repo = Mock(ChunkRepository)
mock_doc_service = Mock(DocumentService)

# Create use case with mocks
use_case = PreviewChunkingUseCase(
    mock_repo,
    mock_doc_service,
    Mock(NotificationService)
)

# Test execution
response = await use_case.execute(request)

# Verify interactions
mock_doc_service.load.assert_called_once()
```

## Integration with Domain Layer

The application layer depends on domain entities and services that are expected to be available in `packages/shared/chunking/domain/`:

- Domain entities (Chunk, ChunkingOperation, etc.)
- Value objects (ChunkConfig, OperationStatus, etc.)
- Domain services (ChunkingStrategy implementations)

## Next Steps

The infrastructure layer (ARCH-003) will:
1. Implement all repository interfaces
2. Implement service interfaces
3. Provide concrete Unit of Work implementation
4. Handle database connections and transactions
5. Integrate with external services

The presentation layer (ARCH-004) will:
1. Create API endpoints that use these use cases
2. Handle HTTP request/response mapping
3. Manage authentication and authorization
4. Provide WebSocket support for real-time updates