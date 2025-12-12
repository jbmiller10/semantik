# Chunking Exception Handling

Structured exception handling and correlation ID middleware for the chunking system.

## Components

### Correlation Middleware (`middleware/correlation.py`)

Request tracing via `X-Correlation-ID` header. Auto-generates UUIDs if not provided.

```python
from webui.middleware.correlation import get_correlation_id
correlation_id = get_correlation_id()
```

### Exception Handlers (`api/chunking_exception_handlers.py`)

| Exception | HTTP Status | Description |
|-----------|-------------|-------------|
| ChunkingMemoryError | 507 | Memory limits exceeded |
| ChunkingTimeoutError | 504 | Operation timeout |
| ChunkingValidationError | 422 | Invalid input |
| ChunkingStrategyError | 501/500 | Strategy not implemented/failed |
| ChunkingResourceLimitError | 503 | Resource limits exceeded |
| ChunkingPartialFailureError | 207 | Partial success |
| ChunkingConfigurationError | 500 | Invalid config |
| ChunkingDependencyError | 503 | External dependency failure |

### Error Classifier (`utils/error_classifier.py`)

Centralized exception → error type translation. Singleton: `get_default_chunking_error_classifier()`.

## Integration

```python
# main.py
from .api.chunking_exception_handlers import register_chunking_exception_handlers
from .middleware.correlation import CorrelationMiddleware, configure_logging_with_correlation

configure_logging_with_correlation()
app.add_middleware(CorrelationMiddleware)  # Add BEFORE other middleware
register_chunking_exception_handlers(app)
```

## Usage

```python
from webui.middleware.correlation import get_correlation_id
from webui.api.chunking_exceptions import ChunkingValidationError

@router.post("/process")
async def process_document(document_id: str):
    if not document_id:
        raise ChunkingValidationError(
            detail="Document ID required",
            correlation_id=get_correlation_id(),
            field_errors={"document_id": ["Required"]},
        )
```

## Testing

```bash
pytest tests/unit/test_correlation_middleware.py tests/unit/test_chunking_exception_handlers.py -v
```
