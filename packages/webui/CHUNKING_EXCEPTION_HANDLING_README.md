# Chunking Exception Handling Implementation

This document describes the exception handling and correlation ID middleware implementation for the chunking system in Semantik.

## Overview

The implementation provides:
1. **Structured exception handling** for all chunking-related errors
2. **Correlation ID middleware** for request tracing across the application
3. **Production-ready error responses** with appropriate HTTP status codes
4. **Security-focused error sanitization** to prevent information leakage

## Components Created

### 1. Correlation Middleware (`/packages/webui/middleware/correlation.py`)

The correlation middleware provides end-to-end request tracing:

- **Automatic ID Generation**: Generates UUID correlation IDs for requests without one
- **Header Propagation**: Extracts correlation IDs from `X-Correlation-ID` request headers
- **Context Management**: Stores correlation IDs in context variables accessible throughout the request
- **Response Headers**: Adds correlation IDs to all response headers
- **Logging Integration**: Automatically includes correlation IDs in all log messages

Key functions:
- `get_correlation_id()`: Get the current correlation ID from context
- `set_correlation_id(id)`: Set a correlation ID in the current context
- `get_or_generate_correlation_id(request)`: Get or generate a correlation ID
- `configure_logging_with_correlation()`: Configure logging to include correlation IDs

### 2. Exception Handlers (`/packages/webui/api/chunking_exception_handlers.py`)

Provides FastAPI exception handlers for each chunking exception type:

| Exception Type | HTTP Status | Description |
|---------------|-------------|-------------|
| `ChunkingMemoryError` | 507 (Insufficient Storage) | Memory limits exceeded |
| `ChunkingTimeoutError` | 504 (Gateway Timeout) | Operation timeout |
| `ChunkingValidationError` | 422 (Unprocessable Entity) | Invalid input parameters |
| `ChunkingStrategyError` | 501/500 | Strategy not implemented or failed |
| `ChunkingResourceLimitError` | 503 (Service Unavailable) | Resource limits exceeded |
| `ChunkingPartialFailureError` | 207 (Multi-Status) | Some items succeeded, others failed |
| `ChunkingConfigurationError` | 500 | Invalid configuration |
| `ChunkingDependencyError` | 503 | External dependency failure |

Features:
- **Structured JSON responses** with error details and recovery hints
- **Error sanitization** in production mode to prevent information leakage
- **Appropriate logging levels** (error for 5xx, warning for 4xx)
- **Retry-After headers** for temporary failures
- **Correlation ID preservation** throughout error handling

### 3. Tests

Comprehensive test coverage for both components:

- **`test_correlation_middleware.py`**: Tests for correlation ID generation, propagation, and logging
- **`test_chunking_exception_handlers.py`**: Tests for all exception handlers and error responses

## Integration Guide

To integrate these components into the main application:

### 1. Update `main.py`

```python
# Add imports
from .api.chunking_exception_handlers import register_chunking_exception_handlers
from .middleware.correlation import CorrelationMiddleware, configure_logging_with_correlation

# In lifespan function, add:
configure_logging_with_correlation()

# In create_app function:
app.add_middleware(CorrelationMiddleware)  # Add BEFORE other middleware
register_chunking_exception_handlers(app)
```

### 2. Use in API Endpoints

```python
from fastapi import Depends
from ..middleware.correlation import get_correlation_id
from ..api.chunking_exceptions import ChunkingValidationError

@router.post("/process")
async def process_document(
    document_id: str,
    correlation_id: str = Depends(get_correlation_id),
):
    if not document_id:
        raise ChunkingValidationError(
            detail="Document ID is required",
            correlation_id=correlation_id,
            field_errors={"document_id": ["This field is required"]},
        )
```

### 3. Use in Services

```python
from ..middleware.correlation import get_correlation_id
from ..api.chunking_exceptions import ChunkingMemoryError

async def process_large_document(self, content: str):
    correlation_id = get_correlation_id()
    
    if memory_usage_too_high():
        raise ChunkingMemoryError(
            detail="Insufficient memory",
            correlation_id=correlation_id,
            operation_id="doc-processing",
            memory_used=current_memory,
            memory_limit=max_memory,
        )
```

## Security Considerations

The implementation includes several security features:

1. **Error Sanitization**: In production mode, sensitive information is automatically redacted:
   - File system paths
   - Database connection strings
   - API keys and tokens
   - Internal service names

2. **Controlled Information Disclosure**: Error responses include only necessary information
3. **Request Context Filtering**: Query parameters are excluded from error responses in production

## Best Practices

1. **Always include correlation IDs** when raising chunking exceptions
2. **Use appropriate exception types** to ensure correct HTTP status codes
3. **Provide helpful recovery hints** in error messages
4. **Log errors at appropriate levels** based on severity
5. **Test error handling paths** to ensure proper behavior

## Example Error Response

```json
{
  "error_code": "CHUNKING_MEMORY_EXCEEDED",
  "detail": "Insufficient memory to process document",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "operation_id": "op-123",
  "type": "ChunkingMemoryError",
  "memory_used_mb": 2048.0,
  "memory_limit_mb": 1024.0,
  "recovery_hint": "Try processing smaller documents or use a more memory-efficient strategy",
  "request": {
    "method": "POST",
    "path": "/api/v2/chunking/process",
    "query_params": null
  }
}
```

## Testing

Run the tests with:

```bash
pytest tests/unit/test_correlation_middleware.py -v
pytest tests/unit/test_chunking_exception_handlers.py -v
```

## Future Enhancements

1. **Distributed Tracing**: Integrate with OpenTelemetry for distributed tracing
2. **Metrics Collection**: Add Prometheus metrics for error rates by type
3. **Rate Limiting**: Add rate limiting based on correlation IDs
4. **Error Recovery**: Implement automatic retry logic for transient failures