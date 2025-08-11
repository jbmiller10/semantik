# BE-003: Implement Exception Translation Layer

## Ticket Information
- **Priority**: CRITICAL
- **Estimated Time**: 2 hours
- **Dependencies**: BE-001, BE-002
- **Risk Level**: MEDIUM - Critical error context being lost
- **Affected Files**:
  - `packages/shared/chunking/infrastructure/exception_translator.py` (new)
  - `packages/webui/api/v2/chunking.py`
  - `packages/webui/services/chunking_service.py`
  - `packages/shared/chunking/domain/exceptions.py`

## Context

Domain exceptions with rich context are being caught as generic exceptions when crossing architectural layers. This loses critical debugging information and makes production issues nearly impossible to diagnose.

### Current Problems

```python
# infrastructure/streaming/processor.py:260
except Exception as e:
    # Save checkpoint on error for resume
    # PROBLEM: Loses domain exception type and context
    logger.error(f"Error: {e}")  # Just the message, no context!
```

## Requirements

1. Create exception translator for each architectural layer
2. Preserve full error context including stack traces
3. Add correlation IDs for request tracing
4. Map domain exceptions to appropriate HTTP status codes
5. Implement structured error responses
6. Add exception chaining for debugging

## Technical Details

### 1. Create Base Exception Hierarchy

```python
# packages/shared/chunking/infrastructure/exceptions.py

from typing import Optional, Dict, Any
from datetime import datetime
import traceback
import uuid

class BaseChunkingException(Exception):
    """Base exception with context preservation"""
    
    def __init__(
        self,
        message: str,
        code: str,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.cause = cause
        self.timestamp = datetime.utcnow()
        self.stack_trace = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/API responses"""
        return {
            "error": {
                "message": self.message,
                "code": self.code,
                "correlation_id": self.correlation_id,
                "timestamp": self.timestamp.isoformat(),
                "details": self.details,
                "cause": str(self.cause) if self.cause else None
            }
        }

# Domain Layer Exceptions
class DomainException(BaseChunkingException):
    """Base for all domain exceptions"""
    pass

class DocumentTooLargeError(DomainException):
    def __init__(self, size: int, max_size: int, **kwargs):
        super().__init__(
            message=f"Document size {size} exceeds maximum {max_size}",
            code="DOCUMENT_TOO_LARGE",
            details={"size": size, "max_size": max_size},
            **kwargs
        )

class InvalidStateTransition(DomainException):
    def __init__(self, current_state: str, attempted_state: str, **kwargs):
        super().__init__(
            message=f"Cannot transition from {current_state} to {attempted_state}",
            code="INVALID_STATE_TRANSITION",
            details={"current": current_state, "attempted": attempted_state},
            **kwargs
        )

class ChunkingStrategyError(DomainException):
    def __init__(self, strategy: str, reason: str, **kwargs):
        super().__init__(
            message=f"Strategy {strategy} failed: {reason}",
            code="CHUNKING_STRATEGY_ERROR",
            details={"strategy": strategy, "reason": reason},
            **kwargs
        )

# Application Layer Exceptions
class ApplicationException(BaseChunkingException):
    """Base for application layer exceptions"""
    pass

class ValidationException(ApplicationException):
    def __init__(self, field: str, value: Any, reason: str, **kwargs):
        super().__init__(
            message=f"Validation failed for {field}: {reason}",
            code="VALIDATION_ERROR",
            details={"field": field, "value": str(value), "reason": reason},
            **kwargs
        )

class ResourceNotFoundException(ApplicationException):
    def __init__(self, resource_type: str, resource_id: str, **kwargs):
        super().__init__(
            message=f"{resource_type} not found: {resource_id}",
            code="RESOURCE_NOT_FOUND",
            details={"resource_type": resource_type, "resource_id": resource_id},
            **kwargs
        )

class PermissionDeniedException(ApplicationException):
    def __init__(self, user_id: str, resource: str, action: str, **kwargs):
        super().__init__(
            message=f"User {user_id} denied {action} on {resource}",
            code="PERMISSION_DENIED",
            details={"user_id": user_id, "resource": resource, "action": action},
            **kwargs
        )

# Infrastructure Layer Exceptions
class InfrastructureException(BaseChunkingException):
    """Base for infrastructure exceptions"""
    pass

class DatabaseException(InfrastructureException):
    def __init__(self, operation: str, table: str, error: str, **kwargs):
        super().__init__(
            message=f"Database error during {operation} on {table}: {error}",
            code="DATABASE_ERROR",
            details={"operation": operation, "table": table, "error": error},
            **kwargs
        )

class ExternalServiceException(InfrastructureException):
    def __init__(self, service: str, operation: str, error: str, **kwargs):
        super().__init__(
            message=f"External service {service} failed during {operation}: {error}",
            code="EXTERNAL_SERVICE_ERROR",
            details={"service": service, "operation": operation, "error": error},
            **kwargs
        )
```

### 2. Create Exception Translator

```python
# packages/shared/chunking/infrastructure/exception_translator.py

from typing import Type, Dict, Callable, Optional
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

class ExceptionTranslator:
    """Translates exceptions between architectural layers"""
    
    def __init__(self):
        # Domain to Application mappings
        self.domain_to_app_map: Dict[Type[Exception], Callable] = {
            DocumentTooLargeError: self._document_too_large_to_app,
            InvalidStateTransition: self._invalid_state_to_app,
            ChunkingStrategyError: self._strategy_error_to_app,
        }
        
        # Application to API mappings
        self.app_to_api_map: Dict[Type[Exception], int] = {
            ValidationException: 400,
            ResourceNotFoundException: 404,
            PermissionDeniedException: 403,
            DocumentTooLargeError: 413,
            InvalidStateTransition: 409,
        }
    
    def translate_domain_to_application(
        self,
        exc: DomainException,
        correlation_id: Optional[str] = None
    ) -> ApplicationException:
        """Translate domain exception to application exception"""
        
        # Log original exception with full context
        logger.error(
            f"Domain exception occurred",
            extra={
                "exception_type": type(exc).__name__,
                "correlation_id": exc.correlation_id,
                "details": exc.to_dict()
            }
        )
        
        # Get specific translator or use default
        translator = self.domain_to_app_map.get(
            type(exc),
            self._default_domain_to_app
        )
        
        return translator(exc, correlation_id)
    
    def translate_application_to_api(
        self,
        exc: ApplicationException
    ) -> HTTPException:
        """Translate application exception to HTTP exception"""
        
        # Get HTTP status code
        status_code = self.app_to_api_map.get(type(exc), 500)
        
        # Create structured error response
        detail = {
            "error": {
                "message": exc.message,
                "code": exc.code,
                "correlation_id": exc.correlation_id,
                "details": exc.details
            }
        }
        
        # Log API error
        logger.warning(
            f"API error response",
            extra={
                "status_code": status_code,
                "correlation_id": exc.correlation_id,
                "error_code": exc.code
            }
        )
        
        return HTTPException(
            status_code=status_code,
            detail=detail
        )
    
    def translate_infrastructure_to_application(
        self,
        exc: Exception,
        context: Dict[str, Any]
    ) -> ApplicationException:
        """Translate infrastructure exception to application exception"""
        
        if isinstance(exc, DatabaseException):
            # Check if it's a not found error
            if "does not exist" in str(exc).lower():
                return ResourceNotFoundException(
                    resource_type=context.get("resource_type", "Resource"),
                    resource_id=context.get("resource_id", "Unknown"),
                    correlation_id=exc.correlation_id,
                    cause=exc
                )
            else:
                # Generic database error
                return ApplicationException(
                    message="Database operation failed",
                    code="DATABASE_ERROR",
                    details={"original_error": str(exc)},
                    correlation_id=exc.correlation_id,
                    cause=exc
                )
        
        elif isinstance(exc, ExternalServiceException):
            return ApplicationException(
                message=f"External service unavailable: {exc.details.get('service')}",
                code="SERVICE_UNAVAILABLE",
                details=exc.details,
                correlation_id=exc.correlation_id,
                cause=exc
            )
        
        else:
            # Unknown infrastructure error
            return ApplicationException(
                message="An infrastructure error occurred",
                code="INFRASTRUCTURE_ERROR",
                details={"error": str(exc)},
                cause=exc
            )
    
    # Specific translators
    def _document_too_large_to_app(
        self,
        exc: DocumentTooLargeError,
        correlation_id: Optional[str]
    ) -> ApplicationException:
        return ValidationException(
            field="document",
            value=f"{exc.details['size']} bytes",
            reason=f"Exceeds maximum size of {exc.details['max_size']} bytes",
            correlation_id=correlation_id or exc.correlation_id,
            cause=exc
        )
    
    def _invalid_state_to_app(
        self,
        exc: InvalidStateTransition,
        correlation_id: Optional[str]
    ) -> ApplicationException:
        return ApplicationException(
            message=exc.message,
            code="INVALID_OPERATION",
            details=exc.details,
            correlation_id=correlation_id or exc.correlation_id,
            cause=exc
        )
    
    def _strategy_error_to_app(
        self,
        exc: ChunkingStrategyError,
        correlation_id: Optional[str]
    ) -> ApplicationException:
        return ApplicationException(
            message=f"Chunking failed: {exc.details.get('reason')}",
            code="CHUNKING_FAILED",
            details=exc.details,
            correlation_id=correlation_id or exc.correlation_id,
            cause=exc
        )
    
    def _default_domain_to_app(
        self,
        exc: DomainException,
        correlation_id: Optional[str]
    ) -> ApplicationException:
        return ApplicationException(
            message=exc.message,
            code=exc.code,
            details=exc.details,
            correlation_id=correlation_id or exc.correlation_id,
            cause=exc
        )
```

### 3. Update Service Layer to Use Translator

```python
# packages/webui/services/chunking_service.py

from packages.shared.chunking.infrastructure.exception_translator import (
    ExceptionTranslator,
    DomainException,
    ApplicationException
)

class ChunkingService:
    def __init__(self, ...):
        # ... existing init ...
        self.exception_translator = ExceptionTranslator()
    
    async def preview_chunks(
        self,
        strategy: str,
        content: Optional[str] = None,
        document_id: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Preview chunking with proper exception handling"""
        
        correlation_id = correlation_id or str(uuid.uuid4())
        
        try:
            # Validation with proper exceptions
            if not content and not document_id:
                raise ValidationException(
                    field="input",
                    value=None,
                    reason="Either content or document_id must be provided",
                    correlation_id=correlation_id
                )
            
            # Document access check
            if document_id:
                try:
                    await self._validate_document_access(document_id)
                    content = await self._load_document_content(document_id)
                except Exception as e:
                    if isinstance(e, ApplicationException):
                        raise
                    else:
                        # Translate infrastructure exception
                        raise self.exception_translator.translate_infrastructure_to_application(
                            e,
                            {"resource_type": "Document", "resource_id": document_id}
                        )
            
            # Execute chunking with domain exception handling
            try:
                chunks = await self._execute_chunking(
                    strategy=strategy,
                    content=content,
                    config=config_overrides
                )
            except DomainException as e:
                # Translate domain exception
                raise self.exception_translator.translate_domain_to_application(
                    e,
                    correlation_id
                )
            except Exception as e:
                # Unexpected error - wrap with context
                raise ApplicationException(
                    message="Unexpected error during chunking",
                    code="CHUNKING_ERROR",
                    details={
                        "strategy": strategy,
                        "error": str(e),
                        "type": type(e).__name__
                    },
                    correlation_id=correlation_id,
                    cause=e
                )
            
            return {
                "chunks": chunks,
                "correlation_id": correlation_id
            }
            
        except ApplicationException:
            # Already translated, just re-raise
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            logger.exception(
                f"Unexpected error in preview_chunks",
                extra={"correlation_id": correlation_id}
            )
            raise ApplicationException(
                message="An unexpected error occurred",
                code="INTERNAL_ERROR",
                correlation_id=correlation_id,
                cause=e
            )
```

### 4. Update API Router with Exception Translation

```python
# packages/webui/api/v2/chunking.py

from packages.shared.chunking.infrastructure.exception_translator import (
    ExceptionTranslator,
    ApplicationException
)

exception_translator = ExceptionTranslator()

@router.post("/preview")
async def preview_chunks(
    request: PreviewRequest,
    service: ChunkingService = Depends(get_chunking_service),
    correlation_id: str = Header(None, alias="X-Correlation-ID")
) -> PreviewResponse:
    """Preview chunking with proper exception translation"""
    
    # Generate correlation ID if not provided
    if not correlation_id:
        correlation_id = str(uuid.uuid4())
    
    try:
        result = await service.preview_chunks(
            strategy=request.strategy,
            content=request.content,
            document_id=request.document_id,
            config_overrides=request.config,
            correlation_id=correlation_id
        )
        
        return PreviewResponse(
            **result,
            correlation_id=correlation_id
        )
        
    except ApplicationException as e:
        # Translate to HTTP exception
        raise exception_translator.translate_application_to_api(e)
    
    except Exception as e:
        # Unexpected error - log and return generic error
        logger.exception(
            f"Unexpected error in preview endpoint",
            extra={"correlation_id": correlation_id}
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "An unexpected error occurred",
                    "code": "INTERNAL_ERROR",
                    "correlation_id": correlation_id
                }
            }
        )

# Add exception handler for all routes
@router.exception_handler(ApplicationException)
async def application_exception_handler(
    request: Request,
    exc: ApplicationException
):
    """Global handler for application exceptions"""
    return JSONResponse(
        status_code=exception_translator.app_to_api_map.get(type(exc), 500),
        content=exc.to_dict()
    )
```

### 5. Add Context Preservation in Infrastructure

```python
# packages/shared/chunking/infrastructure/streaming/processor.py

class StreamingProcessor:
    async def process_chunk(self, data: bytes, correlation_id: str):
        """Process chunk with exception context preservation"""
        
        try:
            # Processing logic...
            pass
            
        except DomainException as e:
            # Domain exception - preserve and re-raise
            logger.error(
                "Domain exception in streaming processor",
                extra={
                    "correlation_id": correlation_id,
                    "exception": e.to_dict()
                }
            )
            raise
            
        except asyncio.TimeoutError as e:
            # Convert to domain exception with context
            raise ChunkingStrategyError(
                strategy="streaming",
                reason="Processing timeout exceeded",
                correlation_id=correlation_id,
                cause=e
            )
            
        except MemoryError as e:
            # Convert to domain exception
            raise DocumentTooLargeError(
                size=len(data),
                max_size=self.max_buffer_size,
                correlation_id=correlation_id,
                cause=e
            )
            
        except Exception as e:
            # Infrastructure exception - wrap with context
            raise InfrastructureException(
                message=f"Streaming processor failed: {str(e)}",
                code="STREAMING_ERROR",
                details={
                    "data_size": len(data),
                    "processor_state": self.get_state()
                },
                correlation_id=correlation_id,
                cause=e
            )
```

## Acceptance Criteria

1. **Exception Translation**
   - [ ] All domain exceptions properly translated
   - [ ] All application exceptions mapped to HTTP codes
   - [ ] Infrastructure exceptions wrapped with context
   - [ ] No generic Exception catches without context

2. **Context Preservation**
   - [ ] Correlation IDs tracked through all layers
   - [ ] Stack traces preserved for debugging
   - [ ] Exception chaining implemented
   - [ ] Original cause accessible

3. **Error Responses**
   - [ ] Structured error format consistent
   - [ ] User-friendly messages
   - [ ] Debug details in development
   - [ ] Sensitive data scrubbed in production

4. **Logging**
   - [ ] All exceptions logged with context
   - [ ] Correlation IDs in all log entries
   - [ ] Stack traces in error logs
   - [ ] Metrics for exception types

## Testing Requirements

1. **Unit Tests**
   ```python
   def test_domain_exception_translation():
       translator = ExceptionTranslator()
       
       domain_exc = DocumentTooLargeError(
           size=1000000,
           max_size=100000
       )
       
       app_exc = translator.translate_domain_to_application(domain_exc)
       
       assert isinstance(app_exc, ValidationException)
       assert app_exc.cause == domain_exc
       assert app_exc.correlation_id == domain_exc.correlation_id
   
   def test_exception_context_preservation():
       exc = ApplicationException(
           message="Test error",
           code="TEST",
           details={"key": "value"},
           cause=ValueError("Original error")
       )
       
       assert exc.cause.args[0] == "Original error"
       assert exc.details["key"] == "value"
       assert exc.correlation_id is not None
   
   async def test_service_exception_handling():
       service = ChunkingService()
       
       with pytest.raises(ApplicationException) as exc_info:
           await service.preview_chunks(
               strategy="invalid",
               content="test"
           )
       
       assert exc_info.value.code == "CHUNKING_FAILED"
       assert exc_info.value.correlation_id is not None
   ```

2. **Integration Tests**
   - Test exception flow through all layers
   - Verify HTTP status codes
   - Check error response format
   - Test correlation ID tracking

## Rollback Plan

1. Keep original exception handling as fallback
2. Feature flag for new exception system
3. Monitor error rates after deployment
4. Revert if error handling degrades

## Success Metrics

- All exceptions have correlation IDs
- No "Unknown error" responses
- Debug time reduced by 50%
- Error categorization accurate
- No sensitive data in error responses

## Notes for LLM Agent

- Preserve all exception context
- Never swallow exceptions silently
- Always add correlation IDs
- Test exception paths thoroughly
- Consider security in error messages
- Log before re-raising
- Use exception chaining for cause tracking