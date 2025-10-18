# Exception Handling & Validation Audit Report
**Semantik Codebase - Comprehensive Security & Error Recovery Analysis**

---

## CRITICAL FINDINGS

### CRITICAL - Silent Failures Without Logging

#### Finding 1: Bare Exception Swallowing in Registry (Metric Registration)
**Location**: `/home/john/semantik/packages/vecpipe/search_api.py:69-71`
```python
try:
    for collector in registry._collector_to_names:
        if hasattr(collector, "_name") and collector._name == name:
            return collector
except AttributeError:
    pass  # BAD: Silent failure, no logging
```
**Risk**: Metric registration silently fails, creating duplicate metrics without any indication to operators
**Consequence**: Prometheus metrics become inconsistent, monitoring breaks silently
**Fix**: Log the AttributeError and add explicit context about fallback behavior
```python
except AttributeError as e:
    logger.debug(f"Registry structure different, creating new metric: {e}")
```

#### Finding 2: Collection Existence Check with Bare Exception
**Location**: `/home/john/semantik/packages/vecpipe/maintenance.py:142-149`
```python
def collection_exists(self, collection_name: str) -> bool:
    """Check if a collection exists in Qdrant"""
    try:
        self.client.get_collection(collection_name)
        return True
    except Exception:  # BAD: Catches all exceptions including network errors
        # Collection doesn't exist or other error
        return False
```
**Risk**: Distinguishes between "not found" and "connection error" as identical - masks real problems
**Consequence**: Silent failures when Qdrant is down appear as empty collections
**Fix**: Catch specific exceptions (e.g., `QdrantException`)
```python
except QdrantException as e:
    if "does not exist" in str(e):
        return False
    logger.error(f"Qdrant error checking collection: {e}")
    raise
```

#### Finding 3: Regex Compilation Silently Failing
**Location**: `/home/john/semantik/packages/shared/text_processing/strategies/hybrid_chunker.py:23-28`
```python
def safe_regex_findall(pattern: str | Pattern[str], text: str, flags: int = 0) -> list[str]:
    """Mock safe regex findall for test compatibility."""
    try:
        if isinstance(pattern, str):
            pattern = re.compile(pattern, flags)
        return pattern.findall(text)
    except Exception:  # BAD: Silent failure
        return []
```
**Risk**: Invalid regex patterns silently return empty results instead of raising
**Consequence**: Parsing silently skips content without error indication
**Fix**: Log and re-raise or at least indicate the failure
```python
except re.error as e:
    logger.error(f"Regex compilation failed for pattern: {pattern}, error: {e}")
    raise
```

---

## HIGH PRIORITY FINDINGS

### HIGH - Overly Broad Exception Catching

#### Finding 4: Generic Exception Masking in API Endpoints
**Location**: `/home/john/semantik/packages/webui/api/v2/chunking.py:123-128`
```python
try:
    strategy_dtos = await service.get_available_strategies_for_api()
    return [cast(StrategyInfo, dto.to_api_model()) for dto in strategy_dtos]
except Exception as e:  # Catches ALL exceptions
    logger.error(f"Failed to list strategies: {e}")
    raise HTTPException(status_code=500, detail="Failed to list strategies") from e
```
**Risk**: All errors (timeout, permission, database, etc.) return generic 500 - client can't distinguish
**Impact**: 
- No specific error codes for different failure modes
- Same response for recoverable vs non-recoverable errors
- Rate limiting indistinguishable from real failures
**Fix**: Catch specific exceptions and return appropriate HTTP status codes
```python
except EntityNotFoundError as e:
    raise HTTPException(status_code=404, detail=str(e)) from e
except TimeoutError as e:
    raise HTTPException(status_code=504, detail="Service timeout") from e
except Exception as e:
    logger.exception("Unexpected error listing strategies")
    raise HTTPException(status_code=500) from e
```

#### Finding 5: Chunking Service Exception Masking
**Location**: `/home/john/semantik/packages/webui/api/v2/chunking.py:332-337`
```python
try:
    compare_dto = await service.compare_strategies_for_api(...)
    return cast(CompareResponse, compare_dto.to_api_model())
except HTTPException:
    raise
except Exception as e:  # BAD: Masks validation, timeout, etc.
    logger.error(f"Strategy comparison failed: {e}")
    raise HTTPException(status_code=500, detail="Failed to compare strategies") from e
```
**Risk**: Doesn't distinguish between:
- Validation errors (should be 400)
- Not found errors (should be 404)
- Timeouts (should be 504)
- Internal errors (500)
**Fix**: Create exception mapping layer (see exception_translator pattern already in codebase)

#### Finding 6: Directory Scan Without Path Validation  
**Location**: `/home/john/semantik/packages/webui/api/v2/directory_scan.py:66-138`
```python
try:
    scan_path = Path(scan_request.path)
    if not scan_path.is_absolute():
        raise ValidationError("Path must be absolute")  # Check is AFTER Path creation
    
    scan_task = asyncio.create_task(service.scan_directory_preview(...))
    # But no further validation before processing
except ValidationError as e:
    raise HTTPException(...) from e
except ValueError as e:  # Catch-all for Path operations
    raise HTTPException(...) from e
except FileNotFoundError as e:  # Good - specific
    raise HTTPException(...) from e
except PermissionError as e:  # Good - specific
    raise HTTPException(...) from e
except Exception as e:  # BAD: Still has generic handler
    logger.error(f"Failed to scan directory: {e}")
    raise HTTPException(...) from e
```
**Risk**: Path operations could raise other OSError subclasses not caught specifically
**Fix**: Explicitly handle OSError and its subclasses

---

### HIGH - Missing Input Validation

#### Finding 7: Vector Dimensions Not Validated at API Layer
**Location**: `/home/john/semantik/packages/vecpipe/search_api.py` (EmbedRequest model - LINE 121)
```python
class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=1000)
    model_name: str = Field(...)  # NO VALIDATION
    batch_size: int = Field(32, ge=1, le=256)
```
**Risk**: 
- Invalid model names accepted at API boundary
- No validation of Qdrant collection compatibility
- Dimension mismatches only caught deep in service layer
**Fix**: Add Pydantic validators or move to dependency injection
```python
@field_validator("model_name")
def validate_model_name(cls, v: str) -> str:
    if v not in SUPPORTED_EMBEDDING_MODELS:
        raise ValueError(f"Unsupported model: {v}")
    return v
```

#### Finding 8: No Bounds Checking on User Inputs
**Location**: `/home/john/semantik/packages/webui/api/v2/chunking.py:442`
```python
chunking_request = ChunkingOperationRequest.model_validate(payload)
# payload.content is never validated for size in schema
# No check before passing to service
```
**Risk**: Large content could be accepted and cause memory issues downstream
**Fix**: Add max_length validation to Pydantic model or add guard at endpoint

#### Finding 9: Operation Config Not Validated
**Location**: `/home/john/semantik/packages/webui/api/internal.py:47-80`
```python
class CompleteReindexRequest(BaseModel):
    new_config: dict[str, Any] | None = None  # No validation of structure
    
    # Only validates IDs and count, not the config content
```
**Risk**: Arbitrary config values could be stored without validation
**Fix**: Use Pydantic model for config structure instead of dict[str, Any]

---

### HIGH - Inconsistent Error Response Formats

#### Finding 10: Duplicate Exception Handling Across Endpoints
**Location**: `/home/john/semantik/packages/webui/api/v2/operations.py:46-80`
```python
try:
    operation = await service.get_operation(...)
except EntityNotFoundError as e:
    raise HTTPException(status_code=404, detail=f"Operation '{operation_uuid}' not found") from e
except AccessDeniedError as e:
    raise HTTPException(status_code=403, detail="You don't have access...") from e
except Exception as e:
    logger.error(f"Failed to get operation: {e}")
    raise HTTPException(status_code=500, detail="Failed to get operation") from e
```
**REPEATED** in nearly every endpoint in collections.py, search.py, documents.py
**Risk**: 
- Code duplication makes maintenance hard
- Inconsistent error messages across endpoints
- Difficult to add global error handling policies
**Fix**: Create middleware or exception handlers for consistent error responses

#### Finding 11: No Error Response Standard
**Location**: `/home/john/semantik/packages/shared/contracts/errors.py` shows good models but NOT USED consistently
```python
class ErrorResponse(BaseModel):
    error: str
    message: str
    details: dict | list | None
    request_id: str | None
    timestamp: str | None
    status_code: int | None
```
**Risk**: Endpoints don't use these models, each creates ad-hoc HTTPException detail strings
**Fix**: Create exception handler middleware that converts ALL exceptions to ErrorResponse models

---

### HIGH - Missing Exception Chains

#### Finding 12: Exception Context Lost in Maintenance Service
**Location**: `/home/john/semantik/packages/vecpipe/maintenance.py:234-238`
```python
try:
    with Path(settings.cleanup_log).open("a") as f:
        f.write(json.dumps(summary) + "\n")
except Exception as e:
    logger.error(f"Failed to write cleanup log: {e}")
    # Exception swallowed here - doesn't propagate or provide context
```
**Risk**: Cleanup failure silently stops, but caller doesn't know cleanup log failed
**Fix**: Re-raise or add more context about consequences
```python
except IOError as e:
    logger.error(f"Failed to write cleanup log: {e} - cleanup operations may not be auditable")
    raise RuntimeError(f"Critical: cleanup log failed: {e}") from e
```

#### Finding 13: Async Exception Handling in Preview Generation
**Location**: `/home/john/semantik/packages/webui/api/v2/chunking.py:253-276`
```python
try:
    preview_dto = await service.preview_chunking(...)
except ApplicationError as e:
    raise exception_translator.translate_application_to_api(e) from e  # GOOD - chains
except HTTPException:
    raise
except Exception:  # BAD - no chain, no context
    logger.exception("Unexpected error in preview endpoint", extra={"correlation_id": correlation_id})
    raise HTTPException(...) from None  # from None loses stack trace
```
**Risk**: Generic exceptions lose their original context with `from None`
**Fix**: Keep exception chain with `from e`

---

## MEDIUM PRIORITY FINDINGS

### MEDIUM - Inconsistent Error Handling Patterns

#### Finding 14: Some Endpoints Check Status, Others Don't
**Location**: Inconsistent across codebase
```python
# GOOD - explicit status checking
if not validation_result.get("valid", True):
    errors = validation_result.get("errors", [])
    raise HTTPException(status_code=400, detail=...)

# BAD - assumes validation_result has these keys
if not preview_dto:  # Type: PreviewResponse | None - not always clear
    raise HTTPException(...)
```
**Risk**: API contracts not enforced - clients can't rely on response structure

#### Finding 15: Exception Throttling in WebSocket
**Location**: `/home/john/semantik/packages/webui/websocket_manager.py:119-120`
```python
for websocket in list(websockets):
    with contextlib.suppress(Exception):
        await websocket.close()  # Silently suppresses ALL exceptions
```
**Risk**: Real errors (connection refused) indistinguishable from already-closed connections
**Fix**: Log specific exception types before suppressing
```python
for websocket in list(websockets):
    try:
        await websocket.close()
    except RuntimeError:
        pass  # Already closed, expected
    except Exception as e:
        logger.warning(f"Error closing websocket: {e}")
```

#### Finding 16: TokenChunker Silent Fallback
**Location**: `/home/john/semantik/packages/shared/text_processing/chunking.py:34-38`
```python
try:
    self.tokenizer = tiktoken.get_encoding(model_name)
except Exception:
    # Fallback to default encoding
    self.tokenizer = tiktoken.get_encoding("cl100k_base")  # No logging
```
**Risk**: Invalid model_name silently falls back - should fail fast
**Fix**: Log the issue and document the fallback
```python
except Exception as e:
    logger.warning(f"Failed to get encoding '{model_name}': {e}, using cl100k_base")
    self.tokenizer = tiktoken.get_encoding("cl100k_base")
```

---

### MEDIUM - Exception Translation Missing

#### Finding 17: ApplicationError Not Consistently Handled
**Location**: `/home/john/semantik/packages/webui/api/v2/chunking.py`
```python
# Some endpoints have this:
except ApplicationError as e:
    raise exception_translator.translate_application_to_api(e) from e

# But not all - many have:
except Exception as e:
    raise HTTPException(status_code=500, ...) from e
```
**Risk**: Some application errors get proper HTTP mapping, others don't
**Fix**: Use exception middleware to catch ALL ApplicationErrors

#### Finding 18: Database Exceptions Not Mapped Consistently
**Location**: `/home/john/semantik/packages/webui/api/v2/operations.py:75-80`
```python
except ValidationError as e:
    raise HTTPException(status_code=400, detail=str(e)) from e
except Exception as e:  # ValidationError can come from DB layer
    logger.error(f"Failed to cancel operation: {e}")
    raise HTTPException(status_code=500, ...) from e
```
**Risk**: Can't distinguish between request validation and database validation errors

---

## LOW PRIORITY FINDINGS

### LOW - Non-Critical Logging Issues

#### Finding 19: Inconsistent Log Levels
**Location**: Multiple files use `.error()` for recoverable failures
```python
# These could be .warning() since operation continues:
except Exception as e:
    logger.error(f"Failed to clear preview cache: {e}")
    # Don't raise error for cache clear failures
```

#### Finding 20: Missing Exception Severity Context
**Location**: Cleanup and maintenance operations don't distinguish severity
```python
logger.error(f"Failed to delete collection {col_name}: {e}")
# Is this critical or just informational?
```

---

## SUMMARY OF ISSUES BY CATEGORY

### By Severity
- **CRITICAL**: 3 findings (Silent failures, bare excepts, swallowed exceptions)
- **HIGH**: 9 findings (Overly broad exceptions, missing validation, inconsistent responses)
- **MEDIUM**: 4 findings (Inconsistent patterns, missing translation, silent fallbacks)
- **LOW**: 2 findings (Log level inconsistencies)

### By Type
- **Bare Exception Handling**: 8 instances (vecpipe, webui API endpoints)
- **Missing Validation**: 4 instances (model names, config structures, content size)
- **Inconsistent Error Mapping**: 7 instances (HTTP exceptions, error responses)
- **Silent Failures**: 5 instances (regex, metrics, collections, cleanup logs)
- **Lost Exception Context**: 3 instances (async handlers, websocket cleanup)

---

## ARCHITECTURAL RECOMMENDATIONS

### 1. Create Global Exception Middleware
```python
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Map all exceptions to standard ErrorResponse
    # Include correlation ID, timestamp
    # Return appropriate HTTP status code
    return ErrorResponse(...)
```

### 2. Use Exception Translation Consistently
The codebase already has `exception_translator` - use it everywhere:
- Create ApplicationError subclasses for domain-specific errors
- Map each to appropriate HTTP status code
- Use in middleware to ensure consistency

### 3. Implement Specific Exception Types
Instead of generic Exception catching:
```python
class ChunkingStrategyError(ApplicationError): pass
class ConfigurationError(ApplicationError): pass
class InsufficientResourcesError(ApplicationError): pass
```

### 4. Add Pre-Validation at API Boundary
Use Pydantic validators on all input models:
- Model names must be in SUPPORTED_MODELS
- Content size must be <= MAX_CONTENT_SIZE
- Collections IDs must be valid UUIDs
- Ensure fail-fast behavior

### 5. Standardize Async Exception Handling
Create helper for graceful resource cleanup:
```python
async def safe_close_resources(resources):
    for resource in resources:
        try:
            await resource.close()
        except (RuntimeError, ConnectionError):
            pass  # Expected for closed connections
        except Exception as e:
            logger.warning(f"Error closing resource: {e}")
```

---

## FIXES BY PRIORITY

### Immediate (This Sprint)
1. Add exception middleware for global error handling
2. Fix bare exception catches in search_api.py and maintenance.py
3. Add input validation to API request models
4. Create application error subclasses and use exception_translator everywhere

### Short-term (Next Sprint)
1. Implement specific exception handlers for each service
2. Add pre-validation guards at API boundaries
3. Standardize error response format across all endpoints
4. Add logging to all silent failure points

### Medium-term (Refactoring)
1. Audit all service layer methods for exception contract clarity
2. Migrate all database exceptions to standard error codes
3. Create comprehensive error handling documentation
4. Add exception handling tests for all error paths

