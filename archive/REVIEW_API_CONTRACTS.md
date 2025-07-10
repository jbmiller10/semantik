# API Contracts Review for Project Semantik

## Executive Summary

The API contracts are well-defined in the `packages/shared/contracts/` directory with proper separation of concerns. The services are communicating via HTTP as intended, though there are some areas where the contracts could be more consistently applied.

## 1. Contract Definition Status ✅

### Location: `packages/shared/contracts/`

The contracts are properly organized into three modules:

1. **search.py** - Search-related contracts
   - `SearchRequest` / `SearchResponse`
   - `BatchSearchRequest` / `BatchSearchResponse`
   - `HybridSearchRequest` / `HybridSearchResponse`
   - `PreloadModelRequest` / `PreloadModelResponse`

2. **jobs.py** - Job management contracts
   - `CreateJobRequest` / `JobResponse`
   - `AddToCollectionRequest`
   - `JobListResponse`
   - `JobStatus` (enum)
   - `JobMetrics` / `JobUpdateRequest` / `JobFilter`

3. **errors.py** - Error response contracts
   - `ErrorResponse` (base class)
   - Specific error types: `ValidationErrorResponse`, `AuthenticationErrorResponse`, etc.
   - Helper functions: `create_validation_error()`, `create_not_found_error()`, etc.

### Key Features of Contracts:

1. **Field Validation**: All contracts use Pydantic with proper field validation
   - Length constraints (min_length, max_length)
   - Range constraints (ge, le)
   - Custom validators for complex logic

2. **Documentation**: Fields include descriptions for API documentation

3. **Backward Compatibility**: 
   - SearchRequest accepts both 'k' and 'top_k' (with 'k' as canonical)
   - JobResponse supports both 'id' and 'job_id', 'error' and 'error_message'

## 2. API Implementation Status

### WebUI → VecPipe Communication ✅

**Configuration**: `packages/shared/config/webui.py`
```python
SEARCH_API_URL: str = "http://localhost:8000"
```

**Implementation**: `packages/webui/api/search.py`
- Properly proxies search requests to vecpipe using HTTP
- Uses httpx AsyncClient for non-blocking requests
- Implements proper timeouts and retry logic for model loading scenarios
- Transforms responses to match frontend expectations

**Example of proper proxying**:
```python
async with httpx.AsyncClient(timeout=timeout) as client:
    response = await client.post(f"{settings.SEARCH_API_URL}/search", json=vector_search_params)
    response.raise_for_status()
```

### VecPipe Search API Implementation ✅

**Implementation**: `packages/vecpipe/search_api.py`
- Properly imports contracts from `shared.contracts.search`
- All endpoints return contract-compliant responses
- Uses FastAPI response_model for automatic validation

**Example**:
```python
@app.post("/search", response_model=SearchResponse)
async def search_post(request: SearchRequest = Body(...)) -> SearchResponse:
    # Implementation follows contract
    return SearchResponse(
        query=request.query,
        results=results,
        num_results=len(results),
        # ... other fields as per contract
    )
```

## 3. Service Communication Patterns ✅

### HTTP-Only Communication
- WebUI communicates with VecPipe exclusively via HTTP REST APIs
- No direct imports between packages (verified via grep)
- Clean separation of concerns

### Shared Dependencies
- Both services import from `shared/` package:
  - Configuration (`shared.config`)
  - Contracts (`shared.contracts`)
  - Embedding service (`shared.embedding`)
  - Metrics (`shared.metrics`)

## 4. Error Handling Analysis ⚠️

### Issue: Inconsistent Error Contract Usage

While error contracts are well-defined, they're not consistently used:

**Current Implementation** (webui/api/search.py):
```python
raise HTTPException(
    status_code=507,
    detail={
        "error": "insufficient_memory",
        "message": error_detail.get("message", "Insufficient GPU memory for reranking"),
        "suggestion": error_detail.get("suggestion", "Try using a smaller model or different quantization")
    }
)
```

**Should Use Contract**:
```python
from shared.contracts.errors import create_insufficient_memory_error

error_response = create_insufficient_memory_error(
    required="4GB",
    available="2GB", 
    suggestion="Try using a smaller model or different quantization"
)
raise HTTPException(status_code=507, detail=error_response.model_dump())
```

## 5. Specific Code Examples

### Proper Contract Usage ✅

**Jobs API** (`webui/api/jobs.py`):
```python
from shared.contracts.jobs import CreateJobRequest as SharedCreateJobRequest
CreateJobRequest = SharedCreateJobRequest  # Direct usage of contract
```

### Service Isolation ❌

**Critical Issue: VecPipe imports from WebUI**

While initially it appeared there were no webui imports, deeper inspection reveals:

```python
# In packages/vecpipe/search_api.py (lines 411, 754):
from webui.api.collection_metadata import get_collection_metadata
metadata = get_collection_metadata(sync_client, collection_name)
```

This is a **violation of service separation** and should be addressed immediately.

### Proper Proxy Pattern ✅

**Search Proxy** (`webui/api/search.py`):
```python
# Receives request using contract
request: SearchRequest

# Transforms and forwards to vecpipe
search_params = {
    "query": request.query,
    "k": request.k,
    "collection": collection_name,
    # ... mapped fields
}

# Makes HTTP call
response = await client.post(f"{settings.SEARCH_API_URL}/search", json=search_params)
```

## 6. Critical Issues to Fix

### 1. Remove WebUI Import from VecPipe (CRITICAL) 🚨

The `collection_metadata` import in `vecpipe/search_api.py` breaks service isolation. Solutions:

**Recommended Solution: Move to Shared Package**
```python
# Move webui/api/collection_metadata.py to shared/database/collection_metadata.py
# Update imports in both services
from shared.database.collection_metadata import get_collection_metadata
```

This module already stores metadata in Qdrant (in a special `_collection_metadata` collection), which is the right approach. It just needs to be in the shared package since both services need it.

**Alternative: Pass Metadata via API**
```python
# WebUI could include model metadata in search requests
# VecPipe uses provided metadata instead of fetching from Qdrant
```

## 7. Other Recommendations for API Improvements

### 1. Consistent Error Response Usage
- Import and use error contracts throughout the codebase
- Replace manual HTTPException detail construction with contract objects
- Example implementation:
  ```python
  from shared.contracts.errors import ErrorResponse, create_validation_error
  
  # For validation errors
  if validation_errors:
      error = create_validation_error(validation_errors)
      raise HTTPException(status_code=400, detail=error.model_dump())
  ```

### 2. Add Response Model Validation to WebUI Endpoints
- WebUI search endpoints should use response_model for validation
- Current: Returns dict[str, Any]
- Recommended: Return contract types directly

### 3. Standardize Error Handling Middleware
- Create FastAPI exception handlers that automatically convert to ErrorResponse
- Ensures all errors follow the contract format

### 4. Add Contract Version Headers
- Include API version in responses (already in contracts)
- Add version negotiation for future compatibility

### 5. Document Service Dependencies
- Create a service dependency diagram
- Document which shared modules each service uses

### 6. Add Contract Tests
- Unit tests to verify contract compatibility
- Integration tests for service communication
- Contract change detection in CI/CD

## 7. Summary

The API contracts in Project Semantik are well-designed and mostly properly implemented:

✅ **Strengths**:
- Clear contract definitions with validation
- HTTP-based service communication (mostly)
- Backward compatibility considerations
- Comprehensive field documentation

❌ **Critical Issues**:
- VecPipe imports from WebUI (`collection_metadata`), breaking service isolation
- This must be fixed to maintain proper architecture boundaries

⚠️ **Areas for Improvement**:
- Inconsistent error contract usage
- Some endpoints return raw dicts instead of contract types
- Missing middleware for standardized error responses

The refactoring has successfully established a clean API contract layer that mostly enables proper service isolation. However, the critical issue of VecPipe importing from WebUI must be resolved to maintain clean architecture boundaries. Once this is fixed and the other recommendations are implemented, the system will have robust and maintainable API contracts that properly enforce service separation.