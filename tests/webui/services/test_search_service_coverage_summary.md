# SearchService Test Coverage Summary

This comprehensive test suite covers all 100 uncovered lines in `packages/webui/services/search_service.py`.

## Test Classes and Coverage

### 1. TestSearchServiceInit (Lines 21-41)
- Tests initialization with default timeout configuration
- Tests initialization with custom timeout and retry multiplier

### 2. TestValidateCollectionAccess (Lines 43-69)
- Successful validation of multiple collections
- EntityNotFoundError handling
- AccessDeniedError handling
- Mixed permissions (partial access)

### 3. TestSearchSingleCollection (Lines 71-189)
- Successful search with all parameters
- Collection not ready (status check)
- Timeout with successful retry (extended timeout calculation)
- Timeout with failed retry
- HTTP error handling for all status codes (404, 403, 429, 500+, etc.)
- Connection errors
- Request errors
- Unexpected exceptions
- Custom timeout parameter usage

### 4. TestHandleHttpError (Lines 155-189)
- All HTTP status codes mapping to appropriate error messages
- Retry suffix handling

### 5. TestMultiCollectionSearch (Lines 192-308)
- Successful multi-collection search with result aggregation
- Hybrid search parameters
- Partial failures (some collections succeed, others fail)
- Result limiting to k parameter
- Empty results handling
- Access denied during validation
- Collection status filtering (not ready collections)
- Error aggregation and reporting

### 6. TestSingleCollectionSearch (Lines 310-397)
- Full parameter coverage including reranking
- Hybrid search mode parameters
- HTTP error handling (404 -> EntityNotFoundError, 403 -> AccessDeniedError)
- General exception propagation
- Access validation

### 7. TestSearchServiceEdgeCases
- Timeout retry with None values (defaults to safe values)
- Mixed collection statuses (ready vs indexing)
- HTTP error after timeout retry
- Parallel execution verification

## Key Testing Patterns

1. **Mocking Strategy**: Uses AsyncMock for async dependencies and httpx.AsyncClient patching
2. **Error Simulation**: Comprehensive HTTP status code testing
3. **Timeout Testing**: Verifies retry logic with extended timeouts
4. **Parallel Execution**: Confirms async gather works correctly
5. **Edge Cases**: None values, partial failures, mixed statuses

## Coverage Details

- All initialization parameters and defaults
- All error handling paths
- All HTTP status codes and their mappings
- Retry logic with timeout calculations
- Result aggregation and sorting
- Collection status filtering
- Parallel search execution
- Parameter passing to vecpipe API

This test suite ensures the SearchService is robust and handles all edge cases appropriately.