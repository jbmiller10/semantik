# Search Reranking Tests

This document describes the comprehensive test suite for the search reranking functionality in Semantik.

## Overview

The reranking feature allows search results to be re-scored using a cross-encoder model (Qwen/Qwen3-Reranker-0.6B by default) to improve relevance. These tests ensure the feature works correctly without requiring GPU or model dependencies in CI/CD environments.

## Test Structure

### Backend Tests

1. **API Endpoint Tests** (`tests/webui/api/v2/test_search.py`)
   - Tests for multi-collection search with/without reranking
   - Tests for single collection search with reranking
   - Verification of score differences when reranking is enabled
   - Parameter passing validation

2. **Service Layer Tests** (`tests/webui/services/test_search_service_reranking.py`)
   - Tests SearchService reranking logic
   - Verifies correct parameters are passed to vecpipe
   - Tests error handling (e.g., insufficient GPU memory)
   - Tests partial failures in multi-collection searches

3. **Integration Tests** (`tests/integration/test_search_reranking_integration.py`)
   - Full flow tests from API to service layer
   - Tests with multiple collections
   - Hybrid search with reranking
   - Error scenario handling

### Frontend Tests

1. **Component Tests** (`apps/webui-react/src/components/__tests__/SearchInterface.reranking.test.tsx`)
   - Tests reranking UI controls
   - Verifies parameter passing to API
   - Tests error handling in UI
   - Tests interaction with hybrid search

2. **Store Tests** (`apps/webui-react/src/stores/__tests__/searchStore.reranking.test.ts`)
   - Tests state management for reranking options
   - Verifies reranking metrics storage
   - Tests parameter persistence

## Running the Tests

### Backend Tests

```bash
# Run all backend reranking tests
uv run pytest tests/ -k "rerank" -v

# Run specific test files
uv run pytest tests/webui/api/v2/test_search.py::TestSearchReranking -v
uv run pytest tests/webui/services/test_search_service_reranking.py -v
uv run pytest tests/integration/test_search_reranking_integration.py -v

# Run with coverage
uv run pytest tests/ -k "rerank" --cov=webui --cov-report=html
```

### Frontend Tests

```bash
# Navigate to the React app directory
cd apps/webui-react

# Run all frontend tests
npm test

# Run reranking-specific tests
npm test -- SearchInterface.reranking
npm test -- searchStore.reranking

# Run with coverage
npm test -- --coverage
```

## Test Fixtures

The test suite includes reusable fixtures in `tests/fixtures/search_reranking_fixtures.py`:

- `create_mock_collection()`: Creates mock collection objects
- `create_search_result()`: Creates search result dictionaries
- `create_vecpipe_response()`: Creates mock vecpipe responses
- Predefined test scenarios and error cases

## Mocking Strategy

All tests use appropriate mocking to avoid external dependencies:

1. **Vecpipe Service**: Mocked using `httpx.AsyncClient` patches
2. **Database**: Mocked using AsyncMock for repositories
3. **Authentication**: Mocked user objects
4. **External APIs**: All HTTP calls are mocked

## CI/CD Considerations

These tests are designed to run in GitHub Actions without requiring:
- GPU hardware
- Downloaded models
- Running vecpipe service
- Qdrant instance

All external dependencies are mocked to ensure tests are:
- Fast (< 1 second per test)
- Deterministic (same results every run)
- Isolated (no side effects between tests)

## Common Test Scenarios

1. **Basic Reranking**: Verify scores change when reranking is enabled
2. **Multi-Collection**: Test reranking across collections with different embedding models
3. **Hybrid Search**: Test reranking with hybrid search enabled
4. **Error Handling**: Test insufficient memory, timeouts, and other failures
5. **Parameter Validation**: Ensure all parameters are passed correctly through the stack

## Adding New Tests

When adding new reranking tests:

1. Use existing fixtures from `search_reranking_fixtures.py`
2. Mock all external dependencies
3. Test both success and failure cases
4. Verify parameter passing at each layer
5. Include descriptive test names and docstrings

## Debugging Test Failures

If tests fail:

1. Check mock setup - ensure all dependencies are properly mocked
2. Verify parameter names match between layers
3. Check for async/await issues in backend tests
4. For frontend tests, check React Testing Library queries
5. Use `-v` flag for verbose output
6. Add `print()` or `console.log()` statements for debugging

## Test Coverage Goals

- Backend: >90% coverage for reranking-related code
- Frontend: >85% coverage for reranking UI and state
- Integration: Key user flows covered end-to-end

Run coverage reports to identify untested code paths.
