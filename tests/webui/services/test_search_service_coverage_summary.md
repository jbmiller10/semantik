# search_service.py Test Coverage

Tests: `tests/webui/services/test_search_service.py`

## Covered Methods

| Method | Tests |
|--------|-------|
| __init__ | default config, custom timeout/retry |
| validate_collection_access | multi-collection, not found, access denied, partial access |
| search_single_collection | all params, not ready, timeout+retry, HTTP errors, connection errors |
| handle_http_error | all status codes, retry suffix |
| multi_collection_search | aggregation, hybrid, partial failures, limiting, empty, access denied |
| single_collection_search | full params+reranking, hybrid, HTTP errors, access validation |

Edge cases: None values, mixed collection statuses, HTTP error after retry, parallel execution
