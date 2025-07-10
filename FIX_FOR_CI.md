# CI Test Fixes Needed

## Issues Found

### 1. Missing Test File
The file `tests/integration/test_search_api_integration.py` exists in CI but not locally. 
It needs to update its imports from:
```python
from packages.vecpipe.config import ...
```
to:
```python
from shared.config import ...
```

### 2. Already Fixed
- Updated `tests/test_reranking_e2e.py` to use correct import path: `from packages.vecpipe.search_api`

### 3. Potential Prometheus Registry Issue  
The error about duplicate timeseries in `test_reranking_e2e.py` suggests that when tests run in a different order in CI, the metrics might be registered multiple times. The code already has a `get_or_create_metric` function that should handle this, so the issue might be:
- The shared registry from `shared.metrics.prometheus` is being imported differently in tests vs production code
- Test isolation issues when running in parallel
- The metrics server being started multiple times in tests

This is likely a test environment issue rather than a code issue, as the test passes locally.

## Recommended Actions

1. Update the missing test file in CI to use the new import paths
2. Consider adding test fixtures to properly isolate Prometheus metrics between tests
3. The embedding service test failure might be a flaky test that needs investigation

## Import Pattern to Fix
Any remaining imports should be updated:
- `from packages.vecpipe.config` → `from shared.config`
- `from vecpipe.config` → `from shared.config`
- `from vecpipe.metrics` → `from shared.metrics.prometheus`
- `from vecpipe.extract_chunks import TokenChunker` → `from shared.text_processing.chunking import TokenChunker`
- `from vecpipe.extract_chunks import extract_text` → `from shared.text_processing.extraction import extract_text`