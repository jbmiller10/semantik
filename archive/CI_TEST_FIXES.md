# CI Test Fixes Summary

## Issue
The CI tests were failing with the following error:
```
ValueError: Duplicated timeseries in CollectorRegistry: {'embedding_oom_errors', 'embedding_oom_errors_total', 'embedding_oom_errors_created'}
```

## Root Cause
The tests were failing due to two issues:

1. **Prometheus Metrics Registration**: The `embedding_service.py` module was registering Prometheus metrics at module import time. When tests imported the module multiple times, it tried to register the same metrics again, causing a duplicate registration error.

2. **Incorrect Import Paths**: Test files were using incorrect module paths for imports and mocking (e.g., `webui.api.documents` instead of `packages.webui.api.documents`).

## Fixes Applied

### 1. Fixed Prometheus Metrics Registration (`packages/webui/embedding_service.py`)
Added error handling to catch duplicate metric registration and reuse existing metrics:

```python
# Check if metrics already exist in registry to avoid duplicates
try:
    oom_errors = Counter(...)
    batch_size_reductions = Counter(...)
except ValueError as e:
    # Metrics already registered, get them from registry
    if "Duplicated timeseries" in str(e):
        # Find existing metrics in registry
        for collector in registry._collector_to_names:
            if hasattr(collector, "_name"):
                if collector._name == "embedding_oom_errors_total":
                    oom_errors = collector
                elif collector._name == "embedding_batch_size_reductions_total":
                    batch_size_reductions = collector
    else:
        raise
```

### 2. Fixed Import Paths in Tests
Updated import paths in test files:
- `tests/test_document_viewer.py`: Fixed import from `webui.api.documents` to `packages.webui.api.documents`
- `tests/test_document_api.py`: Fixed mock paths from `webui.database` to `packages.webui.database`

## Test Results
All tests now pass successfully:
- 20 tests passed
- 0 tests failed
- Code was formatted with `make format`

## Branch Information
- Branch name: `fix/jm-refactor-ui-test-failures`
- Based on: `jm/refactor-ui-to-react`

The fixes ensure that tests can run successfully in CI without duplicate metric registration errors.