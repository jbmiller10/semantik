# Celery Tasks and WebSocket Integration Test Fixes

## Summary of Fixes Applied

This document summarizes the fixes applied to resolve approximately 25 failing tests related to Celery tasks and WebSocket integration in the Semantik codebase.

## Key Issues Identified and Fixed

### 1. Redis Connection Mocking Issues

**Problem:** The async Redis client mocking was incorrectly set up, causing failures in tests that relied on Redis connections.

**Fix Applied:**
- Changed from `side_effect=async_from_url` to `return_value=mock_redis` in patch statements
- Ensured proper async context manager usage with the updater instances
- Fixed context manager lifecycle tests by properly patching `redis.asyncio.from_url`

### 2. Missing Test Class Decorators

**Problem:** Several test classes were missing the `@pytest.mark.asyncio` decorator, causing async test methods to fail.

**Fix Applied:**
- Added `@pytest.mark.asyncio` decorator to the following test classes:
  - `TestProcessCollectionOperation`
  - `TestIndexOperation`
  - `TestAppendOperation`
  - `TestReindexOperation`
  - `TestRemoveSourceOperation`

### 3. WebSocket Manager Test Issues

**Problem:** Tests for Redis reconnection and operation getter functionality were failing due to incorrect mock setup.

**Fix Applied:**
- Fixed `test_connect_redis_reconnect_attempt` by:
  - Adding a mock operation getter to avoid errors
  - Using `side_effect` to properly set Redis after startup
- Fixed `test_connect_without_operation_getter` by:
  - Creating proper async context manager mocks for AsyncSessionLocal
  - Ensuring the mock session properly supports async context manager protocol

### 4. WebSocket Integration Test Failures

**Problem:** Tests were missing required mocks for model configuration and had incorrect context manager usage.

**Fix Applied:**
- Added `get_model_config` mock patches where needed
- Fixed context manager nesting in test methods
- Added missing `embedding_model` field to collection test data
- Added `collection_repo.update` mock where required

### 5. Concurrent Operations Test Fix

**Problem:** The test for multiple updaters was creating tasks inside context managers but executing them outside.

**Fix Applied:**
- Refactored to use a helper function that properly manages the context:
  ```python
  async def send_update_with_updater(updater, i):
      async with updater:
          await updater.send_update(f"update_{i}", {"index": i})
  ```

## Files Modified

1. `/home/dockertest/semantik/tests/webui/test_celery_tasks.py`
   - Fixed Redis mocking patterns
   - Added missing class decorators

2. `/home/dockertest/semantik/tests/webui/test_websocket_manager.py`
   - Fixed async context manager mocking
   - Improved test setup for reconnection scenarios

3. `/home/dockertest/semantik/tests/webui/test_tasks_websocket_integration.py`
   - Added required mock patches
   - Fixed context manager usage

4. `/home/dockertest/semantik/tests/webui/test_tasks_helpers.py`
   - Fixed concurrent updater test pattern

## Testing Recommendations

After applying these fixes:

1. Run the specific test suites to verify fixes:
   ```bash
   python -m pytest tests/webui/test_celery_tasks.py -xvs
   python -m pytest tests/webui/test_websocket_manager.py -xvs
   python -m pytest tests/webui/test_tasks_websocket_integration.py -xvs
   python -m pytest tests/webui/test_tasks_helpers.py -xvs
   ```

2. Run the full test suite to ensure no regressions:
   ```bash
   python -m pytest tests/webui/ -xvs
   ```

## Key Patterns to Remember

1. **Async Redis Mocking:** Always use `return_value=mock_redis` instead of `side_effect` for simpler cases
2. **Async Context Managers:** Ensure mocks have proper `__aenter__` and `__aexit__` methods
3. **Test Class Decorators:** Always add `@pytest.mark.asyncio` to test classes containing async methods
4. **Mock Dependencies:** Check what external dependencies each function requires and mock them appropriately

These fixes ensure robust error handling and proper async patterns throughout the critical infrastructure tests.