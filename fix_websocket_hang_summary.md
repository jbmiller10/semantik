# WebSocket Test Hang Fix Summary

## Issue
The test suite is hanging at 98% completion after running `test_consume_updates_stream_not_exist`. The tests have been running for 2+ hours.

## Root Cause
1. The `_consume_updates` method in the WebSocket manager has an infinite loop that continuously polls Redis
2. Some tests create async tasks that run this method but may not properly clean them up
3. The singleton `ws_manager` instance might have lingering tasks from previous test classes

## Fixes Applied

### 1. Improved Redis Mocking
- Added mock for `xgroup_delconsumer` in `test_consume_updates_stream_not_exist` to ensure proper cleanup

### 2. Added Test Timeouts
- Added `@pytest.mark.timeout(10)` to all tests that create `_consume_updates` tasks:
  - `test_consume_updates`
  - `test_consume_updates_operation_completion`
  - `test_consume_updates_stream_not_exist`
  - `test_consume_updates_message_processing_error`
  - `test_consume_updates_consumer_cleanup`
  - `test_consume_updates_group_creation_retry`

### 3. Enhanced Singleton Cleanup
- Modified the `cleanup_singleton` fixture in `TestWebSocketManagerSingleton` to:
  - Cancel ALL existing tasks before each test (not just new ones)
  - Use shorter timeouts (0.1s instead of 0.5s) for task cancellation
  - Clear all connections and tasks before starting the test

## Recommendations

If tests still hang, consider:

1. **Run tests with pytest-timeout plugin**:
   ```bash
   poetry run pytest tests/webui/test_websocket_manager.py --timeout=60
   ```

2. **Add explicit event loop cleanup between test classes**:
   ```python
   @pytest.fixture(scope="class", autouse=True)
   def cleanup_event_loop():
       yield
       # Get all tasks and cancel them
       loop = asyncio.get_event_loop()
       pending = asyncio.all_tasks(loop)
       for task in pending:
           task.cancel()
   ```

3. **Consider mocking the singleton directly** in tests that don't need it:
   ```python
   with patch("packages.webui.websocket_manager.ws_manager"):
       # Run test
   ```

## Testing the Fix

To verify the fixes work:
```bash
# Run just the problematic test
poetry run pytest tests/webui/test_websocket_manager.py::TestRedisStreamWebSocketManager::test_consume_updates_stream_not_exist -v

# Run the full test file with timeout
poetry run pytest tests/webui/test_websocket_manager.py --timeout=60 -v
```