# WebSocket Test Hanging Fix

## Issue
The WebSocket manager tests were hanging indefinitely after `test_consume_updates_stream_not_exist`.

## Root Cause
The `_consume_updates` method in `websocket_manager.py` has an infinite loop that:
1. Checks if a Redis stream exists using `xinfo_stream`
2. If the stream doesn't exist, it sleeps for 2 seconds and retries
3. This continues forever until the stream exists or the task is cancelled

In the test, we were mocking `xinfo_stream` to always raise an exception, causing the method to loop forever.

## Fix Applied
1. **Updated test fixture**: Added proper cleanup in the `manager` fixture to cancel any remaining tasks
2. **Fixed `test_consume_updates_stream_not_exist`**: Changed the mock to simulate the stream appearing after a few retries
3. **Fixed `test_consume_updates_group_creation_retry`**: Improved the mock to properly simulate NOGROUP error recovery
4. **Added missing decorator**: Added `@pytest.mark.asyncio()` to `test_consume_updates_redis_none`
5. **Added singleton cleanup**: Added cleanup fixture for the singleton tests to prevent background tasks from interfering

## Key Changes
- Tests now properly simulate real-world scenarios where streams may not exist initially but appear later
- All async tasks are properly cancelled and awaited in fixtures
- The singleton WebSocket manager is properly cleaned up between tests

## Testing
After these fixes, the WebSocket tests should complete without hanging. The tests now properly verify the retry logic without getting stuck in infinite loops.