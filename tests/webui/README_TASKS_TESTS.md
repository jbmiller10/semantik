# Celery Tasks Test Suite

This directory contains comprehensive tests for the `packages/webui/tasks.py` module, which handles all asynchronous background processing in Semantik.

## Test Files

### 1. `test_celery_tasks.py` (Main Test Suite)
The primary test file covering core functionality:

- **Process Collection Operation Tests**
  - Test successful task execution for all operation types
  - Test failure handling and rollback scenarios
  - Test task retry logic for transient failures
  - Test immediate task ID recording

- **Operation-Specific Tests**
  - `TestIndexOperation`: Tests initial collection creation in Qdrant
  - `TestAppendOperation`: Tests document scanning and embedding generation
  - `TestReindexOperation`: Tests blue-green reindexing with validation
  - `TestRemoveSourceOperation`: Tests document removal from collections

- **Validation Tests**
  - Reindex validation logic testing
  - Vector count verification
  - Search quality comparison

- **Failure Handling Tests**
  - Task failure handler testing
  - Collection status updates on failure
  - Staging resource cleanup

### 2. `test_tasks_helpers.py` (Helper Functions)
Tests for utility and helper functions:

- **Sanitization Tests**
  - PII removal from error messages
  - Audit log detail sanitization
  - Path and email redaction

- **Audit Logging Tests**
  - Operation audit log creation
  - Collection deletion audit logs
  - Batch audit logging

- **Metrics Recording Tests**
  - Operation metrics recording
  - Collection statistics updates
  - Error metric handling

- **Resource Management Tests**
  - Active collection retrieval
  - Staging resource cleanup
  - Cleanup delay calculations

### 3. `test_tasks_websocket_integration.py` (WebSocket Integration)
Tests for real-time update functionality:

- **Message Flow Tests**
  - Complete operation message sequences
  - Progress update formatting
  - Error message propagation

- **Redis Stream Tests**
  - Stream TTL management
  - Message ordering guarantees
  - Connection pooling

- **Integration Scenarios**
  - Full document processing flow
  - Operation failure scenarios
  - Concurrent update handling

### 4. `test_celery_redis_updates.py` (Existing)
Tests specifically for the `CeleryTaskWithOperationUpdates` class.

### 5. `test_cleanup_tasks.py` (Existing)
Tests for cleanup task functionality.

## Running the Tests

### Run all task tests:
```bash
pytest tests/webui/test_*tasks*.py -v
```

### Run specific test categories:
```bash
# Core functionality only
pytest tests/webui/test_celery_tasks.py -v

# Helper functions only
pytest tests/webui/test_tasks_helpers.py -v

# WebSocket integration only
pytest tests/webui/test_tasks_websocket_integration.py -v
```

### Run with coverage:
```bash
pytest tests/webui/test_*tasks*.py --cov=webui.tasks --cov-report=html
```

## Test Categories

### Unit Tests
- Individual function testing with mocked dependencies
- Edge case handling
- Error condition testing

### Integration Tests
- Multi-component interaction testing
- Database transaction testing
- External service mocking (Qdrant, vecpipe)

### Performance Tests
- Large batch processing
- Concurrent operation handling
- Memory usage validation

## Key Testing Patterns

### 1. Async Testing
All async functions use `@pytest.mark.asyncio` and proper async mocking:
```python
@pytest.mark.asyncio
async def test_async_function():
    mock_repo = AsyncMock()
    await function_under_test(mock_repo)
```

### 2. Repository Mocking
Consistent patterns for mocking database repositories:
```python
mock_session = AsyncMock()
mock_session.__aenter__ = AsyncMock(return_value=mock_session)
mock_session.__aexit__ = AsyncMock(return_value=None)
```

### 3. WebSocket Message Verification
Capturing and verifying Redis stream messages:
```python
captured_messages = []
async def capture_xadd(stream, message, **kwargs):
    captured_messages.append(json.loads(message["message"]))
mock_redis.xadd = capture_xadd
```

### 4. Error Handling Validation
Testing both expected exceptions and graceful degradation:
```python
with pytest.raises(ValueError, match="Expected error"):
    await function_that_should_fail()
```

## Coverage Goals

The test suite aims for:
- **Line Coverage**: >90% of task.py
- **Branch Coverage**: All major code paths tested
- **Critical Path Coverage**: 100% for INDEX, APPEND, REINDEX operations
- **Error Path Coverage**: All error handlers tested

## Adding New Tests

When adding new functionality to tasks.py:
1. Add unit tests for new helper functions
2. Add integration tests for new operation types
3. Add WebSocket message tests for new update types
4. Update this README with new test categories

## Known Limitations

1. Some external service calls (vecpipe API) are mocked rather than integration tested
2. Actual Celery worker execution is not tested (would require Celery test worker)
3. Redis connection failures are simulated, not actual network errors
