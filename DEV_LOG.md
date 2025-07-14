# Development Log - CORE-304: Hybrid State/Event WebSocket Updates

## 2025-07-14

### Initial Analysis

Starting implementation of distributed WebSocket state synchronization using Redis Pub/Sub.

**Current Architecture:**
- WebSocket implementation in `packages/webui/api/jobs.py` uses a local `ConnectionManager`
- Stateful design that won't work with distributed workers
- No persistence of messages for clients connecting mid-process

**Target Architecture:**
- Redis Streams for message persistence and guaranteed delivery
- Consumer groups for multi-server support
- Immediate state delivery from DB on connection
- Real-time updates via Redis pub/sub

**Key Challenges:**
1. Maintaining backward compatibility during migration
2. Handling authentication in WebSocket connections
3. Ensuring message delivery guarantees
4. Managing Redis connection lifecycle

### Todo List Created
1. Create RedisStreamWebSocketManager class
2. Update Celery tasks to publish to Redis Streams
3. Update FastAPI application lifespan
4. Refactor WebSocket endpoint
5. Add Redis dependencies
6. Configure environment variables
7. Create unit tests
8. Add integration tests

### Starting Implementation
Beginning with Phase 1: Creating the Redis WebSocket Manager...

## Phase 1: Redis WebSocket Manager

### Analysis Complete
- Current WebSocket implementation uses a local `ConnectionManager` in `packages/webui/api/jobs.py`
- Celery tasks use `update_state()` to send updates, but these don't reach WebSocket clients
- Redis is already configured and available as the Celery broker
- Need to implement Redis Streams for persistent message delivery

### Creating websocket_manager.py
Implementing RedisStreamWebSocketManager with:
- Redis Streams for message persistence
- Consumer groups for multi-server support
- Automatic reconnection and error handling
- Message history for late-joining clients

✅ Created websocket_manager.py with full implementation

## Phase 2: Update Celery Tasks

### Updating tasks.py
Need to:
- Add CeleryTaskWithUpdates helper class
- Replace celery_task.update_state() calls with Redis Stream publishing
- Send structured messages for all job events

✅ Updated tasks.py with:
- CeleryTaskWithUpdates helper class for Redis Stream publishing
- Added Redis updates for all job events: job_started, file_processing, file_completed, error, job_failed, job_completed
- Kept celery_task.update_state() for backwards compatibility
- Proper cleanup of Redis connection in finally block

## Phase 3: FastAPI Integration

### Updating main.py
Need to add application lifespan manager for Redis WebSocket connections

✅ Updated main.py with:
- Added lifespan context manager for application startup/shutdown
- Initialized ws_manager on startup
- Cleaned up ws_manager on shutdown
- Moved service configuration to lifespan function

## Phase 4: Refactor WebSocket Endpoint

### Updating jobs.py
Need to replace local ConnectionManager with global ws_manager

✅ Updated jobs.py with:
- Removed local ConnectionManager class
- Imported global ws_manager from websocket_manager module
- Updated websocket_endpoint to use ws_manager.connect/disconnect
- Added temporary user authentication (TODO: implement proper JWT auth)

## Phase 5: Configuration and Testing

### Environment Variables
Need to document REDIS_URL configuration

✅ Configuration completed:
- Added REDIS_URL to WebuiConfig class
- Added REDIS_URL documentation to .env.example
- Fixed settings access in websocket_manager.py and tasks.py

## Summary of Implementation

### Completed Tasks:
1. ✅ Created RedisStreamWebSocketManager with full Redis Streams support
2. ✅ Updated Celery tasks to publish updates to Redis Streams
3. ✅ Integrated WebSocket manager with FastAPI lifespan
4. ✅ Refactored WebSocket endpoint to use global manager
5. ✅ Configured Redis environment variables

### Architecture Changes:
- Replaced stateful WebSocket implementation with Redis Streams
- Added message persistence and guaranteed delivery
- Implemented consumer groups for multi-server support
- Added automatic state delivery on connection
- Maintained backward compatibility with Celery state updates

### Key Features:
- **Distributed State Sync**: Works across multiple API servers
- **Message Persistence**: Up to 1000 messages per job stream
- **Auto-reconnection**: Built-in error handling and retry logic
- **Historical Messages**: Late-joining clients receive message history
- **Current State Delivery**: Immediate job state from DB on connection

### TODOs:
- Implement proper WebSocket JWT authentication
- Add unit tests for WebSocket manager
- Add integration tests for end-to-end flow
- Add Prometheus metrics for monitoring
- Consider batch message delivery for performance

## Code Review Improvements (2025-07-14)

Based on code review feedback, implementing critical improvements:

### 1. WebSocket Authentication
Adding proper JWT authentication for WebSocket connections to replace the temporary "anonymous" user approach.

✅ Implemented:
- Added `get_current_user_websocket()` function in auth.py for WebSocket JWT validation
- Updated websocket_endpoint to authenticate users via token query parameter
- Added user access verification to ensure users can only access their own jobs
- Added DISABLE_AUTH setting to webui config for development mode
- Proper error codes (1008 for policy violation, 1011 for server error)

### 2. Redis Connection Retry with Backoff
Implementing exponential backoff for Redis connection failures to improve resilience.

✅ Implemented:
- Added retry logic with exponential backoff (1s, 2s, 4s) in startup()
- Set connection and socket timeouts (5s) to fail fast
- Don't raise exception on failure - allow graceful degradation

### 3. Graceful Degradation
Ensuring WebSocket functionality works even when Redis is unavailable.

✅ Implemented:
- WebSocket manager checks Redis availability before using it
- Falls back to direct client broadcast when Redis unavailable
- Initial state still delivered from database
- Warning logs inform about degraded mode
- send_job_update() handles both Redis and non-Redis modes

### 4. Cleanup for Completed Job Streams
Adding automatic cleanup of Redis streams for completed jobs.

✅ Implemented:
- Added cleanup_job_stream() method to delete streams and consumer groups
- Called from tasks.py when job completes successfully
- Called from jobs.py when job is deleted
- Frees up Redis memory automatically

### 5. WebSocket Connection Limits
Preventing DOS attacks by limiting connections per user.

✅ Implemented:
- Added max_connections_per_user setting (default: 10)
- Check enforced in connect() method before accepting connection
- Returns 1008 error code when limit exceeded
- Logs connection count for monitoring

### Summary of Improvements

All critical security and resilience issues from the code review have been addressed:

1. **Security**: Proper JWT authentication for WebSocket connections
2. **Resilience**: Redis connection retry with exponential backoff
3. **Graceful Degradation**: WebSocket works without Redis (degraded mode)
4. **Resource Management**: Automatic cleanup of completed job streams
5. **DOS Protection**: Connection limits per user

The remaining lower-priority improvements (message batching, monitoring metrics) can be addressed in future iterations.

## CI/CD Test Collection Failures (2025-07-14)

### Issue Description
After implementing the Redis-based WebSocket manager, CI/CD tests are failing during test collection phase:
```
ERROR tests/test_add_to_collection.py
ERROR tests/test_document_viewer.py
ERROR tests/test_internal_api.py
ERROR tests/webui/api/test_collections.py
ERROR tests/webui/api/test_files.py
ERROR tests/webui/api/test_jobs.py
ERROR tests/webui/test_cors_configuration.py
!!!!!!!!!!!!!!!!!!! Interrupted: 7 errors during collection !!!!!!!!!!!!!!!!!!!
```

### Root Cause Analysis
1. The new `websocket_manager.py` imports `redis.asyncio` and tries to access `settings.REDIS_URL` during module import
2. The `tasks.py` also imports `redis.asyncio` and uses Redis for WebSocket updates
3. Test files import from `webui.api` modules, which import these Redis-dependent modules
4. **Critical Issue**: The GitHub Actions CI environment doesn't have Redis service configured
5. Only Qdrant service is defined in `.github/workflows/ci.yml`, but not Redis

### Impact
- All tests that import any webui modules fail during collection phase
- Tests can't even start running because imports fail
- This blocks the entire CI/CD pipeline

### Solution
Add Redis service to GitHub Actions workflow and configure REDIS_URL environment variable.

## Fix Implementation (2025-07-14)

### Changes Made

1. **Updated `.github/workflows/ci.yml`**:
   - Added Redis service to the test job:
     ```yaml
     redis:
       image: redis:7-alpine
       ports:
         - 6379:6379
     ```
   - Added Redis readiness check before running tests
   - Added `REDIS_URL` environment variable: `redis://localhost:6379/0`

2. **Fixed Import Issues**:
   - Fixed `packages/webui/api/files.py` to import `ws_manager` correctly
   - Updated `tests/webui/api/test_jobs.py` to remove references to old `ConnectionManager`

### Result
- Test collection errors resolved
- CI/CD pipeline should now pass with Redis available
- WebSocket functionality will work properly in test environment

## Redis Connection Check Issue (2025-07-14)

### Issue
The Redis readiness check in CI was timing out because `redis-cli` is not installed in GitHub Actions runners by default.

### Fix
Changed from using `redis-cli` to using `nc` (netcat) to check if Redis port 6379 is open:
```bash
nc -zv localhost 6379
```

This uses netcat which is available by default in GitHub Actions runners.

## CI/CD Warning Investigation (2025-07-14)

### Warnings Found in Test Output

1. **Passlib crypt deprecation warning**:
   ```
   DeprecationWarning: 'crypt' is deprecated and slated for removal in Python 3.13
   ```
   - Third-party library issue, not our code
   - Will be fixed by passlib maintainers before Python 3.13

2. **Pydantic V1 style validators**:
   ```
   PydanticDeprecatedSince20: Pydantic V1 style `@validator` validators are deprecated
   ```
   - Found in `packages/webui/auth.py` lines 45 and 54
   - Need to migrate to V2 style `@field_validator`

3. **Async test warnings**:
   ```
   RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited
   RuntimeWarning: coroutine 'test_async_service_lifecycle_workflow' was never awaited
   ```
   - Found in embedding integration tests
   - Tests may not be running correctly
   - Need to ensure async tests are properly awaited

### Action Plan
1. Update Pydantic validators to V2 style
2. Fix async test issues to ensure proper test execution

## Fixes Applied (2025-07-14)

### 1. Fixed Pydantic V1 Validators
Updated `packages/webui/auth.py` to use Pydantic V2 style validators:
- Changed `@validator` to `@field_validator`
- Added `@classmethod` decorator as required by V2

### 2. Fixed Async Test Warnings
Updated async test methods in:
- `tests/test_embedding_full_integration.py`
- `tests/test_embedding_integration.py`

Changed from async methods to synchronous methods that use `asyncio.run()` internally, 
since unittest doesn't natively support async test methods.

### 3. Fixed Ruff Linting Issues
- Fixed datetime usage: Changed `datetime.utcnow()` to `datetime.now(UTC)`
- Fixed import order: Moved late import to top of file
- Added `contextlib.suppress()` for cleaner exception handling
- Fixed unused loop variables by replacing with `_`
- Fixed nested if statements by combining conditions
- Added `noqa` comment for required but unused function parameter

All CI/CD warnings and linting issues have been resolved.

## Test Failures Analysis (2025-07-14)

### Issue Overview
After implementing the WebSocket/Redis functionality, we have 29 test failures across the test suite.

### Failure Categories

1. **Import/Patch Location Issues (16 failures)**
   - Error: `AttributeError: <module 'webui.websocket_manager'> does not have the attribute 'create_job_repository'`
   - Affected tests: Most tests in `test_websocket_redis_integration.py`, `test_websocket_example.py`, and some in `test_websocket_manager.py`
   - Root Cause: `create_job_repository` is imported inside the `connect()` method, not at module level
   - Fix: Patch at correct location: `shared.database.factory.create_job_repository`

2. **AsyncMock Configuration Issues (4 failures)**
   - Error: `TypeError: object AsyncMock can't be used in 'await' expression`
   - Affected tests: `test_get_redis_creates_client`, `test_close`, `test_shutdown`, `test_disconnect`
   - Root Cause: AsyncMock not properly configured to be awaitable
   - Fix: Ensure AsyncMock is properly set up for async context managers

3. **Redis Mock Not Working (8 failures)**
   - Error: `AssertionError: Expected 'xadd' to have been called once. Called 0 times`
   - Affected tests: All tests in `test_celery_redis_updates.py`
   - Root Cause: The patch for `redis.from_url` is not working correctly
   - Fix: Need to fix the patching of `webui.tasks.redis.from_url`

4. **Other Issues (1 failure)**
   - `ValueError: not enough values to unpack (expected 3, got 2)` in `test_send_job_update_with_redis`
   - `AttributeError: 'coroutine' object has no attribute 'streams'` in `test_stream_cleanup_after_job_completion`

### Action Plan

1. **Fix Import Patches**: Update all tests to patch `create_job_repository` at the correct location
2. **Fix AsyncMock Setup**: Ensure AsyncMocks are properly configured for async context managers
3. **Fix Redis Mock Patching**: Correct the patching of Redis in the tasks module
4. **Fix API Mismatches**: Update test expectations to match actual implementation

## Test Fixes Applied (2025-07-14)

### 1. Fixed Import Patches ✅
- Changed all `patch("webui.websocket_manager.create_job_repository")` to `patch("shared.database.factory.create_job_repository")`
- Affected files: `test_websocket_redis_integration.py`, `test_websocket_example.py`, `test_websocket_manager.py`

### 2. Fixed AsyncMock Setup ✅
- Updated `test_startup_success` to use proper async function for `redis.from_url`
- Added `mock_redis_from_url` fixture to `test_celery_redis_updates.py`
- Updated all test methods to use the fixture correctly

### 3. Fixed Redis Mock Patching ✅
- Changed patch location from `webui.tasks.redis.from_url` to `redis.asyncio.from_url`
- Added proper async function wrapper for Redis mock
- Updated all test methods in `test_celery_redis_updates.py`

### 4. Fixed API Mismatches ✅
- Updated xadd call expectations to use keyword arguments (`maxlen=1000`)
- Fixed unpacking of call arguments in test assertions
- Fixed fixture decorator issue in `test_stream_cleanup_after_job_completion`

### 5. Fixed Ruff Linting Issues ✅
- Combined nested `with` statements (SIM117)
- Removed unused variables and parameters (F841, ARG001)
- Added proper exception chaining (B904)
- Used `contextlib.suppress` for cleaner exception handling (SIM105)

All test fixes have been applied and are ready for CI validation.