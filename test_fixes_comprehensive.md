# Comprehensive Test Fixes

## Summary of Issues

After implementing the initial fakeredis mocking, we still have 47 failed tests and 49 errors:

### Main Error Categories:

1. **isinstance() type errors (9 tests)**: 
   - Redis mocking is replacing classes with instances
   - Type guards are failing when checking Redis client types

2. **InvalidSpecError (49 errors)**:
   - Tests trying to create `AsyncMock(spec=redis.Redis)` 
   - But `redis.Redis` has been patched to be a Mock object

3. **AttributeError in WebSocket tests (10+ tests)**:
   - Tests expecting mock methods but getting real fakeredis methods
   - Method attributes like `.return_value` not available on real methods

4. **Integration test failures (3 tests)**:
   - Tests expecting specific Redis stream behaviors
   - fakeredis may not fully implement all stream features

5. **ChunkingStrategy enum issues (3 tests)**:
   - Test expectations don't match new chunking logic
   - Enum value changes or strategy selection logic changed

## Root Cause Analysis

The global mocking approach is too invasive and breaks test infrastructure:
- Patching Redis classes prevents isinstance checks
- Prevents using Redis classes as spec for mocks
- Makes all tests use fakeredis even when they need to test mocking behavior

## Recommended Solution

### 1. Remove Global Mocking
Remove the invasive `mock_redis_globally` fixture and instead:
- Only patch connection creation points
- Leave Redis classes intact for type checking
- Provide opt-in fixtures for tests that need fakeredis

### 2. Create Targeted Fixtures
```python
@pytest.fixture
def use_fakeredis():
    """Opt-in fixture to use fakeredis for a specific test."""
    fake_sync = fakeredis.FakeRedis(decode_responses=True)
    fake_async = fakeredis.aioredis.FakeRedis(decode_responses=True)
    
    with patch('redis.from_url', return_value=fake_sync), \
         patch('redis.asyncio.from_url', return_value=fake_async):
        yield fake_sync, fake_async

@pytest.fixture
def mock_redis_for_type_checking():
    """Mock that passes isinstance checks."""
    mock = MagicMock(spec=redis.Redis)
    mock.__class__ = redis.Redis
    return mock
```

### 3. Fix Specific Test Categories

#### For isinstance() errors:
- Use the simplified patching that only affects connection creation
- Don't patch Redis classes themselves

#### For InvalidSpecError:
- Don't patch Redis classes globally
- Let tests create their own mocks with proper specs

#### For WebSocket tests:
- Update tests to work with fakeredis behavior
- Or provide a flag to use mocks instead of fakeredis

#### For Integration tests:
- Mark tests that need real Redis behavior
- Provide a way to disable mocking for those tests

#### For ChunkingStrategy issues:
- Review and update test expectations
- These are likely legitimate test failures from code changes

## Implementation Plan

### Phase 1: Simplify Global Mocking
```python
# Only patch connection methods, not classes
@pytest.fixture(scope="session", autouse=True)
def configure_test_redis():
    """Configure Redis for testing without breaking type checking."""
    if not os.getenv("USE_REAL_REDIS"):
        # Only patch the connection creation
        fake_sync = fakeredis.FakeRedis(decode_responses=True)
        fake_async = fakeredis.aioredis.FakeRedis(decode_responses=True)
        
        # Minimal patching
        with patch('redis.from_url', return_value=fake_sync), \
             patch('redis.asyncio.from_url', return_value=fake_async):
            yield
    else:
        yield
```

### Phase 2: Update Type Guards
- Make type guards aware of fakeredis in test mode
- Already implemented in `testing_utils.py`

### Phase 3: Fix Individual Tests
- Update WebSocket tests to work with fakeredis
- Fix ChunkingStrategy test expectations
- Mark integration tests that need real Redis

## Test Categories and Solutions

### Tests That Should Use fakeredis:
- Unit tests for services that use Redis
- API endpoint tests
- Most functional tests

### Tests That Should Mock Redis:
- Tests that verify error handling
- Tests that check specific Redis call patterns
- WebSocket manager unit tests

### Tests That Need Real Redis:
- Integration tests for Redis streams
- Performance tests
- End-to-end tests

## Next Steps

1. Implement simplified mocking approach
2. Run tests to identify which specific tests need updates
3. Update tests category by category
4. Document testing patterns for future development