# BE-001: Fix Redis Client Type Mismatch

## Ticket Information
- **Priority**: CRITICAL
- **Estimated Time**: 3 hours
- **Dependencies**: None
- **Risk Level**: HIGH - Causes authentication bypasses and silent failures
- **Affected Files**:
  - `packages/webui/services/chunking_service.py`
  - `packages/webui/chunking_tasks.py`
  - `packages/webui/services/factory.py`
  - `packages/webui/websocket_manager.py`

## Context

The codebase has a critical issue with Redis client type mismatches. Services expect async Redis clients but receive sync clients in multiple places, and Celery tasks incorrectly use `asyncio.run` with sync Redis. This causes:

1. Silent failures in Redis operations
2. Potential security bypasses if Redis-based auth fails
3. Event loop conflicts in Celery workers
4. WebSocket notification failures

### Current Problems

```python
# In chunking_tasks.py line 628
chunking_service = ChunkingService(
    redis_client=None,  # Wrong: passing None instead of proper client
)

# In chunking_service.py
class ChunkingService:
    def __init__(self, redis_client: aioredis.Redis):  # Expects async
        self.redis = redis_client
    
    async def update_progress(self):
        await self.redis.set(...)  # Fails if redis is sync client
```

## Requirements

1. Establish clear separation between async and sync Redis clients
2. Create proper factory methods for service instantiation
3. Fix Celery tasks to use sync Redis correctly
4. Implement bridge pattern for async/sync interop where needed
5. Add runtime type checking to prevent future mismatches
6. Ensure all Redis operations have proper error handling

## Technical Details

### 1. Create Redis Client Manager

```python
# packages/webui/services/redis_manager.py

from typing import Optional, Union
import redis
import aioredis
from dataclasses import dataclass
from contextlib import asynccontextmanager, contextmanager

@dataclass
class RedisConfig:
    url: str
    max_connections: int = 50
    decode_responses: bool = True
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    retry_on_error: list = None

class RedisManager:
    """Manages both sync and async Redis clients with proper typing"""
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self._async_pool: Optional[aioredis.ConnectionPool] = None
        self._sync_pool: Optional[redis.ConnectionPool] = None
    
    @property
    def sync_client(self) -> redis.Redis:
        """Get synchronous Redis client for Celery tasks"""
        if not self._sync_pool:
            self._sync_pool = redis.ConnectionPool.from_url(
                self.config.url,
                max_connections=self.config.max_connections,
                decode_responses=self.config.decode_responses,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                retry_on_error=self.config.retry_on_error or [redis.ConnectionError]
            )
        return redis.Redis(connection_pool=self._sync_pool)
    
    async def async_client(self) -> aioredis.Redis:
        """Get asynchronous Redis client for FastAPI services"""
        if not self._async_pool:
            self._async_pool = aioredis.ConnectionPool.from_url(
                self.config.url,
                max_connections=self.config.max_connections,
                decode_responses=self.config.decode_responses,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout
            )
        return aioredis.Redis(connection_pool=self._async_pool)
    
    @asynccontextmanager
    async def async_transaction(self):
        """Context manager for async Redis transactions"""
        client = await self.async_client()
        async with client.pipeline(transaction=True) as pipe:
            yield pipe
            await pipe.execute()
    
    @contextmanager
    def sync_transaction(self):
        """Context manager for sync Redis transactions"""
        client = self.sync_client
        with client.pipeline(transaction=True) as pipe:
            yield pipe
            pipe.execute()
    
    async def close_async(self):
        """Close async connections"""
        if self._async_pool:
            await self._async_pool.disconnect()
    
    def close_sync(self):
        """Close sync connections"""
        if self._sync_pool:
            self._sync_pool.disconnect()
```

### 2. Fix Service Factory

```python
# packages/webui/services/factory.py

class ServiceFactory:
    """Factory for creating properly configured services"""
    
    _redis_manager: Optional[RedisManager] = None
    
    @classmethod
    def initialize_redis(cls, config: RedisConfig):
        """Initialize Redis manager (call once at startup)"""
        cls._redis_manager = RedisManager(config)
    
    @classmethod
    async def create_chunking_service(
        cls,
        db_session: AsyncSession,
        user_id: Optional[str] = None
    ) -> ChunkingService:
        """Create ChunkingService with proper async Redis client"""
        if not cls._redis_manager:
            raise RuntimeError("Redis not initialized. Call initialize_redis first")
        
        redis_client = await cls._redis_manager.async_client()
        
        # Type verification
        if not isinstance(redis_client, aioredis.Redis):
            raise TypeError(f"Expected aioredis.Redis, got {type(redis_client)}")
        
        return ChunkingService(
            db_session=db_session,
            redis_client=redis_client,
            chunk_repository=ChunkRepository(db_session),
            error_handler=ChunkingErrorHandler(redis_client),
            user_id=user_id
        )
    
    @classmethod
    def create_celery_chunking_service(
        cls,
        db_session_factory
    ) -> 'CeleryChunkingService':
        """Create service for Celery tasks with sync Redis"""
        if not cls._redis_manager:
            raise RuntimeError("Redis not initialized")
        
        redis_client = cls._redis_manager.sync_client
        
        # Type verification
        if not isinstance(redis_client, redis.Redis):
            raise TypeError(f"Expected redis.Redis, got {type(redis_client)}")
        
        return CeleryChunkingService(
            db_session_factory=db_session_factory,
            redis_client=redis_client
        )
```

### 3. Fix Celery Tasks

```python
# packages/webui/chunking_tasks.py

from celery import Task
import redis
from typing import Dict, Any

class ChunkingTask(Task):
    """Base task class with proper Redis client"""
    _redis_client: Optional[redis.Redis] = None
    
    @property
    def redis_client(self) -> redis.Redis:
        if self._redis_client is None:
            # Get sync Redis client from factory
            self._redis_client = ServiceFactory._redis_manager.sync_client
        return self._redis_client

@app.task(base=ChunkingTask, bind=True)
def process_chunking_operation(
    self,
    operation_id: str,
    document_id: str,
    strategy: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Process chunking with proper sync Redis"""
    
    # Use sync Redis directly - no asyncio.run!
    redis = self.redis_client
    
    try:
        # Update status in Redis
        redis.hset(
            f"operation:{operation_id}",
            mapping={
                "status": "processing",
                "started_at": datetime.utcnow().isoformat()
            }
        )
        
        # Send progress update via Redis stream
        redis.xadd(
            f"stream:chunking:{operation_id}",
            {"event": "started", "timestamp": time.time()}
        )
        
        # Process chunking (sync implementation)
        result = process_document_sync(
            document_id=document_id,
            strategy=strategy,
            config=config,
            progress_callback=lambda p: redis.hset(
                f"operation:{operation_id}",
                "progress", p
            )
        )
        
        # Update completion status
        redis.hset(
            f"operation:{operation_id}",
            mapping={
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "result": json.dumps(result)
            }
        )
        
        return result
        
    except Exception as e:
        # Error handling with Redis update
        redis.hset(
            f"operation:{operation_id}",
            mapping={
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.utcnow().isoformat()
            }
        )
        raise
```

### 4. Fix WebSocket Manager

```python
# packages/webui/websocket_manager.py

class ScalableWebSocketManager:
    """WebSocket manager with proper async Redis"""
    
    def __init__(self):
        self.redis_client: Optional[aioredis.Redis] = None
        self.connections: Dict[str, WebSocket] = {}
    
    async def initialize(self):
        """Initialize with async Redis client"""
        self.redis_client = await ServiceFactory._redis_manager.async_client()
        
        # Verify correct type
        if not isinstance(self.redis_client, aioredis.Redis):
            raise TypeError(f"Expected aioredis.Redis, got {type(self.redis_client)}")
    
    async def subscribe_to_updates(self, operation_id: str):
        """Subscribe to Redis streams for operation updates"""
        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")
        
        # Use async Redis for pub/sub
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe(f"chunking:{operation_id}")
        
        async for message in pubsub.listen():
            if message['type'] == 'message':
                await self.broadcast_to_operation(
                    operation_id,
                    message['data']
                )
```

### 5. Add Runtime Type Checking

```python
# packages/webui/services/type_guards.py

from typing import TypeGuard, Any
import redis
import aioredis

def is_async_redis(client: Any) -> TypeGuard[aioredis.Redis]:
    """Type guard for async Redis client"""
    return isinstance(client, aioredis.Redis)

def is_sync_redis(client: Any) -> TypeGuard[redis.Redis]:
    """Type guard for sync Redis client"""
    return isinstance(client, redis.Redis)

def ensure_async_redis(client: Any) -> aioredis.Redis:
    """Ensure client is async Redis, raise TypeError if not"""
    if not is_async_redis(client):
        raise TypeError(
            f"Expected aioredis.Redis, got {type(client).__name__}. "
            "This service requires an async Redis client."
        )
    return client

def ensure_sync_redis(client: Any) -> redis.Redis:
    """Ensure client is sync Redis, raise TypeError if not"""
    if not is_sync_redis(client):
        raise TypeError(
            f"Expected redis.Redis, got {type(client).__name__}. "
            "Celery tasks require a sync Redis client."
        )
    return client
```

## Acceptance Criteria

1. **Type Safety**
   - [ ] All services receive correct Redis client type
   - [ ] Runtime type checking prevents mismatches
   - [ ] Clear error messages when wrong type provided

2. **Celery Tasks**
   - [ ] No more `asyncio.run` in Celery tasks
   - [ ] All tasks use sync Redis client
   - [ ] No event loop errors in workers

3. **FastAPI Services**
   - [ ] All services use async Redis client
   - [ ] WebSocket manager uses async Redis
   - [ ] No blocking Redis calls in async contexts

4. **Error Handling**
   - [ ] All Redis operations have try/except blocks
   - [ ] Connection failures handled gracefully
   - [ ] Clear logging of Redis errors

5. **Performance**
   - [ ] Connection pooling properly configured
   - [ ] No connection leaks
   - [ ] Reduced latency in Redis operations

## Testing Requirements

1. **Unit Tests**
   ```python
   async def test_service_gets_async_redis():
       service = await ServiceFactory.create_chunking_service(session)
       assert is_async_redis(service.redis_client)
   
   def test_celery_gets_sync_redis():
       service = ServiceFactory.create_celery_chunking_service(factory)
       assert is_sync_redis(service.redis_client)
   
   async def test_type_mismatch_raises_error():
       with pytest.raises(TypeError, match="Expected aioredis.Redis"):
           ChunkingService(redis_client=redis.Redis())  # Wrong type
   ```

2. **Integration Tests**
   - Test Celery task execution with sync Redis
   - Test FastAPI endpoints with async Redis
   - Test WebSocket notifications work correctly
   - Test Redis connection recovery

3. **Load Tests**
   - Verify no connection pool exhaustion
   - Test concurrent Celery tasks
   - Test many WebSocket connections

## Rollback Plan

If issues occur:
1. Revert to previous Redis client initialization
2. Monitor for TypeErrors in logs
3. Check Redis connection metrics
4. Verify all services still functional

## Success Metrics

- Zero Redis type mismatch errors in logs
- All Celery tasks complete successfully
- WebSocket notifications delivered reliably
- Redis operation latency < 10ms p99
- No event loop errors in Celery workers

## Notes for LLM Agent

- NEVER mix async and sync Redis clients
- Celery CANNOT use async Redis - it will cause event loop conflicts
- FastAPI services SHOULD use async Redis for performance
- Always add type checking when accepting Redis clients
- Test both client types thoroughly
- Consider using dependency injection for better testability
- Monitor Redis connection pool usage