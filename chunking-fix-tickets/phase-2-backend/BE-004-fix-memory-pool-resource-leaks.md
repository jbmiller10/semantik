# BE-004: Fix Memory Pool Resource Leaks

## Ticket Information
- **Priority**: CRITICAL
- **Estimated Time**: 2 hours
- **Dependencies**: BE-001, BE-002
- **Risk Level**: HIGH - Can cause OOM crashes under load
- **Affected Files**:
  - `packages/shared/chunking/infrastructure/streaming/memory_pool.py`
  - `packages/shared/chunking/infrastructure/streaming/processor.py`
  - `packages/shared/chunking/infrastructure/streaming/window.py`

## Context

The streaming processor's memory pool has a critical resource leak. If an exception occurs between buffer acquisition and the try block, the buffer is never released. Under high load, this causes memory exhaustion within 2-4 hours.

### Current Problem

```python
# infrastructure/streaming/processor.py:246
buffer_id, buffer = self.memory_pool.acquire(size)  # Exception here = leak!
try:
    # Use buffer
    pass
finally:
    self.memory_pool.release(buffer_id)  # Never reached if acquire fails
```

## Requirements

1. Implement context manager pattern for automatic cleanup
2. Add buffer lifecycle tracking
3. Implement automatic leak detection
4. Add memory usage monitoring
5. Create buffer timeout and reclamation
6. Ensure thread-safe operations

## Technical Details

### 1. Implement Safe Memory Pool with Context Manager

```python
# packages/shared/chunking/infrastructure/streaming/memory_pool.py

import asyncio
import threading
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, Set, Tuple, Any
import weakref
import psutil
import logging

logger = logging.getLogger(__name__)

@dataclass
class BufferAllocation:
    """Track buffer allocation metadata"""
    buffer_id: str
    size: int
    allocated_at: datetime
    last_accessed: datetime
    stack_trace: str
    thread_id: int
    owner: Optional[weakref.ref] = None
    
    def is_leaked(self, timeout_seconds: int = 300) -> bool:
        """Check if buffer is likely leaked"""
        age = (datetime.utcnow() - self.allocated_at).total_seconds()
        return age > timeout_seconds and self.owner is None

class ManagedBuffer:
    """Buffer wrapper with automatic cleanup"""
    
    def __init__(self, buffer_id: str, data: bytearray, pool: 'MemoryPool'):
        self.buffer_id = buffer_id
        self.data = data
        self.pool = weakref.ref(pool)
        self._released = False
    
    def __del__(self):
        """Ensure buffer is released on garbage collection"""
        if not self._released:
            pool = self.pool()
            if pool:
                logger.warning(
                    f"Buffer {self.buffer_id} released by garbage collector - "
                    "possible leak detected"
                )
                pool._force_release(self.buffer_id)
    
    def release(self):
        """Explicitly release buffer"""
        if not self._released:
            pool = self.pool()
            if pool:
                pool.release(self.buffer_id)
                self._released = True

class MemoryPool:
    """Thread-safe memory pool with leak prevention"""
    
    def __init__(
        self,
        max_size: int = 100 * 1024 * 1024,  # 100MB
        max_buffer_size: int = 10 * 1024 * 1024,  # 10MB per buffer
        leak_check_interval: int = 60  # seconds
    ):
        self.max_size = max_size
        self.max_buffer_size = max_buffer_size
        self.used_size = 0
        
        # Thread-safe structures
        self._lock = threading.RLock()
        self._buffers: Dict[str, bytearray] = {}
        self._allocations: Dict[str, BufferAllocation] = {}
        self._free_buffers: Set[Tuple[int, str]] = set()  # (size, buffer_id)
        
        # Async support
        self._async_lock = asyncio.Lock()
        self._allocation_event = asyncio.Event()
        
        # Leak detection
        self._leak_check_task = None
        self._leak_check_interval = leak_check_interval
        self._leaked_buffers: Set[str] = set()
        
        # Metrics
        self.allocation_count = 0
        self.release_count = 0
        self.leak_count = 0
        self.reuse_count = 0
    
    @contextmanager
    def acquire_sync(self, size: int, timeout: float = 30.0) -> ManagedBuffer:
        """Synchronous context manager for buffer acquisition"""
        import time
        import traceback
        
        if size > self.max_buffer_size:
            raise ValueError(f"Buffer size {size} exceeds maximum {self.max_buffer_size}")
        
        start_time = time.time()
        buffer_id = None
        
        try:
            while True:
                with self._lock:
                    # Try to reuse existing buffer
                    buffer_id, buffer = self._try_acquire_buffer(size)
                    
                    if buffer_id:
                        # Track allocation
                        self._allocations[buffer_id] = BufferAllocation(
                            buffer_id=buffer_id,
                            size=size,
                            allocated_at=datetime.utcnow(),
                            last_accessed=datetime.utcnow(),
                            stack_trace=traceback.format_stack()[-5:],
                            thread_id=threading.current_thread().ident
                        )
                        
                        self.allocation_count += 1
                        managed_buffer = ManagedBuffer(buffer_id, buffer, self)
                        break
                
                # Check timeout
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Failed to acquire buffer of size {size} after {timeout}s")
                
                # Wait before retry
                time.sleep(0.1)
            
            yield managed_buffer
            
        finally:
            # Always release buffer
            if buffer_id and buffer_id in self._buffers:
                self.release(buffer_id)
    
    @asynccontextmanager
    async def acquire_async(self, size: int, timeout: float = 30.0) -> ManagedBuffer:
        """Asynchronous context manager for buffer acquisition"""
        import traceback
        
        if size > self.max_buffer_size:
            raise ValueError(f"Buffer size {size} exceeds maximum {self.max_buffer_size}")
        
        buffer_id = None
        
        try:
            # Acquire with timeout
            async with asyncio.timeout(timeout):
                while True:
                    async with self._async_lock:
                        buffer_id, buffer = self._try_acquire_buffer(size)
                        
                        if buffer_id:
                            # Track allocation
                            self._allocations[buffer_id] = BufferAllocation(
                                buffer_id=buffer_id,
                                size=size,
                                allocated_at=datetime.utcnow(),
                                last_accessed=datetime.utcnow(),
                                stack_trace=traceback.format_stack()[-5:],
                                thread_id=threading.current_thread().ident
                            )
                            
                            self.allocation_count += 1
                            managed_buffer = ManagedBuffer(buffer_id, buffer, self)
                            break
                    
                    # Wait for buffer to become available
                    self._allocation_event.clear()
                    await asyncio.wait_for(
                        self._allocation_event.wait(),
                        timeout=1.0
                    )
            
            yield managed_buffer
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Failed to acquire buffer of size {size} after {timeout}s")
        
        finally:
            # Always release buffer
            if buffer_id and buffer_id in self._buffers:
                self.release(buffer_id)
    
    def _try_acquire_buffer(self, size: int) -> Tuple[Optional[str], Optional[bytearray]]:
        """Try to acquire a buffer (internal, must be called with lock)"""
        import uuid
        
        # Check if we can allocate
        if self.used_size + size > self.max_size:
            # Try to reclaim leaked buffers
            self._reclaim_leaked_buffers()
            
            if self.used_size + size > self.max_size:
                return None, None
        
        # Try to reuse a free buffer
        for buffer_size, buffer_id in self._free_buffers:
            if buffer_size >= size and buffer_size <= size * 1.5:
                self._free_buffers.remove((buffer_size, buffer_id))
                buffer = self._buffers[buffer_id]
                buffer[:] = bytearray(size)  # Resize
                self.reuse_count += 1
                return buffer_id, buffer
        
        # Allocate new buffer
        buffer_id = str(uuid.uuid4())
        buffer = bytearray(size)
        self._buffers[buffer_id] = buffer
        self.used_size += size
        
        return buffer_id, buffer
    
    def release(self, buffer_id: str):
        """Release a buffer back to the pool"""
        with self._lock:
            if buffer_id not in self._buffers:
                logger.warning(f"Attempted to release unknown buffer {buffer_id}")
                return
            
            if buffer_id in self._allocations:
                del self._allocations[buffer_id]
            
            # Add to free list for reuse
            buffer = self._buffers[buffer_id]
            self._free_buffers.add((len(buffer), buffer_id))
            
            self.release_count += 1
            
            # Notify waiters
            if hasattr(self, '_allocation_event'):
                self._allocation_event.set()
    
    def _force_release(self, buffer_id: str):
        """Force release a buffer (used by garbage collector)"""
        with self._lock:
            if buffer_id in self._buffers:
                buffer = self._buffers.pop(buffer_id)
                self.used_size -= len(buffer)
                
                if buffer_id in self._allocations:
                    del self._allocations[buffer_id]
                
                self.leak_count += 1
    
    def _reclaim_leaked_buffers(self):
        """Reclaim buffers that appear to be leaked"""
        with self._lock:
            current_time = datetime.utcnow()
            leaked = []
            
            for buffer_id, allocation in self._allocations.items():
                if allocation.is_leaked():
                    leaked.append(buffer_id)
                    logger.warning(
                        f"Reclaiming leaked buffer {buffer_id} "
                        f"(age: {(current_time - allocation.allocated_at).total_seconds()}s)"
                    )
            
            for buffer_id in leaked:
                self._force_release(buffer_id)
    
    async def start_leak_detection(self):
        """Start background leak detection task"""
        if self._leak_check_task is None:
            self._leak_check_task = asyncio.create_task(self._leak_detection_loop())
    
    async def stop_leak_detection(self):
        """Stop leak detection task"""
        if self._leak_check_task:
            self._leak_check_task.cancel()
            try:
                await self._leak_check_task
            except asyncio.CancelledError:
                pass
            self._leak_check_task = None
    
    async def _leak_detection_loop(self):
        """Background task to detect and log leaks"""
        while True:
            try:
                await asyncio.sleep(self._leak_check_interval)
                
                with self._lock:
                    leaked = []
                    for buffer_id, allocation in self._allocations.items():
                        if allocation.is_leaked():
                            leaked.append((buffer_id, allocation))
                    
                    if leaked:
                        logger.error(
                            f"Detected {len(leaked)} leaked buffers:\n" +
                            "\n".join([
                                f"  - {bid}: size={alloc.size}, "
                                f"age={(datetime.utcnow() - alloc.allocated_at).total_seconds()}s, "
                                f"thread={alloc.thread_id}"
                                for bid, alloc in leaked
                            ])
                        )
                        
                        # Optionally reclaim
                        for buffer_id, _ in leaked:
                            self._force_release(buffer_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in leak detection: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            process = psutil.Process()
            
            return {
                "max_size": self.max_size,
                "used_size": self.used_size,
                "usage_percent": (self.used_size / self.max_size) * 100,
                "active_buffers": len(self._allocations),
                "free_buffers": len(self._free_buffers),
                "total_buffers": len(self._buffers),
                "allocation_count": self.allocation_count,
                "release_count": self.release_count,
                "leak_count": self.leak_count,
                "reuse_count": self.reuse_count,
                "reuse_rate": (self.reuse_count / max(self.allocation_count, 1)) * 100,
                "process_memory_mb": process.memory_info().rss / 1024 / 1024,
                "oldest_allocation_age": self._get_oldest_allocation_age()
            }
    
    def _get_oldest_allocation_age(self) -> Optional[float]:
        """Get age of oldest allocation in seconds"""
        if not self._allocations:
            return None
        
        current_time = datetime.utcnow()
        oldest = min(
            self._allocations.values(),
            key=lambda a: a.allocated_at
        )
        
        return (current_time - oldest.allocated_at).total_seconds()
```

### 2. Update Streaming Processor to Use Safe Pool

```python
# packages/shared/chunking/infrastructure/streaming/processor.py

class StreamingProcessor:
    def __init__(self, memory_pool: MemoryPool):
        self.memory_pool = memory_pool
        self._processing_stats = {
            "chunks_processed": 0,
            "bytes_processed": 0,
            "errors": 0
        }
    
    async def process_chunk(
        self,
        data: bytes,
        correlation_id: str
    ) -> ProcessingResult:
        """Process chunk with guaranteed buffer cleanup"""
        
        try:
            # Use context manager for automatic cleanup
            async with self.memory_pool.acquire_async(
                size=len(data),
                timeout=10.0
            ) as managed_buffer:
                # Buffer is guaranteed to be released
                buffer = managed_buffer.data
                buffer[:len(data)] = data
                
                # Process data
                result = await self._process_buffer(buffer, correlation_id)
                
                self._processing_stats["chunks_processed"] += 1
                self._processing_stats["bytes_processed"] += len(data)
                
                return result
                
        except TimeoutError as e:
            self._processing_stats["errors"] += 1
            logger.error(
                f"Failed to acquire buffer for chunk processing",
                extra={
                    "correlation_id": correlation_id,
                    "data_size": len(data),
                    "pool_stats": self.memory_pool.get_stats()
                }
            )
            raise ChunkingError("Memory pool exhausted", cause=e)
        
        except Exception as e:
            self._processing_stats["errors"] += 1
            logger.exception(
                f"Error processing chunk",
                extra={"correlation_id": correlation_id}
            )
            raise
    
    async def process_stream(
        self,
        stream: AsyncIterator[bytes],
        correlation_id: str
    ) -> List[ProcessingResult]:
        """Process stream with multiple buffers"""
        
        results = []
        tasks = []
        
        try:
            async for chunk in stream:
                # Process chunks concurrently with proper cleanup
                task = asyncio.create_task(
                    self.process_chunk(chunk, correlation_id)
                )
                tasks.append(task)
                
                # Limit concurrent tasks
                if len(tasks) >= 10:
                    done, tasks = await asyncio.wait(
                        tasks,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    for task in done:
                        try:
                            results.append(await task)
                        except Exception as e:
                            logger.error(f"Chunk processing failed: {e}")
            
            # Wait for remaining tasks
            if tasks:
                done, _ = await asyncio.wait(tasks)
                for task in done:
                    try:
                        results.append(await task)
                    except Exception as e:
                        logger.error(f"Chunk processing failed: {e}")
            
            return results
            
        finally:
            # Cancel any remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
```

### 3. Add Memory Monitoring

```python
# packages/shared/chunking/infrastructure/streaming/memory_monitor.py

import asyncio
import psutil
from typing import Optional, Callable
from dataclasses import dataclass

@dataclass
class MemoryAlert:
    level: str  # "warning", "critical"
    message: str
    usage_percent: float
    details: Dict[str, Any]

class MemoryMonitor:
    """Monitor memory usage and trigger alerts"""
    
    def __init__(
        self,
        memory_pool: MemoryPool,
        warning_threshold: float = 0.8,  # 80%
        critical_threshold: float = 0.95,  # 95%
        check_interval: int = 10  # seconds
    ):
        self.memory_pool = memory_pool
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval
        
        self._monitor_task: Optional[asyncio.Task] = None
        self._alert_callback: Optional[Callable] = None
        self._last_alert_level = None
    
    async def start(self, alert_callback: Optional[Callable] = None):
        """Start monitoring"""
        self._alert_callback = alert_callback
        self._monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop(self):
        """Stop monitoring"""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.check_interval)
                
                # Check memory usage
                stats = self.memory_pool.get_stats()
                usage_percent = stats["usage_percent"] / 100
                
                # Determine alert level
                alert_level = None
                if usage_percent >= self.critical_threshold:
                    alert_level = "critical"
                elif usage_percent >= self.warning_threshold:
                    alert_level = "warning"
                
                # Send alert if level changed
                if alert_level != self._last_alert_level:
                    if alert_level:
                        alert = MemoryAlert(
                            level=alert_level,
                            message=f"Memory pool usage at {usage_percent:.1%}",
                            usage_percent=usage_percent,
                            details=stats
                        )
                        
                        await self._send_alert(alert)
                    
                    self._last_alert_level = alert_level
                
                # Log statistics
                if alert_level:
                    logger.warning(
                        f"Memory pool {alert_level}: {usage_percent:.1%} used",
                        extra=stats
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in memory monitor: {e}")
    
    async def _send_alert(self, alert: MemoryAlert):
        """Send alert to callback"""
        if self._alert_callback:
            try:
                if asyncio.iscoroutinefunction(self._alert_callback):
                    await self._alert_callback(alert)
                else:
                    self._alert_callback(alert)
            except Exception as e:
                logger.exception(f"Error sending alert: {e}")
```

## Acceptance Criteria

1. **Resource Management**
   - [ ] All buffers released even on exceptions
   - [ ] Context managers used for all acquisitions
   - [ ] No manual acquire/release pairs
   - [ ] Garbage collector fallback works

2. **Leak Prevention**
   - [ ] Leaked buffers detected automatically
   - [ ] Leaked buffers reclaimed after timeout
   - [ ] Leak statistics tracked
   - [ ] Stack traces for leak debugging

3. **Thread Safety**
   - [ ] All operations thread-safe
   - [ ] No deadlocks possible
   - [ ] Async and sync modes work
   - [ ] Concurrent access handled

4. **Monitoring**
   - [ ] Memory usage tracked
   - [ ] Alerts on high usage
   - [ ] Statistics available
   - [ ] Oldest allocation tracked

5. **Performance**
   - [ ] Buffer reuse implemented
   - [ ] Allocation fast path optimized
   - [ ] No performance regression
   - [ ] Memory overhead minimal

## Testing Requirements

1. **Unit Tests**
   ```python
   async def test_buffer_cleanup_on_exception():
       pool = MemoryPool(max_size=1024)
       
       with pytest.raises(ValueError):
           async with pool.acquire_async(512) as buffer:
               raise ValueError("Test error")
       
       # Buffer should be released
       assert pool.get_stats()["active_buffers"] == 0
   
   async def test_leak_detection():
       pool = MemoryPool(max_size=1024)
       
       # Simulate leak by not using context manager
       buffer_id, buffer = pool._try_acquire_buffer(512)
       
       # Wait for leak detection
       await asyncio.sleep(1)
       pool._reclaim_leaked_buffers()
       
       assert pool.leak_count == 1
   
   async def test_concurrent_access():
       pool = MemoryPool(max_size=10240)
       
       async def acquire_and_release():
           async with pool.acquire_async(1024) as buffer:
               await asyncio.sleep(0.1)
       
       # Run concurrent acquisitions
       tasks = [acquire_and_release() for _ in range(10)]
       await asyncio.gather(*tasks)
       
       assert pool.get_stats()["active_buffers"] == 0
   ```

2. **Stress Tests**
   - Test with high concurrency
   - Test with memory pressure
   - Test with random failures
   - Test leak detection under load

## Rollback Plan

1. Keep old memory pool implementation
2. Feature flag for new pool
3. Monitor memory metrics closely
4. Quick revert if leaks detected

## Success Metrics

- Zero memory leaks in 24-hour test
- Memory usage stays under limit
- No OOM errors under load
- Buffer reuse rate > 50%
- Leak detection catches 100% of simulated leaks

## Notes for LLM Agent

- Always use context managers
- Never manual acquire/release
- Test exception paths thoroughly
- Monitor memory metrics
- Consider thread safety
- Add comprehensive logging
- Test with memory pressure