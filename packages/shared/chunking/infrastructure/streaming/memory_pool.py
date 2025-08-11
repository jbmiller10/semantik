#!/usr/bin/env python3
"""
Memory pool for efficient buffer management with automatic leak prevention.

This module implements a reusable buffer pool with context managers to ensure
proper cleanup, leak detection, and bounded memory usage during streaming operations.
"""

import asyncio
import logging
import threading
import traceback
import weakref
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import psutil

logger = logging.getLogger(__name__)


@dataclass
class BufferAllocation:
    """Track buffer allocation metadata for leak detection."""

    buffer_id: str
    size: int
    allocated_at: datetime
    last_accessed: datetime
    stack_trace: list[str]
    thread_id: int | None
    owner: weakref.ref | None = None

    def is_leaked(self, timeout_seconds: int = 300) -> bool:
        """
        Check if buffer is likely leaked.

        Args:
            timeout_seconds: Time after which buffer is considered leaked

        Returns:
            True if buffer appears to be leaked
        """
        age = (datetime.now(UTC) - self.allocated_at).total_seconds()
        return age > timeout_seconds and self.owner is None


class ManagedBuffer:
    """Buffer wrapper with automatic cleanup on garbage collection."""

    def __init__(self, buffer_id: str, data: bytearray, pool: "MemoryPool"):
        """
        Initialize managed buffer.

        Args:
            buffer_id: Unique buffer identifier
            data: Buffer data
            pool: Parent memory pool
        """
        self.buffer_id = buffer_id
        self.data = data
        self.pool = weakref.ref(pool)
        self._released = False

    def __del__(self):
        """Ensure buffer is released on garbage collection."""
        if not self._released:
            pool = self.pool()
            if pool:
                logger.warning(
                    f"Buffer {self.buffer_id} released by garbage collector - "
                    "possible leak detected"
                )
                pool._force_release(self.buffer_id)

    def release(self):
        """Explicitly release buffer back to pool."""
        if not self._released:
            pool = self.pool()
            if pool:
                pool.release(self.buffer_id)
                self._released = True


class MemoryPool:
    """Thread-safe memory pool with automatic leak prevention."""

    def __init__(
        self,
        max_size: int = 100 * 1024 * 1024,  # 100MB
        max_buffer_size: int = 10 * 1024 * 1024,  # 10MB per buffer
        buffer_size: int = 64 * 1024,  # Default buffer size for compatibility
        pool_size: int = 10,  # Number of pre-allocated buffers for compatibility
        leak_check_interval: int = 60,  # seconds
    ):
        """
        Initialize memory pool with leak prevention.

        Args:
            max_size: Maximum total memory pool size in bytes
            max_buffer_size: Maximum size per buffer
            buffer_size: Default buffer size (for backward compatibility)
            pool_size: Number of buffers to pre-allocate (for backward compatibility)
            leak_check_interval: Seconds between leak checks
        """
        self.max_size = max_size
        self.max_buffer_size = max_buffer_size
        self.default_buffer_size = buffer_size
        self.initial_pool_size = pool_size
        self.used_size = 0

        # Thread-safe structures
        self._lock = threading.RLock()
        self._buffers: dict[str, bytearray] = {}
        self._allocations: dict[str, BufferAllocation] = {}
        self._free_buffers: set[tuple[int, str]] = set()  # (size, buffer_id)

        # Async support
        self._async_lock: asyncio.Lock | None = None
        self._allocation_event: asyncio.Event | None = None

        # Leak detection
        self._leak_check_task: asyncio.Task | None = None
        self._leak_check_interval = leak_check_interval
        self._leaked_buffers: set[str] = set()

        # Metrics
        self.allocation_count = 0
        self.release_count = 0
        self.leak_count = 0
        self.reuse_count = 0

        # Pre-allocate initial buffers for backward compatibility
        self._preallocate_buffers()

    def _preallocate_buffers(self):
        """Pre-allocate initial buffers for backward compatibility."""
        with self._lock:
            for _ in range(self.initial_pool_size):
                buffer_id = str(uuid4())
                buffer = bytearray(self.default_buffer_size)
                self._buffers[buffer_id] = buffer
                self._free_buffers.add((self.default_buffer_size, buffer_id))
                self.used_size += self.default_buffer_size

    @contextmanager
    def acquire_sync_context(self, size: int = None, timeout: float = 30.0):
        """
        Synchronous context manager for buffer acquisition.

        Args:
            size: Buffer size needed (uses default if None)
            timeout: Maximum time to wait for buffer

        Yields:
            ManagedBuffer with automatic cleanup

        Raises:
            ValueError: If buffer size exceeds maximum
            TimeoutError: If buffer not available within timeout
        """
        import time

        if size is None:
            size = self.default_buffer_size

        if size > self.max_buffer_size:
            raise ValueError(f"Buffer size {size} exceeds maximum {self.max_buffer_size}")

        start_time = time.time()
        buffer_id = None

        try:
            while True:
                with self._lock:
                    # Try to acquire buffer
                    buffer_id, buffer = self._try_acquire_buffer(size)

                    if buffer_id:
                        # Track allocation
                        self._allocations[buffer_id] = BufferAllocation(
                            buffer_id=buffer_id,
                            size=size,
                            allocated_at=datetime.now(UTC),
                            last_accessed=datetime.now(UTC),
                            stack_trace=traceback.format_stack()[-5:],
                            thread_id=threading.current_thread().ident,
                        )

                        self.allocation_count += 1
                        managed_buffer = ManagedBuffer(buffer_id, buffer, self)
                        break

                # Check timeout
                if time.time() - start_time > timeout:
                    raise TimeoutError(
                        f"Failed to acquire buffer of size {size} after {timeout}s"
                    )

                # Wait before retry
                time.sleep(0.1)

            yield managed_buffer

        finally:
            # Always release buffer
            if buffer_id and buffer_id in self._buffers:
                self.release(buffer_id)

    @asynccontextmanager
    async def acquire_async(self, size: int = None, timeout: float = 30.0):
        """
        Asynchronous context manager for buffer acquisition.

        Args:
            size: Buffer size needed (uses default if None)
            timeout: Maximum time to wait for buffer

        Yields:
            ManagedBuffer with automatic cleanup

        Raises:
            ValueError: If buffer size exceeds maximum
            TimeoutError: If buffer not available within timeout
        """
        if size is None:
            size = self.default_buffer_size

        if size > self.max_buffer_size:
            raise ValueError(f"Buffer size {size} exceeds maximum {self.max_buffer_size}")

        # Initialize async structures if needed
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        if self._allocation_event is None:
            self._allocation_event = asyncio.Event()

        buffer_id = None

        try:
            # Acquire with timeout
            async with asyncio.timeout(timeout):
                while True:
                    async with self._async_lock:
                        with self._lock:  # Also use thread lock for thread safety
                            buffer_id, buffer = self._try_acquire_buffer(size)

                            if buffer_id:
                                # Track allocation
                                self._allocations[buffer_id] = BufferAllocation(
                                    buffer_id=buffer_id,
                                    size=size,
                                    allocated_at=datetime.now(UTC),
                                    last_accessed=datetime.now(UTC),
                                    stack_trace=traceback.format_stack()[-5:],
                                    thread_id=threading.current_thread().ident,
                                )

                                self.allocation_count += 1
                                managed_buffer = ManagedBuffer(buffer_id, buffer, self)
                                break

                    # Wait for buffer to become available
                    self._allocation_event.clear()
                    try:
                        await asyncio.wait_for(
                            self._allocation_event.wait(), timeout=1.0
                        )
                    except TimeoutError:
                        continue  # Check again

            yield managed_buffer

        except TimeoutError as e:
            raise TimeoutError(
                f"Failed to acquire buffer of size {size} after {timeout}s"
            ) from e

        finally:
            # Always release buffer
            if buffer_id and buffer_id in self._buffers:
                self.release(buffer_id)

    # Backward compatibility methods
    async def acquire(self, timeout: float = 5.0) -> tuple[str, bytearray]:
        """
        Legacy acquire method for backward compatibility.

        Args:
            timeout: Maximum time to wait for buffer

        Returns:
            Tuple of (buffer_id, buffer)

        Note:
            This method is deprecated. Use acquire_async context manager instead.
        """
        logger.warning(
            "Using deprecated acquire() method. Please use acquire_async() context manager."
        )

        async with self.acquire_async(size=self.default_buffer_size, timeout=timeout) as managed_buffer:
            # Return buffer but keep it tracked - caller must release
            return managed_buffer.buffer_id, managed_buffer.data

    def acquire_sync_legacy(self, timeout: float = 5.0) -> tuple[str, bytearray]:
        """
        Legacy synchronous acquire for backward compatibility.

        Args:
            timeout: Maximum time to wait for buffer

        Returns:
            Tuple of (buffer_id, buffer)

        Note:
            This method is deprecated. Use acquire_sync_context context manager instead.
        """
        logger.warning(
            "Using deprecated acquire_sync() method. Please use acquire_sync_context() context manager."
        )

        import time

        if self.default_buffer_size > self.max_buffer_size:
            raise ValueError(f"Buffer size {self.default_buffer_size} exceeds maximum {self.max_buffer_size}")

        start_time = time.time()

        while True:
            with self._lock:
                # Try to acquire buffer
                buffer_id, buffer = self._try_acquire_buffer(self.default_buffer_size)

                if buffer_id:
                    # Track allocation
                    self._allocations[buffer_id] = BufferAllocation(
                        buffer_id=buffer_id,
                        size=self.default_buffer_size,
                        allocated_at=datetime.now(UTC),
                        last_accessed=datetime.now(UTC),
                        stack_trace=traceback.format_stack()[-5:],
                        thread_id=threading.current_thread().ident,
                    )

                    self.allocation_count += 1
                    return buffer_id, buffer

            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Failed to acquire buffer of size {self.default_buffer_size} after {timeout}s"
                )

            # Wait before retry
            time.sleep(0.1)

    def _try_acquire_buffer(self, size: int) -> tuple[str | None, bytearray | None]:
        """
        Try to acquire a buffer (internal, must be called with lock).

        Args:
            size: Required buffer size

        Returns:
            Tuple of (buffer_id, buffer) or (None, None) if not available
        """
        # First, try to reuse a free buffer
        for buffer_size, buffer_id in list(self._free_buffers):
            if buffer_size >= size and buffer_size <= size * 1.5:
                self._free_buffers.remove((buffer_size, buffer_id))
                buffer = self._buffers[buffer_id]
                # Resize if needed
                if len(buffer) != size:
                    buffer[:] = bytearray(size)
                    self.used_size = self.used_size - buffer_size + size
                self.reuse_count += 1
                return buffer_id, buffer

        # If no free buffer available, check if we can allocate a new one
        if self.used_size + size > self.max_size:
            # Try to reclaim leaked buffers
            self._reclaim_leaked_buffers()

            if self.used_size + size > self.max_size:
                return None, None

        # Allocate new buffer
        buffer_id = str(uuid4())
        buffer = bytearray(size)
        self._buffers[buffer_id] = buffer
        self.used_size += size

        return buffer_id, buffer

    def release(self, buffer_id: str):
        """
        Release a buffer back to the pool.

        Args:
            buffer_id: Buffer identifier to release
        """
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
            if self._allocation_event:
                self._allocation_event.set()

    def _force_release(self, buffer_id: str):
        """
        Force release a buffer (used by garbage collector).

        Args:
            buffer_id: Buffer to force release
        """
        with self._lock:
            if buffer_id in self._buffers:
                buffer = self._buffers.pop(buffer_id)
                self.used_size -= len(buffer)

                if buffer_id in self._allocations:
                    del self._allocations[buffer_id]

                # Remove from free list if present
                self._free_buffers = {
                    (size, bid) for size, bid in self._free_buffers if bid != buffer_id
                }

                self.leak_count += 1

    def _reclaim_leaked_buffers(self):
        """Reclaim buffers that appear to be leaked."""
        with self._lock:
            current_time = datetime.now(UTC)
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
        """Start background leak detection task."""
        if self._leak_check_task is None:
            self._leak_check_task = asyncio.create_task(self._leak_detection_loop())

    async def stop_leak_detection(self):
        """Stop leak detection task."""
        if self._leak_check_task:
            self._leak_check_task.cancel()
            try:
                await self._leak_check_task
            except asyncio.CancelledError:
                pass
            finally:
                self._leak_check_task = None

    async def _leak_detection_loop(self):
        """Background task to detect and log leaks."""
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
                            f"Detected {len(leaked)} leaked buffers:\n"
                            + "\n".join(
                                [
                                    f"  - {bid}: size={alloc.size}, "
                                    f"age={(datetime.now(UTC) - alloc.allocated_at).total_seconds()}s, "
                                    f"thread={alloc.thread_id}"
                                    for bid, alloc in leaked
                                ]
                            )
                        )

                        # Optionally reclaim
                        for buffer_id, _ in leaked:
                            self._force_release(buffer_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in leak detection: {e}")

    def get_stats(self) -> dict[str, Any]:
        """
        Get pool statistics.

        Returns:
            Dictionary with pool metrics
        """
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
                "reuse_rate": (
                    (self.reuse_count / max(self.allocation_count, 1)) * 100
                ),
                "process_memory_mb": process.memory_info().rss / 1024 / 1024,
                "oldest_allocation_age": self._get_oldest_allocation_age(),
            }

    def get_statistics(self) -> dict:
        """
        Get pool usage statistics (backward compatibility).

        Returns:
            Dictionary with usage statistics
        """
        stats = self.get_stats()
        # Map to old format for backward compatibility
        return {
            "pool_size": self.initial_pool_size,
            "buffer_size": self.default_buffer_size,
            "total_memory": stats["used_size"],
            "available": stats["free_buffers"],
            "in_use": stats["active_buffers"],
            "total_acquisitions": stats["allocation_count"],
            "total_releases": stats["release_count"],
            "max_concurrent_usage": stats["active_buffers"],  # Approximation
            "average_wait_time": 0.0,  # Not tracked in new implementation
            "allocation_failures": 0,  # Not tracked in new implementation
            "utilization": stats["usage_percent"] / 100,
        }

    def _get_oldest_allocation_age(self) -> float | None:
        """
        Get age of oldest allocation in seconds.

        Returns:
            Age in seconds or None if no allocations
        """
        if not self._allocations:
            return None

        current_time = datetime.now(UTC)
        oldest = min(self._allocations.values(), key=lambda a: a.allocated_at)

        return (current_time - oldest.allocated_at).total_seconds()

    def clear(self) -> None:
        """
        Clear all buffers and reset pool.

        Raises:
            RuntimeError: If any buffers are still in use
        """
        with self._lock:
            if self._allocations:
                raise RuntimeError(
                    f"Cannot clear pool: {len(self._allocations)} buffers still in use"
                )

            # Clear all buffers
            self._buffers.clear()
            self._free_buffers.clear()
            self._allocations.clear()
            self.used_size = 0

            # Re-preallocate
            self._preallocate_buffers()

            # Reset statistics
            self.allocation_count = 0
            self.release_count = 0
            self.leak_count = 0
            self.reuse_count = 0

    def reset_statistics(self) -> None:
        """Reset usage statistics."""
        with self._lock:
            self.allocation_count = 0
            self.release_count = 0
            self.leak_count = 0
            self.reuse_count = 0

    @property
    def total_memory(self) -> int:
        """
        Calculate total memory allocated by the pool.

        Returns:
            Total bytes allocated
        """
        with self._lock:
            return self.used_size

    @property
    def available_buffers(self) -> int:
        """
        Get number of available buffers.

        Returns:
            Count of available buffers
        """
        with self._lock:
            return len(self._free_buffers)

    @property
    def used_buffers(self) -> int:
        """
        Get number of buffers in use.

        Returns:
            Count of used buffers
        """
        with self._lock:
            return len(self._allocations)

    def __repr__(self) -> str:
        """String representation of the pool."""
        with self._lock:
            return (
                f"MemoryPool(max_size={self.max_size}, "
                f"used={self.used_size}, "
                f"active={len(self._allocations)}, "
                f"free={len(self._free_buffers)})"
            )

    @property
    def buffer_size(self) -> int:
        """Get default buffer size (backward compatibility)."""
        return self.default_buffer_size

    @property
    def pool_size(self) -> int:
        """Get initial pool size (backward compatibility)."""
        return self.initial_pool_size

    def acquire_sync(self, timeout: float = 5.0) -> tuple[str, bytearray]:
        """
        Synchronous acquire for backward compatibility.
        This is the non-context-manager version.
        
        Args:
            timeout: Maximum time to wait for buffer
            
        Returns:
            Tuple of (buffer_id, buffer)
        """
        return self.acquire_sync_legacy(timeout)

    def __enter__(self) -> "MemoryPool":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensure all buffers returned."""
        if self._allocations:
            # Log warning but don't raise - buffers will be GC'd
            logger.warning(
                f"MemoryPool exiting with {len(self._allocations)} buffers still in use"
            )

