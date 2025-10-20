#!/usr/bin/env python3

"""
Unit tests for the memory pool component.

Separated from integration tests to isolate functionality.
"""

import pytest

from packages.shared.chunking.infrastructure.streaming.memory_pool import MemoryPool


class TestMemoryPool:
    """Test cases for MemoryPool class."""

    def test_pool_initialization(self) -> None:
        """Test pool is properly initialized."""
        pool = MemoryPool(buffer_size=1024, pool_size=5)

        assert pool.buffer_size == 1024
        assert pool.pool_size == 5
        assert pool.available_buffers == 5
        assert pool.used_buffers == 0
        assert pool.total_memory == 5 * 1024

    def test_acquire_and_release_sync(self) -> None:
        """Test synchronous buffer acquisition and release."""
        pool = MemoryPool(buffer_size=512, pool_size=3)

        # Acquire a buffer
        buffer_id, buffer = pool.acquire_sync()
        assert isinstance(buffer_id, str)  # UUID string
        assert len(buffer) == 512
        assert pool.used_buffers == 1
        assert pool.available_buffers == 2

        # Release the buffer
        pool.release(buffer_id)
        assert pool.used_buffers == 0
        assert pool.available_buffers == 3

    def test_statistics_tracking(self) -> None:
        """Test that statistics are properly tracked."""
        pool = MemoryPool(buffer_size=256, pool_size=2)

        # Initial stats
        stats = pool.get_statistics()
        assert stats["total_acquisitions"] == 0
        assert stats["total_releases"] == 0
        assert stats["in_use"] == 0
        assert stats["available"] == 2

        # Acquire and check stats
        buffer_id, _ = pool.acquire_sync()
        stats = pool.get_statistics()
        assert stats["total_acquisitions"] == 1
        assert stats["in_use"] == 1
        assert stats["available"] == 1

        # Release and check stats
        pool.release(buffer_id)
        stats = pool.get_statistics()
        assert stats["total_releases"] == 1
        assert stats["in_use"] == 0
        assert stats["available"] == 2

    def test_pool_exhaustion(self) -> None:
        """Test behavior when pool is exhausted."""
        pool = MemoryPool(buffer_size=128, pool_size=2, max_size=256)  # Limit to exactly 2 buffers

        # Acquire all buffers
        buffer1_id, _ = pool.acquire_sync()
        buffer2_id, _ = pool.acquire_sync()

        assert pool.available_buffers == 0
        assert pool.used_buffers == 2

        # Try to acquire when exhausted (should timeout)
        with pytest.raises(TimeoutError):
            pool.acquire_sync(timeout=0.1)

        # Release one and try again
        pool.release(buffer1_id)
        buffer3_id, _ = pool.acquire_sync()
        assert isinstance(buffer3_id, str)  # Should get a buffer (may reuse)

    def test_clear_pool(self) -> None:
        """Test clearing the pool."""
        pool = MemoryPool(buffer_size=256, pool_size=3)

        # Acquire and release to generate statistics
        buffer_id, _ = pool.acquire_sync()
        pool.release(buffer_id)

        # Stats before clear
        stats = pool.get_statistics()
        assert stats["total_acquisitions"] == 1
        assert stats["total_releases"] == 1

        # Clear the pool
        pool.clear()

        # Stats after clear should be reset
        stats = pool.get_statistics()
        assert stats["total_acquisitions"] == 0
        assert stats["total_releases"] == 0
        assert stats["available"] == 3
        assert stats["in_use"] == 0

    def test_clear_with_buffers_in_use(self) -> None:
        """Test that clear fails when buffers are in use."""
        pool = MemoryPool(buffer_size=128, pool_size=2)

        # Acquire a buffer
        buffer_id, _ = pool.acquire_sync()

        # Try to clear (should fail)
        with pytest.raises(RuntimeError, match="Cannot clear pool"):
            pool.clear()

        # Release and then clear should work
        pool.release(buffer_id)
        pool.clear()  # Should not raise

    def test_invalid_release(self) -> None:
        """Test releasing an invalid buffer ID."""
        pool = MemoryPool(buffer_size=128, pool_size=2)

        # Try to release a buffer that wasn't acquired - should just warn
        pool.release("invalid-id")  # Should not raise, just log warning

        # Acquire and release properly
        buffer_id, _ = pool.acquire_sync()
        pool.release(buffer_id)

        # Try to release again - should just warn
        pool.release(buffer_id)  # Should not raise, just log warning

    def test_buffer_reuse(self) -> None:
        """Test that buffers can be reused."""
        pool = MemoryPool(buffer_size=10, pool_size=1)

        # Acquire, modify, and release
        buffer_id, buffer = pool.acquire_sync()
        pool.release(buffer_id)

        # Acquire again - should get a buffer (may be same or new)
        buffer_id2, buffer2 = pool.acquire_sync()
        assert isinstance(buffer_id2, str)
        assert len(buffer2) == 10

    def test_context_manager(self) -> None:
        """Test using MemoryPool as context manager."""
        pool = MemoryPool(buffer_size=128, pool_size=2)

        # Pool should work as a context manager
        with pool as p:
            assert p is pool
            buffer_id, _ = p.acquire_sync()
            p.release(buffer_id)
