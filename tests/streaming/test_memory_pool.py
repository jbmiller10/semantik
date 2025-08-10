#!/usr/bin/env python3
"""
Unit tests for the memory pool component.

Separated from integration tests to isolate functionality.
"""

import pytest

from packages.shared.chunking.infrastructure.streaming.memory_pool import MemoryPool


class TestMemoryPool:
    """Test cases for MemoryPool class."""

    def test_pool_initialization(self):
        """Test pool is properly initialized."""
        pool = MemoryPool(buffer_size=1024, pool_size=5)

        assert pool.buffer_size == 1024
        assert pool.pool_size == 5
        assert pool.available_buffers == 5
        assert pool.used_buffers == 0
        assert pool.total_memory == 5 * 1024

    def test_acquire_and_release_sync(self):
        """Test synchronous buffer acquisition and release."""
        pool = MemoryPool(buffer_size=512, pool_size=3)

        # Acquire a buffer
        buffer_id, buffer = pool.acquire_sync()
        assert buffer_id in range(3)
        assert len(buffer) == 512
        assert pool.used_buffers == 1
        assert pool.available_buffers == 2

        # Release the buffer
        pool.release(buffer_id)
        assert pool.used_buffers == 0
        assert pool.available_buffers == 3

    def test_statistics_tracking(self):
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

    def test_pool_exhaustion(self):
        """Test behavior when pool is exhausted."""
        pool = MemoryPool(buffer_size=128, pool_size=2)

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
        assert buffer3_id == buffer1_id  # Should reuse the released buffer

    def test_clear_pool(self):
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

    def test_clear_with_buffers_in_use(self):
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

    def test_invalid_release(self):
        """Test releasing an invalid buffer ID."""
        pool = MemoryPool(buffer_size=128, pool_size=2)

        # Try to release a buffer that wasn't acquired
        with pytest.raises(ValueError, match="not in use"):
            pool.release(0)

        # Acquire and release properly
        buffer_id, _ = pool.acquire_sync()
        pool.release(buffer_id)

        # Try to release again (should fail)
        with pytest.raises(ValueError, match="not in use"):
            pool.release(buffer_id)

    def test_buffer_clearing_on_acquire(self):
        """Test that buffers are cleared when acquired."""
        pool = MemoryPool(buffer_size=10, pool_size=1)

        # Acquire, modify, and release
        buffer_id, buffer = pool.acquire_sync()
        buffer[0:5] = b"hello"
        pool.release(buffer_id)

        # Acquire again - should be cleared
        buffer_id2, buffer2 = pool.acquire_sync()
        assert buffer_id2 == buffer_id  # Same buffer
        assert buffer2[0:5] == b"\x00\x00\x00\x00\x00"  # Should be cleared

    def test_resize_buffer(self):
        """Test resizing a buffer."""
        pool = MemoryPool(buffer_size=128, pool_size=2)

        # Can't resize a buffer in use
        buffer_id, _ = pool.acquire_sync()
        with pytest.raises(ValueError, match="while in use"):
            pool.resize_buffer(buffer_id, 256)

        pool.release(buffer_id)

        # Now resize should work
        new_buffer = pool.resize_buffer(0, 256)
        assert len(new_buffer) == 256

        # Invalid size should fail
        with pytest.raises(ValueError, match="Invalid buffer size"):
            pool.resize_buffer(0, -1)
