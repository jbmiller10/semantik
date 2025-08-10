#!/usr/bin/env python3
"""
Test memory usage constraints in streaming processor.

This module tests that the streaming processor maintains memory usage
under the specified limits regardless of file size.
"""

import asyncio
import os
import tempfile
import tracemalloc
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from packages.shared.chunking.domain.services.chunking_strategies.base import ChunkingStrategy
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.infrastructure.streaming.memory_pool import MemoryPool
from packages.shared.chunking.infrastructure.streaming.processor import StreamingDocumentProcessor
from packages.shared.chunking.infrastructure.streaming.window import StreamingWindow


class MockChunkingStrategy(ChunkingStrategy):
    """Mock chunking strategy for testing."""

    def __init__(self):
        super().__init__("mock_strategy")  # Pass name to parent constructor

    def chunk(self, text: str, config: ChunkConfig) -> list:
        """Simple chunking by splitting on periods."""
        chunks = []
        sentences = text.split(".")
        for sentence in sentences:
            if sentence.strip():
                chunk = MagicMock()
                chunk.content = sentence.strip()
                chunk.metadata = MagicMock()
                chunk.metadata.token_count = len(sentence.split())
                chunks.append(chunk)
        return chunks

    def validate_content(self, content: str) -> tuple[bool, str | None]:
        """Validate content for testing."""
        if not content:
            return False, "Content is empty"
        return True, None

    def estimate_chunks(self, content_length: int, config: ChunkConfig) -> int:
        """Estimate number of chunks for testing."""
        # Rough estimate: one chunk per 100 characters
        return max(1, content_length // 100)


class TestMemoryUsage:
    """Test suite for memory usage constraints."""

    @pytest.fixture()
    def temp_file(self) -> tuple[str, Path]:
        """Create a temporary file for testing."""
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        yield path
        Path(path).unlink(missing_ok=True)

    @pytest.fixture()
    def mock_config(self) -> ChunkConfig:
        """Create mock chunk configuration."""
        config = MagicMock(spec=ChunkConfig)
        config.min_tokens = 10
        config.max_tokens = 100
        config.estimate_chunks = lambda tokens: tokens // 50
        return config

    async def test_small_file_memory_usage(self, temp_file, mock_config):
        """Test memory usage with small file (< 1MB)."""
        # Create small test file
        content = "This is a test sentence. " * 100  # ~2.5KB
        Path(temp_file).write_text(content)

        processor = StreamingDocumentProcessor()
        strategy = MockChunkingStrategy()

        # Track memory
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]

        chunks = []
        async for chunk in processor.process_document(str(temp_file), strategy, mock_config):
            chunks.append(chunk)

        peak_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        # Memory should be well under limit
        memory_used = peak_memory - start_memory
        assert memory_used < 10 * 1024 * 1024  # Less than 10MB for small file
        assert len(chunks) > 0

    async def test_memory_pool_limits(self):
        """Test memory pool enforces size limits."""
        pool = MemoryPool(buffer_size=1024, pool_size=5)

        # Total memory should be buffer_size * pool_size
        assert pool.total_memory == 1024 * 5

        # Acquire all buffers
        acquired = []
        for _ in range(5):
            buffer_id, buffer = await pool.acquire(timeout=1.0)
            acquired.append(buffer_id)
            assert len(buffer) == 1024

        # Pool should be exhausted
        with pytest.raises(TimeoutError):
            await pool.acquire(timeout=0.1)

        # Release one and acquire should work
        pool.release(acquired[0])
        buffer_id, buffer = await pool.acquire(timeout=0.1)
        assert buffer_id == acquired[0]

    async def test_streaming_window_memory_bounds(self):
        """Test streaming window respects memory limits."""
        max_size = 256 * 1024  # 256KB
        window = StreamingWindow(max_size=max_size)

        # Add data up to limit
        chunk = b"A" * 1024  # 1KB chunks
        for _ in range(256):
            window.append(chunk)

        assert window.size <= max_size

        # Adding more should trigger slide or error
        # Try to add without sliding
        large_chunk = b"B" * max_size
        with pytest.raises(MemoryError):
            window.append(large_chunk)

    async def test_window_sliding_releases_memory(self):
        """Test that sliding window properly releases memory."""
        window = StreamingWindow(max_size=10 * 1024)  # 10KB

        # Fill window
        data = b"A" * 8192  # 8KB
        window.append(data)

        initial_size = window.size
        assert initial_size == 8192

        # Slide window
        released = window.slide(amount=4096)  # Slide 4KB
        assert len(released) == 4096
        assert window.size == initial_size - 4096

        # Add more data
        window.append(b"B" * 2048)
        assert window.size == 6144  # 4KB + 2KB


class TestMemoryPoolConcurrency:
    """Test memory pool under concurrent access."""

    async def test_concurrent_acquire_release(self):
        """Test concurrent acquire/release operations."""
        pool = MemoryPool(buffer_size=1024, pool_size=5)

        async def worker(worker_id: int, iterations: int):
            results = []
            for i in range(iterations):
                buffer_id, buffer = await pool.acquire()
                # Simulate work
                await asyncio.sleep(0.001)
                pool.release(buffer_id)
                results.append((worker_id, i))
            return results

        # Run multiple workers concurrently
        workers = [worker(i, 10) for i in range(10)]
        results = await asyncio.gather(*workers)

        # All workers should complete
        assert len(results) == 10
        assert all(len(r) == 10 for r in results)

        # Pool should be in clean state
        assert pool.available_buffers == pool.pool_size
        assert pool.used_buffers == 0

    async def test_pool_statistics_accuracy(self):
        """Test that pool statistics are accurate under load."""
        pool = MemoryPool(buffer_size=2048, pool_size=3)

        # Get initial stats
        stats = pool.get_statistics()
        assert stats["available"] == 3
        assert stats["in_use"] == 0
        assert stats["total_acquisitions"] == 0

        # Acquire buffers
        b1_id, _ = await pool.acquire()
        b2_id, _ = await pool.acquire()

        stats = pool.get_statistics()
        assert stats["available"] == 1
        assert stats["in_use"] == 2
        assert stats["total_acquisitions"] == 2
        assert stats["max_concurrent_usage"] == 2

        # Release one
        pool.release(b1_id)

        stats = pool.get_statistics()
        assert stats["available"] == 2
        assert stats["in_use"] == 1
        assert stats["total_releases"] == 1

        # Release all
        pool.release(b2_id)

        stats = pool.get_statistics()
        assert stats["available"] == 3
        assert stats["in_use"] == 0
        assert stats["utilization"] == 0.0
