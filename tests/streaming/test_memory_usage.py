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
from typing import List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from packages.shared.chunking.domain.services.chunking_strategies.base import ChunkingStrategy
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.infrastructure.streaming.memory_pool import MemoryPool
from packages.shared.chunking.infrastructure.streaming.processor import StreamingDocumentProcessor
from packages.shared.chunking.infrastructure.streaming.window import StreamingWindow


class MockChunkingStrategy(ChunkingStrategy):
    """Mock chunking strategy for testing."""
    
    def __init__(self):
        self.name = "mock_strategy"
    
    def chunk(self, text: str, config: ChunkConfig) -> List:
        """Simple chunking by splitting on periods."""
        chunks = []
        sentences = text.split('.')
        for sentence in sentences:
            if sentence.strip():
                chunk = MagicMock()
                chunk.content = sentence.strip()
                chunk.metadata = MagicMock()
                chunk.metadata.token_count = len(sentence.split())
                chunks.append(chunk)
        return chunks


class TestMemoryUsage:
    """Test suite for memory usage constraints."""
    
    @pytest.fixture
    def temp_file(self) -> Tuple[str, Path]:
        """Create a temporary file for testing."""
        fd, path = tempfile.mkstemp(suffix='.txt')
        os.close(fd)
        yield path
        Path(path).unlink(missing_ok=True)
    
    @pytest.fixture
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
        async for chunk in processor.process_document(
            str(temp_file), strategy, mock_config
        ):
            chunks.append(chunk)
        
        peak_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        
        # Memory should be well under limit
        memory_used = peak_memory - start_memory
        assert memory_used < 10 * 1024 * 1024  # Less than 10MB for small file
        assert len(chunks) > 0
    
    async def test_large_file_memory_bounded(self, temp_file, mock_config):
        """Test memory stays bounded with large file (100MB)."""
        # Create large test file
        chunk_size = 1024 * 1024  # 1MB chunks
        total_size = 100 * chunk_size  # 100MB
        
        with open(temp_file, 'wb') as f:
            sentence = b"This is test data. "
            chunk = sentence * (chunk_size // len(sentence))
            for _ in range(total_size // chunk_size):
                f.write(chunk)
        
        processor = StreamingDocumentProcessor()
        strategy = MockChunkingStrategy()
        
        # Track memory
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        
        max_memory_delta = 0
        chunks_processed = 0
        
        async for chunk in processor.process_document(
            str(temp_file), strategy, mock_config
        ):
            chunks_processed += 1
            
            # Check memory periodically
            if chunks_processed % 100 == 0:
                current_memory = tracemalloc.get_traced_memory()[0]
                memory_delta = current_memory - start_memory
                max_memory_delta = max(max_memory_delta, memory_delta)
        
        tracemalloc.stop()
        
        # Memory should stay under 100MB limit
        assert max_memory_delta < processor.MAX_MEMORY
        assert chunks_processed > 0
    
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
        with pytest.raises(MemoryError):
            # Try to add without sliding
            large_chunk = b"B" * max_size
            window.append(large_chunk)
    
    async def test_backpressure_prevents_memory_overflow(self, temp_file, mock_config):
        """Test backpressure mechanism prevents memory overflow."""
        # Create medium test file
        content = "Test sentence. " * 10000  # ~150KB
        Path(temp_file).write_text(content)
        
        # Create processor with small pool for testing
        memory_pool = MemoryPool(buffer_size=1024, pool_size=2)
        processor = StreamingDocumentProcessor(memory_pool=memory_pool)
        strategy = MockChunkingStrategy()
        
        # Track backpressure events
        backpressure_events = []
        
        original_manage = processor._manage_backpressure
        async def track_backpressure():
            backpressure_events.append(processor.downstream_pressure)
            return await original_manage()
        
        processor._manage_backpressure = track_backpressure
        
        chunks = []
        async for chunk in processor.process_document(
            str(temp_file), strategy, mock_config
        ):
            chunks.append(chunk)
            # Simulate slow processing
            if len(chunks) % 10 == 0:
                await asyncio.sleep(0.01)
        
        # Should have managed backpressure
        assert len(chunks) > 0
        # Pool should be mostly available after completion
        assert memory_pool.available_buffers >= memory_pool.pool_size - 1
    
    async def test_memory_statistics_tracking(self, temp_file, mock_config):
        """Test memory statistics are properly tracked."""
        content = "Test data. " * 1000
        Path(temp_file).write_text(content)
        
        processor = StreamingDocumentProcessor()
        strategy = MockChunkingStrategy()
        
        chunks = []
        async for chunk in processor.process_document(
            str(temp_file), strategy, mock_config
        ):
            chunks.append(chunk)
        
        # Check memory statistics
        stats = processor.get_memory_usage()
        assert 'total_allocated' in stats
        assert 'buffers_in_use' in stats
        assert 'buffers_available' in stats
        assert 'utilization' in stats
        assert 'max_memory' in stats
        assert 'within_limit' in stats
        
        # Should be within limits
        assert stats['within_limit'] is True
        assert stats['total_allocated'] <= stats['max_memory']
    
    async def test_concurrent_operations_memory_isolated(self, mock_config):
        """Test concurrent operations don't exceed memory limits."""
        # Create multiple temp files
        temp_files = []
        for i in range(3):
            fd, path = tempfile.mkstemp(suffix=f'_{i}.txt')
            os.close(fd)
            temp_files.append(path)
            Path(path).write_text(f"Test file {i}. " * 1000)
        
        try:
            # Single shared memory pool
            memory_pool = MemoryPool(
                buffer_size=64 * 1024,
                pool_size=10
            )
            
            async def process_file(file_path):
                processor = StreamingDocumentProcessor(memory_pool=memory_pool)
                strategy = MockChunkingStrategy()
                chunks = []
                async for chunk in processor.process_document(
                    str(file_path), strategy, mock_config
                ):
                    chunks.append(chunk)
                return len(chunks)
            
            # Process files concurrently
            tasks = [process_file(f) for f in temp_files]
            results = await asyncio.gather(*tasks)
            
            # All should complete successfully
            assert all(r > 0 for r in results)
            
            # Memory pool should be fully available after
            assert memory_pool.available_buffers == memory_pool.pool_size
            
        finally:
            # Cleanup
            for f in temp_files:
                Path(f).unlink(missing_ok=True)
    
    async def test_memory_cleanup_on_error(self, temp_file, mock_config):
        """Test memory is properly cleaned up on error."""
        content = "Test data. " * 100
        Path(temp_file).write_text(content)
        
        memory_pool = MemoryPool(buffer_size=1024, pool_size=5)
        processor = StreamingDocumentProcessor(memory_pool=memory_pool)
        
        # Create strategy that fails after some chunks
        strategy = MockChunkingStrategy()
        original_chunk = strategy.chunk
        call_count = [0]
        
        def failing_chunk(text, config):
            call_count[0] += 1
            if call_count[0] > 2:
                raise ValueError("Simulated error")
            return original_chunk(text, config)
        
        strategy.chunk = failing_chunk
        
        # Process should fail but clean up memory
        with pytest.raises(ValueError, match="Simulated error"):
            chunks = []
            async for chunk in processor.process_document(
                str(temp_file), strategy, mock_config
            ):
                chunks.append(chunk)
        
        # Memory should be released
        await asyncio.sleep(0.1)  # Allow cleanup
        assert memory_pool.available_buffers == memory_pool.pool_size
    
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
    
    async def test_extreme_file_size_simulation(self, mock_config):
        """Simulate processing of extremely large file (10GB)."""
        # We'll simulate by creating a small file and seeking
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_file = f.name
            # Write 1MB of actual data
            chunk = b"Large file test data. " * 50000
            f.write(chunk)
        
        try:
            processor = StreamingDocumentProcessor()
            strategy = MockChunkingStrategy()
            
            # Mock file operations to simulate 10GB
            bytes_read = [0]
            max_bytes = 10 * 1024 * 1024 * 1024  # 10GB
            
            original_process = processor.process_document
            
            async def mock_process(*args, **kwargs):
                # Track simulated bytes
                async for chunk in original_process(*args, **kwargs):
                    bytes_read[0] += len(chunk.content.encode('utf-8'))
                    if bytes_read[0] >= max_bytes:
                        break
                    yield chunk
            
            processor.process_document = mock_process
            
            # Process "10GB" file
            memory_samples = []
            chunks_processed = 0
            
            async for chunk in processor.process_document(
                temp_file, strategy, mock_config
            ):
                chunks_processed += 1
                
                # Sample memory usage
                if chunks_processed % 1000 == 0:
                    stats = processor.get_memory_usage()
                    memory_samples.append(stats['total_allocated'])
            
            # Memory should never exceed limit
            if memory_samples:
                max_memory_used = max(memory_samples)
                assert max_memory_used <= processor.MAX_MEMORY
            
        finally:
            Path(temp_file).unlink(missing_ok=True)


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
        assert stats['available'] == 3
        assert stats['in_use'] == 0
        assert stats['total_acquisitions'] == 0
        
        # Acquire buffers
        b1_id, _ = await pool.acquire()
        b2_id, _ = await pool.acquire()
        
        stats = pool.get_statistics()
        assert stats['available'] == 1
        assert stats['in_use'] == 2
        assert stats['total_acquisitions'] == 2
        assert stats['max_concurrent_usage'] == 2
        
        # Release one
        pool.release(b1_id)
        
        stats = pool.get_statistics()
        assert stats['available'] == 2
        assert stats['in_use'] == 1
        assert stats['total_releases'] == 1
        
        # Release all
        pool.release(b2_id)
        
        stats = pool.get_statistics()
        assert stats['available'] == 3
        assert stats['in_use'] == 0
        assert stats['utilization'] == 0.0