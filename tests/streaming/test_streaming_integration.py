#!/usr/bin/env python3
"""
Integration test for the complete streaming pipeline.

This module tests the end-to-end functionality of the streaming document
processor with real files and strategies.
"""

import asyncio
import os
import tempfile
import tracemalloc
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

from packages.shared.chunking.domain.services.chunking_strategies.base import ChunkingStrategy
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.infrastructure.streaming.checkpoint import CheckpointManager
from packages.shared.chunking.infrastructure.streaming.memory_pool import MemoryPool
from packages.shared.chunking.infrastructure.streaming.processor import StreamingDocumentProcessor
from packages.shared.chunking.infrastructure.streaming.window import StreamingWindow


class SimpleChunkingStrategy(ChunkingStrategy):
    """Simple chunking strategy for integration testing."""
    
    def __init__(self):
        super().__init__("simple_strategy")  # Pass name to parent constructor
    
    def chunk(self, text: str, config: ChunkConfig) -> List:
        """Create chunks by splitting on double newlines."""
        chunks = []
        paragraphs = text.split('\n\n')
        
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                chunk = MagicMock()
                chunk.content = paragraph.strip()
                chunk.metadata = MagicMock()
                chunk.metadata.token_count = len(paragraph.split())
                chunks.append(chunk)
        
        return chunks
    
    def validate_content(self, content: str) -> tuple[bool, str | None]:
        """Validate content for testing."""
        if not content:
            return False, "Content is empty"
        return True, None
    
    def estimate_chunks(self, content_length: int, config: ChunkConfig) -> int:
        """Estimate number of chunks for testing."""
        # Rough estimate: one chunk per 200 characters (paragraphs are larger)
        return max(1, content_length // 200)


class TestStreamingIntegration:
    """Integration tests for the complete streaming pipeline."""
    
    @pytest.fixture
    def mock_config(self) -> ChunkConfig:
        """Create mock chunk configuration."""
        config = MagicMock(spec=ChunkConfig)
        config.min_tokens = 5
        config.max_tokens = 500
        config.estimate_chunks = lambda tokens: max(1, tokens // 100)
        return config
    
    async def test_complete_pipeline_small_file(self, mock_config):
        """Test complete pipeline with a small text file."""
        # Create test file with known content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            content = """This is the first paragraph of our test document.
It contains multiple sentences for testing.

This is the second paragraph with different content.
We want to ensure proper chunking.

And finally, a third paragraph to complete our test.
The streaming processor should handle this efficiently."""
            f.write(content)
            temp_file = f.name
        
        try:
            # Initialize components
            checkpoint_manager = CheckpointManager()
            memory_pool = MemoryPool(buffer_size=1024, pool_size=5)
            processor = StreamingDocumentProcessor(
                checkpoint_manager=checkpoint_manager,
                memory_pool=memory_pool
            )
            strategy = SimpleChunkingStrategy()
            
            # Process document
            chunks = []
            async for chunk in processor.process_document(
                temp_file, strategy, mock_config
            ):
                chunks.append(chunk)
            
            # Verify results
            assert len(chunks) == 3  # Three paragraphs
            assert "first paragraph" in chunks[0].content
            assert "second paragraph" in chunks[1].content
            assert "third paragraph" in chunks[2].content
            
            # Verify memory was properly managed
            stats = memory_pool.get_statistics()
            assert stats['available'] == stats['pool_size']
            assert stats['in_use'] == 0
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    async def test_pipeline_with_utf8_content(self, mock_config):
        """Test pipeline with UTF-8 content including emojis and CJK."""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False) as f:
            content = """English paragraph with simple text.

Français: Café, résumé, naïveté - testing accented characters.

中文段落：这是一个测试文档，包含中文字符。

日本語: これはテストドキュメントです。

Emoji test: 🎉 🎊 🎈 Testing emoji handling! 😊

Mixed: English café 中文 🎉 all together."""
            f.write(content)
            temp_file = f.name
        
        try:
            processor = StreamingDocumentProcessor()
            strategy = SimpleChunkingStrategy()
            
            chunks = []
            async for chunk in processor.process_document(
                temp_file, strategy, mock_config
            ):
                chunks.append(chunk)
            
            # Verify all content types were processed
            assert len(chunks) == 6
            
            # Check specific content preservation
            assert "café" in chunks[1].content or "café" in chunks[5].content
            assert "中文" in chunks[2].content or "中文" in chunks[5].content
            assert "🎉" in chunks[4].content or "🎉" in chunks[5].content
            
            # No corruption should occur
            for chunk in chunks:
                # Should be valid UTF-8
                chunk.content.encode('utf-8')
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    async def test_pipeline_memory_constraints(self, mock_config):
        """Test that pipeline respects memory constraints with large file."""
        # Create a 10MB file
        file_size = 10 * 1024 * 1024
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as f:
            paragraph = b"This is a test paragraph that will be repeated many times. " * 20 + b"\n\n"
            while f.tell() < file_size:
                f.write(paragraph)
            temp_file = f.name
        
        try:
            # Use small memory pool to test constraints
            memory_pool = MemoryPool(buffer_size=64 * 1024, pool_size=3)
            processor = StreamingDocumentProcessor(memory_pool=memory_pool)
            strategy = SimpleChunkingStrategy()
            
            # Track memory usage
            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            
            chunks = []
            max_memory_delta = 0
            
            async for chunk in processor.process_document(
                temp_file, strategy, mock_config
            ):
                chunks.append(chunk)
                
                # Check memory periodically
                if len(chunks) % 10 == 0:
                    current_memory = tracemalloc.get_traced_memory()[0]
                    memory_delta = current_memory - start_memory
                    max_memory_delta = max(max_memory_delta, memory_delta)
            
            tracemalloc.stop()
            
            # Verify processing completed
            assert len(chunks) > 0
            
            # Memory should stay bounded
            assert max_memory_delta < processor.MAX_MEMORY
            
            # Pool should be clean after processing
            stats = memory_pool.get_statistics()
            assert stats['in_use'] == 0
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    async def test_pipeline_concurrent_processing(self, mock_config):
        """Test concurrent document processing."""
        # Create multiple test files
        temp_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{i}.txt', delete=False) as f:
                for j in range(5):
                    f.write(f"File {i}, Paragraph {j}: Test content.\n\n")
                temp_files.append(f.name)
        
        try:
            # Shared memory pool for all processors
            memory_pool = MemoryPool(buffer_size=1024, pool_size=10)
            
            async def process_file(file_path: str, file_id: int):
                processor = StreamingDocumentProcessor(memory_pool=memory_pool)
                strategy = SimpleChunkingStrategy()
                
                chunks = []
                async for chunk in processor.process_document(
                    file_path, strategy, mock_config
                ):
                    chunks.append(chunk)
                
                return file_id, len(chunks)
            
            # Process all files concurrently
            tasks = [process_file(f, i) for i, f in enumerate(temp_files)]
            results = await asyncio.gather(*tasks)
            
            # All files should be processed
            assert len(results) == 3
            for file_id, chunk_count in results:
                assert chunk_count == 5  # Each file has 5 paragraphs
            
            # Memory pool should be clean
            stats = memory_pool.get_statistics()
            assert stats['in_use'] == 0
            assert stats['available'] == stats['pool_size']
            
        finally:
            for f in temp_files:
                Path(f).unlink(missing_ok=True)
    
    async def test_pipeline_error_handling(self, mock_config):
        """Test error handling in the pipeline."""
        # Test with non-existent file
        processor = StreamingDocumentProcessor()
        strategy = SimpleChunkingStrategy()
        
        with pytest.raises(FileNotFoundError):
            async for chunk in processor.process_document(
                "/non/existent/file.txt", strategy, mock_config
            ):
                pass
        
        # Test with directory instead of file
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(Exception):  # Will raise when trying to read directory
                async for chunk in processor.process_document(
                    tmpdir, strategy, mock_config
                ):
                    pass
    
    def test_streaming_window_operations(self):
        """Test streaming window operations."""
        window = StreamingWindow(max_size=1024)
        
        # Test append and decode
        text = "Hello, world! This is a test."
        data = text.encode('utf-8')
        window.append(data)
        
        decoded = window.decode_safe()
        assert decoded == text
        
        # Test sliding
        more_text = " Additional content."
        window.append(more_text.encode('utf-8'))
        
        # Slide window
        removed = window.slide(amount=10)
        assert len(removed) == 10
        assert window.size < len(data) + len(more_text.encode('utf-8'))
        
        # Test capacity
        remaining = window.remaining_capacity()
        assert remaining > 0
        assert remaining == window.max_size - window.size
    
    def test_memory_pool_operations(self):
        """Test memory pool operations."""
        pool = MemoryPool(buffer_size=512, pool_size=3)
        
        # Test statistics
        stats = pool.get_statistics()
        assert stats['pool_size'] == 3
        assert stats['buffer_size'] == 512
        assert stats['available'] == 3
        assert stats['in_use'] == 0
        
        # Test acquire and release using synchronous method
        buffer_id, buffer = pool.acquire_sync()
        assert len(buffer) == 512
        assert pool.used_buffers == 1
        
        pool.release(buffer_id)
        assert pool.available_buffers == 3
        
        # Test clearing
        pool.clear()
        stats = pool.get_statistics()
        assert stats['total_acquisitions'] == 0  # Reset after clear


if __name__ == "__main__":
    # Run a simple test
    async def main():
        test = TestStreamingIntegration()
        config = test.mock_config()
        await test.test_complete_pipeline_small_file(config)
        print("✓ Basic integration test passed")
        
        await test.test_pipeline_with_utf8_content(config)
        print("✓ UTF-8 content test passed")
        
        await test.test_pipeline_memory_constraints(config)
        print("✓ Memory constraints test passed")
        
        print("\nAll integration tests passed successfully!")
    
    asyncio.run(main())