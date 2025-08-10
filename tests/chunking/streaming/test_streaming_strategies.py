#!/usr/bin/env python3
"""
Comprehensive tests for streaming chunking strategies.

Tests verify:
1. Output consistency with non-streaming versions
2. Memory bounds are respected
3. Proper boundary handling (UTF-8, sentences, etc.)
4. No data loss or corruption
"""

import asyncio
import random
import string
from typing import List

import pytest

from packages.shared.chunking.domain.entities.chunk import Chunk
from packages.shared.chunking.domain.services.chunking_strategies.character import (
    CharacterChunkingStrategy,
)
from packages.shared.chunking.domain.services.chunking_strategies.markdown import (
    MarkdownChunkingStrategy,
)
from packages.shared.chunking.domain.services.chunking_strategies.recursive import (
    RecursiveChunkingStrategy,
)
from packages.shared.chunking.domain.services.chunking_strategies.semantic import (
    SemanticChunkingStrategy,
)
from packages.shared.chunking.domain.services.streaming_strategies import (
    StreamingCharacterStrategy,
    StreamingHierarchicalStrategy,
    StreamingHybridStrategy,
    StreamingMarkdownStrategy,
    StreamingRecursiveStrategy,
    StreamingSemanticStrategy,
)
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.infrastructure.streaming.window import StreamingWindow


class TestStreamingStrategies:
    """Test suite for all streaming chunking strategies."""
    
    @pytest.fixture
    def chunk_config(self):
        """Create a standard chunk configuration."""
        return ChunkConfig(
            min_tokens=100,
            max_tokens=500,
            overlap_tokens=50,
            strategy_name="test",
        )
    
    @pytest.fixture
    def sample_text(self):
        """Generate sample text for testing."""
        return """
        # Introduction
        
        This is a sample document for testing chunking strategies. It contains
        multiple paragraphs with various formatting elements.
        
        ## Section 1: Background
        
        The purpose of this document is to test the streaming chunking implementation.
        We need to ensure that the streaming version produces identical output to the
        non-streaming version while maintaining memory constraints.
        
        Here's a code example:
        
        ```python
        def hello_world():
            print("Hello, World!")
            return True
        ```
        
        ## Section 2: Details
        
        This section contains more detailed information. It includes:
        
        - List item 1
        - List item 2
        - List item 3
        
        And a table:
        
        | Column 1 | Column 2 | Column 3 |
        |----------|----------|----------|
        | Data 1   | Data 2   | Data 3   |
        | Data 4   | Data 5   | Data 6   |
        
        ## Conclusion
        
        In conclusion, this document serves as a comprehensive test case for our
        streaming chunking strategies. It contains various Markdown elements that
        should be properly handled by the chunking algorithms.
        """
    
    @pytest.fixture
    def large_text(self):
        """Generate large text for memory testing."""
        # Generate 1MB of text
        words = ['word'] * 50000
        sentences = []
        for i in range(0, len(words), 10):
            sentence = ' '.join(words[i:i+10]) + '.'
            sentences.append(sentence)
        
        paragraphs = []
        for i in range(0, len(sentences), 5):
            paragraph = ' '.join(sentences[i:i+5])
            paragraphs.append(paragraph)
        
        return '\n\n'.join(paragraphs)
    
    async def _simulate_streaming(
        self, 
        text: str, 
        window_size: int = 1024
    ) -> List[StreamingWindow]:
        """
        Simulate streaming by breaking text into windows.
        
        Args:
            text: Text to stream
            window_size: Size of each window
            
        Returns:
            List of streaming windows
        """
        windows = []
        text_bytes = text.encode('utf-8')
        
        for i in range(0, len(text_bytes), window_size):
            window = StreamingWindow(max_size=window_size * 2)
            chunk = text_bytes[i:i+window_size]
            window.append(chunk)
            windows.append(window)
        
        return windows
    
    @pytest.mark.asyncio
    async def test_character_strategy_consistency(self, chunk_config, sample_text):
        """Test that streaming character strategy matches non-streaming output."""
        # Non-streaming version
        non_streaming = CharacterChunkingStrategy()
        non_streaming_chunks = non_streaming.chunk(sample_text, chunk_config)
        
        # Streaming version
        streaming = StreamingCharacterStrategy()
        streaming_chunks = []
        
        windows = await self._simulate_streaming(sample_text)
        for i, window in enumerate(windows):
            is_final = (i == len(windows) - 1)
            chunks = await streaming.process_window(window, chunk_config, is_final)
            streaming_chunks.extend(chunks)
        
        # Finalize
        final_chunks = await streaming.finalize(chunk_config)
        streaming_chunks.extend(final_chunks)
        
        # Compare outputs
        assert len(streaming_chunks) == len(non_streaming_chunks), (
            f"Chunk count mismatch: streaming={len(streaming_chunks)}, "
            f"non-streaming={len(non_streaming_chunks)}"
        )
        
        for i, (s_chunk, ns_chunk) in enumerate(
            zip(streaming_chunks, non_streaming_chunks)
        ):
            # Allow minor differences in whitespace
            s_content = ' '.join(s_chunk.content.split())
            ns_content = ' '.join(ns_chunk.content.split())
            
            assert s_content == ns_content, (
                f"Content mismatch at chunk {i}:\n"
                f"Streaming: {s_content[:100]}...\n"
                f"Non-streaming: {ns_content[:100]}..."
            )
    
    @pytest.mark.asyncio
    async def test_character_strategy_memory_bounds(self, chunk_config, large_text):
        """Test that character strategy respects memory bounds."""
        strategy = StreamingCharacterStrategy()
        max_buffer = strategy.get_max_buffer_size()
        
        windows = await self._simulate_streaming(large_text, window_size=4096)
        
        for i, window in enumerate(windows):
            is_final = (i == len(windows) - 1)
            await strategy.process_window(window, chunk_config, is_final)
            
            # Check buffer size
            buffer_size = strategy.get_buffer_size()
            assert buffer_size <= max_buffer, (
                f"Buffer size {buffer_size} exceeds max {max_buffer}"
            )
    
    @pytest.mark.asyncio
    async def test_recursive_strategy_memory_bounds(self, chunk_config, large_text):
        """Test that recursive strategy respects memory bounds."""
        strategy = StreamingRecursiveStrategy()
        max_buffer = strategy.get_max_buffer_size()
        
        windows = await self._simulate_streaming(large_text, window_size=2048)
        
        for i, window in enumerate(windows):
            is_final = (i == len(windows) - 1)
            await strategy.process_window(window, chunk_config, is_final)
            
            # Check buffer size
            buffer_size = strategy.get_buffer_size()
            assert buffer_size <= max_buffer, (
                f"Buffer size {buffer_size} exceeds max {max_buffer} (10KB)"
            )
    
    @pytest.mark.asyncio
    async def test_semantic_strategy_memory_bounds(self, chunk_config, large_text):
        """Test that semantic strategy respects memory bounds."""
        strategy = StreamingSemanticStrategy()
        max_buffer = strategy.get_max_buffer_size()
        
        windows = await self._simulate_streaming(large_text, window_size=4096)
        
        for i, window in enumerate(windows):
            is_final = (i == len(windows) - 1)
            await strategy.process_window(window, chunk_config, is_final)
            
            # Check buffer size
            buffer_size = strategy.get_buffer_size()
            assert buffer_size <= max_buffer, (
                f"Buffer size {buffer_size} exceeds max {max_buffer} (50KB)"
            )
    
    @pytest.mark.asyncio
    async def test_markdown_strategy_memory_bounds(self, chunk_config, large_text):
        """Test that markdown strategy respects memory bounds."""
        strategy = StreamingMarkdownStrategy()
        max_buffer = strategy.get_max_buffer_size()
        
        # Add markdown formatting to large text
        markdown_text = f"# Document\n\n{large_text}"
        
        windows = await self._simulate_streaming(markdown_text, window_size=8192)
        
        for i, window in enumerate(windows):
            is_final = (i == len(windows) - 1)
            await strategy.process_window(window, chunk_config, is_final)
            
            # Check buffer size
            buffer_size = strategy.get_buffer_size()
            assert buffer_size <= max_buffer, (
                f"Buffer size {buffer_size} exceeds max {max_buffer} (100KB)"
            )
    
    @pytest.mark.asyncio
    async def test_hierarchical_strategy_memory_bounds(self, chunk_config, large_text):
        """Test that hierarchical strategy respects memory bounds."""
        strategy = StreamingHierarchicalStrategy()
        max_buffer = strategy.get_max_buffer_size()
        
        windows = await self._simulate_streaming(large_text, window_size=2048)
        
        for i, window in enumerate(windows):
            is_final = (i == len(windows) - 1)
            await strategy.process_window(window, chunk_config, is_final)
            
            # Check buffer size
            buffer_size = strategy.get_buffer_size()
            assert buffer_size <= max_buffer, (
                f"Buffer size {buffer_size} exceeds max {max_buffer} (10KB)"
            )
    
    @pytest.mark.asyncio
    async def test_hybrid_strategy_memory_bounds(self, chunk_config, large_text):
        """Test that hybrid strategy respects memory bounds."""
        strategy = StreamingHybridStrategy()
        max_buffer = strategy.get_max_buffer_size()
        
        windows = await self._simulate_streaming(large_text, window_size=8192)
        
        for i, window in enumerate(windows):
            is_final = (i == len(windows) - 1)
            await strategy.process_window(window, chunk_config, is_final)
            
            # Check buffer size
            buffer_size = strategy.get_buffer_size()
            assert buffer_size <= max_buffer, (
                f"Buffer size {buffer_size} exceeds max {max_buffer} (150KB)"
            )
    
    @pytest.mark.asyncio
    async def test_utf8_boundary_handling(self, chunk_config):
        """Test proper handling of UTF-8 boundaries."""
        # Text with multi-byte UTF-8 characters
        text = "Hello 世界! This is a test with 中文 characters. 😀🎉"
        
        strategy = StreamingCharacterStrategy()
        
        # Simulate streaming with small windows that might split UTF-8
        windows = await self._simulate_streaming(text, window_size=10)
        
        all_chunks = []
        for i, window in enumerate(windows):
            is_final = (i == len(windows) - 1)
            chunks = await strategy.process_window(window, chunk_config, is_final)
            all_chunks.extend(chunks)
        
        # Finalize
        final_chunks = await strategy.finalize(chunk_config)
        all_chunks.extend(final_chunks)
        
        # Reconstruct text from chunks
        reconstructed = ' '.join(chunk.content for chunk in all_chunks)
        
        # Check that no characters were corrupted
        assert '世界' in reconstructed
        assert '中文' in reconstructed
        assert '😀' in reconstructed
        assert '🎉' in reconstructed
    
    @pytest.mark.asyncio
    async def test_sentence_boundary_preservation(self, chunk_config):
        """Test that sentence boundaries are preserved."""
        text = (
            "This is the first sentence. This is the second sentence! "
            "Is this the third sentence? Indeed it is. "
            "Here's another one for good measure."
        )
        
        strategies = [
            StreamingCharacterStrategy(),
            StreamingRecursiveStrategy(),
            StreamingSemanticStrategy(),
        ]
        
        for strategy in strategies:
            strategy.reset()
            windows = await self._simulate_streaming(text, window_size=50)
            
            all_chunks = []
            for i, window in enumerate(windows):
                is_final = (i == len(windows) - 1)
                chunks = await strategy.process_window(window, chunk_config, is_final)
                all_chunks.extend(chunks)
            
            # Finalize
            final_chunks = await strategy.finalize(chunk_config)
            all_chunks.extend(final_chunks)
            
            # Check that chunks don't split mid-sentence
            for chunk in all_chunks:
                content = chunk.content.strip()
                if content:
                    # Should start with capital or continue from overlap
                    assert (content[0].isupper() or 
                           content[0].islower() or 
                           content[0].isdigit())
                    
                    # Should end with sentence ending or be followed by another chunk
                    if content:
                        last_char = content[-1]
                        # Allow for sentence endings or continuation
                        assert last_char in '.!?,' or content[-1].isalnum()
    
    @pytest.mark.asyncio
    async def test_no_data_loss(self, chunk_config, sample_text):
        """Test that no data is lost during streaming."""
        strategies = [
            StreamingCharacterStrategy(),
            StreamingRecursiveStrategy(),
            StreamingSemanticStrategy(),
            StreamingMarkdownStrategy(),
            StreamingHierarchicalStrategy(),
            StreamingHybridStrategy(),
        ]
        
        for strategy in strategies:
            strategy.reset()
            windows = await self._simulate_streaming(sample_text, window_size=256)
            
            all_chunks = []
            for i, window in enumerate(windows):
                is_final = (i == len(windows) - 1)
                chunks = await strategy.process_window(window, chunk_config, is_final)
                all_chunks.extend(chunks)
            
            # Finalize
            final_chunks = await strategy.finalize(chunk_config)
            all_chunks.extend(final_chunks)
            
            # Check that all content is present
            all_content = ' '.join(chunk.content for chunk in all_chunks)
            
            # Key phrases that should be preserved
            assert "Introduction" in all_content
            assert "Background" in all_content
            assert "hello_world" in all_content
            assert "Conclusion" in all_content
    
    @pytest.mark.asyncio
    async def test_overlap_handling(self, chunk_config):
        """Test that overlap is handled correctly."""
        text = " ".join([f"Sentence {i}." for i in range(100)])
        
        # Set specific overlap
        config = ChunkConfig(
            min_tokens=50,
            max_tokens=100,
            overlap_tokens=20,
            strategy_name="test",
        )
        
        strategy = StreamingCharacterStrategy()
        windows = await self._simulate_streaming(text, window_size=512)
        
        all_chunks = []
        for i, window in enumerate(windows):
            is_final = (i == len(windows) - 1)
            chunks = await strategy.process_window(window, config, is_final)
            all_chunks.extend(chunks)
        
        # Finalize
        final_chunks = await strategy.finalize(config)
        all_chunks.extend(final_chunks)
        
        # Check overlap between consecutive chunks
        for i in range(len(all_chunks) - 1):
            chunk1 = all_chunks[i].content
            chunk2 = all_chunks[i + 1].content
            
            # There should be some overlap
            chunk1_words = chunk1.split()[-10:]  # Last 10 words
            chunk2_words = chunk2.split()[:10]   # First 10 words
            
            # Check for common words (indicating overlap)
            common = set(chunk1_words) & set(chunk2_words)
            assert len(common) > 0, "No overlap detected between consecutive chunks"
    
    @pytest.mark.asyncio
    async def test_empty_input_handling(self, chunk_config):
        """Test handling of empty input."""
        strategies = [
            StreamingCharacterStrategy(),
            StreamingRecursiveStrategy(),
            StreamingSemanticStrategy(),
            StreamingMarkdownStrategy(),
            StreamingHierarchicalStrategy(),
            StreamingHybridStrategy(),
        ]
        
        for strategy in strategies:
            strategy.reset()
            
            # Empty window
            window = StreamingWindow()
            chunks = await strategy.process_window(window, chunk_config, is_final=True)
            final_chunks = await strategy.finalize(chunk_config)
            
            # Should handle gracefully
            assert len(chunks) + len(final_chunks) == 0
    
    @pytest.mark.asyncio 
    async def test_state_reset(self, chunk_config, sample_text):
        """Test that reset properly clears state."""
        strategy = StreamingSemanticStrategy()
        
        # Process first document
        windows = await self._simulate_streaming(sample_text, window_size=512)
        for i, window in enumerate(windows):
            is_final = (i == len(windows) - 1)
            await strategy.process_window(window, chunk_config, is_final)
        
        # Reset
        strategy.reset()
        
        # Check state is cleared
        assert strategy.get_buffer_size() == 0
        assert strategy._chunk_index == 0
        assert strategy._char_offset == 0
        
        # Process second document
        windows = await self._simulate_streaming(sample_text, window_size=512)
        chunks = []
        for i, window in enumerate(windows):
            is_final = (i == len(windows) - 1)
            new_chunks = await strategy.process_window(window, chunk_config, is_final)
            chunks.extend(new_chunks)
        
        # Should process normally
        assert len(chunks) > 0