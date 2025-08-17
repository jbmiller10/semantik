#!/usr/bin/env python3
"""
Example test file showing how to test the unified chunking implementation directly.

This serves as a template for migrating other tests from the old implementation
to the new unified implementation.
"""

import pytest
from packages.shared.chunking.unified.factory import UnifiedChunkingFactory
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.domain.entities.chunk import Chunk


class TestUnifiedChunking:
    """Example tests for unified chunking implementation."""
    
    def test_character_strategy_basic(self):
        """Test character chunking with token-based sizing."""
        # Create config with token-based sizes
        # Old: chunk_size=50 chars → New: ~12 tokens (50/4)
        # Old: overlap=10 chars → New: ~2 tokens (10/4)
        config = ChunkConfig(
            max_tokens=12,  # ~50 characters
            min_tokens=5,   # ~20 characters
            overlap_tokens=2,  # ~10 characters
            strategy_name="character"
        )
        
        # Create chunker using factory
        strategy = UnifiedChunkingFactory.create_strategy("character")
        
        # Test text of ~250 characters (~62 tokens)
        text = "This is a test document. " * 10
        
        # Chunk the text
        chunks = strategy.chunk(text, config)
        
        # Verify we get multiple chunks
        assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
        
        # Verify chunk structure
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.content
            assert chunk.metadata.token_count > 0
            assert chunk.metadata.token_count <= config.max_tokens
    
    def test_recursive_strategy_basic(self):
        """Test recursive chunking with proper sentence boundaries."""
        config = ChunkConfig(
            max_tokens=15,  # ~60 characters
            min_tokens=8,   # ~32 characters
            overlap_tokens=3,  # ~12 characters
            strategy_name="recursive"
        )
        
        strategy = UnifiedChunkingFactory.create("recursive")
        
        text = "This is sentence one. This is sentence two. This is sentence three. " * 5
        chunks = strategy.chunk(text, config)
        
        # Should get multiple chunks
        assert len(chunks) > 1
        
        # Chunks should respect sentence boundaries (end with periods)
        for chunk in chunks:
            # Most chunks should end with sentence boundaries
            content = chunk.content.strip()
            if content and not content.endswith("..."):
                # Check for common sentence endings
                assert any(content.endswith(end) for end in ['.', '!', '?', '...']), \
                    f"Chunk doesn't end with sentence boundary: {content[-20:]}"
    
    def test_hierarchical_strategy_basic(self):
        """Test hierarchical chunking with parent-child relationships."""
        config = ChunkConfig(
            max_tokens=50,  # Parent level
            min_tokens=20,
            overlap_tokens=5,
            strategy_name="hierarchical",
            custom_attributes={
                "hierarchy_levels": 3,
                "level_sizes": [50, 25, 12]  # Token sizes for each level
            }
        )
        
        strategy = UnifiedChunkingFactory.create("hierarchical")
        
        text = """
        Machine learning is a field of artificial intelligence.
        It enables systems to learn from data.
        Neural networks are inspired by the human brain.
        They consist of interconnected nodes called neurons.
        """
        
        chunks = strategy.chunk(text, config)
        
        # Should create chunks
        assert len(chunks) > 0
        
        # Check for hierarchical metadata
        for chunk in chunks:
            assert chunk.metadata.strategy_name == "hierarchical"
            # Hierarchical chunks may have parent/child relationships in custom_attributes
            if chunk.metadata.custom_attributes:
                # May have hierarchy_level, parent_chunk_id, etc.
                pass
    
    def test_config_validation(self):
        """Test that invalid configs are properly rejected."""
        # Test overlap >= min_tokens (invalid)
        with pytest.raises(Exception):  # Should raise InvalidConfigurationError
            ChunkConfig(
                max_tokens=100,
                min_tokens=50,
                overlap_tokens=50,  # Equal to min_tokens - invalid!
                strategy_name="character"
            )
        
        # Test overlap >= max_tokens (invalid)
        with pytest.raises(Exception):
            ChunkConfig(
                max_tokens=100,
                min_tokens=50,
                overlap_tokens=100,  # Equal to max_tokens - invalid!
                strategy_name="character"
            )
    
    def test_estimate_chunks(self):
        """Test chunk estimation for planning."""
        config = ChunkConfig(
            max_tokens=100,
            min_tokens=50,
            overlap_tokens=20,
            strategy_name="character"
        )
        
        strategy = UnifiedChunkingFactory.create("character")
        
        # Text of ~1000 characters (~250 tokens)
        text_length = 1000
        estimated = strategy.estimate_chunks(text_length, config)
        
        # With 100 token chunks and 20 token overlap:
        # First chunk: 100 tokens
        # Subsequent chunks: 80 new tokens each (100 - 20 overlap)
        # 250 tokens total: 100 + (150/80) ≈ 3 chunks
        assert estimated >= 2
        assert estimated <= 5  # Reasonable bounds

    async def test_async_chunking(self):
        """Test async chunking interface."""
        config = ChunkConfig(
            max_tokens=20,
            min_tokens=10,
            overlap_tokens=5,
            strategy_name="character"
        )
        
        strategy = UnifiedChunkingFactory.create("character")
        
        text = "This is a test document with multiple sentences that need to be chunked properly."
        
        # Test async interface
        chunks = await strategy.chunk_async(text, config)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)


# Example of how to migrate an existing test
def migrate_old_test_example():
    """
    Example showing how to migrate from old to new implementation.
    """
    # OLD TEST:
    # chunker = CharacterChunker(chunk_size=50, chunk_overlap=10)
    # chunks = chunker.chunk_text(text, "doc_id")
    
    # NEW TEST:
    config = ChunkConfig(
        max_tokens=12,  # 50 chars ÷ 4 ≈ 12 tokens
        min_tokens=5,
        overlap_tokens=2,  # 10 chars ÷ 4 ≈ 2 tokens
        strategy_name="character"
    )
    strategy = UnifiedChunkingFactory.create("character")
    chunks = strategy.chunk(text, config)
    
    # Convert to old format if needed for compatibility
    # chunk_results = [
    #     ChunkResult(
    #         chunk_id=chunk.metadata.chunk_id,
    #         text=chunk.content,
    #         start_offset=chunk.metadata.start_offset,
    #         end_offset=chunk.metadata.end_offset,
    #         metadata={"strategy": "character", ...}
    #     )
    #     for chunk in chunks
    # ]