#!/usr/bin/env python3

"""Unit tests for hierarchical chunker security and robustness using the unified factory.

This module tests security-related features including:
1. Configuration validation to prevent invalid settings
2. Input handling to prevent issues with malicious inputs
3. Memory-efficient processing for large documents
"""

import pytest

from shared.chunking.unified.factory import (
    TextProcessingStrategyAdapter,
    UnifiedChunkingFactory,
)


def _create_hierarchical_chunker(**params: int) -> TextProcessingStrategyAdapter:
    """Create a hierarchical chunker using the unified factory."""
    unified_strategy = UnifiedChunkingFactory.create_strategy(
        "hierarchical",
        use_llama_index=True,
        **params,
    )
    return TextProcessingStrategyAdapter(unified_strategy, **params)


class TestHierarchicalChunkerSecurity:
    """Test suite for hierarchical chunker security features using unified factory."""

    def test_config_validation_rejects_invalid_token_sizes(self) -> None:
        """Test that invalid token configuration is rejected."""
        chunker = _create_hierarchical_chunker()

        # Invalid: negative max_tokens
        assert not chunker.validate_config({"max_tokens": -100})

        # Invalid: overlap larger than reasonable for tokens
        assert not chunker.validate_config({"max_tokens": 100, "min_tokens": 50, "overlap_tokens": 60})

    def test_config_validation_accepts_valid_config(self) -> None:
        """Test that valid configurations are accepted."""
        chunker = _create_hierarchical_chunker()

        # Valid config
        assert chunker.validate_config({"max_tokens": 512, "min_tokens": 100, "overlap_tokens": 25})

    def test_malicious_input_handling(self) -> None:
        """Test handling of potentially malicious inputs."""
        chunker = _create_hierarchical_chunker()

        # Test null bytes
        text_with_null = "Normal text\x00with null bytes"
        chunks = chunker.chunk_text(text_with_null, "test_doc")
        assert len(chunks) > 0

        # Test very long lines
        long_line = "a" * 10000  # Long but within limits
        chunks = chunker.chunk_text(long_line, "test_doc")
        assert len(chunks) > 0

        # Test deeply nested structure simulation
        nested_text = "Start\n" + "\n".join(["  " * i + f"Level {i}" for i in range(50)])
        chunks = chunker.chunk_text(nested_text, "test_doc")
        assert len(chunks) > 0

    def test_memory_efficient_processing(self) -> None:
        """Test that chunker processes large texts memory-efficiently."""
        chunker = _create_hierarchical_chunker(max_tokens=200, min_tokens=50, overlap_tokens=10)

        # Create a large but valid document
        sections = []
        for i in range(100):
            sections.append(f"Section {i}: " + "content " * 100)
        large_doc = "\n\n".join(sections)

        # Should process without memory issues
        chunks = chunker.chunk_text(large_doc, "test_doc")
        assert len(chunks) > 0

    def test_whitespace_only_input(self) -> None:
        """Test handling of whitespace-only input."""
        chunker = _create_hierarchical_chunker()

        # Various whitespace inputs
        whitespace_inputs = [
            "   ",
            "\n\n\n",
            "\t\t\t",
            "   \n   \t   ",
            " " * 1000,
        ]

        for ws_input in whitespace_inputs:
            chunks = chunker.chunk_text(ws_input, "test_doc")
            assert len(chunks) == 0  # Should return empty list

    def test_empty_text_handling(self) -> None:
        """Test handling of empty text."""
        chunker = _create_hierarchical_chunker()
        chunks = chunker.chunk_text("", "test_doc")
        assert chunks == []

    @pytest.mark.asyncio()
    async def test_async_large_document_processing(self) -> None:
        """Test that large documents are processed correctly in async mode."""
        chunker = _create_hierarchical_chunker(max_tokens=200, min_tokens=50)

        # Create text with realistic content
        sections = []
        for i in range(110):
            sections.append(f"Section {i}: " + "content " * 100)
        large_text = "\n\n".join(sections)

        # Should process without error
        chunks = await chunker.chunk_text_async(large_text, "test_doc")
        assert len(chunks) > 0

    def test_chunk_metadata_integrity(self) -> None:
        """Test that chunks have proper metadata."""
        chunker = _create_hierarchical_chunker()

        text = " ".join(f"Sentence {i}. " for i in range(50))
        chunks = chunker.chunk_text(text, "test_doc")

        assert len(chunks) > 0
        for chunk in chunks:
            # All chunks should have basic metadata
            assert "hierarchy_level" in chunk.metadata
            assert isinstance(chunk.metadata["hierarchy_level"], int)
            assert chunk.metadata["hierarchy_level"] >= 0
