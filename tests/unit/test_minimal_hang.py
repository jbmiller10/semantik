#!/usr/bin/env python3
"""Minimal test to ensure hierarchical chunker handles edge case inputs."""

from shared.chunking.unified.factory import TextProcessingStrategyAdapter, UnifiedChunkingFactory


def _create_hierarchical_chunker() -> TextProcessingStrategyAdapter:
    """Create a hierarchical chunker using the unified factory."""
    unified_strategy = UnifiedChunkingFactory.create_strategy(
        "hierarchical",
        use_llama_index=True,
    )
    return TextProcessingStrategyAdapter(unified_strategy)


def test_hierarchical_chunker_edge_cases() -> None:
    """Test the hierarchical chunker handles edge case inputs without hanging."""
    chunker = _create_hierarchical_chunker()

    # Test null bytes
    text_with_null = "Normal text\x00with null bytes"
    chunks = chunker.chunk_text(text_with_null, "test_doc")
    assert len(chunks) > 0

    # Test very long lines
    long_line = "a" * 10000
    chunks = chunker.chunk_text(long_line, "test_doc")
    assert len(chunks) > 0

    # Test deeply nested structure
    nested_text = "Start\n" + "\n".join(["  " * i + f"Level {i}" for i in range(50)])
    chunks = chunker.chunk_text(nested_text, "test_doc")
    assert len(chunks) > 0
