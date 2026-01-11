"""Focused unit tests for HierarchicalChunker validation logic."""

from __future__ import annotations

import pytest

from shared.text_processing.base_chunker import ChunkResult
from shared.text_processing.strategies.hierarchical_chunker import (
    MAX_HIERARCHY_DEPTH,
    MAX_TEXT_LENGTH,
    HierarchicalChunker,
)


def test_hierarchical_chunker_rejects_duplicate_sizes() -> None:
    """Chunk sizes must be strictly descending."""
    with pytest.raises(ValueError, match="Chunk sizes must be in descending order"):
        HierarchicalChunker(chunk_sizes=[512, 512, 128])


def test_hierarchical_chunker_rejects_excessive_depth() -> None:
    """The implementation guards against runaway hierarchy depth."""
    excessive = [max(128, 2048 - i * 32) for i in range(MAX_HIERARCHY_DEPTH + 1)]
    with pytest.raises(ValueError, match="Too many hierarchy levels"):
        HierarchicalChunker(chunk_sizes=excessive)


@pytest.mark.parametrize("invalid_size", [0, -5])
def test_hierarchical_chunker_validates_chunk_size_values(invalid_size: int) -> None:
    """Chunk sizes must be positive integers."""
    with pytest.raises(ValueError, match="Must be positive"):
        HierarchicalChunker(chunk_sizes=[1024, invalid_size, 128])


@pytest.mark.anyio()
async def test_hierarchical_chunker_async_emits_leaf_and_parent_chunks() -> None:
    """Async chunking should return structured results with hierarchy metadata."""
    # Use larger chunk sizes to avoid fallback to character chunking
    # chunk_sizes are in characters; max_tokens = max(chunk_sizes) // 4
    chunker = HierarchicalChunker(chunk_sizes=[1024, 512, 256])
    text = " ".join(f"Sentence {i}. " for i in range(80))

    chunks = await chunker.chunk_text_async(text, "hierarchical-doc")

    assert chunks
    assert all(isinstance(chunk, ChunkResult) for chunk in chunks)
    # Verify hierarchy metadata is set (not None)
    hierarchy_levels = {chunk.metadata.get("hierarchy_level") for chunk in chunks}
    assert None not in hierarchy_levels, "All chunks should have hierarchy_level set"
    assert all(isinstance(level, int) for level in hierarchy_levels)
    # Verify is_leaf is set for all chunks
    assert all("is_leaf" in chunk.metadata for chunk in chunks)
    assert any(chunk.metadata.get("is_leaf") for chunk in chunks)


def test_hierarchical_chunker_rejects_excessive_text_length() -> None:
    """Large payloads should be rejected early to guard memory usage."""
    chunker = HierarchicalChunker()
    text = "a" * (MAX_TEXT_LENGTH + 1)
    with pytest.raises(ValueError, match="Text too large"):
        chunker.chunk_text(text, "oversized-document")
