"""Focused unit tests for HierarchicalChunker validation logic."""

from __future__ import annotations

import pytest

from packages.shared.text_processing.base_chunker import ChunkResult
from packages.shared.text_processing.strategies.hierarchical_chunker import (
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
    with pytest.raises(ValueError):
        HierarchicalChunker(chunk_sizes=[1024, invalid_size, 128])


@pytest.mark.anyio()
async def test_hierarchical_chunker_async_emits_leaf_and_parent_chunks() -> None:
    """Async chunking should return structured results with hierarchy metadata."""
    chunker = HierarchicalChunker(chunk_sizes=[256, 128, 64])
    text = " ".join(f"Sentence {i}. " for i in range(80))

    chunks = await chunker.chunk_text_async(text, "hierarchical-doc")

    assert chunks
    assert all(isinstance(chunk, ChunkResult) for chunk in chunks)
    hierarchy_levels = {chunk.metadata.get("hierarchy_level") for chunk in chunks}
    assert hierarchy_levels.issuperset({0, 1})
    assert any(chunk.metadata.get("is_leaf") for chunk in chunks)


def test_hierarchical_chunker_rejects_excessive_text_length() -> None:
    """Large payloads should be rejected early to guard memory usage."""
    chunker = HierarchicalChunker()
    text = "a" * (MAX_TEXT_LENGTH + 1)
    with pytest.raises(ValueError, match="Text too large"):
        chunker.chunk_text(text, "oversized-document")
