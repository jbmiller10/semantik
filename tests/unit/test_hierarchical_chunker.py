#!/usr/bin/env python3
"""Focused unit tests for HierarchicalChunker core behavior."""

from __future__ import annotations

import pytest

from packages.shared.text_processing.base_chunker import ChunkResult
from packages.shared.text_processing.strategies.hierarchical_chunker import HierarchicalChunker


def test_hierarchical_chunker_orders_chunk_sizes() -> None:
    """Chunk sizes supplied out of order should be normalized to descending order."""
    chunker = HierarchicalChunker(chunk_sizes=[256, 1024, 512])

    assert chunker.chunk_sizes == [1024, 512, 256]
    assert chunker.chunk_overlap == 20


def test_hierarchical_chunker_chunk_text_emits_hierarchy_metadata() -> None:
    """Chunking a structured document should produce parent and leaf metadata."""
    text = "\n".join(
        [
            "# Title",
            "",
            "Section one explains the basics.",
            "Section two offers more depth and examples.",
            "Section three wraps up with conclusions.",
        ]
    )

    chunker = HierarchicalChunker(chunk_sizes=[200, 100, 50], chunk_overlap=10)
    chunks = chunker.chunk_text(text, "doc-1")

    assert chunks
    assert all(isinstance(chunk, ChunkResult) for chunk in chunks)

    hierarchy_levels = {chunk.metadata.get("hierarchy_level") for chunk in chunks}
    assert hierarchy_levels.issuperset({0, 1})

    parents = [chunk for chunk in chunks if chunk.metadata.get("child_chunk_ids")]
    leaves = [chunk for chunk in chunks if chunk.metadata.get("is_leaf")]

    assert parents, "Expected parent chunks to be present"
    assert leaves, "Expected leaf chunks to be present"
    assert any(chunk.metadata.get("parent_chunk_id") for chunk in leaves)


@pytest.mark.parametrize(
    "chunk_sizes",
    [
        [],  # Empty configuration
        [512, 512, 128],  # Duplicate levels
        [0, 256, 128],  # Zero entry
    ],
)
def test_hierarchical_chunker_invalid_configuration(chunk_sizes: list[int]) -> None:
    """Invalid configurations should raise ValueError with descriptive messages."""
    with pytest.raises(ValueError, match="chunk_sizes"):
        HierarchicalChunker(chunk_sizes=chunk_sizes)


@pytest.mark.anyio()
async def test_chunk_text_async_matches_sync_output() -> None:
    """Async chunking should mirror synchronous chunking output."""
    text = " ".join(f"Sentence {i} for async validation." for i in range(60))

    chunker = HierarchicalChunker(chunk_sizes=[180, 90])

    sync_chunks = chunker.chunk_text(text, "doc-sync")
    async_chunks = await chunker.chunk_text_async(text, "doc-async")

    assert len(async_chunks) == len(sync_chunks)
    assert [chunk.text for chunk in async_chunks] == [chunk.text for chunk in sync_chunks]
