"""Focused unit tests for hierarchical chunker validation logic using the unified factory."""

from __future__ import annotations

import pytest

from shared.chunking.unified.factory import (
    TextProcessingStrategyAdapter,
    UnifiedChunkingFactory,
)
from shared.text_processing.base_chunker import ChunkResult


def _create_hierarchical_chunker(**params: int) -> TextProcessingStrategyAdapter:
    """Create a hierarchical chunker using the unified factory."""
    unified_strategy = UnifiedChunkingFactory.create_strategy(
        "hierarchical",
        use_llama_index=True,
        **params,
    )
    return TextProcessingStrategyAdapter(unified_strategy, **params)


def test_hierarchical_chunker_validates_invalid_token_config() -> None:
    """Config validation should reject invalid token parameters."""
    chunker = _create_hierarchical_chunker()

    # Invalid: overlap too large
    assert not chunker.validate_config({"max_tokens": 100, "min_tokens": 50, "overlap_tokens": 60})

    # Invalid: negative values
    assert not chunker.validate_config({"max_tokens": -100, "min_tokens": 50})

    # Invalid: min > max
    assert not chunker.validate_config({"max_tokens": 50, "min_tokens": 100})


def test_hierarchical_chunker_accepts_valid_config() -> None:
    """Config validation should accept valid configurations."""
    chunker = _create_hierarchical_chunker()

    # Valid config
    assert chunker.validate_config({"max_tokens": 512, "min_tokens": 100, "overlap_tokens": 25})


@pytest.mark.anyio()
async def test_hierarchical_chunker_async_emits_hierarchy_metadata() -> None:
    """Async chunking should return structured results with hierarchy metadata."""
    chunker = _create_hierarchical_chunker(max_tokens=1024, min_tokens=25, overlap_tokens=10)
    text = " ".join(f"Sentence {i}. " for i in range(50))

    chunks = await chunker.chunk_text_async(text, "hierarchical-doc")

    assert chunks
    assert all(isinstance(chunk, ChunkResult) for chunk in chunks)
    # Verify hierarchy metadata is set
    hierarchy_levels = {chunk.metadata.get("hierarchy_level") for chunk in chunks}
    assert None not in hierarchy_levels, "All chunks should have hierarchy_level set"
    assert all(isinstance(level, int) for level in hierarchy_levels)


def test_hierarchical_chunker_handles_empty_text() -> None:
    """Empty text should return empty list."""
    chunker = _create_hierarchical_chunker()
    chunks = chunker.chunk_text("", "empty-doc")
    assert chunks == []


def test_hierarchical_chunker_handles_whitespace_only() -> None:
    """Whitespace-only text should return empty list."""
    chunker = _create_hierarchical_chunker()
    chunks = chunker.chunk_text("   \n\t  ", "whitespace-doc")
    assert chunks == []
