"""Integration tests for the unified chunking factory."""

from __future__ import annotations

import pytest

from shared.chunking.unified.factory import (
    TextProcessingStrategyAdapter,
    UnifiedChunkingFactory,
)

pytestmark = pytest.mark.integration


def test_factory_creates_each_strategy() -> None:
    """Factory should construct each registered strategy with defaults."""
    strategies = ["character", "recursive", "markdown", "semantic", "hierarchical", "hybrid"]

    for strategy_name in strategies:
        strategy = UnifiedChunkingFactory.create_strategy(strategy_name)
        assert strategy is not None
        assert strategy.name == strategy_name


def test_factory_rejects_unknown_strategy() -> None:
    """Factory should raise if an unknown strategy is requested."""
    with pytest.raises(ValueError, match="not a valid ChunkingStrategyType"):
        UnifiedChunkingFactory.create_strategy("does_not_exist")


def test_get_available_strategies_lists_all() -> None:
    """Factory should expose the registered strategy names."""
    strategies = UnifiedChunkingFactory.get_available_strategies()
    expected = {"character", "recursive", "markdown", "semantic", "hierarchical", "hybrid"}
    assert expected.issubset(set(strategies))


@pytest.mark.parametrize(
    ("strategy", "small_text", "large_text"),
    [
        ("character", 200, 10_000),
        ("recursive", 200, 10_000),
        ("markdown", 150, 5_000),
        ("semantic", 200, 10_000),
        ("hierarchical", 200, 10_000),
    ],
)
def test_estimate_chunks_scales_with_length(strategy: str, small_text: int, large_text: int) -> None:
    """Chunk estimations should scale roughly with input length."""
    unified_strategy = UnifiedChunkingFactory.create_strategy(strategy)
    chunker = TextProcessingStrategyAdapter(unified_strategy)

    small_estimate = chunker.estimate_chunks(small_text, {})
    large_estimate = chunker.estimate_chunks(large_text, {})

    assert small_estimate >= 1
    assert large_estimate >= small_estimate


def test_validate_config_examples() -> None:
    """Representative config validation calls across strategies."""
    # Character chunker validation
    char_strategy = UnifiedChunkingFactory.create_strategy("character")
    char_chunker = TextProcessingStrategyAdapter(char_strategy)
    assert char_chunker.validate_config({"chunk_size": 256, "chunk_overlap": 32})
    assert not char_chunker.validate_config({"chunk_size": -1})

    # Recursive chunker validation
    recursive_strategy = UnifiedChunkingFactory.create_strategy("recursive")
    recursive_chunker = TextProcessingStrategyAdapter(recursive_strategy)
    assert not recursive_chunker.validate_config({"chunk_overlap": 9999})

    # Semantic chunker validation
    semantic_strategy = UnifiedChunkingFactory.create_strategy("semantic")
    semantic_chunker = TextProcessingStrategyAdapter(semantic_strategy)
    assert not semantic_chunker.validate_config({"breakpoint_percentile_threshold": 200})
    assert not semantic_chunker.validate_config({"buffer_size": 0})

    # Hierarchical chunker validation - test with token-based params
    hierarchical_strategy = UnifiedChunkingFactory.create_strategy("hierarchical")
    hierarchical_chunker = TextProcessingStrategyAdapter(hierarchical_strategy)
    # Valid config with reasonable token sizes
    assert hierarchical_chunker.validate_config({"max_tokens": 512, "min_tokens": 100, "overlap_tokens": 25})
    # Invalid: overlap too large
    assert not hierarchical_chunker.validate_config({"max_tokens": 512, "min_tokens": 100, "overlap_tokens": 150})
