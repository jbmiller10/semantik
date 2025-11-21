"""Integration tests for the chunking factory wiring."""

from __future__ import annotations

import pytest
from shared.text_processing.chunking_factory import ChunkingFactory
from shared.text_processing.strategies.character_chunker import CharacterChunker
from shared.text_processing.strategies.hierarchical_chunker import HierarchicalChunker
from shared.text_processing.strategies.hybrid_chunker import HybridChunker
from shared.text_processing.strategies.markdown_chunker import MarkdownChunker
from shared.text_processing.strategies.recursive_chunker import RecursiveChunker
from shared.text_processing.strategies.semantic_chunker import SemanticChunker

pytestmark = pytest.mark.integration


def test_factory_creates_each_strategy() -> None:
    """Factory should construct each registered strategy with defaults."""
    expectations = {
        "character": CharacterChunker,
        "recursive": RecursiveChunker,
        "markdown": MarkdownChunker,
        "semantic": SemanticChunker,
        "hierarchical": HierarchicalChunker,
        "hybrid": HybridChunker,
    }

    for strategy, expected_type in expectations.items():
        chunker = ChunkingFactory.create_chunker({"strategy": strategy})
        assert isinstance(chunker, expected_type)


def test_factory_rejects_unknown_strategy() -> None:
    """Factory should raise if an unknown strategy is requested."""
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        ChunkingFactory.create_chunker({"strategy": "does_not_exist"})


def test_get_available_strategies_lists_all() -> None:
    """Factory should expose the registered strategy names."""
    strategies = ChunkingFactory.get_available_strategies()
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
    chunker = ChunkingFactory.create_chunker({"strategy": strategy})

    small_estimate = chunker.estimate_chunks(small_text, {})
    large_estimate = chunker.estimate_chunks(large_text, {})

    assert small_estimate >= 1
    assert large_estimate >= small_estimate


def test_validate_config_examples() -> None:
    """Representative config validation calls across strategies."""
    char_chunker = ChunkingFactory.create_chunker({"strategy": "character"})
    assert char_chunker.validate_config({"chunk_size": 256, "chunk_overlap": 32})
    assert not char_chunker.validate_config({"chunk_size": -1})

    recursive_chunker = ChunkingFactory.create_chunker({"strategy": "recursive"})
    assert not recursive_chunker.validate_config({"chunk_overlap": 9999})

    semantic_chunker = ChunkingFactory.create_chunker({"strategy": "semantic"})
    assert not semantic_chunker.validate_config({"breakpoint_percentile_threshold": 200})
    assert not semantic_chunker.validate_config({"buffer_size": 0})

    hierarchical_chunker = ChunkingFactory.create_chunker({"strategy": "hierarchical"})
    assert hierarchical_chunker.validate_config({"chunk_sizes": [512, 256, 128]})
    assert not hierarchical_chunker.validate_config({"chunk_sizes": []})
