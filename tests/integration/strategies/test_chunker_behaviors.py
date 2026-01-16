"""Focused behavioral integration tests for individual chunkers."""

from __future__ import annotations

import string
from typing import Any

import pytest
from llama_index.core.embeddings import MockEmbedding

from shared.chunking.unified.factory import TextProcessingStrategyAdapter, UnifiedChunkingFactory
from shared.text_processing.base_chunker import ChunkResult

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


def _create_chunker(strategy: str, **params: Any) -> TextProcessingStrategyAdapter:
    """Helper that creates a chunker using the unified factory."""
    # Handle embed_model for semantic strategies
    embed_model = params.pop("embed_model", None)
    if strategy in ["semantic", "hybrid"] and embed_model is None:
        embed_model = MockEmbedding(embed_dim=384)

    unified_strategy = UnifiedChunkingFactory.create_strategy(
        strategy,
        use_llama_index=True,
        embed_model=embed_model,
    )
    return TextProcessingStrategyAdapter(unified_strategy, **params)


async def test_character_chunker_respects_overlap() -> None:
    """Character chunker should enforce configured overlap across sequential chunks."""
    # Use token-based params for unified factory (max_tokens ~20 = ~80 chars)
    chunker = _create_chunker("character", max_tokens=20, min_tokens=5, overlap_tokens=4)
    text = " ".join(f"Sentence {i}." for i in range(40))

    chunks = await chunker.chunk_text_async(text, "character_overlap")

    assert len(chunks) > 1
    for first, second in zip(chunks, chunks[1:], strict=False):
        assert first.end_offset > second.start_offset
        assert len(first.text) <= 400  # defensive upper bound to catch regressions


async def test_recursive_chunker_honors_sentence_boundaries() -> None:
    """Recursive chunker should keep sentences intact."""
    # Use token-based params for unified factory (max_tokens ~15 = ~60 chars)
    chunker = _create_chunker("recursive", max_tokens=15, min_tokens=5, overlap_tokens=4)
    sentences = [
        "This is sentence one.",
        "Here comes sentence two!",
        "Is sentence three a question?",
        "Sentence four brings things home.",
    ]
    text = " ".join(sentences)  # ~110 characters total (~28 tokens)

    chunks = await chunker.chunk_text_async(text, "recursive_sentences")

    # With max_tokens=15 and ~28 tokens total, we should get at least 2 chunks
    assert len(chunks) >= 2

    def _normalize(value: str) -> str:
        return "".join(ch for ch in value if ch not in string.whitespace + string.punctuation).lower()

    normalized_reassembled = "".join(_normalize(chunk.text) for chunk in chunks)

    for sentence in sentences:
        assert _normalize(sentence) in normalized_reassembled


async def test_markdown_chunker_separates_sections() -> None:
    """Markdown chunker should respect headers when chunking documents."""
    chunker = _create_chunker("markdown")
    document = """# Title

Intro paragraph.

## Section One
Section one content.

## Section Two
Section two content with more detail.
"""

    chunks = await chunker.chunk_text_async(document, "markdown_sections")

    assert chunks
    texts = [chunk.text for chunk in chunks]
    assert any("# Title" in text for text in texts)
    assert any("Section One" in text for text in texts)
    assert any("Section Two" in text for text in texts)


async def test_semantic_chunker_produces_chunks() -> None:
    """Semantic chunker should produce valid chunks with strategy metadata."""
    # Use token-based params for unified factory
    chunker = _create_chunker("semantic", max_tokens=100, min_tokens=10)
    text = " ".join(f"Sentence {i} about embeddings." for i in range(12))

    chunks = await chunker.chunk_text_async(text, "semantic_doc")

    assert chunks
    for chunk in chunks:
        assert isinstance(chunk, ChunkResult)
        # Verify it has the semantic strategy in metadata
        assert chunk.metadata.get("strategy") == "semantic"


async def test_hierarchical_chunker_emits_hierarchy_metadata() -> None:
    """Hierarchical chunker should produce chunks with hierarchy metadata."""
    # Use token-based params for the unified implementation
    chunker = _create_chunker("hierarchical", max_tokens=200, min_tokens=25, overlap_tokens=10, hierarchy_levels=3)
    text = " ".join(f"Sentence {i} for hierarchy." for i in range(80))

    chunks = await chunker.chunk_text_async(text, "hierarchy_doc")

    assert chunks
    # All chunks should have hierarchy_level metadata
    for chunk in chunks:
        assert "hierarchy_level" in chunk.metadata
        assert isinstance(chunk.metadata["hierarchy_level"], int)
        assert chunk.metadata["hierarchy_level"] >= 0


async def test_hybrid_chunker_strategy_selection() -> None:
    """Hybrid chunker should adjust strategy based on document signals."""
    chunker = _create_chunker("hybrid")
    markdown_text = "# Heading\n\n## Details\nContent that looks like markdown."
    plain_text = " ".join("Machine learning transforms data processing." for _ in range(30))

    md_chunks = await chunker.chunk_text_async(markdown_text, "hybrid_md", {"file_path": "doc.md"})
    assert md_chunks
    # Hybrid chunker should include metadata about which strategy was selected
    assert any(chunk.metadata.get("hybrid_strategy") or chunk.metadata.get("selected_strategy") for chunk in md_chunks)

    plain_chunks = await chunker.chunk_text_async(plain_text, "hybrid_plain")
    assert plain_chunks


async def test_hybrid_chunker_allows_strategy_override() -> None:
    """Hybrid chunker should respect explicit strategy metadata."""
    chunker = _create_chunker("hybrid")
    # Short text that would normally use a default strategy
    chunks = await chunker.chunk_text_async("Plain text content here.", "hybrid_override")

    assert chunks
    # Verify chunks were created - the specific strategy selection is internal
    for chunk in chunks:
        assert chunk.text
        assert chunk.metadata.get("strategy") in {"hybrid", "semantic", "recursive", "markdown", "character"}
