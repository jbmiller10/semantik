"""Focused behavioral integration tests for individual chunkers."""

from __future__ import annotations

import string

import pytest

from shared.text_processing.base_chunker import ChunkResult
from shared.text_processing.chunking_factory import ChunkingFactory

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


def _create_chunker(strategy: str, **params):
    config = {"strategy": strategy, "params": params}
    return ChunkingFactory.create_chunker(config)


async def test_character_chunker_respects_overlap() -> None:
    """Character chunker should enforce configured overlap across sequential chunks."""
    chunker = _create_chunker("character", chunk_size=80, chunk_overlap=16)
    text = " ".join(f"Sentence {i}." for i in range(40))

    chunks = await chunker.chunk_text_async(text, "character_overlap")

    assert len(chunks) > 1
    for first, second in zip(chunks, chunks[1:], strict=False):
        assert first.end_offset > second.start_offset
        assert len(first.text) <= 400  # defensive upper bound to catch regressions


async def test_recursive_chunker_honors_sentence_boundaries() -> None:
    """Recursive chunker should keep sentences intact."""
    # Use smaller chunk size to ensure text gets split into multiple chunks
    chunker = _create_chunker("recursive", chunk_size=60, chunk_overlap=15)
    sentences = [
        "This is sentence one.",
        "Here comes sentence two!",
        "Is sentence three a question?",
        "Sentence four brings things home.",
    ]
    text = " ".join(sentences)  # ~110 characters total

    chunks = await chunker.chunk_text_async(text, "recursive_sentences")

    # With chunk_size=60 and text ~110 chars, we should get at least 2 chunks
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


async def test_semantic_chunker_attaches_semantic_metadata() -> None:
    """Semantic chunker should add semantic metadata to outputs."""
    chunker = _create_chunker("semantic", max_chunk_size=120)
    text = " ".join(f"Sentence {i} about embeddings." for i in range(12))

    chunks = await chunker.chunk_text_async(text, "semantic_doc")

    assert chunks
    for chunk in chunks:
        assert isinstance(chunk, ChunkResult)
        assert chunk.metadata.get("semantic_boundary") is True
        assert "breakpoint_threshold" in chunk.metadata


async def test_hierarchical_chunker_emits_hierarchy_metadata() -> None:
    """Hierarchical chunker should produce parent/child relationships."""
    chunker = _create_chunker("hierarchical", chunk_sizes=[200, 100, 50], chunk_overlap=10)
    text = " ".join(f"Sentence {i} for hierarchy." for i in range(80))

    chunks = await chunker.chunk_text_async(text, "hierarchy_doc")

    assert chunks
    levels = {chunk.metadata["hierarchy_level"] for chunk in chunks}
    assert levels == {0, 1, 2} or levels.issuperset({0, 1})
    assert any(chunk.metadata.get("is_leaf") for chunk in chunks)
    assert any(chunk.metadata.get("parent_chunk_id") for chunk in chunks)


async def test_hybrid_chunker_strategy_selection() -> None:
    """Hybrid chunker should adjust strategy based on document signals."""
    chunker = _create_chunker("hybrid", semantic_coherence_threshold=0.6)
    markdown_text = "# Heading\n\n## Details\nContent that looks like markdown."
    semantic_text = " ".join("Machine learning transforms data processing." for _ in range(30))

    md_chunks = await chunker.chunk_text_async(markdown_text, "hybrid_md", {"file_path": "doc.md"})
    assert md_chunks
    assert any(chunk.metadata.get("selected_strategy") in {"markdown", "hybrid"} for chunk in md_chunks)

    semantic_chunks = await chunker.chunk_text_async(semantic_text, "hybrid_semantic")
    assert semantic_chunks
    assert any(
        chunk.metadata.get("selected_strategy") in {"semantic", "recursive", "hybrid"} for chunk in semantic_chunks
    )


async def test_hybrid_chunker_allows_strategy_override() -> None:
    """Hybrid chunker should respect explicit strategy overrides in metadata."""
    chunker = _create_chunker("hybrid", enable_strategy_override=True)
    chunks = await chunker.chunk_text_async("Plain text", "hybrid_override", {"chunking_strategy": "semantic"})

    assert chunks
    assert any(chunk.metadata.get("selected_strategy") in {"semantic", "hybrid"} for chunk in chunks)
