"""Integration tests covering shared chunker edge cases."""

from __future__ import annotations

from typing import Any

import pytest

from shared.chunking.unified.factory import (
    TextProcessingStrategyAdapter,
    UnifiedChunkingFactory,
)
from shared.text_processing.base_chunker import ChunkResult

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

EDGE_CASES: dict[str, str] = {
    "empty": "",
    "whitespace": "   \n\t  ",
    "unicode": "Hello 世界! 🚀🌟 — café naïve résumé.",
    "null_bytes": "Hello\x00World",
    "long_line": " ".join("longword" for _ in range(2000)),
}

STRATEGIES: tuple[str, ...] = ("character", "recursive", "markdown", "semantic", "hierarchical", "hybrid")


def _create_chunker(strategy: str) -> TextProcessingStrategyAdapter:
    """Helper that creates a chunker using the unified factory."""
    unified_strategy = UnifiedChunkingFactory.create_strategy(strategy, use_llama_index=True)
    return TextProcessingStrategyAdapter(unified_strategy)


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize(("edge_case_name", "text"), EDGE_CASES.items())
async def test_chunkers_handle_edge_cases(strategy: str, edge_case_name: str, text: str) -> None:
    """All strategies should gracefully handle canonical edge cases."""
    chunker = _create_chunker(strategy)

    chunks = await chunker.chunk_text_async(text, f"{strategy}_{edge_case_name}")

    if edge_case_name in {"empty", "whitespace"}:
        assert chunks == []
        return

    assert chunks, f"{strategy} returned no chunks for {edge_case_name}"
    for chunk in chunks:
        assert isinstance(chunk, ChunkResult)
        assert chunk.text is not None
        assert chunk.end_offset >= chunk.start_offset


async def test_recursive_chunker_preserves_metadata() -> None:
    """Recursive chunker should keep arbitrary metadata intact."""
    chunker = _create_chunker("recursive")
    metadata: dict[str, Any] = {"source": "edge_case.txt", "author": "testsuite", "custom_field": 42}

    chunks = await chunker.chunk_text_async("Sentence one. Sentence two.", "meta_doc", metadata)

    assert chunks
    for chunk in chunks:
        assert chunk.metadata["strategy"] == "recursive"
        assert chunk.metadata["source"] == metadata["source"]
        assert chunk.metadata["author"] == metadata["author"]
        assert chunk.metadata["custom_field"] == metadata["custom_field"]


async def test_character_chunker_generates_unique_ids() -> None:
    """Character chunker should generate deterministic, unique chunk ids."""
    chunker = _create_chunker("character")

    chunks = await chunker.chunk_text_async("Test text. " * 20, "doc123")

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
    for index, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"doc123_{index:04d}"
