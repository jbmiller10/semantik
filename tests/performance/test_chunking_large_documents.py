"""Performance-styled checks for large document chunking."""

from __future__ import annotations

import pytest

from shared.chunking.unified.factory import (
    TextProcessingStrategyAdapter,
    UnifiedChunkingFactory,
)

pytestmark = [pytest.mark.performance, pytest.mark.anyio]


async def test_character_chunker_large_document_handles_offsets() -> None:
    """Character chunker should process ~1MB documents within size bounds."""
    unified_strategy = UnifiedChunkingFactory.create_strategy("character", use_llama_index=True)
    chunker = TextProcessingStrategyAdapter(
        unified_strategy,
        chunk_size=1300,  # mirrors production default (approx 250 tokens)
        chunk_overlap=200,
    )

    large_text = "This is a test sentence. " * 40_000  # ~1MB of text
    chunks = await chunker.chunk_text_async(large_text, "large_doc")

    assert len(chunks) > 50
    assert all(len(chunk.text) <= 5000 for chunk in chunks)

    for first, second in zip(chunks, chunks[1:], strict=False):
        assert first.end_offset <= second.end_offset
