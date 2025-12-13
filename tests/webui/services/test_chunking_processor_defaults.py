"""Tests for ChunkingProcessor default config handling."""

import pytest

from webui.services.chunking.processor import ChunkingProcessor


@pytest.mark.asyncio()
async def test_recursive_default_overlap_does_not_trigger_fallback() -> None:
    processor = ChunkingProcessor()

    # Use defaults commonly produced by the strategy registry / UI.
    config = {"chunk_size": 1000, "chunk_overlap": 200}
    content = ("hello world\n" * 1000).strip()

    chunks = await processor.process_document(content, "recursive", config, use_fallback=False)

    assert chunks
    assert {chunk["strategy"] for chunk in chunks} == {"recursive"}

