#!/usr/bin/env python3
"""Slim unit tests validating HybridChunker strategy selection."""

from __future__ import annotations

import pytest

from packages.shared.text_processing.base_chunker import ChunkResult
from packages.shared.text_processing.strategies.hybrid_chunker import HybridChunker


def test_hybrid_chunker_identifies_markdown_documents() -> None:
    """Hybrid chunker should choose markdown-aware strategy for markdown-heavy documents."""
    markdown_text = """# Title

## Section

- Bullet one
- Bullet two

```python
print("hello world")
```"""

    chunker = HybridChunker()
    chunks = chunker.chunk_text(markdown_text, "doc-md")

    assert chunks
    assert any(isinstance(chunk, ChunkResult) for chunk in chunks)
    selected_strategies = {chunk.metadata.get("selected_strategy") for chunk in chunks}
    assert "markdown" in selected_strategies or "hybrid" in selected_strategies


def test_hybrid_chunker_respects_override_metadata() -> None:
    """Providing chunking_strategy metadata should override automatic selection."""
    chunker = HybridChunker(enable_strategy_override=True)

    chunks = chunker.chunk_text("Plain text content", "doc-override", {"chunking_strategy": "semantic"})
    selected_strategies = {chunk.metadata.get("selected_strategy") for chunk in chunks}

    assert "semantic" in selected_strategies or "hybrid" in selected_strategies


@pytest.mark.anyio()
async def test_hybrid_chunker_async_matches_sync_output() -> None:
    """Async chunking should produce the same results as synchronous chunking."""
    text = " ".join(f"Sentence {i} about data quality." for i in range(40))

    chunker = HybridChunker()

    sync_chunks = chunker.chunk_text(text, "doc-sync")
    async_chunks = await chunker.chunk_text_async(text, "doc-async")

    assert len(async_chunks) == len(sync_chunks)
    assert [chunk.text for chunk in async_chunks] == [chunk.text for chunk in sync_chunks]
