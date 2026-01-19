"""Tests for streaming recursive chunking strategy."""

from __future__ import annotations

import pytest

from shared.chunking.domain.services.streaming_strategies.recursive import StreamingRecursiveStrategy
from shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from shared.chunking.infrastructure.streaming.window import StreamingWindow

pytestmark = [pytest.mark.anyio]


async def test_streaming_recursive_does_not_truncate_long_paragraph() -> None:
    """Streaming recursive chunker must not truncate when exceeding max_tokens."""
    strategy = StreamingRecursiveStrategy()
    config = ChunkConfig(strategy_name="recursive", min_tokens=15, max_tokens=20, overlap_tokens=0)

    text = "Header\n\n" + ("word " * 200)
    window = StreamingWindow(max_size=256 * 1024)
    window.append(text.encode("utf-8"))

    chunks = await strategy.process_window(window, config, is_final=True)
    chunks += await strategy.finalize(config)

    combined_words = " ".join(chunk.content for chunk in chunks).split()
    assert "Header" in combined_words
    assert combined_words.count("word") == 200

