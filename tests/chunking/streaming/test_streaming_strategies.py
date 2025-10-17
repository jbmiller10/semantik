#!/usr/bin/env python3
"""Smoke tests ensuring streaming chunking strategies align with batch implementations."""

from __future__ import annotations

import pytest

from packages.shared.chunking.domain.services.chunking_strategies import CharacterChunkingStrategy
from packages.shared.chunking.domain.services.streaming_strategies import (
    StreamingCharacterStrategy,
    StreamingMarkdownStrategy,
)
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.infrastructure.streaming.window import StreamingWindow


async def _simulate_streaming(text: str, window_size: int = 2048) -> list[StreamingWindow]:
    """Utility that splits text into streaming windows for integration-style testing."""
    encoded = text.encode("utf-8")
    windows: list[StreamingWindow] = []
    buffer = StreamingWindow(max_size=window_size * 2)

    for idx in range(0, len(encoded), window_size):
        chunk = encoded[idx : idx + window_size]
        try:
            buffer.append(chunk)
        except MemoryError:
            windows.append(buffer)
            buffer = StreamingWindow(max_size=window_size * 2)
            buffer.append(chunk)

    if buffer.size > 0:
        windows.append(buffer)

    return windows


@pytest.mark.anyio()
async def test_streaming_character_matches_batch_output() -> None:
    """Streaming character strategy should yield the same chunks as the batch strategy."""
    text = " ".join(f"Sentence {i} for streaming parity checks." for i in range(120))
    config = ChunkConfig(min_tokens=80, max_tokens=160, overlap_tokens=32, strategy_name="character")

    batch_strategy = CharacterChunkingStrategy()
    batch_chunks = batch_strategy.chunk(text, config)

    streaming_strategy = StreamingCharacterStrategy()
    streaming_chunks = []
    windows = await _simulate_streaming(text, window_size=1024)
    for index, window in enumerate(windows):
        streaming_chunks.extend(
            await streaming_strategy.process_window(window, config, is_final=index == len(windows) - 1)
        )

    assert [chunk.text for chunk in streaming_chunks] == [chunk.text for chunk in batch_chunks]


@pytest.mark.anyio()
async def test_streaming_markdown_handles_incremental_updates() -> None:
    """Streaming markdown strategy should accumulate state across windows."""
    text = """# Title

## Section One
Content in section one.

## Section Two
Further content with lists.

- bullet one
- bullet two

```python
print("Streaming!")
```
"""

    config = ChunkConfig(min_tokens=32, max_tokens=128, overlap_tokens=16, strategy_name="markdown")
    streaming_strategy = StreamingMarkdownStrategy()
    windows = await _simulate_streaming(text, window_size=256)

    collected = []
    for index, window in enumerate(windows):
        collected.extend(await streaming_strategy.process_window(window, config, is_final=index == len(windows) - 1))

    assert collected
    assert any("# Title" in chunk.text for chunk in collected)
    assert any("Section Two" in chunk.text for chunk in collected)
