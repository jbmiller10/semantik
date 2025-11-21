"""Integration tests for streaming chunking strategies using real implementations."""

from __future__ import annotations

import re

import pytest

from shared.chunking.domain.services.chunking_strategies import CharacterChunkingStrategy
from shared.chunking.domain.services.streaming_strategies import StreamingCharacterStrategy, StreamingMarkdownStrategy
from shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from shared.chunking.infrastructure.streaming.window import StreamingWindow

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


def _windowed_bytes(payload: str, window_size: int) -> list[StreamingWindow]:
    """Split payload into streaming windows."""
    encoded = payload.encode("utf-8")
    windows: list[StreamingWindow] = []
    current = StreamingWindow(max_size=window_size * 4)

    for offset in range(0, len(encoded), window_size):
        chunk = encoded[offset : offset + window_size]
        try:
            current.append(chunk)
        except MemoryError:
            windows.append(current)
            current = StreamingWindow(max_size=window_size * 4)
            current.append(chunk)

    if current.size > 0:
        windows.append(current)
    return windows


async def _collect_streaming_chunks(strategy, text: str, config: ChunkConfig) -> list:
    """Run the streaming strategy over the simulated windows."""
    windows = _windowed_bytes(text, window_size=4096)
    collected: list = []

    for index, window in enumerate(windows):
        final_window = index == len(windows) - 1
        chunks = await strategy.process_window(window, config, final_window)
        collected.extend(chunks)

    if hasattr(strategy, "finalize"):
        finalize_result = await strategy.finalize(config)
        collected.extend(finalize_result)

    return collected


def _normalize_text(value: str) -> str:
    """Normalize whitespace for comparing streaming outputs."""
    return re.sub(r"\s+", " ", value).strip()


@pytest.fixture()
def chunk_config() -> ChunkConfig:
    """Standard chunk configuration for streaming comparisons."""
    return ChunkConfig(
        min_tokens=120,
        max_tokens=360,
        overlap_tokens=32,
        strategy_name="streaming-test",
    )


@pytest.fixture()
def markdown_text() -> str:
    """Sample markdown document with structure."""
    sections = [
        "# Title",
        "Intro paragraph with context.",
        "## Section One",
        "Details about section one.",
        "## Section Two",
        "More information with **bold** text and a list:",
        "- bullet 1",
        "- bullet 2",
        "```python",
        "def hello():",
        "    return 'world'",
        "```",
        "### Subsection",
        "Nested details.",
    ]
    return "\n\n".join(sections)


async def test_character_streaming_matches_non_streaming(chunk_config: ChunkConfig, markdown_text: str) -> None:
    """Streaming character strategy should mirror non-streaming outputs."""
    baseline = CharacterChunkingStrategy()
    expected = baseline.chunk(markdown_text, chunk_config)

    streaming = StreamingCharacterStrategy()
    observed = await _collect_streaming_chunks(streaming, markdown_text, chunk_config)

    assert observed
    assert len(observed) == len(expected)
    expected_texts = [_normalize_text(chunk.content) for chunk in expected]
    observed_texts = [_normalize_text(chunk.content) for chunk in observed]
    assert observed_texts == expected_texts


async def test_markdown_streaming_preserves_structure(chunk_config: ChunkConfig, markdown_text: str) -> None:
    """Streaming markdown strategy should emit structured chunks."""
    streaming = StreamingMarkdownStrategy()
    chunks = await _collect_streaming_chunks(streaming, markdown_text, chunk_config)

    assert chunks
    concatenated = "\n".join(chunk.content for chunk in chunks)
    assert "# Title" in concatenated
    assert "## Section One" in concatenated
    assert "### Subsection" in concatenated
