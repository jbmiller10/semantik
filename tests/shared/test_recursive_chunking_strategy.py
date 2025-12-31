"""Tests for the recursive chunking strategy helpers."""

from unittest.mock import MagicMock, patch

import pytest

from shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from shared.chunking.unified.recursive_strategy import RecursiveChunkingStrategy


@pytest.mark.asyncio()
async def test_recursive_chunking_async_delegates_to_sync() -> None:
    strategy = RecursiveChunkingStrategy()
    config = ChunkConfig("recursive", min_tokens=10, max_tokens=50, overlap_tokens=5)

    chunks = [MagicMock()]  # Sentinel return value

    with patch.object(strategy, "chunk", return_value=chunks) as mock_chunk:
        result = await strategy.chunk_async("sample text", config)

    mock_chunk.assert_called_once_with("sample text", config, None)
    assert result == chunks


def test_recursive_chunking_offsets_are_valid() -> None:
    strategy = RecursiveChunkingStrategy()
    config = ChunkConfig("recursive", min_tokens=5, max_tokens=20, overlap_tokens=2)
    content = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu."

    chunks = strategy.chunk(content, config)

    assert chunks
    for chunk in chunks:
        assert 0 <= chunk.metadata.start_offset < chunk.metadata.end_offset <= len(content)


def test_recursive_chunking_rejects_equal_min_max_tokens() -> None:
    strategy = RecursiveChunkingStrategy()
    config = ChunkConfig("recursive", min_tokens=50, max_tokens=50, overlap_tokens=5)

    with pytest.raises(ValueError, match=r"min_tokens .* must be less than max_tokens"):
        strategy.chunk("Short content to trigger validation.", config)
