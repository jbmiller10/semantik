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
