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


class TestRecursiveChunkingEdgeCases:
    """Tests for edge cases in recursive chunking."""

    def test_chunk_returns_empty_list_for_empty_content(self) -> None:
        """Test chunk() returns empty list for empty content."""
        strategy = RecursiveChunkingStrategy()
        config = ChunkConfig("recursive", min_tokens=10, max_tokens=100, overlap_tokens=5)

        result = strategy.chunk("", config)

        assert result == []

    def test_chunk_returns_empty_list_for_whitespace_only(self) -> None:
        """Test chunk() returns empty list for whitespace-only content."""
        strategy = RecursiveChunkingStrategy()
        config = ChunkConfig("recursive", min_tokens=10, max_tokens=100, overlap_tokens=5)

        result = strategy.chunk("   \n\n\t  ", config)

        # After cleaning, whitespace-only content should be empty
        assert len(result) == 0 or all(c.content.strip() == "" for c in result)

    def test_recursive_split_respects_separators(self) -> None:
        """Test _recursive_split() uses configured separators in order."""
        strategy = RecursiveChunkingStrategy()

        # Content with paragraph breaks
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

        splits = strategy._recursive_split(
            content,
            strategy.separators,
            max_size=30,
            min_size=5,
        )

        # Should split on paragraph breaks
        assert len(splits) >= 2
        for split in splits:
            assert len(split) <= 30

    def test_recursive_split_handles_no_separator_match(self) -> None:
        """Test _recursive_split() falls back to character split when no separator matches."""
        strategy = RecursiveChunkingStrategy()

        # Content with no separators
        content = "abcdefghijklmnopqrstuvwxyz" * 10  # 260 chars, no separators

        splits = strategy._recursive_split(
            content,
            ["|||"],  # Separator that won't match
            max_size=50,
            min_size=10,
        )

        # Should still produce splits using force_split_by_size
        assert len(splits) > 0
        for split in splits:
            assert len(split) <= 50

    def test_validate_content_rejects_empty(self) -> None:
        """Test validate_content() rejects empty content."""
        strategy = RecursiveChunkingStrategy()

        is_valid, error = strategy.validate_content("")

        assert is_valid is False
        assert error == "Content cannot be empty"

    def test_validate_content_rejects_too_large(self) -> None:
        """Test validate_content() rejects content exceeding 50MB."""
        strategy = RecursiveChunkingStrategy()

        large_content = "x" * (50_000_001)  # Just over 50MB

        is_valid, error = strategy.validate_content(large_content)

        assert is_valid is False
        assert "Content too large" in error

    def test_validate_content_accepts_valid(self) -> None:
        """Test validate_content() accepts valid content."""
        strategy = RecursiveChunkingStrategy()

        is_valid, error = strategy.validate_content("Valid content here.")

        assert is_valid is True
        assert error is None

    def test_estimate_chunks_returns_zero_for_empty(self) -> None:
        """Test estimate_chunks() returns 0 for empty content."""
        strategy = RecursiveChunkingStrategy()
        config = ChunkConfig("recursive", min_tokens=10, max_tokens=100, overlap_tokens=5)

        result = strategy.estimate_chunks(0, config)

        assert result == 0

    def test_estimate_chunks_returns_at_least_one(self) -> None:
        """Test estimate_chunks() returns at least 1 for non-empty content."""
        strategy = RecursiveChunkingStrategy()
        config = ChunkConfig("recursive", min_tokens=10, max_tokens=100, overlap_tokens=5)

        result = strategy.estimate_chunks(100, config)

        assert result >= 1


class TestRecursiveChunkingLlamaIndexFallback:
    """Tests for LlamaIndex fallback behavior."""

    def test_llama_index_not_enabled_by_default(self) -> None:
        """Test LlamaIndex is not enabled by default."""
        strategy = RecursiveChunkingStrategy()

        assert strategy._use_llama_index is False
        assert strategy._llama_available is False

    def test_chunk_with_llama_index_returns_none_when_disabled(self) -> None:
        """Test _chunk_with_llama_index() returns None when LlamaIndex is not enabled."""
        strategy = RecursiveChunkingStrategy(use_llama_index=False)
        config = ChunkConfig("recursive", min_tokens=10, max_tokens=100, overlap_tokens=5)

        result = strategy._chunk_with_llama_index("test content", config)

        assert result is None


class TestRecursiveChunkingConfigValidation:
    """Tests for configuration validation."""

    def test_rejects_overlap_greater_than_min_tokens(self) -> None:
        """Test _validate_config() rejects overlap >= min_tokens."""
        from shared.chunking.domain.exceptions import InvalidConfigurationError

        # ChunkConfig validates this at construction time
        with pytest.raises(InvalidConfigurationError, match=r"overlap_tokens must be less than min_tokens"):
            ChunkConfig("recursive", min_tokens=10, max_tokens=100, overlap_tokens=15)
