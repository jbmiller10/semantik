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


class TestRecursiveChunkingLimits:
    """Tests for recursion and iteration limits."""

    def test_max_recursion_depth_limit(self) -> None:
        """Test that max recursion depth is enforced."""
        strategy = RecursiveChunkingStrategy()

        # Simulate deep recursion by calling with high depth
        result = strategy._recursive_split(
            "a" * 100,
            strategy.separators,
            max_size=10,
            min_size=1,
            depth=strategy.MAX_RECURSION_DEPTH + 1,
        )

        # Should return text as-is when depth exceeded
        assert result == ["a" * 100]

    def test_max_iterations_limit(self) -> None:
        """Test that max iterations is enforced."""
        strategy = RecursiveChunkingStrategy()

        # Pre-set iteration count to near the limit
        iteration_count = [strategy.MAX_SPLIT_ITERATIONS + 1]

        result = strategy._recursive_split(
            "a" * 100,
            strategy.separators,
            max_size=10,
            min_size=1,
            _iteration_count=iteration_count,
        )

        # Should return text as-is when iterations exceeded
        assert result == ["a" * 100]


class TestRecursiveChunkingForceSplit:
    """Tests for _force_split_by_size method."""

    def test_force_split_creates_chunks_of_max_size(self) -> None:
        """Test _force_split_by_size creates chunks of max_size."""
        strategy = RecursiveChunkingStrategy()

        content = "a" * 100
        result = strategy._force_split_by_size(content, max_size=25, min_size=10)

        assert len(result) == 4
        for chunk in result:
            assert len(chunk) == 25

    def test_force_split_merges_small_final_chunk(self) -> None:
        """Test _force_split_by_size merges chunks smaller than min_size."""
        strategy = RecursiveChunkingStrategy()

        content = "a" * 35  # Will split to 25 + 10
        result = strategy._force_split_by_size(content, max_size=25, min_size=15)

        # The 10-char remainder is below min_size, so should be merged with previous
        assert len(result) == 1
        assert len(result[0]) == 35

    def test_force_split_empty_content(self) -> None:
        """Test _force_split_by_size handles empty content."""
        strategy = RecursiveChunkingStrategy()

        result = strategy._force_split_by_size("", max_size=25, min_size=10)

        assert result == []


class TestRecursiveChunkingProgressCallback:
    """Tests for progress callback functionality."""

    def test_progress_callback_called_during_chunking(self) -> None:
        """Test progress callback is called during chunking."""
        strategy = RecursiveChunkingStrategy()
        config = ChunkConfig("recursive", min_tokens=5, max_tokens=20, overlap_tokens=2)
        content = "Word one. Word two. Word three. Word four. Word five."

        progress_values = []

        def callback(progress: float) -> None:
            progress_values.append(progress)

        strategy.chunk(content, config, progress_callback=callback)

        # Progress should be called at least once
        assert len(progress_values) >= 1
        # Progress should be between 0 and 100
        for p in progress_values:
            assert 0 <= p <= 100


class TestRecursiveChunkingSmallContent:
    """Tests for handling small content."""

    def test_small_content_not_discarded(self) -> None:
        """Test small content is not discarded when below min size."""
        strategy = RecursiveChunkingStrategy()
        config = ChunkConfig("recursive", min_tokens=100, max_tokens=1000, overlap_tokens=10)

        # Content smaller than min_tokens * 4 chars
        content = "Short text."

        result = strategy.chunk(content, config)

        # Should still produce a chunk for small content
        assert len(result) >= 1
        assert result[0].content.strip() == content

    def test_single_word_content(self) -> None:
        """Test single word content produces a chunk."""
        strategy = RecursiveChunkingStrategy()
        config = ChunkConfig("recursive", min_tokens=5, max_tokens=100, overlap_tokens=2)

        result = strategy.chunk("Hello", config)

        assert len(result) >= 1


class TestRecursiveChunkingOffsetCalculation:
    """Tests for offset calculation edge cases."""

    def test_chunk_with_overlap_preserves_context(self) -> None:
        """Test chunks with overlap include context from previous chunk."""
        strategy = RecursiveChunkingStrategy()
        config = ChunkConfig("recursive", min_tokens=5, max_tokens=15, overlap_tokens=3)

        # Content with clear paragraph breaks
        content = "First part here. Second part here. Third part here. Fourth part here."

        chunks = strategy.chunk(content, config)

        # All chunks should have valid offsets
        for chunk in chunks:
            assert chunk.metadata.start_offset >= 0
            assert chunk.metadata.end_offset <= len(content)
            assert chunk.metadata.start_offset < chunk.metadata.end_offset

    def test_offset_calculation_with_special_characters(self) -> None:
        """Test offset calculation handles special characters."""
        strategy = RecursiveChunkingStrategy()
        config = ChunkConfig("recursive", min_tokens=5, max_tokens=30, overlap_tokens=3)

        content = "Line1\n\nLine2\n\nLine3\n\nLine4"

        chunks = strategy.chunk(content, config)

        for chunk in chunks:
            assert chunk.metadata.start_offset >= 0
            assert chunk.metadata.end_offset <= len(content)


class TestRecursiveChunkingInitialization:
    """Tests for strategy initialization."""

    def test_default_separators(self) -> None:
        """Test default separators are set correctly."""
        strategy = RecursiveChunkingStrategy()

        assert len(strategy.separators) > 0
        assert "\n\n" in strategy.separators
        assert ". " in strategy.separators
        assert " " in strategy.separators
        assert "" in strategy.separators  # Last resort

    def test_llama_index_fallback_when_not_available(self) -> None:
        """Test chunking works without LlamaIndex by falling back to domain implementation."""
        # Even if use_llama_index=True is requested, if llama_index isn't installed,
        # the strategy should fall back to domain implementation.
        # We can verify this by checking that chunking still works.
        strategy = RecursiveChunkingStrategy(use_llama_index=False)
        config = ChunkConfig("recursive", min_tokens=5, max_tokens=50, overlap_tokens=2)

        # Should still produce chunks using domain implementation
        result = strategy.chunk("This is test content to chunk.", config)

        assert len(result) >= 1

    def test_init_llama_splitter_returns_none_when_disabled(self) -> None:
        """Test _init_llama_splitter returns None when LlamaIndex disabled."""
        strategy = RecursiveChunkingStrategy(use_llama_index=False)
        config = ChunkConfig("recursive", min_tokens=10, max_tokens=100, overlap_tokens=5)

        result = strategy._init_llama_splitter(config)

        assert result is None


class TestSplitToTokenLimit:
    """Tests for split_to_token_limit and oversized word handling."""

    def test_split_to_token_limit_normal_text(self) -> None:
        """Test split_to_token_limit with normal text splits at word boundaries."""
        strategy = RecursiveChunkingStrategy()

        text = "The quick brown fox jumps over the lazy dog"
        result = strategy.split_to_token_limit(text, max_tokens=5)

        assert len(result) >= 1
        for chunk in result:
            assert strategy.count_tokens(chunk) <= 5

    def test_split_to_token_limit_oversized_word(self) -> None:
        """Test split_to_token_limit handles single word exceeding max_tokens."""
        strategy = RecursiveChunkingStrategy()

        # Create a very long "word" (no spaces) that exceeds token limit
        # Using a long URL-like string
        long_word = "https://example.com/" + "a" * 500
        text = f"Before {long_word} after"

        result = strategy.split_to_token_limit(text, max_tokens=20)

        assert len(result) >= 1
        for chunk in result:
            token_count = strategy.count_tokens(chunk)
            assert token_count <= 20, f"Chunk has {token_count} tokens, expected <= 20"

    def test_split_to_token_limit_only_oversized_word(self) -> None:
        """Test split_to_token_limit handles text that is only an oversized word."""
        strategy = RecursiveChunkingStrategy()

        # A single long string with no whitespace
        long_word = "x" * 1000

        result = strategy.split_to_token_limit(long_word, max_tokens=10)

        assert len(result) >= 1
        for chunk in result:
            token_count = strategy.count_tokens(chunk)
            assert token_count <= 10, f"Chunk has {token_count} tokens, expected <= 10"

    def test_split_to_token_limit_multiple_oversized_words(self) -> None:
        """Test split_to_token_limit handles multiple oversized words."""
        strategy = RecursiveChunkingStrategy()

        long_word1 = "abc" * 200
        long_word2 = "xyz" * 200
        text = f"{long_word1} normal words here {long_word2}"

        result = strategy.split_to_token_limit(text, max_tokens=15)

        assert len(result) >= 1
        for chunk in result:
            token_count = strategy.count_tokens(chunk)
            assert token_count <= 15, f"Chunk has {token_count} tokens, expected <= 15"

    def test_split_to_token_limit_empty_text(self) -> None:
        """Test split_to_token_limit handles empty text."""
        strategy = RecursiveChunkingStrategy()

        result = strategy.split_to_token_limit("", max_tokens=10)

        assert result == []

    def test_split_to_token_limit_zero_max_tokens(self) -> None:
        """Test split_to_token_limit handles zero max_tokens."""
        strategy = RecursiveChunkingStrategy()

        result = strategy.split_to_token_limit("some text", max_tokens=0)

        assert result == []

    def test_split_to_token_limit_text_within_limit(self) -> None:
        """Test split_to_token_limit returns text as-is when within limit."""
        strategy = RecursiveChunkingStrategy()

        text = "Short text"
        result = strategy.split_to_token_limit(text, max_tokens=100)

        assert result == [text]


class TestSplitWordByTokens:
    """Tests for _split_word_by_tokens helper method."""

    def test_split_word_by_tokens_short_word(self) -> None:
        """Test _split_word_by_tokens returns short word as-is."""
        strategy = RecursiveChunkingStrategy()

        result = strategy._split_word_by_tokens("hello", max_tokens=100)

        assert result == ["hello"]

    def test_split_word_by_tokens_long_word(self) -> None:
        """Test _split_word_by_tokens splits long word at token boundaries."""
        strategy = RecursiveChunkingStrategy()

        # Create a word that will definitely exceed the token limit
        long_word = "supercalifragilisticexpialidocious" * 10

        result = strategy._split_word_by_tokens(long_word, max_tokens=5)

        assert len(result) >= 1
        for chunk in result:
            token_count = strategy.count_tokens(chunk)
            assert token_count <= 5, f"Chunk has {token_count} tokens, expected <= 5"

        # Verify all content is preserved
        reconstructed = "".join(result)
        assert reconstructed == long_word

    def test_split_word_by_tokens_empty_word(self) -> None:
        """Test _split_word_by_tokens handles empty word."""
        strategy = RecursiveChunkingStrategy()

        result = strategy._split_word_by_tokens("", max_tokens=10)

        assert result == []

    def test_split_word_by_tokens_zero_max_tokens(self) -> None:
        """Test _split_word_by_tokens handles zero max_tokens."""
        strategy = RecursiveChunkingStrategy()

        result = strategy._split_word_by_tokens("word", max_tokens=0)

        assert result == []

    def test_split_word_by_tokens_base64_like_content(self) -> None:
        """Test _split_word_by_tokens handles base64-like content."""
        strategy = RecursiveChunkingStrategy()

        # Simulate base64 encoded content (no whitespace)
        import base64

        original = b"This is some test content that will be encoded" * 20
        base64_content = base64.b64encode(original).decode("ascii")

        result = strategy._split_word_by_tokens(base64_content, max_tokens=10)

        assert len(result) >= 1
        for chunk in result:
            token_count = strategy.count_tokens(chunk)
            assert token_count <= 10


class TestRecursiveChunkingOversizedContent:
    """Integration tests for chunking content with oversized words."""

    def test_chunking_document_with_long_url(self) -> None:
        """Test chunking a document containing a very long URL."""
        strategy = RecursiveChunkingStrategy()
        config = ChunkConfig("recursive", min_tokens=5, max_tokens=50, overlap_tokens=2)

        # Simulate a document with a very long URL
        long_url = "https://example.com/path/" + "param=value&" * 100
        content = f"Check out this link: {long_url} for more information."

        # This should not raise ChunkSizeViolationError
        chunks = strategy.chunk(content, config)

        assert len(chunks) >= 1
        for chunk in chunks:
            token_count = strategy.count_tokens(chunk.content)
            assert token_count <= config.max_tokens, f"Chunk has {token_count} tokens, expected <= {config.max_tokens}"

    def test_chunking_minified_code(self) -> None:
        """Test chunking minified code without whitespace."""
        strategy = RecursiveChunkingStrategy()
        config = ChunkConfig("recursive", min_tokens=5, max_tokens=30, overlap_tokens=2)

        # Simulate minified JavaScript (no whitespace)
        minified = "function(){" + "var x=1;y=2;z=x+y;console.log(z);" * 50 + "}"
        content = f"Here is some code:\n{minified}\nEnd of code."

        chunks = strategy.chunk(content, config)

        assert len(chunks) >= 1
        for chunk in chunks:
            token_count = strategy.count_tokens(chunk.content)
            assert token_count <= config.max_tokens, f"Chunk has {token_count} tokens, expected <= {config.max_tokens}"
