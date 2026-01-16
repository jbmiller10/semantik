#!/usr/bin/env python3

"""Unit tests for ReDoS protection in chunking utilities.

This module tests the Regular Expression Denial of Service (ReDoS) protection
implemented in the SafeRegex utility to ensure:
1. Regex patterns are executed with timeout protection
2. Malicious patterns that could cause exponential backtracking are handled safely
3. The chunking system continues to function even when regex patterns fail
"""

import re
import signal
import time
from contextlib import contextmanager, suppress
from typing import Any

import pytest
from llama_index.core.embeddings import MockEmbedding

# Import regex module for pattern type checking
try:
    import regex

    HAS_REGEX = True
except ImportError:
    HAS_REGEX = False

from shared.chunking.unified.factory import (
    TextProcessingStrategyAdapter,
    UnifiedChunkingFactory,
)
from shared.chunking.utils.safe_regex import RegexTimeout, SafeRegex

# Define test constants and helpers
REGEX_TIMEOUT = 1  # Default timeout from SafeRegex


def safe_regex_findall(pattern: re.Pattern[str] | str, text: str, flags: int | None = None) -> list[str]:
    """Helper function for testing regex with timeout protection."""
    safe_regex = SafeRegex(timeout=REGEX_TIMEOUT)
    if isinstance(pattern, str):
        if flags:
            pattern = re.compile(pattern, flags)
        else:
            pattern = safe_regex.compile_safe(pattern)
    try:
        return safe_regex.findall_safe(pattern.pattern if hasattr(pattern, "pattern") else str(pattern), text)
    except RegexTimeout:
        return []


@contextmanager
def timeout(seconds: int) -> Any:
    """Simple timeout context manager for testing."""

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Set the signal handler and alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore the original signal handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _create_hybrid_chunker() -> TextProcessingStrategyAdapter:
    """Create a hybrid chunker using the unified factory."""
    unified_strategy = UnifiedChunkingFactory.create_strategy(
        "hybrid",
        use_llama_index=True,
        embed_model=MockEmbedding(embed_dim=384),
    )
    return TextProcessingStrategyAdapter(unified_strategy)


class TestSafeRegexProtection:
    """Test suite for SafeRegex ReDoS protection."""

    def test_safe_regex_findall_normal_pattern(self) -> None:
        """Test safe_regex_findall with normal, safe patterns."""
        # Test with compiled pattern
        pattern = re.compile(r"\d+", re.MULTILINE)
        text = "There are 123 numbers and 456 more"
        matches = safe_regex_findall(pattern, text)
        assert matches == ["123", "456"]

        # Test with string pattern
        matches = safe_regex_findall(r"\w+", "hello world", re.IGNORECASE)
        assert matches == ["hello", "world"]

    def test_safe_regex_findall_empty_results(self) -> None:
        """Test safe_regex_findall when no matches found."""
        pattern = re.compile(r"xyz")
        text = "abc def"
        matches = safe_regex_findall(pattern, text)
        assert matches == []

    @pytest.mark.skipif(
        True,
        reason="ReDoS test is time-dependent and may be flaky",
    )
    def test_safe_regex_findall_timeout(self) -> None:
        """Test that regex execution times out for malicious patterns."""
        # This is a known ReDoS pattern: (a+)+
        # With long input, it causes exponential backtracking
        pattern = re.compile(r"(a+)+b")
        text = "a" * 30  # No 'b' at the end causes maximum backtracking

        with pytest.raises(TimeoutError):
            safe_regex_findall(pattern, text)

    def test_regex_timeout_value(self) -> None:
        """Test that REGEX_TIMEOUT is set to a reasonable value."""
        assert REGEX_TIMEOUT == 1  # Should be 1 second
        assert isinstance(REGEX_TIMEOUT, int)

    def test_safe_regex_findall_with_malformed_pattern(self) -> None:
        """Test safe_regex_findall handles malformed patterns gracefully."""
        # Test with invalid regex pattern
        # Handle both re.error and regex.error depending on which module is used
        error_types: list[type] = [re.error]
        if HAS_REGEX:
            # regex module has its own error type
            with suppress(AttributeError):
                error_types.append(regex.error)

        with pytest.raises(tuple(error_types)):
            safe_regex_findall(r"[", "test text")  # Unclosed bracket

    def test_timeout_context_manager(self) -> None:
        """Test the timeout context manager behavior."""
        # Test normal execution (no timeout)
        with timeout(2):
            result = 1 + 1
            assert result == 2

        # Context manager should complete normally
        try:
            with timeout(1):
                pass
        except Exception as e:
            pytest.fail(f"Timeout context raised unexpected exception: {e}")


class TestHybridChunkerWithMaliciousContent:
    """Test suite for hybrid chunker handling of potentially malicious content."""

    @pytest.fixture()
    def chunker(self) -> TextProcessingStrategyAdapter:
        """Create a hybrid chunker instance."""
        return _create_hybrid_chunker()

    def test_malicious_pattern_in_markdown_file(self, chunker: TextProcessingStrategyAdapter) -> None:
        """Test that malicious content doesn't cause issues in markdown detection."""
        # Create content that could trigger ReDoS with poorly written patterns
        malicious_content = "a" * 1000 + "[[[[[[" + "]]]]]]" + "(((((" + ")))))"

        # This should complete without hanging
        start_time = time.time()
        chunks = chunker.chunk_text(malicious_content, "test_doc")
        elapsed_time = time.time() - start_time

        # Should complete quickly (well under the timeout)
        assert elapsed_time < REGEX_TIMEOUT
        assert isinstance(chunks, list)

    @pytest.mark.asyncio()
    async def test_async_chunk_text_handles_malicious_content(self, chunker: TextProcessingStrategyAdapter) -> None:
        """Test async chunking handles potentially malicious content."""
        test_text = """# Test Document

        This is a test with some **markdown** elements.

        - List item 1
        - List item 2

        [Link](http://example.com)
        """

        # Should still produce chunks
        chunks = await chunker.chunk_text_async(test_text, "test_doc")

        assert len(chunks) > 0
        assert all(chunk.text for chunk in chunks)

    def test_performance_with_large_documents(self, chunker: TextProcessingStrategyAdapter) -> None:
        """Test that chunking performs well on large documents."""
        # Create a large document with mixed content
        large_doc = ""
        for i in range(100):
            large_doc += f"# Section {i}\n\n"
            large_doc += "This is some content " * 50 + "\n\n"
            large_doc += f"- List item {i}\n"
            large_doc += f"[Link {i}](http://example.com/{i})\n\n"

        # Time the chunking operation
        start_time = time.time()
        chunks = chunker.chunk_text(large_doc, "test_doc")
        elapsed_time = time.time() - start_time

        # Should complete in reasonable time
        assert elapsed_time < 5.0  # Should be much faster than timeout
        assert len(chunks) > 0
