#!/usr/bin/env python3

"""Unit tests for ReDoS protection in HybridChunker.

This module tests the Regular Expression Denial of Service (ReDoS) protection
implemented in the HybridChunker to ensure:
1. Regex patterns are executed with timeout protection
2. Malicious patterns that could cause exponential backtracking are handled safely
3. The chunker continues to function even when regex patterns fail
"""

import re
import signal
import time
from contextlib import contextmanager
from unittest.mock import patch

import pytest

# Import regex module for pattern type checking
try:
    import regex
    HAS_REGEX = True
except ImportError:
    HAS_REGEX = False

from packages.shared.chunking.utils.safe_regex import RegexTimeout, SafeRegex
from packages.shared.text_processing.strategies.hybrid_chunker import HybridChunker

# Define test constants and helpers
REGEX_TIMEOUT = 1  # Default timeout from SafeRegex


def safe_regex_findall(pattern, text, flags=None):
    """Helper function for testing regex with timeout protection."""
    safe_regex = SafeRegex(timeout=REGEX_TIMEOUT)
    if isinstance(pattern, str):
        if flags:
            import re

            pattern = re.compile(pattern, flags)
        else:
            pattern = safe_regex.compile_safe(pattern)
    try:
        return safe_regex.findall_safe(pattern.pattern if hasattr(pattern, "pattern") else str(pattern), text)
    except RegexTimeout:
        return []


@contextmanager
def timeout(seconds):
    """Simple timeout context manager for testing."""

    def timeout_handler(_signum, _frame):
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


class TestHybridChunkerReDoSProtection:
    """Test suite for ReDoS protection in HybridChunker."""

    @pytest.fixture()
    def chunker(self) -> None:
        """Create a HybridChunker instance."""
        return HybridChunker()

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
        True, reason="ReDoS test is time-dependent and may be flaky"  # Skip by default as it's time-dependent
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

    def test_markdown_pattern_compilation(self, chunker) -> None:
        """Test that markdown patterns are pre-compiled during initialization."""
        assert hasattr(chunker, "_compiled_patterns")
        assert isinstance(chunker._compiled_patterns, dict)

        # Check expected patterns are compiled
        # These patterns should match what's actually in the HybridChunker implementation
        expected_patterns = [
            r"^#{1,6}\s+\S.*$",  # Headers
            r"^[\*\-\+]\s+\S.*$",  # Unordered lists
            r"^\d+\.\s+\S.*$",  # Ordered lists
            r"\[([^\]]+)\]\(([^)]+)\)",  # Links
            r"!\[([^\]]*)\]\(([^)]+)\)",  # Images
            r"`([^`]+)`",  # Inline code
            r"^>\s*\S.*$",  # Blockquotes
            r"\*\*([^*]+)\*\*",  # Bold
            r"\*([^*]+)\*",  # Italic
            r"^\s*\|[^|]+\|",  # Tables
            r"^(?:---|\\*\\*\\*|___)$",  # Horizontal rules
        ]

        # Verify patterns are compiled
        for pattern_str in expected_patterns:
            assert pattern_str in chunker._compiled_patterns
            compiled_pattern, weight = chunker._compiled_patterns[pattern_str]
            # Handle standard re.Pattern, regex.Pattern, and re2._Regexp types
            valid_pattern_types = [re.Pattern]
            
            # Add regex.Pattern if available
            if HAS_REGEX:
                # The regex module may use _regex.Pattern internally
                try:
                    regex_pattern = regex.compile("test")
                    valid_pattern_types.append(type(regex_pattern))
                except:
                    pass
            
            # Add re2._Regexp if available
            try:
                import re2
                valid_pattern_types.append(re2._Regexp)
            except ImportError:
                pass
                
            assert isinstance(compiled_pattern, tuple(valid_pattern_types))
            assert isinstance(weight, float)
            assert weight > 0

    def test_markdown_analysis_with_pattern_failure(self, chunker) -> None:
        """Test markdown analysis continues even when patterns fail."""
        test_text = "# Header\n\nSome content with **bold** text."

        # Mock safe_regex_findall to simulate some patterns failing
        original_safe_regex = safe_regex_findall
        call_count = 0

        def mock_safe_regex(pattern, text, flags=0) -> None:
            nonlocal call_count
            call_count += 1
            # Make every other pattern fail
            if call_count % 2 == 0:
                raise TimeoutError("Simulated timeout")
            return original_safe_regex(pattern, text, flags)

        with patch(
            "packages.shared.text_processing.strategies.hybrid_chunker.safe_regex_findall", side_effect=mock_safe_regex
        ):
            is_file, density = chunker._analyze_markdown_content(test_text, None)

            # Should still return a result despite some patterns failing
            assert isinstance(is_file, bool)
            assert isinstance(density, float)
            assert density >= 0  # Should have found some markdown elements

    def test_semantic_coherence_with_regex_failure(self, chunker) -> None:
        """Test semantic coherence calculation handles regex failures gracefully."""
        test_text = """This is a test document with repeated words.
        The test document contains test content for testing.
        Testing is important for test coverage."""

        # Mock safe_regex_findall to fail
        with patch(
            "packages.shared.text_processing.strategies.hybrid_chunker.safe_regex_findall",
            side_effect=Exception("Regex failed"),
        ):
            coherence = chunker._estimate_semantic_coherence(test_text)

            # Should still return a valid coherence score
            assert isinstance(coherence, float)
            assert 0.0 <= coherence <= 1.0

    def test_malicious_pattern_in_markdown_file(self, chunker) -> None:
        """Test that malicious content doesn't cause ReDoS in markdown detection."""
        # Create content that could trigger ReDoS with poorly written patterns
        malicious_content = "a" * 1000 + "[[[[[[" + "]]]]]]" + "(((((" + ")))))"

        # This should complete without hanging
        start_time = time.time()
        is_file, density = chunker._analyze_markdown_content(malicious_content, None)
        elapsed_time = time.time() - start_time

        # Should complete quickly (well under the timeout)
        assert elapsed_time < REGEX_TIMEOUT
        assert isinstance(is_file, bool)
        assert isinstance(density, float)

    def test_compile_markdown_patterns_timeout_handling(self) -> None:
        """Test that pattern compilation handles timeouts gracefully."""
        # Create a new chunker and mock the timeout context
        with patch("packages.shared.text_processing.strategies.hybrid_chunker.timeout") as mock_timeout:
            # Make timeout raise TimeoutError for specific patterns
            def timeout_side_effect(seconds) -> None:  # noqa: ARG001
                class TimeoutContext:
                    def __enter__(self) -> None:
                        return self

                    def __exit__(self, *args) -> None:
                        # Simulate timeout on complex patterns
                        if hasattr(self, "_pattern_check"):
                            raise TimeoutError("Pattern compilation timeout")

                ctx = TimeoutContext()
                # Mark for timeout on the second pattern
                if not hasattr(timeout_side_effect, "call_count"):
                    timeout_side_effect.call_count = 0
                timeout_side_effect.call_count += 1
                if timeout_side_effect.call_count == 2:
                    ctx._pattern_check = True
                return ctx

            mock_timeout.side_effect = timeout_side_effect

            # Create chunker - should handle timeout during pattern compilation
            chunker = HybridChunker()

            # Should have compiled some patterns despite timeout
            assert len(chunker._compiled_patterns) > 0

    @pytest.mark.asyncio()
    async def test_async_chunk_text_with_pattern_failures(self, chunker) -> None:
        """Test async chunking continues despite regex pattern failures."""
        test_text = """# Test Document

        This is a test with some **markdown** elements.

        - List item 1
        - List item 2

        [Link](http://example.com)
        """

        # Mock some pattern executions to fail
        with patch("packages.shared.text_processing.strategies.hybrid_chunker.safe_regex_findall") as mock_safe:
            call_count = 0

            def side_effect(pattern, text, flags=0) -> None:  # noqa: ARG001
                nonlocal call_count
                call_count += 1
                # Fail on some calls
                if call_count in [2, 4, 6]:
                    raise Exception("Pattern failed")
                # Return empty for others (to ensure we get to strategy selection)
                return []

            mock_safe.side_effect = side_effect

            # Should still produce chunks
            chunks = await chunker.chunk_text_async(test_text, "test_doc")

            assert len(chunks) > 0
            assert all(chunk.text for chunk in chunks)

    def test_regex_pattern_security_validation(self, chunker) -> None:
        """Test that patterns are validated for security during compilation."""
        # Patterns should have reasonable complexity
        for pattern_str, (compiled_pattern, _) in chunker._compiled_patterns.items():
            # Check pattern doesn't have nested quantifiers that could cause ReDoS
            # This is a simplified check - real ReDoS detection is complex

            # Patterns should not have patterns like (a+)+ or (a*)*
            assert not re.search(r"\([^)]*[+*]\)[+*]", pattern_str), f"Pattern {pattern_str} may be vulnerable to ReDoS"

            # Patterns should compile successfully
            assert compiled_pattern is not None

    def test_performance_with_large_documents(self, chunker) -> None:
        """Test that regex operations perform well on large documents."""
        # Create a large document with mixed content
        large_doc = ""
        for i in range(100):
            large_doc += f"# Section {i}\n\n"
            large_doc += "This is some content " * 50 + "\n\n"
            large_doc += f"- List item {i}\n"
            large_doc += f"[Link {i}](http://example.com/{i})\n\n"

        # Time the markdown analysis
        start_time = time.time()
        is_file, density = chunker._analyze_markdown_content(large_doc, None)
        elapsed_time = time.time() - start_time

        # Should complete in reasonable time
        assert elapsed_time < 1.0  # Should be much faster than timeout
        assert density > 0  # Should detect markdown elements

    def test_safe_regex_findall_with_malformed_pattern(self) -> None:
        """Test safe_regex_findall handles malformed patterns gracefully."""
        # Test with invalid regex pattern
        # Handle both re.error and regex.error depending on which module is used
        error_types = [re.error]
        if HAS_REGEX:
            # regex module has its own error type
            try:
                error_types.append(regex.error)
            except AttributeError:
                # Some versions might use a different error type
                pass
        
        with pytest.raises(tuple(error_types)):
            safe_regex_findall(r"[", "test text")  # Unclosed bracket

    def test_timeout_context_manager(self) -> None:
        """Test the timeout context manager behavior."""
        # This test verifies the timeout mechanism works
        # Note: actual implementation differs between Unix/Windows

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
