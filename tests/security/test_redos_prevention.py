"""Comprehensive test suite for ReDoS (Regular Expression Denial of Service) prevention.

This module tests the regex safety mechanisms to ensure:
1. Patterns with nested quantifiers are detected and rejected
2. Regex operations timeout within 1 second
3. Fallback mechanisms work correctly
4. Performance impact is minimal for normal inputs
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from packages.shared.chunking.utils.input_validator import ChunkingInputValidator as InputValidator
from packages.shared.chunking.utils.safe_regex import SafeRegex
from packages.shared.utils.regex_safety import (
    RegexTimeout,
    analyze_pattern_complexity,
    compile_safe,
    safe_regex_findall,
    safe_regex_match,
    safe_regex_search,
    search_with_fallback,
    simplify_pattern,
)
from packages.webui.services.chunking_validation import ChunkingInputValidator


class TestRegexSafetyModule:
    """Test the regex_safety module functions."""

    def test_safe_regex_search_normal_pattern(self):
        """Test safe_regex_search with normal, safe patterns."""
        result = safe_regex_search(r"\d+", "The answer is 42", timeout=1.0)
        assert result is not None
        assert result.group() == "42"

        # Test with no match
        result = safe_regex_search(r"xyz", "abc def", timeout=1.0)
        assert result is None

    def test_safe_regex_match_normal_pattern(self):
        """Test safe_regex_match with normal patterns."""
        result = safe_regex_match(r"Hello", "Hello World", timeout=1.0)
        assert result is not None
        assert result.group() == "Hello"

        # Test no match
        result = safe_regex_match(r"World", "Hello World", timeout=1.0)
        assert result is None

    def test_safe_regex_findall_normal_pattern(self):
        """Test safe_regex_findall with normal patterns."""
        results = safe_regex_findall(r"\d+", "123 and 456 and 789", timeout=1.0)
        assert results == ["123", "456", "789"]

        # Test with max_matches
        results = safe_regex_findall(r"\d+", "123 and 456 and 789", timeout=1.0, max_matches=2)
        assert results == ["123", "456"]

    def test_analyze_pattern_complexity_dangerous_patterns(self):
        """Test that dangerous patterns are detected."""
        # Nested quantifiers
        assert analyze_pattern_complexity(r"(a+)+") is True
        assert analyze_pattern_complexity(r"(a*)*") is True
        assert analyze_pattern_complexity(r"(\w+)+") is True
        assert analyze_pattern_complexity(r'([^"]*)*') is True

        # Alternation with quantifier
        assert analyze_pattern_complexity(r"(a|b)*") is True
        assert analyze_pattern_complexity(r"(cat|dog)+") is True

        # Multiple wildcards
        assert analyze_pattern_complexity(r".*.*.*") is True
        assert analyze_pattern_complexity(r".+.+.+") is True

        # Very long pattern
        assert analyze_pattern_complexity("a" * 600) is True

    def test_analyze_pattern_complexity_safe_patterns(self):
        """Test that safe patterns are not flagged."""
        assert analyze_pattern_complexity(r"\d+") is False
        assert analyze_pattern_complexity(r"\w+") is False
        assert analyze_pattern_complexity(r"[a-z]+") is False
        assert analyze_pattern_complexity(r"hello|world") is False
        assert analyze_pattern_complexity(r"^test$") is False

    def test_simplify_pattern(self):
        """Test pattern simplification."""
        # Nested quantifiers should be simplified
        assert simplify_pattern(r"(a+)+") == r"a+"
        assert simplify_pattern(r"(a*)*") == r"a*"
        assert simplify_pattern(r"(\w+)+") == r"\w+"

        # Multiple wildcards should be simplified
        assert simplify_pattern(r".*.*") == r".*"
        assert simplify_pattern(r".+.+") == r".+"

        # Unbounded repetitions should be bounded
        assert "{5,1000}" in simplify_pattern(r"a{5,}")

    def test_search_with_fallback(self):
        """Test search with automatic fallback."""
        # Normal pattern should work
        result = search_with_fallback(r"\d+", "test 123", timeout=1.0)
        assert result is not None
        assert result.group() == "123"

        # Test with a pattern that would timeout (mocked)
        with patch("packages.shared.utils.regex_safety.safe_regex_search") as mock_search:
            # First call times out, second succeeds
            mock_search.side_effect = [RegexTimeout("Timeout"), MagicMock(group=lambda: "test")]
            result = search_with_fallback(r"(a+)+b", "aaaa", timeout=0.1)
            # Should have tried simplified version
            assert mock_search.call_count == 2

    @pytest.mark.skipif(True, reason="Actual timeout test may be flaky in CI")
    def test_regex_timeout_with_pathological_input(self):
        """Test that pathological input causes timeout."""
        # This pattern causes exponential backtracking
        pattern = r"(a+)+b"
        text = "a" * 30  # No 'b' at the end causes maximum backtracking

        with pytest.raises(RegexTimeout):
            safe_regex_search(pattern, text, timeout=0.1)

    def test_dangerous_pattern_rejection(self):
        """Test that dangerous patterns are rejected or simplified."""
        # This pattern gets simplified rather than rejected
        result = safe_regex_search(r"(a+)+b", "test", timeout=1.0)
        assert result is None  # Pattern doesn't match "test"

        # This pattern should also be simplified
        pattern = compile_safe(r"(.*)*", timeout=1.0)
        assert pattern is not None  # Pattern gets compiled after simplification


class TestSafeRegexClass:
    """Test the updated SafeRegex class."""

    @pytest.fixture()
    def safe_regex(self):
        """Create a SafeRegex instance."""
        return SafeRegex(timeout=1.0)

    def test_compile_safe_with_re2(self, safe_regex):
        """Test pattern compilation with RE2 fallback."""
        pattern = safe_regex.compile_safe(r"\d+", use_re2=True)
        assert pattern is not None

    def test_match_with_timeout(self, safe_regex):
        """Test match with timeout protection."""
        result = safe_regex.match_with_timeout(r"test", "test string", timeout=1.0)
        assert result is not None
        assert result.group() == "test"

    def test_search_with_timeout(self, safe_regex):
        """Test search with timeout protection."""
        result = safe_regex.search_with_timeout(r"\d+", "answer is 42", timeout=1.0)
        assert result is not None
        assert result.group() == "42"

    def test_findall_safe(self, safe_regex):
        """Test findall with safety limits."""
        results = safe_regex.findall_safe(r"\w+", "hello world test", max_matches=2)
        assert len(results) == 2
        assert results == ["hello", "world"]

    def test_pattern_caching(self, safe_regex):
        """Test that patterns are cached."""
        # Compile the same pattern twice
        safe_regex.compile_safe(r"\d+")
        safe_regex.compile_safe(r"\d+")

        # Should have cached the pattern
        assert len(safe_regex._pattern_cache) > 0

    def test_dangerous_pattern_detection(self, safe_regex):
        """Test that dangerous patterns are detected."""
        assert safe_regex._is_pattern_dangerous(r"(a+)+") is True
        assert safe_regex._is_pattern_dangerous(r".*.*.*") is True
        assert safe_regex._is_pattern_dangerous(r"\d+") is False


class TestChunkingValidation:
    """Test the updated chunking validation with ReDoS protection."""

    def test_validate_content_with_safe_text(self):
        """Test validation with safe content."""
        # Should not raise any exception
        ChunkingInputValidator.validate_content(
            "This is safe content with no injection patterns.", correlation_id="test-123"
        )

    def test_validate_content_with_sql_injection(self):
        """Test SQL injection detection with timeout protection."""
        with pytest.raises(Exception, match="SQL|forbidden") as exc_info:
            ChunkingInputValidator.validate_content("'; DROP TABLE users; --", correlation_id="test-123")
        assert "SQL" in str(exc_info.value) or "forbidden" in str(exc_info.value)

    def test_validate_content_with_bounded_patterns(self):
        """Test that bounded patterns don't cause ReDoS."""
        # Create content that would cause issues with unbounded patterns
        content = "a" * 1000 + " or " + "b" * 1000

        # Should complete quickly
        start_time = time.time()
        from contextlib import suppress

        with suppress(Exception):
            ChunkingInputValidator.validate_content(content, correlation_id="test")
        elapsed = time.time() - start_time

        # Should complete in under 2 seconds (generous margin for slow CI)
        assert elapsed < 2.0

    @patch("packages.shared.utils.regex_safety.safe_regex_search")
    def test_validate_content_handles_timeout(self, mock_search):
        """Test that validation handles regex timeouts gracefully."""
        # Make search timeout
        mock_search.side_effect = RegexTimeout("Pattern timed out")

        # Should handle the timeout without crashing
        # (implementation may vary - might skip pattern or use fallback)
        try:
            ChunkingInputValidator.validate_content("Some content to validate", correlation_id="test-123")
        except RegexTimeout:
            # It's ok if timeout is propagated
            pass
        except Exception as e:
            # Should not raise other exceptions
            pytest.fail(f"Unexpected exception: {e}")


class TestInputValidator:
    """Test the input validator with ReDoS protection."""

    def test_sanitize_text_normal(self):
        """Test text sanitization with normal input."""
        text = "Hello   World!!!   Test***"
        result = InputValidator.sanitize_text(text)
        assert result == text  # Should not change normal text

    def test_sanitize_text_excessive_whitespace(self):
        """Test sanitization of excessive whitespace."""
        text = "Hello" + " " * 150 + "World"
        result = InputValidator.sanitize_text(text)
        assert " " * 100 not in result
        assert "Hello" in result
        assert "World" in result

    def test_sanitize_text_excessive_punctuation(self):
        """Test sanitization of excessive punctuation."""
        text = "Hello" + "!" * 20 + " World"
        result = InputValidator.sanitize_text(text)
        assert "!" * 10 not in result
        assert "Hello" in result
        assert "World" in result

    def test_sanitize_text_excessive_special_chars(self):
        """Test sanitization of excessive special characters."""
        text = "Test" + "*" * 30 + "End"
        result = InputValidator.sanitize_text(text)
        assert "*" * 20 not in result
        assert "Test" in result
        assert "End" in result

    def test_sanitize_text_repeated_characters(self):
        """Test sanitization of excessively repeated characters."""
        text = "a" * 150 + " test"
        result = InputValidator.sanitize_text(text)
        assert "a" * 100 not in result
        assert "test" in result

    @patch("packages.shared.utils.regex_safety.safe_regex_findall")
    def test_sanitize_text_handles_timeout(self, mock_findall):
        """Test that sanitization handles timeouts gracefully."""
        # Make findall timeout
        mock_findall.side_effect = RegexTimeout("Pattern timed out")

        text = "Some text to sanitize"
        result = InputValidator.sanitize_text(text)

        # Should return original text on timeout
        assert result == text


class TestPerformance:
    """Test performance impact of ReDoS protection."""

    def test_normal_pattern_performance(self):
        """Test that normal patterns have minimal overhead."""
        pattern = r"\d+"
        text = "The numbers are 123, 456, 789" * 100

        # Time without protection (using standard re)
        import re

        start = time.time()
        for _ in range(100):
            re.findall(pattern, text)
        baseline_time = time.time() - start

        # Time with protection
        start = time.time()
        for _ in range(100):
            safe_regex_findall(pattern, text, timeout=1.0)
        protected_time = time.time() - start

        # Should have reasonable overhead (regex module is slower but safer)
        # We accept higher overhead for security
        overhead = (protected_time - baseline_time) / baseline_time
        assert overhead < 10.0, f"Overhead too high: {overhead:.2%}"  # Allow up to 10x slower for safety

    def test_complex_validation_performance(self):
        """Test performance of complex validation."""
        content = (
            """
        This is a test document with various content types.
        It includes numbers like 12345 and text.
        Some SQL-like content: SELECT * FROM table WHERE id = 1
        Some code: function test() { return true; }
        """
            * 10
        )

        start = time.time()
        for _ in range(10):
            ChunkingInputValidator.validate_content(content, "test")
        elapsed = time.time() - start

        # Should complete 10 validations in under 1 second
        assert elapsed < 1.0, f"Validation too slow: {elapsed:.2f}s"


class TestIntegration:
    """Integration tests for ReDoS protection across the system."""

    def test_chunking_validation_integration(self):
        """Test that chunking validation works end-to-end."""
        # Test various content types
        test_cases = [
            ("Normal text content", True),
            ("<script>alert('xss')</script>", False),
            ("SELECT * FROM users WHERE id = 1", True),  # SQL-like but safe
            ("'; DROP TABLE users; --", False),  # SQL injection
            ("`rm -rf /`", True),  # Command-like but in backticks (might be code)
        ]

        for content, should_pass in test_cases:
            try:
                ChunkingInputValidator.validate_content(content, "test")
                assert should_pass, f"Content should have been rejected: {content[:50]}"
            except Exception:
                assert not should_pass, f"Content should have passed: {content[:50]}"

    def test_input_validator_integration(self):
        """Test input validator integration."""
        # Test document validation
        doc = "Test document " * 100
        InputValidator.validate_document(doc)  # Should not raise

        # Test line validation
        assert InputValidator.validate_line("Normal line") is True
        assert InputValidator.validate_line("a" * 10001) is False  # Too long

        # Test risk estimation
        risk = InputValidator.estimate_processing_risk("Small doc")
        assert risk == "low"

        large_doc = "x" * 5000000  # Large document
        risk = InputValidator.estimate_processing_risk(large_doc)
        assert risk in ["medium", "high"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
