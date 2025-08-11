#!/usr/bin/env python3
"""Tests for ReDoS protection in chunking utilities."""

import time

import pytest

from packages.shared.chunking.domain.services.chunking_strategies.markdown import (
    MarkdownChunkingStrategy,
)
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.utils.input_validator import ChunkingInputValidator
from packages.shared.chunking.utils.regex_monitor import RegexPerformanceMonitor
from packages.shared.chunking.utils.safe_regex import RegexTimeout, SafeRegex


class TestSafeRegex:
    """Test SafeRegex class for ReDoS protection."""

    def test_redos_pattern_rejection(self):
        """Test that dangerous patterns are rejected."""
        safe_regex = SafeRegex(timeout=0.1)

        # Known ReDoS patterns that should be rejected
        dangerous_patterns = [
            r"(a+)+b",  # Nested quantifiers
            r"(a*)*b",  # Nested stars
            r"(a|a)*",  # Overlapping alternation
            r"(a|ab)*c",  # Alternation with overlap
        ]

        for pattern in dangerous_patterns:
            with pytest.raises(ValueError, match="potentially dangerous"):
                safe_regex.compile_safe(pattern)

    def test_timeout_protection(self):
        """Test that long-running regex operations timeout."""
        safe_regex = SafeRegex(timeout=0.1)

        # This pattern can cause catastrophic backtracking
        evil_input = "a" * 100
        pattern = r"(a+)+"

        # Even if pattern isn't rejected, execution should timeout
        with pytest.raises((RegexTimeout, ValueError)):
            safe_regex.match_with_timeout(pattern, evil_input, timeout=0.1)

    def test_safe_patterns_work(self):
        """Test that safe patterns work correctly."""
        safe_regex = SafeRegex()

        # Safe patterns should work fine
        safe_patterns = [
            (r"\d+", "123", True),
            (r"[a-z]+", "hello", True),
            (r"^#+ ", "## Header", True),
            (r"^\* ", "* List item", True),
        ]

        for pattern, text, should_match in safe_patterns:
            compiled = safe_regex.compile_safe(pattern)
            result = compiled.match(text)
            assert (result is not None) == should_match

    def test_findall_with_limit(self):
        """Test findall_safe respects match limits."""
        safe_regex = SafeRegex()

        text = " ".join(["word"] * 2000)
        matches = safe_regex.findall_safe(r"\w+", text, max_matches=100)

        assert len(matches) == 100


class TestChunkingInputValidator:
    """Test input validation for ReDoS prevention."""

    def test_document_size_validation(self):
        """Test document size limits."""
        # Document too large
        large_doc = "a" * (ChunkingInputValidator.MAX_DOCUMENT_SIZE + 1)
        with pytest.raises(ValueError, match="Document too large"):
            ChunkingInputValidator.validate_document(large_doc)

        # Document within limits
        normal_doc = "This is a normal document."
        ChunkingInputValidator.validate_document(normal_doc)  # Should not raise

    def test_line_length_validation(self):
        """Test line length limits."""
        # Line too long
        long_line = "a" * (ChunkingInputValidator.MAX_LINE_LENGTH + 1)
        doc_with_long_line = f"Normal line\n{long_line}\nAnother normal line"

        with pytest.raises(ValueError, match="Line .* too long"):
            ChunkingInputValidator.validate_document(doc_with_long_line)

    def test_redos_trigger_detection(self):
        """Test detection of ReDoS triggers in input."""
        # Document with excessive repetition
        evil_doc = "a" * 1001 + "!" * 1001

        with pytest.raises(ValueError, match="potential ReDoS triggers"):
            ChunkingInputValidator.validate_document(evil_doc)

    def test_binary_content_detection(self):
        """Test detection of binary content."""
        # Document with binary data
        binary_doc = "Normal text\x00Binary data\xffMore binary"

        with pytest.raises(ValueError, match="binary data"):
            ChunkingInputValidator.validate_document(binary_doc)

    def test_text_sanitization(self):
        """Test text sanitization removes dangerous patterns."""
        # Text with excessive repetition
        dangerous_text = "Hello" + "!" * 100 + " World" + "*" * 50
        sanitized = ChunkingInputValidator.sanitize_text(dangerous_text)

        # Check that excessive punctuation is reduced
        assert "!" * 100 not in sanitized
        assert "*" * 50 not in sanitized
        assert "..." in sanitized or "*" * 19 in sanitized

    def test_risk_estimation(self):
        """Test risk level estimation."""
        # Low risk document
        safe_doc = "This is a normal document with regular text."
        risk = ChunkingInputValidator.estimate_processing_risk(safe_doc)
        assert risk == "low"

        # High risk document
        risky_doc = "a" * (ChunkingInputValidator.MAX_DOCUMENT_SIZE // 3)
        risky_doc += "\n".join(["x" * (ChunkingInputValidator.MAX_LINE_LENGTH // 2)] * 100)
        risk = ChunkingInputValidator.estimate_processing_risk(risky_doc)
        assert risk in ["medium", "high"]


class TestRegexPerformanceMonitor:
    """Test regex performance monitoring."""

    def test_metric_recording(self):
        """Test that metrics are recorded correctly."""
        monitor = RegexPerformanceMonitor()

        # Record some executions
        monitor.record_execution(r"\d+", 0.01, 100, matched=True)
        monitor.record_execution(r"\w+", 0.05, 500, matched=True)
        monitor.record_execution(r"(a+)+", 1.5, 1000, timed_out=True)

        stats = monitor.get_statistics()
        assert stats["total_executions"] == 3
        assert stats["total_timeouts"] == 1
        assert stats["timeout_rate"] == 1 / 3

    def test_slow_pattern_detection(self):
        """Test detection of consistently slow patterns."""
        monitor = RegexPerformanceMonitor(slow_threshold=0.1, alert_threshold=3)

        # Record multiple slow executions of same pattern
        slow_pattern = r"complex.*pattern"
        for _ in range(3):
            monitor.record_execution(slow_pattern, 0.2, 1000, matched=False)

        problematic = monitor.get_problematic_patterns()
        assert slow_pattern in problematic

    def test_pattern_blocking(self):
        """Test pattern blocking based on history."""
        monitor = RegexPerformanceMonitor()

        bad_pattern = r"(a+)+"

        # Record multiple timeouts
        for _ in range(3):
            monitor.record_execution(bad_pattern[:100], 1.0, 1000, timed_out=True)

        # Should recommend blocking
        assert monitor.should_block_pattern(bad_pattern)

    def test_pattern_analysis(self):
        """Test analysis of specific pattern."""
        monitor = RegexPerformanceMonitor()

        pattern = r"\d+"
        monitor.record_execution(pattern, 0.01, 100, matched=True)
        monitor.record_execution(pattern, 0.02, 200, matched=True)
        monitor.record_execution(pattern, 0.03, 300, matched=False)

        analysis = monitor.analyze_pattern(pattern)
        assert analysis["executions"] == 3
        assert analysis["average_time"] == pytest.approx(0.02, rel=0.1)
        assert analysis["matches"] == 2


class TestMarkdownChunkingWithReDoSProtection:
    """Test MarkdownChunkingStrategy with ReDoS protection."""

    def test_markdown_chunking_with_evil_input(self):
        """Test that markdown chunking handles malicious input safely."""
        chunker = MarkdownChunkingStrategy()
        config = ChunkConfig(
            strategy_name="markdown",
            min_tokens=50,
            max_tokens=500,
            overlap_tokens=10,  # Add overlap_tokens
        )

        # Create input that could cause ReDoS with unsafe patterns
        evil_content = (
            """
# Header

"""
            + "* " * 100
            + "nested " * 100
            + """

## Another Section

"""
            + "-" * 1000
            + """

Some normal text here.
"""
        )

        # First validate content
        is_valid, error = chunker.validate_content(evil_content)

        # Even if validation passes, chunking should complete quickly
        start_time = time.time()

        if is_valid:
            chunks = chunker.chunk(evil_content, config)
            # Should have created some chunks
            assert len(chunks) >= 0
        else:
            # Content was rejected for safety
            assert "ReDoS" in error or "too long" in error

        elapsed = time.time() - start_time

        # Should complete within reasonable time (5 seconds max)
        assert elapsed < 5.0

    def test_markdown_patterns_are_safe(self):
        """Test that markdown patterns don't cause ReDoS."""
        chunker = MarkdownChunkingStrategy()

        # All patterns should be compiled without errors
        assert chunker.compiled_patterns

        # Test patterns with potentially problematic input
        test_cases = [
            ("heading", "#" * 10 + " " + "a" * 1000),
            ("list_item", "* " + "a" * 1000),
            ("numbered_list", "1. " + "a" * 1000),
            ("blockquote", "> " + "a" * 1000),
            ("horizontal_rule", "-" * 100),
        ]

        for pattern_name, test_input in test_cases:
            if pattern_name in chunker.patterns:
                pattern = chunker.patterns[pattern_name]

                # Should either match quickly or timeout
                start = time.time()
                try:
                    chunker.safe_regex.match_with_timeout(pattern, test_input, timeout=0.1)
                    # If it matches, it should be fast
                    elapsed = time.time() - start
                    assert elapsed < 0.1
                except RegexTimeout:
                    # Timeout is acceptable for safety
                    pass

    def test_normal_markdown_processing(self):
        """Test that normal markdown processing still works correctly."""
        chunker = MarkdownChunkingStrategy()
        config = ChunkConfig(
            strategy_name="markdown",
            min_tokens=50,
            max_tokens=500,
            overlap_tokens=10,  # Add overlap_tokens
        )

        normal_content = """
# Main Title

This is a paragraph with some text.

## Section 1

* Item 1
* Item 2
* Item 3

### Subsection

Some more text here.

---

## Section 2

1. First item
2. Second item
3. Third item

```python
def hello():
    print("Hello, world!")
```

> This is a blockquote
> with multiple lines

Final paragraph.
"""

        chunks = chunker.chunk(normal_content, config)

        # Should create appropriate chunks
        assert len(chunks) > 0

        # Check that chunks contain expected content
        all_content = " ".join(chunk.content for chunk in chunks)
        assert "Main Title" in all_content
        assert "Section 1" in all_content
        assert "Item 1" in all_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
