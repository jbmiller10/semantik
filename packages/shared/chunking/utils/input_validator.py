#!/usr/bin/env python3
"""Input validation for chunking operations to prevent ReDoS attacks."""

import logging

from packages.shared.utils.regex_safety import RegexTimeout, safe_regex_findall

logger = logging.getLogger(__name__)


class ChunkingInputValidator:
    """Validate input before regex processing."""

    MAX_DOCUMENT_SIZE = 10_000_000  # 10MB
    MAX_LINE_LENGTH = 10_000
    MAX_WORD_LENGTH = 100

    @classmethod
    def validate_document(cls, text: str) -> None:
        """Validate document is safe for regex processing.

        Args:
            text: Document text to validate

        Raises:
            ValueError: If document is invalid
        """
        # Check size first
        if len(text) > cls.MAX_DOCUMENT_SIZE:
            raise ValueError(f"Document too large: {len(text)} > {cls.MAX_DOCUMENT_SIZE}")

        # Check for binary content early
        if "\x00" in text or "\xff" in text:
            raise ValueError("Document appears to contain binary data")

        # Check line lengths before ReDoS triggers
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if len(line) > cls.MAX_LINE_LENGTH:
                raise ValueError(f"Line {i} too long: {len(line)} > {cls.MAX_LINE_LENGTH}")

        # Check for ReDoS triggers last (most expensive check)
        if cls._contains_redos_triggers(text):
            raise ValueError("Document contains potential ReDoS triggers")

    @classmethod
    def validate_line(cls, line: str) -> bool:
        """Validate a single line for safe processing.

        Args:
            line: Line to validate

        Returns:
            True if line is safe to process
        """
        if len(line) > cls.MAX_LINE_LENGTH:
            return False

        # Check for excessive repetition
        if cls._has_excessive_repetition(line):
            return False

        return True

    @classmethod
    def sanitize_text(cls, text: str) -> str:
        """Remove potential ReDoS triggers from text.

        Args:
            text: Text to sanitize

        Returns:
            Sanitized text
        """
        try:
            # Remove excessive whitespace with timeout protection
            matches = safe_regex_findall(r"\s{100,}", text, timeout=0.5, max_matches=100)
            for match in matches:
                text = text.replace(match, " " * 99)

            # Remove excessive punctuation
            matches = safe_regex_findall(r"[.!?]{10,}", text, timeout=0.5, max_matches=100)
            for match in matches:
                text = text.replace(match, "...")

            # Remove excessive special characters
            matches = safe_regex_findall(r"[*#\-]{20,}", text, timeout=0.5, max_matches=100)
            for match in matches:
                text = text.replace(match, match[0] * 19)

            # Remove excessive repeated characters - simplified pattern
            # Use a simpler approach to avoid complex backreferences
            import re

            for char in set(text):
                if char.isalnum():  # Only process alphanumeric to avoid special char issues
                    # Replace 100+ repeated chars with 99 of them
                    pattern = re.escape(char) + "{100,}"
                    text = re.sub(pattern, char * 99, text)

            return text
        except RegexTimeout:
            logger.warning("Regex timeout during text sanitization, returning original text")
            return text
        except Exception as e:
            logger.error(f"Error during text sanitization: {e}")
            return text

    @classmethod
    def _contains_redos_triggers(cls, text: str) -> bool:
        """Check for common ReDoS trigger patterns.

        Args:
            text: Text to check

        Returns:
            True if text contains potential ReDoS triggers
        """
        # Simple check for excessive repetition without regex
        # Check for runs of the same character
        max_repetition = 0
        current_char = None
        current_count = 0

        for char in text:
            if char == current_char:
                current_count += 1
                max_repetition = max(max_repetition, current_count)
            else:
                current_char = char
                current_count = 1

        # Consider more than 1000 repeated characters as a trigger
        if max_repetition > 1000:
            return True

        # Check for specific patterns using simple string operations
        # Check for excessive 'a' characters (common ReDoS payload)
        if text.count("a") > 1000 and "a" * 1000 in text:
            return True

        # Check for excessive whitespace
        if "  " * 500 in text or "\n" * 500 in text:
            return True

        # Check for excessive special characters in sequence
        special_chars = ["*", "+", "?", "{", "}", "[", "]", "(", ")"]
        return any(char * 100 in text for char in special_chars)

    @classmethod
    def _has_excessive_repetition(cls, text: str) -> bool:
        """Check if text has excessive character repetition.

        Args:
            text: Text to check

        Returns:
            True if text has excessive repetition
        """
        # Check for runs of same character
        max_repetition = 0
        current_char = None
        current_count = 0

        for char in text:
            if char == current_char:
                current_count += 1
                max_repetition = max(max_repetition, current_count)
            else:
                current_char = char
                current_count = 1

        # Consider more than 100 repeated characters excessive
        return max_repetition > 100

    @classmethod
    def estimate_processing_risk(cls, text: str) -> str:
        """Estimate the risk level of processing this text with regex.

        Args:
            text: Text to analyze

        Returns:
            Risk level: "low", "medium", or "high"
        """
        risk_score = 0

        # Check document size
        if len(text) > cls.MAX_DOCUMENT_SIZE / 2:
            risk_score += 2
        elif len(text) > cls.MAX_DOCUMENT_SIZE / 4:
            risk_score += 1

        # Check line lengths
        lines = text.split("\n")
        long_lines = sum(1 for line in lines if len(line) > cls.MAX_LINE_LENGTH / 2)
        if long_lines > len(lines) * 0.1:
            risk_score += 1

        # Check for patterns that might cause backtracking using string operations
        # Look for sequences of quantifier characters
        quantifier_sequences = 0
        for i in range(len(text) - 2):
            if text[i : i + 3] in ["***", "+++", "???", "*+?", "+*?", "?*+"]:
                quantifier_sequences += 1
                if quantifier_sequences > 5:  # Multiple sequences found
                    risk_score += 1
                    break

        # Check for deeply nested structures
        if text.count("(") > 100 or text.count("[") > 100:
            risk_score += 1

        # Determine risk level
        if risk_score >= 3:
            return "high"
        if risk_score >= 1:
            return "medium"
        return "low"
