"""
Centralized metadata sanitization for XSS prevention.

This module provides comprehensive HTML escaping and validation
to prevent XSS attacks in metadata fields throughout the application.
"""

import html
from typing import Any

from packages.shared.utils.regex_safety import safe_regex_search


class MetadataSanitizer:
    """Provides comprehensive sanitization for metadata to prevent XSS attacks."""

    # Patterns that indicate potential XSS attempts - simplified to avoid ReDoS
    DANGEROUS_PATTERNS = [
        # JavaScript event handlers
        r"\bon\w+\s*=",
        # JavaScript protocols
        r"javascript:",
        r"vbscript:",
        r"data:text/html",
        # Script tags (even encoded) - bounded whitespace
        r"<\s{0,5}script",
        r"<\s{0,5}/\s{0,5}script",
        # Other dangerous tags - bounded whitespace
        r"<\s{0,5}iframe",
        r"<\s{0,5}object",
        r"<\s{0,5}embed",
        r"<\s{0,5}applet",
        r"<\s{0,5}meta",
        r"<\s{0,5}link",
        r"<\s{0,5}style",
        r"<\s{0,5}svg",
        r"<\s{0,5}marquee",
        r"<\s{0,5}form",
        # CSS expressions - bounded whitespace
        r"expression\s{0,5}\(",
        # Import/require statements - simplified and bounded
        r"import\s{1,10}[^\s]{1,50}\s{1,10}from",
        r"require\s{0,5}\(",
    ]

    @classmethod
    def sanitize_string(cls, value: str, max_length: int = 1000) -> str:
        """
        Sanitize a string value for safe HTML display.

        Args:
            value: The string to sanitize
            max_length: Maximum allowed length (default 1000)

        Returns:
            Sanitized string safe for HTML display
        """
        if not value:
            return ""

        # Remove null bytes
        value = value.replace("\x00", "")

        # Truncate if too long
        if len(value) > max_length:
            value = value[:max_length]

        # Check for dangerous patterns with timeout protection
        value_lower = value.lower()
        for pattern in cls.DANGEROUS_PATTERNS:
            # Use safe regex with 0.5 second timeout for each pattern
            if safe_regex_search(pattern, value_lower, timeout=0.5):
                # Return empty string for dangerous content
                # In production, you might want to log this
                return "[Content removed for security]"

        # Comprehensive HTML escaping
        # quote=True escapes quotes as well, preventing attribute injection
        return html.escape(value, quote=True)

    @classmethod
    def sanitize_metadata(cls, metadata: dict[str, Any], max_key_length: int = 100) -> dict[str, Any]:
        """
        Sanitize a metadata dictionary to prevent XSS attacks.

        Args:
            metadata: The metadata dictionary to sanitize
            max_key_length: Maximum allowed key length (default 100)

        Returns:
            Sanitized metadata dictionary
        """
        if not metadata:
            return {}

        sanitized = {}
        for key, value in metadata.items():
            # Sanitize and limit key length
            key_str = str(key)
            if len(key_str) > max_key_length:
                continue

            # Sanitize the key itself
            safe_key = cls.sanitize_string(key_str, max_key_length)
            if not safe_key or safe_key == "[Content removed for security]":
                continue

            # Sanitize the value based on type
            if isinstance(value, str):
                safe_value: Any = cls.sanitize_string(value)
            elif isinstance(value, int | float | bool):
                # Numeric and boolean values are safe
                safe_value = value
            elif isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                safe_value = cls.sanitize_metadata(value, max_key_length)
            elif isinstance(value, list):
                # Sanitize list items
                safe_value = cls._sanitize_list(value)
            else:
                # Convert other types to string and sanitize
                safe_value = cls.sanitize_string(str(value))

            sanitized[safe_key] = safe_value

        return sanitized

    @classmethod
    def _sanitize_list(cls, items: list) -> list[Any]:
        """
        Sanitize a list of items.

        Args:
            items: List to sanitize

        Returns:
            Sanitized list
        """
        sanitized: list[Any] = []
        for item in items[:100]:  # Limit list size to prevent DOS
            if isinstance(item, str):
                sanitized.append(cls.sanitize_string(item))
            elif isinstance(item, int | float | bool):
                sanitized.append(item)
            elif isinstance(item, dict):
                sanitized.append(cls.sanitize_metadata(item))
            elif isinstance(item, list):
                sanitized.append(cls._sanitize_list(item))
            else:
                sanitized.append(cls.sanitize_string(str(item)))
        return sanitized

    @classmethod
    def validate_no_xss(cls, value: str) -> bool:
        """
        Validate that a string contains no XSS patterns.

        Args:
            value: String to validate

        Returns:
            True if safe, False if XSS patterns detected
        """
        if not value:
            return True

        # Check for dangerous patterns using safe regex search
        value_lower = value.lower()
        for pattern in cls.DANGEROUS_PATTERNS:
            if safe_regex_search(pattern, value_lower, timeout=0.5):
                return False
        return True

    @classmethod
    def escape_for_json(cls, value: str) -> str:
        """
        Escape a string for safe inclusion in JSON responses.

        Args:
            value: String to escape

        Returns:
            Escaped string safe for JSON
        """
        if not value:
            return ""

        # First apply HTML escaping
        escaped = cls.sanitize_string(value)

        # Additional JSON-specific escaping
        # Escape forward slashes to prevent </script> injection in JSON
        return escaped.replace("/", "\\/")
