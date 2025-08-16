"""
Centralized metadata sanitization for XSS prevention.

This module provides comprehensive HTML escaping and validation
to prevent XSS attacks in metadata fields throughout the application.
"""

import html
import re
from typing import Any


class MetadataSanitizer:
    """Provides comprehensive sanitization for metadata to prevent XSS attacks."""

    # Patterns that indicate potential XSS attempts
    DANGEROUS_PATTERNS = [
        # JavaScript event handlers
        re.compile(r"\bon\w+\s*=", re.IGNORECASE),
        # JavaScript protocols
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"vbscript:", re.IGNORECASE),
        re.compile(r"data:text/html", re.IGNORECASE),
        # Script tags (even encoded)
        re.compile(r"<\s*script", re.IGNORECASE),
        re.compile(r"<\s*/\s*script", re.IGNORECASE),
        # Other dangerous tags
        re.compile(r"<\s*iframe", re.IGNORECASE),
        re.compile(r"<\s*object", re.IGNORECASE),
        re.compile(r"<\s*embed", re.IGNORECASE),
        re.compile(r"<\s*applet", re.IGNORECASE),
        re.compile(r"<\s*meta", re.IGNORECASE),
        re.compile(r"<\s*link", re.IGNORECASE),
        re.compile(r"<\s*style", re.IGNORECASE),
        re.compile(r"<\s*svg", re.IGNORECASE),
        re.compile(r"<\s*marquee", re.IGNORECASE),
        re.compile(r"<\s*form", re.IGNORECASE),
        # CSS expressions
        re.compile(r"expression\s*\(", re.IGNORECASE),
        # Import/require statements (context dependent)
        re.compile(r"import\s+.*from", re.IGNORECASE),
        re.compile(r"require\s*\(", re.IGNORECASE),
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

        # Check for dangerous patterns and reject if found
        for pattern in cls.DANGEROUS_PATTERNS:
            if pattern.search(value):
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
                sanitized[safe_key] = cls.sanitize_string(value)
            elif isinstance(value, int | float | bool):
                # Numeric and boolean values are safe
                sanitized[safe_key] = value
            elif isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                sanitized[safe_key] = cls.sanitize_metadata(value, max_key_length)
            elif isinstance(value, list):
                # Sanitize list items
                sanitized[safe_key] = cls._sanitize_list(value)
            else:
                # Convert other types to string and sanitize
                sanitized[safe_key] = cls.sanitize_string(str(value))

        return sanitized

    @classmethod
    def _sanitize_list(cls, items: list) -> list:
        """
        Sanitize a list of items.

        Args:
            items: List to sanitize

        Returns:
            Sanitized list
        """
        sanitized = []
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

        # Check for dangerous patterns
        return all(not pattern.search(value) for pattern in cls.DANGEROUS_PATTERNS)

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
