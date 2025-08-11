"""
Comprehensive input validation for chunking operations.

This module provides security-focused validation for all chunking inputs
to prevent injection attacks, resource exhaustion, and other vulnerabilities.
"""

import re
from typing import Any

from packages.shared.chunking.infrastructure.exceptions import ValidationError


class ChunkingInputValidator:
    """Validates and sanitizes inputs for chunking operations."""

    # Dangerous patterns that could indicate injection attempts
    DANGEROUS_PATTERNS = [
        r"<script[^>]*>.*?</script>",  # Script tags
        r"javascript:",  # JavaScript protocol
        r"on\w+\s*=",  # Event handlers
        r"data:text/html",  # Data URLs with HTML
        r"<iframe[^>]*>",  # IFrame tags
        r"<object[^>]*>",  # Object tags
        r"<embed[^>]*>",  # Embed tags
        r"<applet[^>]*>",  # Applet tags
        r"<meta[^>]*>",  # Meta tags
        r"<link[^>]*>",  # Link tags
        r"vbscript:",  # VBScript protocol
        r"<svg[^>]*onload",  # SVG with onload
        r"<marquee[^>]*>",  # Marquee tags (annoying)
        r"<form[^>]*>",  # Form tags
        r"expression\s*\(",  # CSS expressions
        r"import\s+\{.*\}\s+from",  # ES6 imports (context dependent)
        r"require\s*\(",  # CommonJS require (context dependent)
    ]

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b.*\b(from|into|where|table|database)\b",
        r"(;|\||&&)\s*(shutdown|drop|delete|truncate)",
        r'(\'|")\s*or\s+[\'"]*\d+[\'"]*\s*=\s*[\'"]*\d+',
        r'(\'|")\s*or\s+(\'|").*\1\s*=\s*\1',
        r'(\'|");?\s*(drop|delete|truncate|alter|create|insert|update|union|select)',
    ]

    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|]\s*(ls|cat|grep|find|curl|wget|bash|sh|python|perl|ruby|php)",
        r"`[^`]*`",  # Backticks
        r"\$\([^)]+\)",  # Command substitution
        r"\$\{[^}]+\}",  # Variable substitution
        r">\s*/dev/null",  # Output redirection
        r"2>&1",  # Error redirection
    ]

    @classmethod
    def validate_config(cls, strategy: str, config: dict[str, Any] | None) -> tuple[bool, list[str]]:
        """Validate chunking configuration.

        Args:
            strategy: The chunking strategy
            config: Configuration dictionary

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []

        if not strategy:
            errors.append("Strategy is required")
            return False, errors

        # Basic validation for common config parameters
        if config:
            if "chunk_size" in config:
                chunk_size = config["chunk_size"]
                if not isinstance(chunk_size, int) or chunk_size <= 0:
                    errors.append("chunk_size must be a positive integer")
                elif chunk_size > 100000:
                    errors.append("chunk_size too large (max 100000)")

            if "chunk_overlap" in config:
                overlap = config["chunk_overlap"]
                if not isinstance(overlap, int) or overlap < 0:
                    errors.append("chunk_overlap must be a non-negative integer")

                # Check overlap is less than chunk_size if both exist
                if "chunk_size" in config and isinstance(config["chunk_size"], int) and overlap >= config["chunk_size"]:
                    errors.append("chunk_overlap must be less than chunk_size")

        return len(errors) == 0, errors

    @classmethod
    def validate_content(cls, content: str, correlation_id: str) -> None:
        """
        Validate content for security issues.

        Args:
            content: The content to validate
            correlation_id: Correlation ID for error tracking

        Raises:
            ValidationError: If validation fails
        """
        if not content:
            return

        # Check for null bytes
        if "\x00" in content:
            raise ValidationError(
                field="content",
                value="<content with null bytes>",
                reason="Content contains null bytes",
                correlation_id=correlation_id,
            )

        # Check for dangerous HTML/JS patterns
        content_lower = content.lower()
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE | re.DOTALL):
                raise ValidationError(
                    field="content",
                    value=f"<content matching pattern: {pattern}>",
                    reason="Content contains forbidden patterns",
                    correlation_id=correlation_id,
                )

        # Check for SQL injection patterns
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                raise ValidationError(
                    field="content",
                    value=f"<content with SQL pattern: {pattern}>",
                    reason="Content contains SQL-like patterns",
                    correlation_id=correlation_id,
                )

        # Check for command injection patterns (be careful with code files)
        for pattern in cls.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, content, re.MULTILINE):
                # For now, just log a warning - code files may contain these
                pass  # Consider context before rejecting

        # Check encoding
        try:
            content.encode("utf-8")
        except UnicodeError as e:
            raise ValidationError(
                field="content",
                value="<content with invalid encoding>",
                reason=f"Encoding error: {str(e)}",
                correlation_id=correlation_id,
                cause=e,
            ) from e

    @classmethod
    def validate_chunk_size(cls, chunk_size: int, correlation_id: str) -> None:
        """
        Validate chunk size is within acceptable bounds.

        Args:
            chunk_size: The chunk size to validate
            correlation_id: Correlation ID for error tracking

        Raises:
            ValidationError: If validation fails
        """
        if chunk_size < 100:
            raise ValidationError(
                field="chunk_size",
                value=chunk_size,
                reason="Minimum chunk size is 100",
                correlation_id=correlation_id,
            )

        if chunk_size > 4096:
            raise ValidationError(
                field="chunk_size",
                value=chunk_size,
                reason="Maximum chunk size is 4096",
                correlation_id=correlation_id,
            )

    @classmethod
    def validate_overlap(cls, overlap: int, chunk_size: int, correlation_id: str) -> None:
        """
        Validate chunk overlap is valid.

        Args:
            overlap: The overlap size
            chunk_size: The chunk size
            correlation_id: Correlation ID for error tracking

        Raises:
            ValidationError: If validation fails
        """
        if overlap < 0:
            raise ValidationError(
                field="overlap",
                value=overlap,
                reason="Overlap must be >= 0",
                correlation_id=correlation_id,
            )

        if overlap >= chunk_size:
            raise ValidationError(
                field="overlap",
                value=overlap,
                reason=f"Overlap must be less than chunk size ({chunk_size})",
                correlation_id=correlation_id,
            )

    @classmethod
    def validate_file_type(cls, file_type: str, correlation_id: str) -> None:
        """
        Validate file type is acceptable.

        Args:
            file_type: The file type/extension
            correlation_id: Correlation ID for error tracking

        Raises:
            ValidationError: If validation fails
        """
        # Prevent path traversal in file type
        if "/" in file_type or "\\" in file_type or ".." in file_type:
            raise ValidationError(
                field="file_type",
                value=file_type,
                reason="File type contains invalid characters",
                correlation_id=correlation_id,
            )

        # Limit file type length
        if len(file_type) > 10:
            raise ValidationError(
                field="file_type",
                value=file_type,
                reason="File type must be <= 10 characters",
                correlation_id=correlation_id,
            )

    @classmethod
    def sanitize_metadata(cls, metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Sanitize metadata dictionary to remove potentially dangerous values.

        Args:
            metadata: The metadata to sanitize

        Returns:
            Sanitized metadata dictionary
        """
        if not metadata:
            return {}

        sanitized = {}
        for key, value in metadata.items():
            # Limit key length
            if len(str(key)) > 100:
                continue

            # Sanitize string values
            if isinstance(value, str):
                # Remove null bytes
                value = value.replace("\x00", "")
                # Limit string length
                if len(value) > 1000:
                    value = value[:1000]
                # Basic HTML escape
                value = value.replace("<", "&lt;").replace(">", "&gt;")

            sanitized[key] = value

        return sanitized

    @classmethod
    def validate_priority(cls, priority: int, correlation_id: str) -> None:
        """
        Validate operation priority.

        Args:
            priority: The priority value
            correlation_id: Correlation ID for error tracking

        Raises:
            ValidationError: If validation fails
        """
        if priority < 1 or priority > 10:
            raise ValidationError(
                field="priority",
                value=priority,
                reason="Priority must be between 1 and 10",
                correlation_id=correlation_id,
            )
