#!/usr/bin/env python3
"""
Security validation for chunking operations.

This module provides security validation to prevent malicious inputs
and ensure safe chunking parameters.
"""

import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import unquote

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""


class ChunkingSecurityValidator:
    """Validate chunking requests for security."""

    # Configurable limits
    MAX_CHUNK_SIZE = 10000
    MIN_CHUNK_SIZE = 50
    MAX_DOCUMENT_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_CHUNKS_PER_DOCUMENT = 50000
    MAX_PREVIEW_SIZE = 1 * 1024 * 1024  # 1MB for preview

    @staticmethod
    def validate_chunk_params(params: dict[str, Any]) -> None:
        """Validate chunking parameters are within safe bounds.

        Args:
            params: Parameters dictionary to validate

        Raises:
            ValidationError: If parameters are invalid
        """
        # Validate chunk_size if present
        chunk_size = params.get("chunk_size")
        if chunk_size is not None:
            if not isinstance(chunk_size, int):
                raise ValidationError(f"chunk_size must be an integer, got {type(chunk_size).__name__}")

            if not (ChunkingSecurityValidator.MIN_CHUNK_SIZE <= chunk_size <= ChunkingSecurityValidator.MAX_CHUNK_SIZE):
                raise ValidationError(
                    f"chunk_size must be between {ChunkingSecurityValidator.MIN_CHUNK_SIZE} "
                    f"and {ChunkingSecurityValidator.MAX_CHUNK_SIZE}, got {chunk_size}"
                )

        # Validate chunk_overlap if present
        chunk_overlap = params.get("chunk_overlap")
        if chunk_overlap is not None:
            if not isinstance(chunk_overlap, int):
                raise ValidationError(f"chunk_overlap must be an integer, got {type(chunk_overlap).__name__}")

            if chunk_overlap < 0:
                raise ValidationError(f"chunk_overlap must be non-negative, got {chunk_overlap}")

            # Validate overlap against chunk size
            if chunk_size and chunk_overlap >= chunk_size:
                raise ValidationError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")

        # Validate other parameters based on strategy
        # For semantic chunking
        if "breakpoint_percentile_threshold" in params:
            threshold = params["breakpoint_percentile_threshold"]
            if not isinstance(threshold, int | float):
                raise ValidationError(
                    f"breakpoint_percentile_threshold must be a number, got {type(threshold).__name__}"
                )
            if not 0 <= threshold <= 100:
                raise ValidationError(f"breakpoint_percentile_threshold must be between 0 and 100, got {threshold}")

        # For hierarchical chunking
        if "chunk_sizes" in params:
            chunk_sizes = params["chunk_sizes"]
            if not isinstance(chunk_sizes, list):
                raise ValidationError(f"chunk_sizes must be a list, got {type(chunk_sizes).__name__}")
            if not all(isinstance(size, int) and size > 0 for size in chunk_sizes):
                raise ValidationError("All chunk_sizes must be positive integers")
            if len(chunk_sizes) > 5:
                raise ValidationError("Maximum 5 hierarchical levels allowed")

        logger.debug(f"Validated chunking params: {params}")

    @staticmethod
    def validate_document_size(size: int, is_preview: bool = False) -> None:
        """Prevent processing of oversized documents.

        Args:
            size: Document size in bytes
            is_preview: Whether this is a preview request

        Raises:
            ValidationError: If document is too large
        """
        max_size = (
            ChunkingSecurityValidator.MAX_PREVIEW_SIZE if is_preview else ChunkingSecurityValidator.MAX_DOCUMENT_SIZE
        )

        if size > max_size:
            size_mb = size / (1024 * 1024)
            max_size_mb = max_size / (1024 * 1024)
            raise ValidationError(f"Document too large: {size_mb:.1f}MB exceeds maximum of {max_size_mb:.1f}MB")

        logger.debug(f"Validated document size: {size} bytes")

    @staticmethod
    def validate_strategy_name(strategy: str) -> None:
        """Validate strategy name to prevent injection.

        Args:
            strategy: Strategy name to validate

        Raises:
            ValidationError: If strategy name is invalid
        """
        # Allow only alphanumeric and underscore
        if not strategy.replace("_", "").isalnum():
            raise ValidationError(
                f"Invalid strategy name: {strategy}. Only alphanumeric characters and underscores allowed."
            )

        # Check length
        if len(strategy) > 50:
            raise ValidationError(f"Strategy name too long: {len(strategy)} characters (maximum 50 allowed)")

    # Path traversal patterns - comprehensive list of dangerous patterns
    TRAVERSAL_PATTERNS = [
        # Unix/Linux patterns
        re.compile(r"\.\.\/"),  # ../
        re.compile(r"\.\.\\"),  # ..\
        re.compile(r"^\/"),  # Absolute path /
        re.compile(r"^~"),  # Home directory ~
        # Windows patterns
        re.compile(r"^[a-zA-Z]:"),  # Drive letters C:
        re.compile(r"^\\\\"),  # UNC paths \\server
        re.compile(r"^\\"),  # Windows absolute \
        # Encoded variants (case-insensitive)
        re.compile(r"%2e%2e[%2f%5c]", re.I),  # URL encoded ../ or ..\
        re.compile(r"%252e%252e", re.I),  # Double encoded
        re.compile(r"%25252e%25252e", re.I),  # Triple encoded
        re.compile(r"%00", re.I),  # URL encoded null byte
        re.compile(r"%25%30%30", re.I),  # Double encoded null byte
        # Unicode/special character attacks
        re.compile(r"[\u0000]"),  # Null bytes
        re.compile(r"[\u202e]"),  # Right-to-left override
        re.compile(r"[\ufeff]"),  # Zero-width no-break space
        re.compile(r"[\uff0e\uff0f]"),  # Full-width dot and slash
    ]

    # Additional dangerous character patterns
    DANGEROUS_CHARS = [
        "\x00",  # Null byte
        "\r",  # Carriage return
        "\n",  # Newline
        "\u202e",  # Right-to-left override
        "\ufeff",  # Zero-width no-break space
    ]

    @staticmethod
    def validate_file_paths(file_paths: list[str], base_dir: str | None = None) -> None:
        """Validate file paths to prevent directory traversal attacks.

        This method implements comprehensive OWASP-compliant path validation including:
        - Multiple URL encoding layer detection
        - Unicode normalization
        - Null byte detection
        - Windows and Unix path traversal patterns
        - Symlink resolution
        - Base directory containment verification

        Args:
            file_paths: List of file paths to validate
            base_dir: Optional base directory for containment verification

        Raises:
            ValidationError: If any path is invalid or potentially malicious
        """
        if not isinstance(file_paths, list):
            raise ValidationError("file_paths must be a list")

        if len(file_paths) > 1000:
            raise ValidationError(f"Too many file paths: {len(file_paths)} (maximum 1000 allowed)")

        for original_path in file_paths:
            if not isinstance(original_path, str):
                raise ValidationError(f"File path must be string, got {type(original_path).__name__}")

            # Check for empty string
            if not original_path:
                raise ValidationError("Invalid file path")

            # Length check on original path
            if len(original_path) > 1000:
                raise ValidationError(f"File path too long: {len(original_path)} characters (maximum 1000 allowed)")

            # Step 1: Decode multiple URL encoding layers (up to 3 rounds)
            decoded_path = original_path
            for _ in range(3):
                try:
                    new_decoded = unquote(decoded_path, errors="strict")
                    if new_decoded == decoded_path:
                        break  # No more encoding layers
                    decoded_path = new_decoded
                except Exception as e:
                    # Invalid encoding - reject
                    raise ValidationError("Invalid file path") from e

            # Step 2: Normalize Unicode to prevent homograph attacks
            try:
                normalized_path = unicodedata.normalize("NFC", decoded_path)
                # Also convert full-width characters to ASCII equivalents
                # Full-width dot (．) to regular dot (.)
                normalized_path = normalized_path.replace("\uff0e", ".")
                # Full-width slash (／) to regular slash (/)
                normalized_path = normalized_path.replace("\uff0f", "/")
                # Full-width backslash (＼) to regular backslash (\)
                normalized_path = normalized_path.replace("\uff3c", "\\")
            except Exception as e:
                raise ValidationError("Invalid file path") from e

            # Step 3: Check for null bytes and dangerous characters
            for char in ChunkingSecurityValidator.DANGEROUS_CHARS:
                if char in normalized_path:
                    raise ValidationError("Invalid file path")

            # Step 4: Check against all traversal patterns
            for pattern in ChunkingSecurityValidator.TRAVERSAL_PATTERNS:
                if pattern.search(normalized_path):
                    raise ValidationError("Invalid file path")

            # Step 5: Additional checks for sneaky patterns
            # Check for just dots - but allow dots in filenames
            if normalized_path in [".", "..", "...", "...."]:
                raise ValidationError("Invalid file path")

            # Check for path starting/ending with dots (suspicious)
            if re.search(r"^\.{2,}[/\\]", normalized_path) or re.search(r"[/\\]\.{2,}$", normalized_path):
                raise ValidationError("Invalid file path")

            # Check for backslash in Unix environments (suspicious)
            if os.name != "nt" and "\\" in normalized_path:
                raise ValidationError("Invalid file path")

            # Step 6: If base_dir provided, verify path stays within it
            if base_dir:
                try:
                    base_path = Path(base_dir).resolve()
                    # Construct the full path and resolve it
                    full_path = (base_path / normalized_path).resolve()

                    # Verify the resolved path is within the base directory
                    # Use os.path.commonpath for reliable containment check
                    try:
                        common = os.path.commonpath([str(base_path), str(full_path)])
                        if common != str(base_path):
                            raise ValidationError("Invalid file path")
                    except ValueError as e:
                        # Paths are on different drives (Windows) or invalid
                        raise ValidationError("Invalid file path") from e

                except Exception as e:
                    # Any path resolution error is a security risk
                    raise ValidationError("Invalid file path") from e

            # Log successful validation (without revealing the path)
            logger.debug("File path validated successfully")

    @staticmethod
    def sanitize_text_for_preview(text: str, max_length: int = 200) -> str:
        """Sanitize text for safe preview display.

        Args:
            text: Text to sanitize
            max_length: Maximum length for preview

        Returns:
            Sanitized text safe for display
        """
        # Remove any potential HTML/script tags
        import re

        text = re.sub(r"<[^>]+>", "", text)

        # Limit length
        if len(text) > max_length:
            text = text[:max_length] + "..."

        # Escape special characters for JSON safety
        text = text.replace("\\", "\\\\")
        text = text.replace('"', '\\"')
        text = text.replace("\n", "\\n")
        text = text.replace("\r", "\\r")
        text = text.replace("\t", "\\t")

        return text  # noqa: RET504

    @staticmethod
    def validate_collection_config(config: dict[str, Any]) -> None:
        """Validate complete collection chunking configuration.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValidationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValidationError(f"Config must be a dictionary, got {type(config).__name__}")

        # Validate strategy
        strategy = config.get("strategy")
        if not strategy:
            raise ValidationError("Configuration must include 'strategy' field")

        ChunkingSecurityValidator.validate_strategy_name(strategy)

        # Validate params if present
        params = config.get("params", {})
        if not isinstance(params, dict):
            raise ValidationError(f"params must be a dictionary, got {type(params).__name__}")

        ChunkingSecurityValidator.validate_chunk_params(params)

        # Check for unknown fields (potential injection)
        allowed_fields = {"strategy", "params", "metadata"}
        unknown_fields = set(config.keys()) - allowed_fields
        if unknown_fields:
            logger.warning(f"Unknown fields in config will be ignored: {unknown_fields}")

    @staticmethod
    def estimate_memory_usage(
        text_length: int,
        chunk_size: int,
        strategy: str,
    ) -> int:
        """Estimate memory usage for chunking operation.

        Args:
            text_length: Length of text in characters
            chunk_size: Chunk size parameter
            strategy: Chunking strategy name

        Returns:
            Estimated memory usage in bytes
        """
        # Base memory for text
        base_memory = text_length * 4  # Assume 4 bytes per character

        # Strategy-specific multipliers
        multipliers = {
            "character": 1.5,
            "recursive": 2.0,
            "markdown": 2.5,
            "semantic": 5.0,  # Embeddings require more memory
            "hierarchical": 3.0,
            "hybrid": 2.5,
        }

        multiplier = multipliers.get(strategy, 2.0)

        # Estimate based on chunk count
        estimated_chunks = max(1, text_length // chunk_size)
        chunk_overhead = estimated_chunks * 1024  # ~1KB per chunk object

        total_memory = int((base_memory * multiplier) + chunk_overhead)

        logger.debug(
            f"Estimated memory usage: {total_memory / (1024 * 1024):.1f}MB "
            f"for {text_length} chars with {strategy} strategy"
        )

        return total_memory
