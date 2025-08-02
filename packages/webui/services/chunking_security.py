#!/usr/bin/env python3
"""
Security validation for chunking operations.

This module provides security validation to prevent malicious inputs
and ensure safe chunking parameters.
"""

import logging
from typing import Any

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
                raise ValidationError(
                    f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
                )

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

    @staticmethod
    def validate_file_paths(file_paths: list[str]) -> None:
        """Validate file paths to prevent directory traversal.

        Args:
            file_paths: List of file paths to validate

        Raises:
            ValidationError: If any path is invalid
        """
        if not isinstance(file_paths, list):
            raise ValidationError("file_paths must be a list")

        if len(file_paths) > 1000:
            raise ValidationError(f"Too many file paths: {len(file_paths)} (maximum 1000 allowed)")

        for path in file_paths:
            if not isinstance(path, str):
                raise ValidationError(f"File path must be string, got {type(path).__name__}")

            # Check for directory traversal attempts
            if ".." in path or path.startswith("/"):
                raise ValidationError(
                    f"Invalid file path: {path}. Absolute paths and parent directory references not allowed."
                )

            # Check length
            if len(path) > 1000:
                raise ValidationError(f"File path too long: {len(path)} characters (maximum 1000 allowed)")

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
