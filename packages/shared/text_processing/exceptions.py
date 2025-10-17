"""
Exception hierarchy for text processing and chunking operations.

This module defines custom exceptions for better error handling and debugging
in the text processing pipeline.
"""


class TextProcessingError(Exception):
    """Base exception for all text processing errors."""


class ChunkingError(TextProcessingError):
    """Base exception for chunking-related errors."""


class EmbeddingError(TextProcessingError):
    """Base exception for embedding-related errors."""


class ValidationError(TextProcessingError):
    """Base exception for validation errors."""


# Specific chunking errors
class ChunkSizeError(ChunkingError):
    """Raised when chunk size constraints are violated."""


class HierarchyDepthError(ChunkingError):
    """Raised when hierarchy depth exceeds maximum allowed."""


class TextLengthError(ChunkingError):
    """Raised when input text exceeds maximum allowed length."""


# Specific embedding errors
class TransientEmbeddingError(EmbeddingError):
    """Transient errors that may succeed on retry (e.g., OOM, temporary API failures)."""


class PermanentEmbeddingError(EmbeddingError):
    """Permanent errors that won't succeed on retry (e.g., invalid model, corrupted data)."""


class DimensionMismatchError(EmbeddingError):
    """Raised when embedding dimensions don't match expected values."""


class EmbeddingServiceNotInitializedError(EmbeddingError):
    """Raised when embedding service is used before initialization."""


# Specific validation errors
class ConfigValidationError(ValidationError):
    """Raised when configuration validation fails."""


class RegexTimeoutError(ValidationError):
    """Raised when regex execution exceeds timeout limit."""


# Factory errors
class ChunkerCreationError(ChunkingError):
    """Raised when chunker creation fails in the factory."""


class UnknownStrategyError(ChunkerCreationError):
    """Raised when an unknown chunking strategy is requested."""
