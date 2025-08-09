#!/usr/bin/env python3
"""
Domain-specific exceptions for chunking operations.

These exceptions represent business rule violations and domain errors,
not technical infrastructure failures.
"""

from typing import Any


class ChunkingDomainError(Exception):
    """Base exception for all chunking domain errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize domain error with message and optional details."""
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidStateError(ChunkingDomainError):
    """Raised when an operation is attempted in an invalid state."""


class InvalidChunkError(ChunkingDomainError):
    """Raised when a chunk doesn't meet business requirements."""


class InvalidConfigurationError(ChunkingDomainError):
    """Raised when chunking configuration violates business rules."""


class DocumentTooLargeError(ChunkingDomainError):
    """Raised when document exceeds size limits."""

    def __init__(self, size: int, max_size: int) -> None:
        """Initialize with size information."""
        super().__init__(
            f"Document size {size} exceeds maximum {max_size}",
            {"size": size, "max_size": max_size},
        )
        self.size = size
        self.max_size = max_size


class ChunkSizeViolationError(ChunkingDomainError):
    """Raised when chunk size violates min/max constraints."""

    def __init__(self, chunk_size: int, min_size: int, max_size: int) -> None:
        """Initialize with size constraints."""
        super().__init__(
            f"Chunk size {chunk_size} not in range [{min_size}, {max_size}]",
            {"chunk_size": chunk_size, "min_size": min_size, "max_size": max_size},
        )
        self.chunk_size = chunk_size
        self.min_size = min_size
        self.max_size = max_size


class OverlapConfigurationError(ChunkingDomainError):
    """Raised when overlap configuration is invalid."""

    def __init__(self, overlap: int, chunk_size: int) -> None:
        """Initialize with overlap information."""
        super().__init__(
            f"Overlap {overlap} must be less than chunk size {chunk_size}",
            {"overlap": overlap, "chunk_size": chunk_size},
        )
        self.overlap = overlap
        self.chunk_size = chunk_size


class StrategyNotFoundError(ChunkingDomainError, ValueError):
    """Raised when a requested chunking strategy is not found."""

    def __init__(self, strategy_name: str) -> None:
        """Initialize with strategy name."""
        super().__init__(
            f"Strategy '{strategy_name}' not found",
            {"strategy_name": strategy_name},
        )
        self.strategy_name = strategy_name
