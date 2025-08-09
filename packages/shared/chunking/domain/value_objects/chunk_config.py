#!/usr/bin/env python3
"""
Immutable chunking configuration value object.

This module defines the configuration for chunking operations with built-in
validation and business rule enforcement.
"""

from dataclasses import dataclass

from packages.shared.chunking.domain.exceptions import (
    InvalidConfigurationError,
    OverlapConfigurationError,
)


@dataclass(frozen=True)
class ChunkConfig:
    """
    Immutable configuration for chunking operations.

    This value object encapsulates all configuration parameters needed for
    chunking and ensures they satisfy business rules.
    """

    strategy_name: str  # Name of the strategy to use
    min_tokens: int = 100
    max_tokens: int = 1000
    overlap_tokens: int = 50

    # Additional parameters for specific strategies
    separator: str = " "  # For character-based chunking
    preserve_structure: bool = True  # For markdown/document structure
    semantic_threshold: float = 0.7  # For semantic chunking (0.0 to 1.0)
    hierarchy_levels: int = 3  # For hierarchical chunking

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate token constraints
        if self.min_tokens <= 0:
            raise InvalidConfigurationError(
                "Minimum tokens must be positive",
                {"min_tokens": self.min_tokens},
            )

        if self.max_tokens <= 0:
            raise InvalidConfigurationError(
                "Maximum tokens must be positive",
                {"max_tokens": self.max_tokens},
            )

        if self.min_tokens > self.max_tokens:
            raise InvalidConfigurationError(
                f"Minimum tokens ({self.min_tokens}) cannot exceed maximum ({self.max_tokens})",
                {"min_tokens": self.min_tokens, "max_tokens": self.max_tokens},
            )

        # Validate overlap
        if self.overlap_tokens < 0:
            raise InvalidConfigurationError(
                "Overlap tokens cannot be negative",
                {"overlap_tokens": self.overlap_tokens},
            )

        if self.overlap_tokens >= self.min_tokens:
            raise OverlapConfigurationError(self.overlap_tokens, self.min_tokens)

        # Validate semantic threshold
        if not 0.0 <= self.semantic_threshold <= 1.0:
            raise InvalidConfigurationError(
                f"Semantic threshold must be between 0.0 and 1.0, got {self.semantic_threshold}",
                {"semantic_threshold": self.semantic_threshold},
            )

        # Validate hierarchy levels
        if self.hierarchy_levels < 1 or self.hierarchy_levels > 10:
            raise InvalidConfigurationError(
                f"Hierarchy levels must be between 1 and 10, got {self.hierarchy_levels}",
                {"hierarchy_levels": self.hierarchy_levels},
            )

    def calculate_effective_chunk_size(self) -> int:
        """
        Calculate the effective chunk size considering overlap.

        Returns:
            Effective size for subsequent chunks after the first
        """
        return self.max_tokens - self.overlap_tokens

    def estimate_chunks(self, total_tokens: int) -> int:
        """
        Estimate the number of chunks for a given token count.

        Args:
            total_tokens: Total number of tokens in the document

        Returns:
            Estimated number of chunks
        """
        if total_tokens <= self.max_tokens:
            return 1

        # First chunk takes max_tokens, subsequent chunks take effective size
        effective_size = self.calculate_effective_chunk_size()
        remaining_tokens = total_tokens - self.max_tokens

        return 1 + ((remaining_tokens + effective_size - 1) // effective_size)

    def calculate_overlap_percentage(self) -> float:
        """
        Calculate overlap as a percentage of chunk size.

        Returns:
            Overlap percentage (0-100)
        """
        if self.max_tokens == 0:
            return 0.0
        return (self.overlap_tokens / self.max_tokens) * 100
