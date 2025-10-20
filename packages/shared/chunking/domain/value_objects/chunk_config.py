#!/usr/bin/env python3
"""
Immutable chunking configuration value object.

This module defines the configuration for chunking operations with built-in
validation and business rule enforcement.
"""

from typing import Any

from packages.shared.chunking.domain.exceptions import InvalidConfigurationError


class ChunkConfig:
    """
    Immutable configuration for chunking operations.

    This value object encapsulates all configuration parameters needed for
    chunking and ensures they satisfy business rules.
    """

    # Whitelist of allowed additional parameters for security
    ALLOWED_ADDITIONAL_PARAMS = {
        "chunk_size",  # Legacy parameter alias
        "chunk_overlap",  # Legacy parameter alias
        "encoding",  # Text encoding parameter
        "language",  # Language hint for processing
        "preserve_whitespace",  # Whether to preserve whitespace
        "max_retries",  # Retry configuration
        "timeout",  # Operation timeout
        "batch_size",  # Batch processing size
        "metadata",  # Additional metadata
        "similarity_threshold",  # Threshold for semantic similarity (alias for semantic_threshold)
        "hierarchy_level",  # Number of hierarchy levels (alias for hierarchy_levels)
        "strategies",  # List of strategies for hybrid chunking
        "weights",  # Weights for hybrid chunking strategies
        "adaptive_weights",  # Enable adaptive weight adjustment for hybrid chunking
        "chunk_sizes",  # Hierarchy sizes for hierarchical chunking
        "custom_attributes",  # Custom attributes to pass through to chunks
    }

    def __init__(
        self,
        strategy_name: str,
        min_tokens: int = 100,
        max_tokens: int = 1000,
        overlap_tokens: int = 50,
        separator: str = " ",
        preserve_structure: bool = True,
        semantic_threshold: float = 0.7,
        hierarchy_levels: int = 3,
        similarity_threshold: float | None = None,
        hierarchy_level: int | None = None,
        strategies: list[str] | None = None,
        weights: dict[str, float] | None = None,
        **kwargs: Any,
    ):
        """Initialize configuration with validation.

        Args:
            strategy_name: Name of the strategy to use
            min_tokens: Minimum tokens per chunk
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Overlap between chunks
            separator: Separator for character-based chunking
            preserve_structure: Whether to preserve document structure
            semantic_threshold: Threshold for semantic chunking (0.0 to 1.0)
            hierarchy_levels: Number of hierarchy levels for hierarchical chunking
            similarity_threshold: Alias for semantic_threshold
            hierarchy_level: Alias for hierarchy_levels
            strategies: List of strategies for hybrid chunking
            weights: Weights for hybrid chunking strategies
            **kwargs: Additional parameters stored in additional_params
        """
        # Set core attributes
        self.strategy_name = strategy_name
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.separator = separator
        self.preserve_structure = preserve_structure

        # Handle aliases and store extra params
        self.additional_params: dict[str, Any] = {}

        # Handle similarity_threshold alias
        if similarity_threshold is not None:
            self.semantic_threshold = similarity_threshold
            self.additional_params["similarity_threshold"] = similarity_threshold
        else:
            self.semantic_threshold = semantic_threshold

        # Handle hierarchy_level alias
        if hierarchy_level is not None:
            self.hierarchy_levels = hierarchy_level
            self.additional_params["hierarchy_level"] = hierarchy_level
        else:
            self.hierarchy_levels = hierarchy_levels

        # Set optional parameters
        self.similarity_threshold = similarity_threshold
        self.hierarchy_level = hierarchy_level
        self.strategies = strategies
        self.weights = weights

        # Validate and store additional kwargs in additional_params
        # Security: Only allow whitelisted parameters to prevent injection
        for key, value in kwargs.items():
            if key not in self.ALLOWED_ADDITIONAL_PARAMS:
                raise InvalidConfigurationError(
                    f"Unknown configuration parameter '{key}'. Allowed additional parameters: {', '.join(sorted(self.ALLOWED_ADDITIONAL_PARAMS))}",
                    {"parameter": key, "value": value},
                )
            self.additional_params[key] = value

        # Run validation
        self._validate()

    def _validate(self) -> None:
        """Validate configuration after initialization."""

        # Validate strategy name
        if not self.strategy_name:
            raise InvalidConfigurationError(
                "strategy_name cannot be empty",
                {"strategy_name": self.strategy_name},
            )

        # Validate token constraints
        if self.min_tokens <= 0:
            raise InvalidConfigurationError(
                "min_tokens must be positive",
                {"min_tokens": self.min_tokens},
            )

        if self.max_tokens <= 0:
            raise InvalidConfigurationError(
                "max_tokens must be positive",
                {"max_tokens": self.max_tokens},
            )

        if self.min_tokens > self.max_tokens:
            raise InvalidConfigurationError(
                "min_tokens cannot be greater than max_tokens",
                {"min_tokens": self.min_tokens, "max_tokens": self.max_tokens},
            )

        # Validate overlap
        if self.overlap_tokens < 0:
            raise InvalidConfigurationError(
                "overlap_tokens cannot be negative",
                {"overlap_tokens": self.overlap_tokens},
            )

        if self.overlap_tokens >= self.min_tokens:
            raise InvalidConfigurationError(
                "overlap_tokens must be less than min_tokens",
                {"overlap_tokens": self.overlap_tokens, "min_tokens": self.min_tokens},
            )

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

        # Validate hybrid chunking parameters if provided
        if self.strategies is not None and len(self.strategies) < 1:
            raise InvalidConfigurationError(
                "At least one strategy must be specified for hybrid chunking",
                {"strategies": self.strategies},
            )

        if self.weights is not None:
            total_weight = sum(self.weights.values())
            if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
                raise InvalidConfigurationError(
                    f"Weights must sum to 1.0, got {total_weight}",
                    {"weights": self.weights},
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

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        result = {
            "strategy_name": self.strategy_name,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "overlap_tokens": self.overlap_tokens,
            "separator": self.separator,
            "preserve_structure": self.preserve_structure,
            "semantic_threshold": self.semantic_threshold,
            "hierarchy_levels": self.hierarchy_levels,
        }

        # Include optional fields if set
        if self.similarity_threshold is not None:
            result["similarity_threshold"] = self.similarity_threshold
        if self.hierarchy_level is not None:
            result["hierarchy_level"] = self.hierarchy_level
        if self.strategies is not None:
            result["strategies"] = self.strategies
        if self.weights is not None:
            result["weights"] = self.weights

        # Include any additional parameters
        result.update(self.additional_params)

        return result

    def __eq__(self, other: object) -> bool:
        """Check equality with another ChunkConfig.

        Args:
            other: Object to compare with

        Returns:
            True if configurations are equal
        """
        if not isinstance(other, ChunkConfig):
            return False

        return (
            self.strategy_name == other.strategy_name
            and self.min_tokens == other.min_tokens
            and self.max_tokens == other.max_tokens
            and self.overlap_tokens == other.overlap_tokens
            and self.separator == other.separator
            and self.preserve_structure == other.preserve_structure
            and self.semantic_threshold == other.semantic_threshold
            and self.hierarchy_levels == other.hierarchy_levels
            and self.similarity_threshold == other.similarity_threshold
            and self.hierarchy_level == other.hierarchy_level
            and self.strategies == other.strategies
            and self.weights == other.weights
            and self.additional_params == other.additional_params
        )

    def __hash__(self) -> int:
        """Generate hash for the configuration.

        Returns:
            Hash value
        """
        return hash(
            (
                self.strategy_name,
                self.min_tokens,
                self.max_tokens,
                self.overlap_tokens,
                self.separator,
                self.preserve_structure,
                self.semantic_threshold,
                self.hierarchy_levels,
                self.similarity_threshold,
                self.hierarchy_level,
                tuple(self.strategies) if self.strategies else None,
                tuple(sorted(self.weights.items())) if self.weights else None,
                tuple(sorted(self.additional_params.items())),
            )
        )
