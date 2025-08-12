"""
Factory for creating chunking strategy instances.

This module provides a centralized factory for instantiating chunking strategies,
removing the direct dependency and logic from routers.
"""

from typing import Any, Dict, List, Optional, Type

from packages.shared.chunking.domain.services.chunking_strategies import (
    STRATEGY_REGISTRY,
    get_strategy,
)
from packages.shared.chunking.infrastructure.exceptions import ChunkingStrategyError
from packages.webui.api.v2.chunking_schemas import ChunkingStrategy as ChunkingStrategyEnum


class ChunkingStrategyFactory:
    """Factory for creating chunking strategy instances."""

    # Mapping from API strategy names to internal strategy names
    _strategy_mapping = {
        ChunkingStrategyEnum.FIXED_SIZE: "character",
        ChunkingStrategyEnum.RECURSIVE: "recursive",
        ChunkingStrategyEnum.DOCUMENT_STRUCTURE: "markdown",
        ChunkingStrategyEnum.SEMANTIC: "semantic",
        ChunkingStrategyEnum.SLIDING_WINDOW: "character",  # Uses character with overlap
        ChunkingStrategyEnum.HYBRID: "hybrid",
    }

    # Reverse mapping for convenience
    _reverse_mapping = {
        "character": ChunkingStrategyEnum.FIXED_SIZE,
        "recursive": ChunkingStrategyEnum.RECURSIVE,
        "markdown": ChunkingStrategyEnum.DOCUMENT_STRUCTURE,
        "semantic": ChunkingStrategyEnum.SEMANTIC,
        "hierarchical": ChunkingStrategyEnum.HYBRID,  # Map hierarchical to hybrid
        "hybrid": ChunkingStrategyEnum.HYBRID,
    }

    @classmethod
    def create_strategy(
        cls,
        strategy_name: str | ChunkingStrategyEnum,
        config: Dict[str, Any],  # noqa: ARG003
        correlation_id: Optional[str] = None,
    ) -> Any:
        """
        Create a chunking strategy instance.

        Args:
            strategy_name: Name of the strategy (string or enum)
            config: Configuration for the strategy
            correlation_id: Optional correlation ID for error tracking

        Returns:
            Configured strategy instance

        Raises:
            ChunkingStrategyError: If strategy is unknown or initialization fails
        """
        # Convert enum to internal name if needed
        if isinstance(strategy_name, ChunkingStrategyEnum):
            internal_name = cls._strategy_mapping.get(strategy_name)
            if not internal_name:
                raise ChunkingStrategyError(
                    strategy=strategy_name.value,
                    reason=f"No implementation for strategy: {strategy_name.value}",
                    correlation_id=correlation_id or "unknown",
                )
            strategy_key = internal_name
        else:
            # Handle string input
            strategy_key = cls._normalize_strategy_name(strategy_name)

        # Check if strategy exists in registry
        if strategy_key not in STRATEGY_REGISTRY:
            available = cls.get_available_strategies()
            raise ChunkingStrategyError(
                strategy=str(strategy_name),
                reason=f"Unknown strategy: {strategy_name}. Available: {', '.join(available)}",
                correlation_id=correlation_id or "unknown",
            )

        try:
            # Get strategy from shared registry
            return get_strategy(strategy_key)

            # Note: The shared strategies don't take config in constructor,
            # they use it in the chunk() method. So we just return the strategy.

        except Exception as e:
            raise ChunkingStrategyError(
                strategy=str(strategy_name),
                reason=f"Failed to initialize strategy {strategy_name}: {str(e)}",
                correlation_id=correlation_id or "unknown",
                cause=e,
            ) from e

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available strategy names."""
        # Return API-level strategy names
        return [s.value for s in ChunkingStrategyEnum]

    @classmethod
    def get_internal_strategies(cls) -> List[str]:
        """Get list of internal strategy implementations."""
        return list(STRATEGY_REGISTRY.keys())

    @classmethod
    def register_strategy(
        cls,
        name: str,
        strategy_class: Type[Any],
        api_enum: Optional[ChunkingStrategyEnum] = None,
    ) -> None:
        """
        Register a custom strategy.

        Args:
            name: Internal name for the strategy
            strategy_class: Strategy class to register
            api_enum: Optional API enum to map to
        """
        # Register in shared registry
        STRATEGY_REGISTRY[name] = strategy_class

        # Update mappings if API enum provided
        if api_enum:
            cls._strategy_mapping[api_enum] = name
            cls._reverse_mapping[name] = api_enum

    @classmethod
    def normalize_strategy_name(cls, name: str) -> str:
        """Normalize strategy name variations to internal names.

        Public method for normalizing strategy names before persistence.

        Args:
            name: User-provided strategy name

        Returns:
            Normalized internal strategy name

        Raises:
            ChunkingStrategyError: If the strategy name is unknown or invalid
        """
        normalized = cls._normalize_strategy_name(name)

        # Validate that the normalized name exists in registry
        if normalized not in STRATEGY_REGISTRY:
            available = cls.get_available_strategies()
            raise ChunkingStrategyError(
                strategy=name,
                reason=f"Unknown strategy: {name}. Available: {', '.join(available)}",
                correlation_id="validation",
            )

        return normalized

    @classmethod
    def _normalize_strategy_name(cls, name: str) -> str:
        """Internal normalize strategy name variations to internal names.

        Args:
            name: Strategy name to normalize

        Returns:
            Normalized internal strategy name (may not be valid)
        """
        name = name.lower().strip()

        # Direct mapping
        direct_mappings = {
            "fixed": "character",
            "fixed_size": "character",
            "character": "character",
            "char": "character",
            "recursive": "recursive",
            "recursive_text": "recursive",
            "markdown": "markdown",
            "document": "markdown",
            "document_structure": "markdown",
            "semantic": "semantic",
            "ai": "semantic",
            "sliding": "character",
            "sliding_window": "character",
            "window": "character",
            "hybrid": "hybrid",
            "mixed": "hybrid",
            "hierarchical": "hierarchical",
        }

        # Return mapped name or original (validation happens in public method)
        return direct_mappings.get(name, name)

    @classmethod
    def get_strategy_info(cls, strategy_name: str | ChunkingStrategyEnum) -> Dict[str, Any]:
        """
        Get information about a strategy.

        Args:
            strategy_name: Strategy name or enum

        Returns:
            Dictionary with strategy information
        """
        # Normalize name
        if isinstance(strategy_name, ChunkingStrategyEnum):
            internal_name = cls._strategy_mapping.get(strategy_name, "")
            api_name = strategy_name.value
        else:
            internal_name = cls._normalize_strategy_name(strategy_name)
            api_name = cls._reverse_mapping.get(internal_name, strategy_name)
            if isinstance(api_name, ChunkingStrategyEnum):
                api_name = api_name.value

        return {
            "api_name": api_name,
            "internal_name": internal_name,
            "available": internal_name in STRATEGY_REGISTRY,
            "description": cls._get_strategy_description(internal_name),
        }

    @classmethod
    def _get_strategy_description(cls, internal_name: str) -> str:
        """Get description for a strategy."""
        descriptions = {
            "character": "Simple fixed-size chunking with configurable overlap",
            "recursive": "Recursively splits text using multiple separators",
            "markdown": "Splits based on markdown/document structure",
            "semantic": "Uses embeddings to find natural semantic boundaries",
            "hierarchical": "Creates parent-child chunk relationships",
            "hybrid": "Combines multiple strategies based on content",
        }
        return descriptions.get(internal_name, "Custom strategy")

    @classmethod
    def validate_strategy_compatibility(
        cls,
        strategy_name: str | ChunkingStrategyEnum,
        file_type: Optional[str] = None,
        content_type: Optional[str] = None,  # noqa: ARG003
    ) -> Dict[str, Any]:
        """
        Validate if a strategy is compatible with given content.

        Args:
            strategy_name: Strategy to validate
            file_type: File extension (e.g., '.pdf')
            content_type: MIME type

        Returns:
            Validation result with compatibility info
        """
        # Get strategy info
        if isinstance(strategy_name, ChunkingStrategyEnum):
            strategy = strategy_name
        else:
            # Try to map string to enum
            internal = cls._normalize_strategy_name(strategy_name)
            strategy = cls._reverse_mapping.get(internal)
            if not strategy:
                return {
                    "compatible": False,
                    "reason": f"Unknown strategy: {strategy_name}",
                    "recommendations": ["Use 'recursive' as a general-purpose strategy"],
                }

        # Check compatibility
        compatibility_issues = []
        recommendations = []

        # Document structure strategy needs structured content
        if (
            strategy == ChunkingStrategyEnum.DOCUMENT_STRUCTURE
            and file_type
            and file_type not in [".md", ".markdown", ".rst", ".tex", ".html"]
        ):
            compatibility_issues.append(f"Document structure strategy may not work well with {file_type} files")
            recommendations.append("Consider using 'recursive' or 'semantic' strategy")

        # Semantic strategy needs embedding support
        if strategy == ChunkingStrategyEnum.SEMANTIC:
            # In production, check if embedding service is available
            pass  # Assume available for now

        # Sliding window works best with continuous text
        if strategy == ChunkingStrategyEnum.SLIDING_WINDOW and file_type and file_type in [".json", ".xml", ".yaml"]:
            compatibility_issues.append("Sliding window may break structure in structured files")
            recommendations.append("Use 'recursive' for structured data files")

        return {
            "compatible": len(compatibility_issues) == 0,
            "issues": compatibility_issues,
            "recommendations": recommendations,
        }

    @classmethod
    def suggest_fallback_strategy(
        cls,
        failed_strategy: str | ChunkingStrategyEnum,
        error_type: Optional[str] = None,
    ) -> ChunkingStrategyEnum:
        """
        Suggest a fallback strategy when one fails.

        Args:
            failed_strategy: Strategy that failed
            error_type: Type of error that occurred

        Returns:
            Suggested fallback strategy
        """
        # Memory errors -> simpler strategy
        if error_type and "memory" in error_type.lower():
            return ChunkingStrategyEnum.FIXED_SIZE

        # Timeout errors -> faster strategy
        if error_type and "timeout" in error_type.lower():
            return ChunkingStrategyEnum.RECURSIVE

        # Strategy-specific fallbacks
        if isinstance(failed_strategy, ChunkingStrategyEnum):
            strategy = failed_strategy
        else:
            internal = cls._normalize_strategy_name(failed_strategy)
            strategy = cls._reverse_mapping.get(internal)

        fallback_map = {
            ChunkingStrategyEnum.SEMANTIC: ChunkingStrategyEnum.RECURSIVE,
            ChunkingStrategyEnum.HYBRID: ChunkingStrategyEnum.RECURSIVE,
            ChunkingStrategyEnum.DOCUMENT_STRUCTURE: ChunkingStrategyEnum.RECURSIVE,
            ChunkingStrategyEnum.SLIDING_WINDOW: ChunkingStrategyEnum.FIXED_SIZE,
            ChunkingStrategyEnum.RECURSIVE: ChunkingStrategyEnum.FIXED_SIZE,
            ChunkingStrategyEnum.FIXED_SIZE: ChunkingStrategyEnum.FIXED_SIZE,  # No fallback
        }

        return fallback_map.get(strategy, ChunkingStrategyEnum.RECURSIVE)
