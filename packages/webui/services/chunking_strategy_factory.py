"""
Factory for creating chunking strategy instances.

This module provides a centralized factory for instantiating chunking strategies,
removing the direct dependency and logic from routers.

Plugin Configuration:
    External chunking plugins can receive configuration from the shared plugin
    state file. The factory loads plugin config and calls configure() on
    strategies that support it.
"""

import logging
from threading import Lock
from typing import Any

from shared.chunking.domain.services.chunking_strategies import STRATEGY_REGISTRY, get_strategy, register_strategy
from shared.chunking.infrastructure.exceptions import ChunkingStrategyError
from shared.chunking.types import ChunkingStrategy as ChunkingStrategyEnum
from shared.plugins.state import get_plugin_config
from webui.services.chunking.strategy_registry import (
    get_api_to_internal_map,
    get_internal_to_primary_api_map,
    get_strategy_metadata,
    resolve_api_identifier,
    resolve_internal_strategy_name,
)

logger = logging.getLogger(__name__)
_REGISTRY_LOCK = Lock()


class ChunkingStrategyFactory:
    """Factory for creating chunking strategy instances."""

    _api_to_internal: dict[str, str] = get_api_to_internal_map().copy()
    _internal_to_api_enum: dict[str, ChunkingStrategyEnum] = {
        internal: ChunkingStrategyEnum(api_id)
        for internal, api_id in get_internal_to_primary_api_map().items()
        if api_id in ChunkingStrategyEnum._value2member_map_
    }

    @classmethod
    def create_strategy(
        cls,
        strategy_name: str | ChunkingStrategyEnum,
        config: dict[str, Any],  # noqa: ARG003 - Per-operation config, used in chunk() method
        correlation_id: str | None = None,
    ) -> Any:
        """
        Create a chunking strategy instance.

        Args:
            strategy_name: Name of the strategy (string or enum)
            config: Per-operation configuration (passed to chunk() method)
            correlation_id: Optional correlation ID for error tracking

        Returns:
            Configured strategy instance

        Raises:
            ChunkingStrategyError: If strategy is unknown or initialization fails

        Note:
            Plugin configuration (API keys, global settings) is loaded from the
            shared plugin state file and applied via the strategy's configure()
            method if available. Per-operation config is passed to chunk().
        """
        # Convert enum to internal name if needed
        if isinstance(strategy_name, ChunkingStrategyEnum):
            internal_name = cls._api_to_internal.get(strategy_name.value)
            if not internal_name:
                raise ChunkingStrategyError(
                    strategy=strategy_name.value,
                    reason=f"No implementation for strategy: {strategy_name.value}",
                    correlation_id=correlation_id or "unknown",
                )
            strategy_key = internal_name
        else:
            # Handle string input
            internal = resolve_internal_strategy_name(strategy_name)
            strategy_key = internal or cls._normalize_strategy_name(str(strategy_name))

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
            strategy = get_strategy(strategy_key)

            # Load plugin config from state file and configure if supported
            plugin_config = get_plugin_config(strategy_key, resolve_secrets=True)
            if plugin_config and hasattr(strategy, "configure"):
                strategy.configure(plugin_config)
                logger.debug("Applied plugin config to chunking strategy '%s'", strategy_key)

            return strategy

        except Exception as e:
            raise ChunkingStrategyError(
                strategy=str(strategy_name),
                reason=f"Failed to initialize strategy {strategy_name}: {str(e)}",
                correlation_id=correlation_id or "unknown",
                cause=e,
            ) from e

    @classmethod
    def get_available_strategies(cls) -> list[str]:
        """Get list of available strategy names."""
        # Return API-level strategy names
        return [s.value for s in ChunkingStrategyEnum]

    @classmethod
    def get_internal_strategies(cls) -> list[str]:
        """Get list of internal strategy implementations."""
        return list(STRATEGY_REGISTRY.keys())

    @classmethod
    def register_strategy(
        cls,
        name: str,
        strategy_class: type[Any],
        api_enum: ChunkingStrategyEnum | None = None,
    ) -> None:
        """
        Register a custom strategy.

        Args:
            name: Internal name for the strategy
            strategy_class: Strategy class to register
            api_enum: Optional API enum to map to
        """
        # Register in shared registry
        registered = register_strategy(name, strategy_class)
        if not registered:
            return

        # Update mappings if API enum provided
        if api_enum:
            with _REGISTRY_LOCK:
                if api_enum.value in cls._api_to_internal and cls._api_to_internal[api_enum.value] != name:
                    logger.warning(
                        "Chunking strategy API id '%s' already mapped to '%s', skipping '%s'",
                        api_enum.value,
                        cls._api_to_internal[api_enum.value],
                        name,
                    )
                    return
                if name in cls._internal_to_api_enum and cls._internal_to_api_enum[name] != api_enum:
                    logger.warning(
                        "Chunking strategy '%s' already mapped to API id '%s', skipping '%s'",
                        name,
                        cls._internal_to_api_enum[name].value,
                        api_enum.value,
                    )
                    return
                cls._api_to_internal[api_enum.value] = name
                cls._internal_to_api_enum[name] = api_enum

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
        internal: str | None = resolve_internal_strategy_name(name)
        if internal:
            return internal
        return str(name).lower().strip()

    @classmethod
    def get_strategy_info(cls, strategy_name: str | ChunkingStrategyEnum) -> dict[str, Any]:
        """
        Get information about a strategy.

        Args:
            strategy_name: Strategy name or enum

        Returns:
            Dictionary with strategy information
        """
        # Normalize name
        if isinstance(strategy_name, ChunkingStrategyEnum):
            api_enum = strategy_name
            api_name = strategy_name.value
            internal_name = cls._api_to_internal.get(api_name, api_name)
        else:
            internal_name = cls._normalize_strategy_name(strategy_name)
            resolved_api = resolve_api_identifier(strategy_name)
            if resolved_api:
                api_name = resolved_api
                try:
                    api_enum = ChunkingStrategyEnum(resolved_api)
                except ValueError:
                    api_enum = None
            else:
                api_enum = cls._internal_to_api_enum.get(internal_name)
                api_name = api_enum.value if api_enum else str(strategy_name)

        metadata = get_strategy_metadata(api_enum if api_enum else api_name)

        return {
            "api_name": api_name,
            "internal_name": internal_name,
            "available": internal_name in STRATEGY_REGISTRY,
            "description": metadata.get("description", "Custom strategy"),
        }

    @classmethod
    def validate_strategy_compatibility(
        cls,
        strategy_name: str | ChunkingStrategyEnum,
        file_type: str | None = None,
        content_type: str | None = None,  # noqa: ARG003
    ) -> dict[str, Any]:
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
            strategy: ChunkingStrategyEnum | None = strategy_name
        else:
            # Try to map string to enum
            internal = cls._normalize_strategy_name(strategy_name)
            strategy = cls._internal_to_api_enum.get(internal)
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
        error_type: str | None = None,
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
            strategy: ChunkingStrategyEnum | None = failed_strategy
        else:
            internal = cls._normalize_strategy_name(failed_strategy)
            strategy = cls._internal_to_api_enum.get(internal)

        fallback_map = {
            ChunkingStrategyEnum.SEMANTIC: ChunkingStrategyEnum.RECURSIVE,
            ChunkingStrategyEnum.HYBRID: ChunkingStrategyEnum.RECURSIVE,
            ChunkingStrategyEnum.DOCUMENT_STRUCTURE: ChunkingStrategyEnum.RECURSIVE,
            ChunkingStrategyEnum.SLIDING_WINDOW: ChunkingStrategyEnum.FIXED_SIZE,
            ChunkingStrategyEnum.RECURSIVE: ChunkingStrategyEnum.FIXED_SIZE,
            ChunkingStrategyEnum.FIXED_SIZE: ChunkingStrategyEnum.FIXED_SIZE,  # No fallback
        }

        if strategy:
            return fallback_map.get(strategy, ChunkingStrategyEnum.RECURSIVE)
        return ChunkingStrategyEnum.RECURSIVE
