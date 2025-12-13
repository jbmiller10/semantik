"""Factory for creating document source connectors."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shared.connectors.base import BaseConnector

logger = logging.getLogger(__name__)

# Registry maps source_type to either:
# - A connector class (type[BaseConnector])
# - A tuple of (module_path, class_name) for lazy loading
_CONNECTOR_REGISTRY: dict[str, type[BaseConnector] | tuple[str, str]] = {}


class ConnectorFactory:
    """Factory for creating connector instances by source type.

    Example:
        ```python
        connector = ConnectorFactory.get_connector(
            source_type="directory",
            config={"path": "/data/docs", "recursive": True}
        )
        ```
    """

    @classmethod
    def _resolve_connector_class(cls, entry: type[BaseConnector] | tuple[str, str]) -> type[BaseConnector]:
        """Resolve a registry entry to a connector class.

        Handles both direct class references and lazy (module, class_name) tuples.
        """
        if isinstance(entry, tuple):
            import importlib

            module_path, class_name = entry
            module = importlib.import_module(module_path)
            return getattr(module, class_name)  # type: ignore[no-any-return]
        return entry

    @classmethod
    def get_connector(
        cls,
        source_type: str,
        config: dict[str, Any],
    ) -> BaseConnector:
        """Create a connector instance for the given source type.

        Args:
            source_type: Type of source (e.g., "directory", "web", "slack").
                         Case-insensitive.
            config: Connector-specific configuration dictionary.

        Returns:
            BaseConnector: Configured connector instance.

        Raises:
            ValueError: If source_type is unknown.
        """
        normalized_type = source_type.lower().strip()

        if normalized_type not in _CONNECTOR_REGISTRY:
            available = list(_CONNECTOR_REGISTRY.keys()) or ["none registered"]
            raise ValueError(f"Unknown source type: {source_type!r}. Available types: {', '.join(available)}")

        entry = _CONNECTOR_REGISTRY[normalized_type]
        connector_cls = cls._resolve_connector_class(entry)
        logger.debug(f"Creating connector for source_type={normalized_type}")
        return connector_cls(config)

    @classmethod
    def register_connector(
        cls,
        source_type: str,
        connector_cls: type[BaseConnector],
    ) -> None:
        """Register a connector class for a source type.

        Args:
            source_type: Type identifier (will be lowercased).
            connector_cls: Connector class to register.
        """
        normalized_type = source_type.lower().strip()
        _CONNECTOR_REGISTRY[normalized_type] = connector_cls
        logger.debug(f"Registered connector: {normalized_type} -> {connector_cls.__name__}")

    @classmethod
    def register_connector_lazy(
        cls,
        source_type: str,
        module_path: str,
        class_name: str,
    ) -> None:
        """Register a connector class lazily by module path.

        The connector class will only be imported when first used.

        Args:
            source_type: Type identifier (will be lowercased).
            module_path: Full module path (e.g., "shared.connectors.git").
            class_name: Class name within the module (e.g., "GitConnector").
        """
        normalized_type = source_type.lower().strip()
        _CONNECTOR_REGISTRY[normalized_type] = (module_path, class_name)
        logger.debug(f"Registered connector (lazy): {normalized_type} -> {module_path}.{class_name}")

    @classmethod
    def list_available_types(cls) -> list[str]:
        """List all registered source types.

        Returns:
            List of available source type identifiers.
        """
        return list(_CONNECTOR_REGISTRY.keys())


def register_connector(
    source_type: str,
    connector_cls: type[BaseConnector],
) -> None:
    """Module-level convenience function for connector registration.

    Args:
        source_type: Type identifier.
        connector_cls: Connector class to register.
    """
    ConnectorFactory.register_connector(source_type, connector_cls)


def register_connector_lazy(
    source_type: str,
    module_path: str,
    class_name: str,
) -> None:
    """Module-level convenience function for lazy connector registration.

    Args:
        source_type: Type identifier.
        module_path: Full module path.
        class_name: Class name within the module.
    """
    ConnectorFactory.register_connector_lazy(source_type, module_path, class_name)


# Register built-in connectors lazily to avoid import-time dependencies
register_connector_lazy("directory", "shared.connectors.local", "LocalFileConnector")
register_connector_lazy("git", "shared.connectors.git", "GitConnector")
register_connector_lazy("imap", "shared.connectors.imap", "ImapConnector")
