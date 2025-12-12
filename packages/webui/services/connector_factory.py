"""Factory for creating document source connectors."""

import logging
from typing import Any

from shared.connectors.base import BaseConnector
from shared.connectors.git import GitConnector
from shared.connectors.imap import ImapConnector
from shared.connectors.local import LocalFileConnector

logger = logging.getLogger(__name__)

# Static registry mapping source_type to connector class
# LocalFileConnector will be added in Ticket 6
_CONNECTOR_REGISTRY: dict[str, type[BaseConnector]] = {}


class ConnectorFactory:
    """Factory for creating connector instances by source type.

    Example:
        ```python
        connector = ConnectorFactory.get_connector(
            source_type="directory",
            config={"source_path": "/data/docs", "recursive": True}
        )
        ```
    """

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

        connector_cls = _CONNECTOR_REGISTRY[normalized_type]
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


# Register built-in connectors
register_connector("directory", LocalFileConnector)
register_connector("git", GitConnector)
register_connector("imap", ImapConnector)
