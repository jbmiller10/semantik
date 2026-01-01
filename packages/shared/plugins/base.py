"""Unified base class for Semantik plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from .manifest import PluginManifest


class SemanticPlugin(ABC):
    """Universal base for all Semantik plugins.

    Plugins can receive configuration in two ways:
    1. Constructor: `__init__(config=...)` - config available immediately
    2. Initialize: `initialize(config=...)` - for async setup

    Constructor config takes precedence over initialize config.
    """

    PLUGIN_TYPE: ClassVar[str]
    PLUGIN_ID: ClassVar[str]
    PLUGIN_VERSION: ClassVar[str] = "0.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize plugin with optional configuration.

        Args:
            config: Plugin configuration dictionary. May contain resolved
                    secrets from environment variables.
        """
        self._config: dict[str, Any] = config or {}

    @property
    def config(self) -> dict[str, Any]:
        """Return the plugin configuration."""
        return self._config

    @classmethod
    @abstractmethod
    def get_manifest(cls) -> PluginManifest:
        """Return plugin metadata for discovery and UI."""

    @classmethod
    def get_config_schema(cls) -> dict[str, Any] | None:
        """Return JSON Schema for plugin configuration, or None if unused."""
        return None

    @classmethod
    async def health_check(cls, config: dict[str, Any] | None = None) -> bool:  # noqa: ARG003
        """Return True if the plugin is healthy.

        Override for custom health checks. Keep this fast and non-blocking.

        Args:
            config: Optional plugin configuration for config-dependent checks.
                    For example, checking if a required API key is present.
        """
        return True

    async def initialize(self, config: dict[str, Any] | None = None) -> None:
        """Initialize plugin resources with optional configuration.

        If config was provided to constructor, it takes precedence.
        This method is for async initialization after construction.

        Args:
            config: Plugin configuration (merged with constructor config)
        """
        if config:
            # Merge with constructor config (constructor takes precedence)
            self._config = {**config, **self._config}

    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        return
