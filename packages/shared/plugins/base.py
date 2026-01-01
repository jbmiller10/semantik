"""Unified base class for Semantik plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from .manifest import PluginManifest


class SemanticPlugin(ABC):
    """Universal base for all Semantik plugins."""

    PLUGIN_TYPE: ClassVar[str]
    PLUGIN_ID: ClassVar[str]
    PLUGIN_VERSION: ClassVar[str] = "0.0.0"

    @classmethod
    @abstractmethod
    def get_manifest(cls) -> PluginManifest:
        """Return plugin metadata for discovery and UI."""

    @classmethod
    def get_config_schema(cls) -> dict[str, Any] | None:
        """Return JSON Schema for plugin configuration, or None if unused."""
        return None

    @classmethod
    async def health_check(cls) -> bool:
        """Return True if the plugin is healthy.

        Override for custom health checks. Keep this fast and non-blocking.
        """
        return True

    async def initialize(self, config: dict[str, Any] | None = None) -> None:  # noqa: ARG002
        """Initialize plugin resources with optional configuration."""
        return

    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        return
