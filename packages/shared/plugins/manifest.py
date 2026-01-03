"""Plugin manifest definitions for Semantik's unified plugin system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from packaging.version import Version


@dataclass(frozen=True)
class PluginDependency:
    """Plugin dependency specification.

    Allows plugins to declare dependencies on other plugins with
    optional version constraints.

    Examples:
        # Any version of another-plugin
        PluginDependency("another-plugin")

        # At least version 1.0.0
        PluginDependency("another-plugin", min_version="1.0.0")

        # Version range
        PluginDependency("another-plugin", min_version="1.0.0", max_version="2.0.0")

        # Optional dependency
        PluginDependency("optional-plugin", optional=True)
    """

    plugin_id: str
    min_version: str | None = None
    max_version: str | None = None
    optional: bool = False

    def check_version(self, actual_version: str) -> tuple[bool, str | None]:
        """Check if an actual version satisfies this dependency's constraints.

        Args:
            actual_version: The version string to check.

        Returns:
            Tuple of (satisfied, error_message).
            error_message is None if satisfied, otherwise describes the failure.
        """
        try:
            actual = Version(actual_version)

            if self.min_version:
                min_ver = Version(self.min_version)
                if actual < min_ver:
                    return False, f"requires >= {self.min_version}, got {actual_version}"

            if self.max_version:
                max_ver = Version(self.max_version)
                if actual >= max_ver:
                    return False, f"requires < {self.max_version}, got {actual_version}"

            return True, None
        except Exception as e:
            return False, f"version check failed: {e}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        result: dict[str, Any] = {"plugin_id": self.plugin_id}
        if self.min_version:
            result["min_version"] = self.min_version
        if self.max_version:
            result["max_version"] = self.max_version
        if self.optional:
            result["optional"] = self.optional
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any] | str) -> PluginDependency:
        """Create a PluginDependency from a dict or string.

        Args:
            data: Either a plugin_id string or a dict with dependency fields.

        Returns:
            PluginDependency instance.
        """
        if isinstance(data, str):
            return cls(plugin_id=data)
        return cls(
            plugin_id=data["plugin_id"],
            min_version=data.get("min_version"),
            max_version=data.get("max_version"),
            optional=data.get("optional", False),
        )


@dataclass(frozen=True)
class PluginManifest:
    """Canonical metadata for a Semantik plugin."""

    id: str
    type: str
    version: str
    display_name: str
    description: str
    author: str | None = None
    license: str | None = None
    homepage: str | None = None
    requires: list[str] = field(default_factory=list)
    semantik_version: str | None = None
    capabilities: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the manifest."""

        return {
            "id": self.id,
            "type": self.type,
            "version": self.version,
            "display_name": self.display_name,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "homepage": self.homepage,
            "requires": list(self.requires),
            "semantik_version": self.semantik_version,
            "capabilities": dict(self.capabilities),
        }
