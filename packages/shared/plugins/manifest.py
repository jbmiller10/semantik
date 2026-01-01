"""Plugin manifest definitions for Semantik's unified plugin system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
