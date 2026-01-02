"""Protocols for plugin runtime validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .manifest import PluginManifest


@runtime_checkable
class PluginProtocol(Protocol):
    """Minimal protocol all plugins should satisfy."""

    PLUGIN_TYPE: ClassVar[str]
    PLUGIN_ID: ClassVar[str]
    PLUGIN_VERSION: ClassVar[str]

    @classmethod
    def get_manifest(cls) -> PluginManifest: ...
