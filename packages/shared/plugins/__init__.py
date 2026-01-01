"""Unified plugin system exports."""

from .adapters import get_config_schema
from .base import SemanticPlugin
from .loader import ENTRYPOINT_GROUP, load_plugins
from .manifest import PluginManifest
from .registry import PluginRecord, PluginRegistry, PluginSource, plugin_registry
from .types import ChunkingPlugin, ConnectorPlugin, EmbeddingPlugin

__all__ = [
    "ENTRYPOINT_GROUP",
    "PluginManifest",
    "SemanticPlugin",
    "PluginRecord",
    "PluginRegistry",
    "PluginSource",
    "plugin_registry",
    "load_plugins",
    "get_config_schema",
    "EmbeddingPlugin",
    "ChunkingPlugin",
    "ConnectorPlugin",
]
