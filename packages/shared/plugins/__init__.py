"""Unified plugin system exports."""

from .adapters import get_config_schema
from .base import SemanticPlugin
from .dto_adapters import (
    ValidationError,
    coerce_to_agent_message,
    coerce_to_ingested_document,
)
from .loader import ENTRYPOINT_GROUP, load_plugins
from .manifest import PluginManifest
from .registry import PluginRecord, PluginRegistry, PluginSource, plugin_registry
from .security import (
    SENSITIVE_ENV_PATTERNS,
    audit_log,
    get_sanitized_environment,
)
from .state import (
    PluginState,
    PluginStateConfig,
    get_disabled_plugin_ids,
    get_plugin_config,
    get_state_file_path,
    read_state,
    resolve_env_vars,
    write_state,
)
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
    # State file management
    "PluginState",
    "PluginStateConfig",
    "read_state",
    "write_state",
    "get_state_file_path",
    "resolve_env_vars",
    "get_plugin_config",
    "get_disabled_plugin_ids",
    # Security utilities
    "SENSITIVE_ENV_PATTERNS",
    "audit_log",
    "get_sanitized_environment",
    # DTO adapters
    "ValidationError",
    "coerce_to_ingested_document",
    "coerce_to_agent_message",
]
