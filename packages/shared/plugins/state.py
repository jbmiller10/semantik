"""Plugin state file management.

This module provides a shared state file mechanism for plugin configuration
and enable/disable status to be communicated between WebUI and VecPipe.

WebUI writes the state file when plugin config changes, VecPipe reads it
at startup. This avoids VecPipe needing direct database access.

Secrets policy: No raw secrets in state file - only env var references.
Keys ending in '_env' are treated as env var names and resolved at runtime.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# Default state file path (on shared volume in Docker)
DEFAULT_STATE_FILE_PATH = Path("/data/plugin_state.json")

# Current schema version - bump when making breaking changes
SCHEMA_VERSION = 1


class PluginStateConfig(BaseModel):
    """Configuration for a single plugin."""

    enabled: bool = True
    config: dict[str, Any] = Field(default_factory=dict)


class PluginState(BaseModel):
    """State file schema for plugin configuration."""

    version: int = SCHEMA_VERSION
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    disabled_ids: list[str] = Field(default_factory=list)
    configs: dict[str, PluginStateConfig] = Field(default_factory=dict)

    @classmethod
    def create_empty(cls) -> PluginState:
        """Create an empty state (all plugins enabled, no config)."""
        return cls(
            version=SCHEMA_VERSION,
            updated_at=datetime.now(UTC).isoformat(),
            disabled_ids=[],
            configs={},
        )


def get_state_file_path() -> Path:
    """Get the configured state file path.

    Checks PLUGIN_STATE_FILE env var, falls back to default.
    """
    env_path = os.environ.get("PLUGIN_STATE_FILE")
    if env_path:
        return Path(env_path)
    return DEFAULT_STATE_FILE_PATH


def write_state(state: PluginState) -> None:
    """Atomically write plugin state to file.

    Uses temp file + rename pattern for atomic write (no partial writes).
    Creates parent directories if they don't exist.

    Args:
        state: The plugin state to write

    Raises:
        OSError: If write fails
    """
    path = get_state_file_path()

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory for atomic rename
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=".plugin_state_",
        suffix=".tmp",
    )
    tmp_path_obj = Path(tmp_path)

    try:
        # Write JSON to temp file
        with os.fdopen(fd, "w") as f:
            json.dump(state.model_dump(), f, indent=2)

        # Atomic rename (same filesystem)
        tmp_path_obj.rename(path)
        logger.debug("Plugin state written to %s", path)

    except Exception:
        # Clean up temp file on failure
        tmp_path_obj.unlink(missing_ok=True)
        raise


def read_state() -> PluginState | None:
    """Read plugin state from file.

    Returns None if file is missing or invalid (graceful fallback).
    Missing state = all plugins enabled, no config.

    Returns:
        PluginState if file exists and is valid, None otherwise
    """
    path = get_state_file_path()

    if not path.exists():
        logger.debug("Plugin state file not found at %s", path)
        return None

    try:
        with path.open() as f:
            data = json.load(f)

        state = PluginState.model_validate(data)

        # Log warning if schema version is newer than we support
        if state.version > SCHEMA_VERSION:
            logger.warning(
                "Plugin state file has newer schema version %d (supported: %d). "
                "Some features may not work correctly.",
                state.version,
                SCHEMA_VERSION,
            )

        return state

    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON in plugin state file %s: %s", path, exc)
        return None
    except ValidationError as exc:
        logger.warning("Plugin state file schema validation failed: %s", exc)
        return None
    except OSError as exc:
        logger.warning("Failed to read plugin state file %s: %s", path, exc)
        return None


def resolve_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve environment variable references in plugin config.

    Keys ending in '_env' are treated as env var names. The value is the
    name of the environment variable to read. The resolved value replaces
    the _env key with the base key name.

    Example:
        {"api_key_env": "OPENAI_API_KEY", "model": "gpt-4"}
        -> {"api_key": "sk-...", "model": "gpt-4"}

    If the env var is not set, the key is omitted (not exposed).

    Args:
        config: Plugin configuration dictionary

    Returns:
        New dict with _env references resolved to actual values
    """
    if not config:
        return {}

    resolved: dict[str, Any] = {}

    for key, value in config.items():
        if key.endswith("_env") and isinstance(value, str):
            # This is an env var reference
            base_key = key[:-4]  # Remove '_env' suffix
            env_value = os.environ.get(value)
            if env_value:
                resolved[base_key] = env_value
            else:
                logger.debug(
                    "Environment variable %s not set for config key %s",
                    value,
                    base_key,
                )
            # Don't include the _env reference itself in resolved config
        else:
            # Regular config value - pass through
            resolved[key] = value

    return resolved


def get_plugin_config(plugin_id: str, resolve_secrets: bool = True) -> dict[str, Any]:
    """Get configuration for a specific plugin from state file.

    Convenience function that reads state and extracts config for one plugin.

    Args:
        plugin_id: The plugin ID to get config for
        resolve_secrets: If True, resolve _env references to actual values

    Returns:
        Plugin config dict, or empty dict if not found
    """
    state = read_state()
    if state is None:
        return {}

    plugin_config = state.configs.get(plugin_id)
    if plugin_config is None:
        return {}

    config = plugin_config.config
    if resolve_secrets:
        return resolve_env_vars(config)
    return dict(config)


def get_disabled_plugin_ids() -> set[str]:
    """Get set of disabled plugin IDs from state file.

    Convenience function for VecPipe startup.

    Returns:
        Set of disabled plugin IDs, or empty set if state file missing
    """
    state = read_state()
    if state is None:
        return set()
    return set(state.disabled_ids)
