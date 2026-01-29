"""Thread-safe registry for loaded plugins."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .manifest import PluginManifest

logger = logging.getLogger(__name__)


class PluginSource(str, Enum):
    """Origin for a plugin registration."""

    BUILTIN = "builtin"
    EXTERNAL = "external"


@dataclass(frozen=True)
class PluginRecord:
    """Captured metadata for a registered plugin.

    The plugin_class field stores the plugin class which may be either:
    - An ABC-based plugin (inherits from SemanticPlugin or type-specific base)
    - A Protocol-based plugin (structurally satisfies type-specific Protocol)

    Protocol validation is performed by the loader before registration.
    """

    plugin_type: str
    plugin_id: str
    plugin_version: str
    manifest: PluginManifest
    plugin_class: type  # Validated against protocol by loader
    source: PluginSource
    entry_point: str | None = None


@dataclass
class PluginRegistry:
    """Thread-safe registry for all loaded plugins.

    Stores PluginRecord instances indexed by type and ID. Plugin classes
    are validated against their type-specific Protocol (ConnectorProtocol,
    EmbeddingProtocol, etc.) by the loader before registration.

    Both ABC-based (built-in) and Protocol-based (external) plugins are
    supported - the registry is agnostic to implementation style.
    """

    _plugins: dict[str, dict[str, PluginRecord]] = field(default_factory=dict)
    _loaded_types: set[str] = field(default_factory=set)
    _disabled_ids: set[str] = field(default_factory=set)
    _lock: RLock = field(default_factory=RLock)

    def register(self, record: PluginRecord) -> bool:
        """Register a plugin record.

        Returns True if newly registered, False if skipped (same class already registered).

        Raises:
            PluginDuplicateError: If plugin ID conflicts with another type or class.
        """
        from .exceptions import PluginDuplicateError

        with self._lock:
            plugin_type = record.plugin_type
            plugin_id = record.plugin_id

            # Ensure plugin_id is globally unique across all types
            for existing_type, records in self._plugins.items():
                if existing_type == plugin_type:
                    continue
                if plugin_id in records:
                    raise PluginDuplicateError(
                        f"Plugin id '{plugin_id}' already registered for type '{existing_type}', "
                        f"cannot register for type '{plugin_type}'",
                        plugin_id=plugin_id,
                        plugin_type=plugin_type,
                        error_code="PLUGIN_ID_CONFLICT",
                        details={"existing_type": existing_type},
                    )

            bucket = self._plugins.setdefault(plugin_type, {})
            existing = bucket.get(plugin_id)
            if existing:
                if existing.plugin_class is record.plugin_class:
                    # Same class re-registered = idempotent, not an error
                    logger.debug(
                        "Plugin '%s/%s' already registered, skipping duplicate",
                        plugin_type,
                        plugin_id,
                    )
                    return False
                # Different class with same ID = conflict error
                raise PluginDuplicateError(
                    f"Plugin conflict: '{plugin_type}/{plugin_id}' already registered with "
                    f"class {existing.plugin_class.__name__}, cannot register {record.plugin_class.__name__}",
                    plugin_id=plugin_id,
                    plugin_type=plugin_type,
                    error_code="PLUGIN_CLASS_CONFLICT",
                    details={
                        "existing_class": existing.plugin_class.__name__,
                        "new_class": record.plugin_class.__name__,
                    },
                )

            bucket[plugin_id] = record
            logger.info("Registered plugin: %s/%s", plugin_type, plugin_id)
            return True

    def get(self, plugin_type: str, plugin_id: str) -> PluginRecord | None:
        """Get a specific plugin by type and ID."""
        with self._lock:
            return self._plugins.get(plugin_type, {}).get(plugin_id)

    def get_by_type(self, plugin_type: str) -> dict[str, PluginRecord]:
        """Get all plugins of a specific type."""
        with self._lock:
            return dict(self._plugins.get(plugin_type, {}))

    def get_all(self) -> dict[str, dict[str, PluginRecord]]:
        """Get all plugins grouped by type."""
        with self._lock:
            return {ptype: dict(records) for ptype, records in self._plugins.items()}

    def list_types(self) -> list[str]:
        """List all plugin types that have registrations."""
        with self._lock:
            return list(self._plugins.keys())

    def list_records(
        self,
        *,
        plugin_type: str | None = None,
        source: PluginSource | None = None,
    ) -> list[PluginRecord]:
        """List plugin records with optional filtering."""
        with self._lock:
            records: list[PluginRecord] = []
            if plugin_type is not None:
                bucket = self._plugins.get(plugin_type, {})
                records = list(bucket.values())
            else:
                for bucket in self._plugins.values():
                    records.extend(bucket.values())

        if source is not None:
            records = [record for record in records if record.source == source]
        return records

    def list_ids(
        self,
        *,
        plugin_type: str | None = None,
        source: PluginSource | None = None,
    ) -> list[str]:
        """List plugin IDs with optional filtering."""
        return [record.plugin_id for record in self.list_records(plugin_type=plugin_type, source=source)]

    def find_by_id(self, plugin_id: str) -> PluginRecord | None:
        """Find a plugin record by ID across all types."""
        with self._lock:
            for records in self._plugins.values():
                if plugin_id in records:
                    return records[plugin_id]
        return None

    def is_loaded(self, plugin_types: Iterable[str]) -> bool:
        """Return True if the given plugin types have been loaded."""
        requested = set(plugin_types)
        with self._lock:
            return requested.issubset(self._loaded_types)

    def mark_loaded(self, plugin_types: Iterable[str]) -> None:
        """Mark plugin types as loaded."""
        with self._lock:
            self._loaded_types.update(plugin_types)

    def set_disabled(self, plugin_ids: Iterable[str]) -> None:
        """Record which external plugin ids are disabled."""
        with self._lock:
            self._disabled_ids = set(plugin_ids)

    def is_disabled(self, plugin_id: str) -> bool:
        """Return True if the plugin id is marked disabled."""
        with self._lock:
            return plugin_id in self._disabled_ids

    def disabled_ids(self) -> set[str]:
        """Return a snapshot of disabled plugin ids."""
        with self._lock:
            return set(self._disabled_ids)

    def loaded_types(self) -> set[str]:
        """Return a snapshot of loaded plugin types."""
        with self._lock:
            return set(self._loaded_types)

    def reset(self) -> None:
        """Clear all plugin registrations and loaded state (testing only)."""
        with self._lock:
            self._plugins.clear()
            self._loaded_types.clear()
            self._disabled_ids.clear()

    def get_parser_emitted_fields(self, plugin_id: str) -> list[str]:
        """Get the parsed.* fields emitted by a parser plugin.

        Used by the pipeline editor to determine which predicate fields
        are available for routing based on the source parser node.

        Args:
            plugin_id: The parser plugin ID (e.g., "text", "unstructured")

        Returns:
            List of field names (e.g., ["detected_language", "approx_token_count"])
            without the "metadata.parsed." prefix.
            Returns empty list if plugin not found or has no EMITTED_FIELDS.
        """
        with self._lock:
            record = self._plugins.get("parser", {}).get(plugin_id)
            if record and hasattr(record.plugin_class, "EMITTED_FIELDS"):
                return list(record.plugin_class.EMITTED_FIELDS)
            return []


plugin_registry = PluginRegistry()
