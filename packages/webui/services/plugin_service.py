"""Service layer for plugin management."""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from shared.database.repositories.plugin_config_repository import PluginConfigRepository
from shared.plugins.adapters import get_config_schema
from shared.plugins.loader import load_plugins
from shared.plugins.registry import PluginRecord, PluginSource, plugin_registry
from shared.plugins.security import audit_log
from shared.plugins.state import PluginState, PluginStateConfig, resolve_env_vars, write_state

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

_DEFAULT_PLUGIN_TYPES = {"embedding", "chunking", "connector", "reranker", "extractor"}

# Timeout for individual plugin health checks (seconds)
HEALTH_CHECK_TIMEOUT_SECONDS = 10.0


def _coerce_type(value: Any, schema_type: str) -> bool:
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "null":
        return value is None
    return True


def _validate_string_constraints(value: str, schema: dict[str, Any], path: str) -> list[str]:
    """Validate string-specific constraints: minLength, maxLength, pattern."""
    errors: list[str] = []

    if "minLength" in schema:
        min_len = schema["minLength"]
        if len(value) < min_len:
            errors.append(f"{path}: string length {len(value)} is less than minimum {min_len}")

    if "maxLength" in schema:
        max_len = schema["maxLength"]
        if len(value) > max_len:
            errors.append(f"{path}: string length {len(value)} exceeds maximum {max_len}")

    if "pattern" in schema:
        pattern = schema["pattern"]
        try:
            if not re.search(pattern, value):
                errors.append(f"{path}: string does not match pattern '{pattern}'")
        except re.error as e:
            # Invalid regex pattern in schema - log but don't fail validation
            logger.warning("Invalid regex pattern '%s' in schema at %s: %s", pattern, path, e)

    return errors


def _validate_number_constraints(value: int | float, schema: dict[str, Any], path: str) -> list[str]:
    """Validate number-specific constraints: minimum, maximum, exclusiveMinimum, exclusiveMaximum."""
    errors: list[str] = []

    if "minimum" in schema:
        minimum = schema["minimum"]
        if value < minimum:
            errors.append(f"{path}: value {value} is less than minimum {minimum}")

    if "maximum" in schema:
        maximum = schema["maximum"]
        if value > maximum:
            errors.append(f"{path}: value {value} exceeds maximum {maximum}")

    if "exclusiveMinimum" in schema:
        exc_min = schema["exclusiveMinimum"]
        if value <= exc_min:
            errors.append(f"{path}: value {value} must be greater than {exc_min}")

    if "exclusiveMaximum" in schema:
        exc_max = schema["exclusiveMaximum"]
        if value >= exc_max:
            errors.append(f"{path}: value {value} must be less than {exc_max}")

    return errors


def _validate_array_constraints(value: list, schema: dict[str, Any], path: str) -> list[str]:
    """Validate array-specific constraints: minItems, maxItems, uniqueItems, items."""
    errors: list[str] = []

    if "minItems" in schema:
        min_items = schema["minItems"]
        if len(value) < min_items:
            errors.append(f"{path}: array has {len(value)} items, minimum is {min_items}")

    if "maxItems" in schema:
        max_items = schema["maxItems"]
        if len(value) > max_items:
            errors.append(f"{path}: array has {len(value)} items, maximum is {max_items}")

    if schema.get("uniqueItems"):
        # Check for duplicates (works for hashable items)
        try:
            seen = set()
            for idx, item in enumerate(value):
                # Convert to tuple for hashability if it's a list
                hashable = tuple(item) if isinstance(item, list) else item
                if hashable in seen:
                    errors.append(f"{path}[{idx}]: duplicate item not allowed")
                    break
                seen.add(hashable)
        except TypeError:
            # Item not hashable (e.g., dict), skip uniqueness check
            pass

    # Validate items against items schema
    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        for idx, item in enumerate(value):
            errors.extend(_validate_value(item, item_schema, f"{path}[{idx}]"))

    return errors


def _validate_value(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    """Validate a value against a JSON Schema definition.

    Supports a substantial subset of JSON Schema draft-07:
    - type validation (string, integer, number, boolean, object, array, null)
    - enum validation
    - string constraints: minLength, maxLength, pattern
    - number constraints: minimum, maximum, exclusiveMinimum, exclusiveMaximum
    - array constraints: minItems, maxItems, uniqueItems, items
    - object constraints: properties, required, additionalProperties
    - nested validation
    """
    errors: list[str] = []
    schema_type = schema.get("type")

    # Type validation
    if schema_type and not _coerce_type(value, schema_type):
        errors.append(f"{path}: expected {schema_type}")
        return errors

    # Enum validation
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value must be one of {schema['enum']}")
        return errors

    # Const validation (exact value match)
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: value must be {schema['const']!r}")
        return errors

    # Type-specific constraint validation
    if schema_type == "string" and isinstance(value, str):
        errors.extend(_validate_string_constraints(value, schema, path))

    elif schema_type in ("integer", "number") and isinstance(value, int | float):
        errors.extend(_validate_number_constraints(value, schema, path))

    elif schema_type == "array" and isinstance(value, list):
        errors.extend(_validate_array_constraints(value, schema, path))

    elif schema_type == "object" and isinstance(value, dict):
        props = schema.get("properties", {})
        required = schema.get("required", [])

        # Check required fields
        for key in required:
            if key not in value:
                errors.append(f"{path}.{key}: field is required")

        # Validate properties
        for key, val in value.items():
            if key in props and isinstance(props[key], dict):
                errors.extend(_validate_value(val, props[key], f"{path}.{key}"))
            elif props and schema.get("additionalProperties") is False:
                errors.append(f"{path}.{key}: additional properties are not allowed")

    return errors


def validate_config_schema(config: dict[str, Any], schema: dict[str, Any] | None) -> list[str]:
    """Validate config against a JSON Schema (substantial subset of draft-07).

    Supported features:
    - type: string, integer, number, boolean, object, array, null
    - enum, const
    - String: minLength, maxLength, pattern
    - Number: minimum, maximum, exclusiveMinimum, exclusiveMaximum
    - Array: minItems, maxItems, uniqueItems, items
    - Object: properties, required, additionalProperties

    Not supported: allOf, anyOf, oneOf, not, $ref, format, dependencies
    """
    if schema is None:
        return []
    if not isinstance(config, dict):
        return ["config must be an object"]
    if schema.get("type") and schema.get("type") != "object":
        return ["schema type must be 'object'"]
    return _validate_value(config, schema, "config")


class PluginService:
    """Service for plugin management operations."""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session
        self.repo = PluginConfigRepository(db_session)

    async def list_plugins(
        self,
        *,
        plugin_type: str | None = None,
        enabled: bool | None = None,
        include_health: bool = False,
    ) -> list[dict[str, Any]]:
        load_plugins(plugin_types={plugin_type} if plugin_type else _DEFAULT_PLUGIN_TYPES)

        records = plugin_registry.list_records(source=PluginSource.EXTERNAL, plugin_type=plugin_type)
        configs = await self.repo.list_configs(plugin_type=plugin_type)
        config_map = {config.id: config for config in configs}

        results: list[dict[str, Any]] = []
        for record in records:
            config_row = config_map.get(record.plugin_id)
            is_enabled = config_row.enabled if config_row else True
            if enabled is not None and is_enabled is not enabled:
                continue

            results.append(self._build_plugin_payload(record, config_row))

        if include_health and results:
            await self._refresh_health(records, config_map)
            # Re-fetch configs after health refresh commit to avoid expired ORM objects
            configs = await self.repo.list_configs(plugin_type=plugin_type)
            config_map = {config.id: config for config in configs}
            results = [self._build_plugin_payload(r, config_map.get(r.plugin_id)) for r in records]
            if enabled is not None:
                results = [payload for payload in results if payload["enabled"] is enabled]

        return results

    async def get_plugin(self, plugin_id: str) -> dict[str, Any] | None:
        record = self._get_external_record(plugin_id)
        if record is None:
            return None
        config_row = await self.repo.get_config(plugin_id)
        return self._build_plugin_payload(record, config_row)

    async def update_config(self, plugin_id: str, config: dict[str, Any]) -> dict[str, Any] | None:
        record = self._get_external_record(plugin_id)
        if record is None:
            return None
        schema = get_config_schema(record.plugin_class)
        errors = validate_config_schema(config, schema)
        if errors:
            raise ValueError("; ".join(errors))

        config_row = await self.repo.upsert_config(
            plugin_id=plugin_id,
            plugin_type=record.plugin_type,
            config=config,
        )
        await self.db_session.commit()

        audit_log(
            plugin_id,
            "plugin.config.updated",
            {"plugin_type": record.plugin_type, "config_keys": list(config.keys())},
        )

        # Sync state file for VecPipe
        await self._sync_state_file()

        return self._build_plugin_payload(record, config_row)

    async def set_enabled(self, plugin_id: str, enabled: bool) -> dict[str, Any] | None:
        record = self._get_external_record(plugin_id)
        if record is None:
            return None
        config_row = await self.repo.upsert_config(
            plugin_id=plugin_id,
            plugin_type=record.plugin_type,
            enabled=enabled,
        )
        await self.db_session.commit()

        audit_log(
            plugin_id,
            "plugin.enabled" if enabled else "plugin.disabled",
            {"plugin_type": record.plugin_type},
        )

        # Sync state file for VecPipe
        await self._sync_state_file()

        payload = self._build_plugin_payload(record, config_row)
        payload["requires_restart"] = True
        return payload

    async def get_manifest(self, plugin_id: str) -> dict[str, Any] | None:
        record = self._get_external_record(plugin_id)
        if record is None:
            return None
        return record.manifest.to_dict()

    async def get_config_schema(self, plugin_id: str) -> dict[str, Any] | None:
        record = self._get_external_record(plugin_id)
        if record is None:
            return None
        return get_config_schema(record.plugin_class)

    async def check_health(self, plugin_id: str) -> dict[str, Any] | None:
        record = self._get_external_record(plugin_id)
        if record is None:
            return None
        config_row = await self._check_and_update_health(record)
        await self.db_session.commit()

        health_status = config_row.health_status if config_row else "unknown"
        audit_log(
            plugin_id,
            "plugin.health_check",
            {"plugin_type": record.plugin_type, "status": health_status},
        )

        return {
            "plugin_id": record.plugin_id,
            "health_status": health_status,
            "last_health_check": config_row.last_health_check if config_row else None,
            "error_message": config_row.error_message if config_row else None,
        }

    def _get_external_record(self, plugin_id: str) -> PluginRecord | None:
        load_plugins(plugin_types=_DEFAULT_PLUGIN_TYPES)
        record = plugin_registry.find_by_id(plugin_id)
        if record is None:
            return None
        if record.source != PluginSource.EXTERNAL:
            return None
        return record

    def _build_plugin_payload(self, record: PluginRecord, config_row: Any | None) -> dict[str, Any]:
        enabled = config_row.enabled if config_row else True
        config = dict(config_row.config) if config_row and config_row.config is not None else {}
        health_status = config_row.health_status if config_row else "unknown"
        last_check = config_row.last_health_check if config_row else None
        error_message = config_row.error_message if config_row else None

        return {
            "id": record.plugin_id,
            "type": record.plugin_type,
            "version": record.plugin_version,
            "manifest": record.manifest.to_dict(),
            "enabled": enabled,
            "config": config,
            "health_status": health_status,
            "last_health_check": last_check,
            "error_message": error_message,
        }

    async def _sync_state_file(self) -> None:
        """Synchronize plugin state to shared file for VecPipe.

        Writes a JSON state file containing:
        - List of disabled plugin IDs
        - Plugin configurations (with env var references, not secrets)

        This file is read by VecPipe at startup to determine which plugins
        to load and how to configure them.
        """
        try:
            # Get all plugin configs from database
            configs = await self.repo.list_configs()

            # Build disabled IDs list
            disabled_ids = [c.id for c in configs if not c.enabled]

            # Build config map
            config_map: dict[str, PluginStateConfig] = {}
            for c in configs:
                config_map[c.id] = PluginStateConfig(
                    enabled=c.enabled,
                    config=dict(c.config) if c.config else {},
                )

            # Write state file atomically
            state = PluginState(
                version=1,
                updated_at=datetime.now(UTC).isoformat(),
                disabled_ids=disabled_ids,
                configs=config_map,
            )
            write_state(state)
            logger.debug("Plugin state file synced with %d configs", len(config_map))

        except Exception as exc:
            # Log but don't fail the operation - state file is a best-effort feature
            logger.warning("Failed to sync plugin state file: %s", exc)

    async def _refresh_health(
        self,
        records: Iterable[PluginRecord],
        config_map: dict[str, Any],
    ) -> None:
        tasks = []
        for record in records:
            config_row = config_map.get(record.plugin_id)
            enabled = config_row.enabled if config_row else True
            if not enabled:
                continue
            # Pass config to health check
            plugin_config = dict(config_row.config) if config_row and config_row.config else {}
            tasks.append(self._check_and_update_health(record, plugin_config))

        if not tasks:
            return

        # Calculate batch timeout based on number of tasks plus buffer
        batch_timeout = HEALTH_CHECK_TIMEOUT_SECONDS * len(tasks) + 5.0

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=batch_timeout,
            )
        except TimeoutError:
            logger.warning("Batch health refresh timed out after %.1fs", batch_timeout)
            return

        for updated in results:
            # Skip exceptions returned by gather(return_exceptions=True)
            if isinstance(updated, BaseException):
                logger.warning("Health check task failed: %s", updated)
                continue
            if updated is not None:
                config_map[updated.id] = updated
        await self.db_session.commit()

    async def _check_and_update_health(
        self,
        record: PluginRecord,
        plugin_config: dict[str, Any] | None = None,
    ) -> Any | None:
        status, error = await self._run_health_check(record, plugin_config)
        try:
            return await self.repo.update_health(
                plugin_id=record.plugin_id,
                plugin_type=record.plugin_type,
                status=status,
                error_message=error,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to update health for %s: %s", record.plugin_id, exc)
            return None

    async def _run_health_check(
        self,
        record: PluginRecord,
        plugin_config: dict[str, Any] | None = None,
    ) -> tuple[str, str | None]:
        health_fn = getattr(record.plugin_class, "health_check", None)
        if health_fn is None:
            return "unknown", "Plugin does not define health_check method"
        if not callable(health_fn):
            return "unknown", "health_check is not callable"

        # Check if the method can be called without an instance
        # (must be classmethod or staticmethod, not instance method)
        try:
            sig = inspect.signature(health_fn)
            # For classmethods, the first parameter is automatically bound
            # For staticmethods, there are no required parameters
            # For instance methods, there's an unbound 'self' parameter
            params = list(sig.parameters.values())
            if params and params[0].name == "self" and params[0].default is inspect.Parameter.empty:
                return "unknown", "health_check must be @classmethod or @staticmethod, not instance method"
        except (ValueError, TypeError):
            # signature() can fail for some built-in types, proceed anyway
            sig = None

        # Resolve env var references in config for health check
        resolved_config = resolve_env_vars(plugin_config) if plugin_config else None

        # Check if health_check accepts config parameter (backward compatibility)
        accepts_config = False
        if sig is not None:
            try:
                param_names = [p.name for p in sig.parameters.values()]
                accepts_config = "config" in param_names
            except (ValueError, TypeError):
                pass

        try:
            result = health_fn(config=resolved_config) if accepts_config else health_fn()
        except TypeError as exc:
            # Could be signature mismatch or other type error
            return "unknown", f"health_check() call failed: {exc}"
        except Exception as exc:
            return "unhealthy", str(exc)

        if inspect.isawaitable(result):
            try:
                result = await asyncio.wait_for(
                    result,
                    timeout=HEALTH_CHECK_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                return "unhealthy", f"Health check timed out after {HEALTH_CHECK_TIMEOUT_SECONDS}s"
            except Exception as exc:
                return "unhealthy", str(exc)

        return ("healthy" if bool(result) else "unhealthy"), None
