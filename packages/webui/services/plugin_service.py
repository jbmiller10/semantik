"""Service layer for plugin management."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Iterable

from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.repositories.plugin_config_repository import PluginConfigRepository
from shared.plugins.adapters import get_config_schema
from shared.plugins.loader import load_plugins
from shared.plugins.registry import PluginRecord, PluginSource, plugin_registry

logger = logging.getLogger(__name__)

_DEFAULT_PLUGIN_TYPES = {"embedding", "chunking", "connector"}


def _coerce_type(value: Any, schema_type: str) -> bool:
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    return True


def _validate_value(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    schema_type = schema.get("type")
    if schema_type and not _coerce_type(value, schema_type):
        errors.append(f"{path}: expected {schema_type}")
        return errors

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value must be one of {schema['enum']}")
        return errors

    if schema_type == "array":
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                errors.extend(_validate_value(item, item_schema, f"{path}[{idx}]"))

    if schema_type == "object" and isinstance(value, dict):
        props = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{path}.{key}: field is required")
        for key, val in value.items():
            if key in props and isinstance(props[key], dict):
                errors.extend(_validate_value(val, props[key], f"{path}.{key}"))
            elif props and schema.get("additionalProperties") is False:
                errors.append(f"{path}.{key}: additional properties are not allowed")
    return errors


def validate_config_schema(config: dict[str, Any], schema: dict[str, Any] | None) -> list[str]:
    """Validate config against a JSON Schema (subset)."""
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
        return {
            "plugin_id": record.plugin_id,
            "health_status": config_row.health_status if config_row else "unknown",
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
            tasks.append(self._check_and_update_health(record))

        if not tasks:
            return

        results = await asyncio.gather(*tasks)
        for updated in results:
            if updated is not None:
                config_map[updated.id] = updated
        await self.db_session.commit()

    async def _check_and_update_health(self, record: PluginRecord) -> Any | None:
        status, error = await self._run_health_check(record)
        try:
            updated = await self.repo.update_health(
                plugin_id=record.plugin_id,
                plugin_type=record.plugin_type,
                status=status,
                error_message=error,
            )
            return updated
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to update health for %s: %s", record.plugin_id, exc)
            return None

    async def _run_health_check(self, record: PluginRecord) -> tuple[str, str | None]:
        health_fn = getattr(record.plugin_class, "health_check", None)
        if not callable(health_fn):
            return "unknown", None
        try:
            result = health_fn()
        except TypeError:
            return "unknown", None
        except Exception as exc:
            return "unhealthy", str(exc)

        if inspect.isawaitable(result):
            try:
                result = await result
            except Exception as exc:
                return "unhealthy", str(exc)

        return ("healthy" if bool(result) else "unhealthy"), None
