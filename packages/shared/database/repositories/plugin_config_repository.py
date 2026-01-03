"""Repository implementation for PluginConfig model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from shared.database.db_retry import with_db_retry
from shared.database.exceptions import DatabaseOperationError
from shared.database.models import PluginConfig

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class PluginConfigRepository:
    """Repository for PluginConfig model operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def list_configs(
        self,
        *,
        plugin_type: str | None = None,
        enabled: bool | None = None,
    ) -> list[PluginConfig]:
        """List plugin configs with optional filters."""
        try:
            stmt = select(PluginConfig)
            if plugin_type is not None:
                stmt = stmt.where(PluginConfig.type == plugin_type)
            if enabled is not None:
                stmt = stmt.where(PluginConfig.enabled.is_(enabled))
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as exc:
            logger.error("Failed to list plugin configs: %s", exc, exc_info=True)
            raise DatabaseOperationError("list", "PluginConfig", str(exc)) from exc

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get_config(self, plugin_id: str) -> PluginConfig | None:
        """Fetch a plugin config by plugin id."""
        try:
            result = await self.session.execute(select(PluginConfig).where(PluginConfig.id == plugin_id))
            return result.scalar_one_or_none()
        except Exception as exc:
            logger.error("Failed to get plugin config '%s': %s", plugin_id, exc, exc_info=True)
            raise DatabaseOperationError("get", "PluginConfig", str(exc)) from exc

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def upsert_config(
        self,
        *,
        plugin_id: str,
        plugin_type: str,
        enabled: bool | None = None,
        config: dict[str, Any] | None = None,
    ) -> PluginConfig:
        """Create or update a plugin config record atomically.

        Uses PostgreSQL ON CONFLICT to avoid race conditions when multiple
        requests try to create the same plugin config concurrently.
        """
        try:
            # Build insert values with defaults for new records
            insert_values = {
                "id": plugin_id,
                "type": plugin_type,
                "enabled": enabled if enabled is not None else True,
                "config": config if config is not None else {},
            }

            # Build update set - always update type, conditionally update enabled/config
            insert_stmt = pg_insert(PluginConfig).values(**insert_values)

            # On conflict, update fields based on what was provided
            update_set: dict[str, Any] = {
                "type": insert_stmt.excluded.type,
                "updated_at": func.now(),
            }
            if enabled is not None:
                update_set["enabled"] = insert_stmt.excluded.enabled
            if config is not None:
                update_set["config"] = insert_stmt.excluded.config

            upsert_stmt = insert_stmt.on_conflict_do_update(
                index_elements=["id"],
                set_=update_set,
            ).returning(PluginConfig)

            result = await self.session.execute(upsert_stmt)
            await self.session.flush()
            return result.scalar_one()
        except Exception as exc:
            logger.error("Failed to upsert plugin config '%s': %s", plugin_id, exc, exc_info=True)
            raise DatabaseOperationError("upsert", "PluginConfig", str(exc)) from exc

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def update_health(
        self,
        *,
        plugin_id: str,
        plugin_type: str,
        status: str,
        error_message: str | None,
    ) -> PluginConfig:
        """Update health status for a plugin config (creates record if missing)."""
        record = await self.upsert_config(plugin_id=plugin_id, plugin_type=plugin_type)
        record.health_status = status
        record.error_message = error_message
        record.last_health_check = func.now()
        await self.session.flush()
        return record

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def list_disabled_ids(self, *, plugin_types: Iterable[str] | None = None) -> set[str]:
        """Return plugin ids that are explicitly disabled."""
        try:
            stmt = select(PluginConfig.id).where(PluginConfig.enabled.is_(False))
            if plugin_types is not None:
                stmt = stmt.where(PluginConfig.type.in_(list(plugin_types)))
            result = await self.session.execute(stmt)
            return {row[0] for row in result.all()}
        except Exception as exc:
            logger.error("Failed to list disabled plugin ids: %s", exc, exc_info=True)
            raise DatabaseOperationError("list", "PluginConfig", str(exc)) from exc
