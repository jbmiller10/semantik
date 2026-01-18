"""Repository implementation for system settings key-value store."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, select

from shared.database.db_retry import with_db_retry
from shared.database.exceptions import DatabaseOperationError
from shared.database.models import SystemSettings

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class SystemSettingsRepository:
    """Repository for system-wide settings management.

    This repository manages key-value configuration settings that can be
    modified by administrators through the UI. Values are stored as JSON
    and a null value means "use environment variable fallback".

    Example:
        ```python
        repo = SystemSettingsRepository(session)

        # Get a single setting
        max_collections = await repo.get_setting("max_collections_per_user")

        # Get all settings as a dict
        all_settings = await repo.get_all_settings()

        # Update a single setting
        await repo.set_setting("max_collections_per_user", 20, user_id=admin_id)

        # Bulk update multiple settings
        updated_keys = await repo.set_settings(
            {"max_collections_per_user": 20, "cache_ttl_seconds": 600},
            user_id=admin_id,
        )
        ```
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with database session.

        Args:
            session: AsyncSession instance for database operations
        """
        self.session = session

    # =========================================================================
    # Read Operations
    # =========================================================================

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get_setting(self, key: str) -> Any | None:
        """Get a single setting value by key.

        Args:
            key: The setting key (e.g., "max_collections_per_user")

        Returns:
            The setting value, or None if not found or value is JSON null

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            result = await self.session.execute(select(SystemSettings).where(SystemSettings.key == key))
            setting = result.scalar_one_or_none()

            if setting is None:
                return None

            # JSON null means "use env var fallback"
            return setting.value

        except Exception as e:
            logger.error("Failed to get setting '%s': %s", key, e, exc_info=True)
            raise DatabaseOperationError("get", "SystemSettings", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get_all_settings(self) -> dict[str, Any]:
        """Get all system settings as a dictionary.

        Returns:
            Dict mapping setting keys to their values.
            Keys with JSON null values are included (value will be None).

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            result = await self.session.execute(select(SystemSettings))
            settings = result.scalars().all()

            return {s.key: s.value for s in settings}

        except Exception as e:
            logger.error("Failed to get all settings: %s", e, exc_info=True)
            raise DatabaseOperationError("get_all", "SystemSettings", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get_settings_with_metadata(self) -> dict[str, dict[str, Any]]:
        """Get all settings with their metadata (updated_at, updated_by).

        Returns:
            Dict mapping setting keys to dicts containing:
            - value: The setting value
            - updated_at: When the setting was last updated
            - updated_by: User ID who last updated (or None)

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            result = await self.session.execute(select(SystemSettings))
            settings = result.scalars().all()

            return {
                s.key: {
                    "value": s.value,
                    "updated_at": s.updated_at.isoformat() if s.updated_at else None,
                    "updated_by": s.updated_by,
                }
                for s in settings
            }

        except Exception as e:
            logger.error("Failed to get settings with metadata: %s", e, exc_info=True)
            raise DatabaseOperationError("get_all", "SystemSettings", str(e)) from e

    # =========================================================================
    # Write Operations
    # =========================================================================

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def set_setting(
        self,
        key: str,
        value: Any,
        user_id: int | None = None,
    ) -> None:
        """Set a single setting value.

        Creates the setting if it doesn't exist, updates if it does.

        Args:
            key: The setting key (max 64 chars)
            value: The setting value (must be JSON-serializable, None = use env fallback)
            user_id: ID of the user making the change (for audit trail)

        Raises:
            DatabaseOperationError: For database errors
        """
        if len(key) > 64:
            raise DatabaseOperationError("set", "SystemSettings", f"Key '{key}' exceeds 64 character limit")

        try:
            result = await self.session.execute(select(SystemSettings).where(SystemSettings.key == key))
            setting = result.scalar_one_or_none()

            if setting is None:
                # Create new setting
                setting = SystemSettings(
                    key=key,
                    value=value,
                    updated_by=user_id,
                )
                self.session.add(setting)
                logger.info(f"Created system setting: {key}")
            else:
                # Update existing setting
                setting.value = value
                setting.updated_at = datetime.now(UTC)
                setting.updated_by = user_id
                logger.info(f"Updated system setting: {key}")

            await self.session.flush()

        except DatabaseOperationError:
            raise
        except Exception as e:
            logger.error("Failed to set setting '%s': %s", key, e, exc_info=True)
            raise DatabaseOperationError("set", "SystemSettings", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def set_settings(
        self,
        settings: dict[str, Any],
        user_id: int | None = None,
    ) -> list[str]:
        """Bulk update multiple settings.

        Creates settings that don't exist, updates those that do.

        Args:
            settings: Dict mapping keys to values
            user_id: ID of the user making the change (for audit trail)

        Returns:
            List of keys that were updated

        Raises:
            DatabaseOperationError: For database errors
        """
        updated_keys: list[str] = []

        try:
            # Validate all keys first
            for key in settings:
                if len(key) > 64:
                    raise DatabaseOperationError(
                        "set_settings",
                        "SystemSettings",
                        f"Key '{key}' exceeds 64 character limit",
                    )

            # Get all existing settings for these keys
            result = await self.session.execute(select(SystemSettings).where(SystemSettings.key.in_(settings.keys())))
            existing = {s.key: s for s in result.scalars().all()}

            now = datetime.now(UTC)

            for key, value in settings.items():
                if key in existing:
                    # Update existing
                    setting = existing[key]
                    setting.value = value
                    setting.updated_at = now
                    setting.updated_by = user_id
                else:
                    # Create new
                    setting = SystemSettings(
                        key=key,
                        value=value,
                        updated_at=now,
                        updated_by=user_id,
                    )
                    self.session.add(setting)

                updated_keys.append(key)

            await self.session.flush()
            logger.info(f"Updated {len(updated_keys)} system settings")

            return updated_keys

        except DatabaseOperationError:
            raise
        except Exception as e:
            logger.error("Failed to bulk update settings: %s", e, exc_info=True)
            raise DatabaseOperationError("set_settings", "SystemSettings", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def delete_setting(self, key: str) -> bool:
        """Delete a setting by key.

        Args:
            key: The setting key to delete

        Returns:
            True if setting was deleted, False if it didn't exist

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            result = await self.session.execute(delete(SystemSettings).where(SystemSettings.key == key))
            deleted = (result.rowcount or 0) > 0

            if deleted:
                await self.session.flush()
                logger.info(f"Deleted system setting: {key}")

            return deleted

        except Exception as e:
            logger.error("Failed to delete setting '%s': %s", key, e, exc_info=True)
            raise DatabaseOperationError("delete", "SystemSettings", str(e)) from e
