"""Service for system settings with caching and env var fallback.

This service provides a caching layer for system settings, checking the
database first and falling back to environment variables when a setting
is not configured (JSON null) or missing.

Note: Cache is process-local. In multi-worker deployments, changes may
take up to CACHE_TTL seconds to propagate to other workers.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from shared.database.repositories.system_settings_repository import SystemSettingsRepository

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# Default values for settings (used when env var also not set)
SYSTEM_SETTING_DEFAULTS: dict[str, Any] = {
    # Resource Limits
    "max_collections_per_user": 10,
    "max_storage_gb_per_user": 50,
    "max_document_size_mb": 100,
    "max_artifact_size_mb": 50,
    # Performance
    "cache_ttl_seconds": 300,
    "model_unload_timeout_seconds": 300,
    "search_candidate_multiplier": 3,
    # GPU & Memory
    "gpu_memory_reserve_percent": 0.10,
    "gpu_memory_max_percent": 0.90,
    "cpu_memory_reserve_percent": 0.20,
    "cpu_memory_max_percent": 0.50,
    "enable_cpu_offload": True,
    "eviction_idle_threshold_seconds": 120,
    # Search & Reranking
    "rerank_candidate_multiplier": 5,
    "rerank_min_candidates": 20,
    "rerank_max_candidates": 200,
    "rerank_hybrid_weight": 0.3,
}


def _parse_env_value(key: str, value: str) -> Any:
    """Parse environment variable string to appropriate type.

    Uses the default value's type as a hint for parsing.
    """
    default = SYSTEM_SETTING_DEFAULTS.get(key)

    if default is None:
        return value

    if isinstance(default, bool):
        return value.lower() in ("true", "1", "yes", "on")
    if isinstance(default, int):
        try:
            return int(value)
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(value)
        except ValueError:
            return default

    return value


class SystemSettingsService:
    """Service for system settings with caching and env var fallback.

    Provides a simple interface for getting system settings with the following
    precedence:
    1. Cached value from database (if not None/null)
    2. Environment variable (if set)
    3. Default value from SYSTEM_SETTING_DEFAULTS

    The cache is refreshed automatically when it expires, with a lock to
    prevent thundering herd problems.

    Example:
        ```python
        service = SystemSettingsService(session_factory)

        # Get a single setting (uses cache)
        max_collections = await service.get_setting("max_collections_per_user")

        # Get setting with custom default
        timeout = await service.get_setting("custom_timeout", default=60)

        # Force cache refresh
        service.invalidate_cache()
        ```
    """

    CACHE_TTL = timedelta(seconds=60)

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        """Initialize the service with a session factory.

        Args:
            session_factory: Async context manager that yields AsyncSession instances.
        """
        self._session_factory = session_factory
        self._cache: dict[str, Any] = {}
        self._cache_expiry: datetime = datetime.min
        self._lock = asyncio.Lock()

    async def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value with caching and env var fallback.

        Precedence:
        1. Cached database value (if not None)
        2. Environment variable (parsed to appropriate type)
        3. Provided default or SYSTEM_SETTING_DEFAULTS

        Args:
            key: The setting key (e.g., "max_collections_per_user")
            default: Custom default if setting not found anywhere

        Returns:
            The setting value
        """
        await self._ensure_cache_fresh()

        # Check cache first (database values)
        cached_value = self._cache.get(key)
        if cached_value is not None:
            return cached_value

        # Fall back to environment variable
        env_key = key.upper()
        env_value = os.getenv(env_key)
        if env_value is not None:
            return _parse_env_value(key, env_value)

        # Fall back to defaults
        if default is not None:
            return default

        return SYSTEM_SETTING_DEFAULTS.get(key)

    async def get_all_settings(self) -> dict[str, Any]:
        """Get all settings with their effective values.

        Returns a dict with all known settings, each resolved through
        the cache -> env var -> default precedence chain.

        Returns:
            Dict mapping setting keys to their effective values
        """
        await self._ensure_cache_fresh()

        result: dict[str, Any] = {}

        for key in SYSTEM_SETTING_DEFAULTS:
            result[key] = await self.get_setting(key)

        # Also include any extra keys from cache not in defaults
        for key in self._cache:
            if key not in result:
                result[key] = self._cache[key]

        return result

    async def _ensure_cache_fresh(self) -> None:
        """Refresh cache if expired, with lock to prevent thundering herd."""
        if datetime.now(UTC) <= self._cache_expiry:
            return

        async with self._lock:
            # Double-check after acquiring lock (another coroutine may have refreshed)
            if datetime.now(UTC) <= self._cache_expiry:
                return
            await self._refresh_cache()

    async def _refresh_cache(self) -> None:
        """Refresh the cache from the database."""
        try:
            async with self._session_factory() as session:
                repo = SystemSettingsRepository(session)
                self._cache = await repo.get_all_settings()
                self._cache_expiry = datetime.now(UTC) + self.CACHE_TTL
                logger.debug("System settings cache refreshed with %d entries", len(self._cache))
        except Exception as e:
            logger.warning("Failed to refresh system settings cache: %s", e)
            # Keep using stale cache if refresh fails, but try again soon
            self._cache_expiry = datetime.now(UTC) + timedelta(seconds=5)

    def invalidate_cache(self) -> None:
        """Force cache refresh on next access."""
        self._cache_expiry = datetime.min
        logger.debug("System settings cache invalidated")


# Module-level service instance (lazy initialized)
_service_instance: SystemSettingsService | None = None


def get_system_settings_service(session_factory: Callable[[], AsyncSession]) -> SystemSettingsService:
    """Get or create the singleton SystemSettingsService instance.

    Args:
        session_factory: Async context manager that yields AsyncSession instances.

    Returns:
        The singleton SystemSettingsService instance.
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = SystemSettingsService(session_factory)
    return _service_instance


def reset_service_instance() -> None:
    """Reset the singleton instance (for testing)."""
    global _service_instance
    _service_instance = None
