"""Remote plugin registry client with caching.

This module provides functions to fetch and cache plugin registry data
from a remote source, with fallback to a bundled registry for offline use.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# Default registry URL (GitHub raw content)
DEFAULT_REGISTRY_URL = "https://raw.githubusercontent.com/semantik/plugin-registry/main/registry.yaml"

# Cache duration (1 hour default)
DEFAULT_CACHE_DURATION_SECONDS = 3600

# Path to bundled registry
BUNDLED_REGISTRY_PATH = Path(__file__).parent / "data" / "registry.yaml"


class RegistryPlugin(BaseModel):
    """A plugin entry from the remote registry."""

    id: str
    type: str  # embedding, chunking, connector, reranker, extractor
    name: str
    description: str
    author: str
    repository: str
    pypi: str | None = None  # PyPI package name (legacy, optional)
    install_command: str | None = None  # pip install command (e.g., git+https://...)
    verified: bool = False
    min_semantik_version: str | None = None
    tags: list[str] = Field(default_factory=list)


class PluginRegistry(BaseModel):
    """Schema for the remote registry YAML."""

    registry_version: str
    last_updated: str
    plugins: list[RegistryPlugin] = Field(default_factory=list)


@dataclass
class RegistryCache:
    """In-memory cache for the registry."""

    registry: PluginRegistry | None = None
    fetched_at: datetime | None = None
    source: str | None = None  # "remote" or "bundled"
    cache_duration: timedelta = field(default_factory=lambda: timedelta(seconds=DEFAULT_CACHE_DURATION_SECONDS))

    def is_valid(self) -> bool:
        """Return True if cache is still valid."""
        if self.registry is None or self.fetched_at is None:
            return False
        return datetime.now(UTC) - self.fetched_at < self.cache_duration

    def invalidate(self) -> None:
        """Clear the cache."""
        self.registry = None
        self.fetched_at = None
        self.source = None


# Global cache instance
_registry_cache = RegistryCache()
_cache_lock = asyncio.Lock()
_inflight_fetch: asyncio.Task[PluginRegistry] | None = None


def get_registry_url() -> str:
    """Get the registry URL from environment or default."""
    return os.environ.get("SEMANTIK_PLUGIN_REGISTRY_URL", DEFAULT_REGISTRY_URL)


def load_bundled_registry() -> PluginRegistry:
    """Load the bundled registry from package data.

    This always succeeds as it reads from bundled files.

    Returns:
        PluginRegistry loaded from bundled YAML.

    Raises:
        FileNotFoundError: If bundled registry is missing (should never happen).
        ValidationError: If bundled registry is malformed.
    """
    logger.debug("Loading bundled registry from %s", BUNDLED_REGISTRY_PATH)
    with BUNDLED_REGISTRY_PATH.open() as f:
        content = yaml.safe_load(f)
    return PluginRegistry.model_validate(content)


async def _fetch_remote_registry(
    url: str,
    timeout: float = 10.0,
) -> PluginRegistry | None:
    """Fetch registry from remote URL.

    Args:
        url: URL to fetch registry from.
        timeout: Request timeout in seconds.

    Returns:
        PluginRegistry if successful, None on error.
    """
    logger.info("Fetching plugin registry from %s", url)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Parse YAML content
            content = yaml.safe_load(response.text)
            registry = PluginRegistry.model_validate(content)

            logger.info(
                "Fetched registry with %d plugins (version %s)",
                len(registry.plugins),
                registry.registry_version,
            )
            return registry

    except httpx.HTTPStatusError as exc:
        logger.warning("Failed to fetch registry: HTTP %d", exc.response.status_code)
        return None
    except httpx.RequestError as exc:
        logger.warning("Failed to fetch registry: %s", exc)
        return None
    except yaml.YAMLError as exc:
        logger.warning("Invalid YAML in registry: %s", exc)
        return None
    except ValidationError as exc:
        logger.warning("Registry validation failed: %s", exc)
        return None


async def fetch_registry(
    force_refresh: bool = False,
    timeout: float = 10.0,
) -> PluginRegistry:
    """Fetch the plugin registry, with caching and fallback.

    Loading strategy:
    1. Return cached registry if valid and not forcing refresh
    2. Try fetching from remote URL
    3. Fall back to bundled registry if remote fails

    Args:
        force_refresh: If True, bypass cache and fetch fresh data.
        timeout: Request timeout in seconds.

    Returns:
        PluginRegistry from cache, remote, or bundled fallback.
    """
    global _inflight_fetch

    async def _fetch_and_update_cache() -> PluginRegistry:
        """Fetch registry from remote (or bundled fallback) and update cache."""
        url = get_registry_url()
        registry = await _fetch_remote_registry(url, timeout)
        source = "remote"

        if registry is None:
            logger.info("Using bundled registry as fallback")
            try:
                registry = load_bundled_registry()
                source = "bundled"
            except Exception as exc:
                # This should never happen if the bundled file exists
                logger.error("Failed to load bundled registry: %s", exc)
                raise

        # Update cache under lock
        async with _cache_lock:
            _registry_cache.registry = registry
            _registry_cache.fetched_at = datetime.now(UTC)
            _registry_cache.source = source
        return registry

    # Single-flight: ensure only one fetch runs at a time
    async with _cache_lock:
        if not force_refresh and _registry_cache.is_valid():
            logger.debug("Returning cached registry (source: %s)", _registry_cache.source)
            return _registry_cache.registry  # type: ignore[return-value]

        if _inflight_fetch is not None and not _inflight_fetch.done():
            task = _inflight_fetch
        else:
            task = asyncio.create_task(_fetch_and_update_cache())
            _inflight_fetch = task

    try:
        return await task
    finally:
        async with _cache_lock:
            if _inflight_fetch is task and task.done():
                _inflight_fetch = None


def get_cached_registry() -> PluginRegistry | None:
    """Get the cached registry without fetching.

    Returns:
        Cached PluginRegistry or None if cache is empty/invalid.
    """
    if _registry_cache.is_valid():
        return _registry_cache.registry
    return None


def get_registry_source() -> str | None:
    """Get the source of the current cached registry.

    Returns:
        "remote", "bundled", or None if no cache.
    """
    if _registry_cache.is_valid():
        return _registry_cache.source
    return None


def invalidate_registry_cache() -> None:
    """Invalidate the registry cache."""
    global _inflight_fetch
    _registry_cache.invalidate()
    _inflight_fetch = None


async def list_available_plugins(
    plugin_type: str | None = None,
    verified_only: bool = False,
    force_refresh: bool = False,
) -> list[RegistryPlugin]:
    """List plugins from the registry with optional filtering.

    Args:
        plugin_type: Filter by plugin type (e.g., "embedding").
        verified_only: Only return verified plugins.
        force_refresh: Force cache refresh.

    Returns:
        List of matching plugins.
    """
    registry = await fetch_registry(force_refresh=force_refresh)
    plugins = registry.plugins

    if plugin_type:
        plugins = [p for p in plugins if p.type == plugin_type]

    if verified_only:
        plugins = [p for p in plugins if p.verified]

    return plugins


async def get_registry_metadata() -> dict[str, Any]:
    """Get registry metadata (version, last updated, source).

    Returns:
        Dict with registry_version, last_updated, and source.
    """
    registry = await fetch_registry()
    return {
        "registry_version": registry.registry_version,
        "last_updated": registry.last_updated,
        "source": get_registry_source(),
    }
