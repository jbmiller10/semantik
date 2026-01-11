"""TTL cache for collection info and metadata.

Reduces redundant Qdrant calls by caching collection dimensions and metadata
with configurable TTL. Thread-safe for async usage via simple dict + timestamps.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Cache TTL in seconds (5 minutes)
CACHE_TTL_SECONDS = 300

# Maximum entries per cache to prevent unbounded growth
MAX_CACHE_ENTRIES = 100

# Cache structure: {collection_name: (timestamp, vector_dim, collection_info)}
_collection_info_cache: dict[str, tuple[float, int, dict[str, Any] | None]] = {}

_collection_info_lock = threading.Lock()

# Cache structure: {collection_name: (timestamp, metadata_dict)}
_collection_metadata_cache: dict[str, tuple[float, dict[str, Any] | None]] = {}

_collection_metadata_lock = threading.Lock()


class _CacheMiss:
    """Sentinel class for cache miss detection.

    Using a dedicated singleton class ensures:
    1. Stable identity across module reloads
    2. Cannot be confused with legitimate cached values (like empty dicts)
    3. Clear semantics via __repr__
    4. Thread-safe singleton via __new__

    This pattern is preferred over using a dict sentinel because identity
    comparison (`is`) with a dict can fail under module reloading, pickling,
    or certain testing frameworks.
    """

    _instance: _CacheMiss | None = None

    def __new__(cls) -> _CacheMiss:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<CacheMiss>"

    def __reduce__(self) -> tuple[type[_CacheMiss], tuple[()]]:
        """Support pickling by returning the class constructor.

        This ensures unpickling returns the same singleton instance,
        which is important for Celery workers that may pickle/unpickle.
        """
        return (self.__class__, ())


# Module-level singleton instance for cache miss detection
_METADATA_CACHE_MISS: _CacheMiss = _CacheMiss()


def _is_expired(timestamp: float) -> bool:
    """Check if a cache entry has expired."""
    return (time.monotonic() - timestamp) > CACHE_TTL_SECONDS


def _evict_if_needed(cache: dict[str, Any]) -> None:
    """Evict oldest entries if cache is at or exceeds max size (to make room for new entry)."""
    if len(cache) < MAX_CACHE_ENTRIES:
        return
    # Sort by timestamp (oldest first) and remove enough to make room for one new entry
    sorted_keys = sorted(cache.keys(), key=lambda k: cache[k][0])
    # Remove entries to get below max (making room for the incoming entry)
    for key in sorted_keys[: len(cache) - MAX_CACHE_ENTRIES + 1]:
        del cache[key]


def get_collection_info(name: str) -> tuple[int, dict[str, Any] | None] | None:
    """Return cached (vector_dim, info) or None if expired/missing."""
    with _collection_info_lock:
        entry = _collection_info_cache.get(name)
    if entry is None:
        logger.debug("Cache miss for collection info: %s", name)
        return None

    timestamp, vector_dim, info = entry
    if _is_expired(timestamp):
        logger.debug("Cache expired for collection info: %s", name)
        with _collection_info_lock:
            _collection_info_cache.pop(name, None)
        return None

    logger.debug("Cache hit for collection info: %s", name)
    return vector_dim, info


def set_collection_info(name: str, dim: int, info: dict[str, Any] | None) -> None:
    """Cache collection info with TTL."""
    with _collection_info_lock:
        _evict_if_needed(_collection_info_cache)
        _collection_info_cache[name] = (time.monotonic(), dim, info)
    logger.debug("Cached collection info for: %s (dim=%d)", name, dim)


def get_collection_metadata(name: str) -> dict[str, Any] | None | _CacheMiss:
    """Return cached metadata or sentinel if expired/missing.

    Returns:
        - The metadata dict if found and not expired
        - None if collection explicitly has no metadata (cached None)
        - _CacheMiss sentinel if cache miss or expired

    Use is_cache_miss() to check for cache miss vs cached None value.
    """
    with _collection_metadata_lock:
        entry = _collection_metadata_cache.get(name)
    if entry is None:
        logger.debug("Cache miss for collection metadata: %s", name)
        return _METADATA_CACHE_MISS

    timestamp, metadata = entry
    if _is_expired(timestamp):
        logger.debug("Cache expired for collection metadata: %s", name)
        with _collection_metadata_lock:
            _collection_metadata_cache.pop(name, None)
        return _METADATA_CACHE_MISS

    logger.debug("Cache hit for collection metadata: %s", name)
    return metadata


def is_cache_miss(value: Any) -> bool:
    """Check if the value represents a cache miss.

    Uses isinstance check for robustness across module reloading scenarios.
    This is preferred over identity check (`is`) which can fail if the
    sentinel instance is recreated.
    """
    return isinstance(value, _CacheMiss)


def set_collection_metadata(name: str, metadata: dict[str, Any] | None) -> None:
    """Cache metadata with TTL."""
    with _collection_metadata_lock:
        _evict_if_needed(_collection_metadata_cache)
        _collection_metadata_cache[name] = (time.monotonic(), metadata)
    logger.debug("Cached collection metadata for: %s", name)


def clear_cache() -> None:
    """Clear all caches (for testing)."""
    with _collection_info_lock:
        _collection_info_cache.clear()
    with _collection_metadata_lock:
        _collection_metadata_cache.clear()
    logger.debug("Cleared all collection caches")


__all__ = [
    "CACHE_TTL_SECONDS",
    "get_collection_info",
    "set_collection_info",
    "get_collection_metadata",
    "set_collection_metadata",
    "is_cache_miss",
    "clear_cache",
]
