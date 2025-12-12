"""TTL cache for collection info and metadata.

Reduces redundant Qdrant calls by caching collection dimensions and metadata
with configurable TTL. Thread-safe for async usage via simple dict + timestamps.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Cache TTL in seconds (5 minutes)
CACHE_TTL_SECONDS = 300

# Maximum entries per cache to prevent unbounded growth
MAX_CACHE_ENTRIES = 100

# Cache structure: {collection_name: (timestamp, vector_dim, collection_info)}
_collection_info_cache: dict[str, tuple[float, int, dict[str, Any] | None]] = {}

# Cache structure: {collection_name: (timestamp, metadata_dict)}
_collection_metadata_cache: dict[str, tuple[float, dict[str, Any] | None]] = {}


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
    entry = _collection_info_cache.get(name)
    if entry is None:
        logger.debug("Cache miss for collection info: %s", name)
        return None

    timestamp, vector_dim, info = entry
    if _is_expired(timestamp):
        logger.debug("Cache expired for collection info: %s", name)
        del _collection_info_cache[name]
        return None

    logger.debug("Cache hit for collection info: %s", name)
    return vector_dim, info


def set_collection_info(name: str, dim: int, info: dict[str, Any] | None) -> None:
    """Cache collection info with TTL."""
    _evict_if_needed(_collection_info_cache)
    _collection_info_cache[name] = (time.monotonic(), dim, info)
    logger.debug("Cached collection info for: %s (dim=%d)", name, dim)


def get_collection_metadata(name: str) -> dict[str, Any] | None:
    """Return cached metadata or None if expired/missing.

    Returns the metadata dict if found and not expired, otherwise None.
    Note: A cached None value (collection has no metadata) is returned as-is,
    distinguishable from cache miss via the _METADATA_CACHE_MISS sentinel.
    """
    entry = _collection_metadata_cache.get(name)
    if entry is None:
        logger.debug("Cache miss for collection metadata: %s", name)
        return _METADATA_CACHE_MISS

    timestamp, metadata = entry
    if _is_expired(timestamp):
        logger.debug("Cache expired for collection metadata: %s", name)
        del _collection_metadata_cache[name]
        return _METADATA_CACHE_MISS

    logger.debug("Cache hit for collection metadata: %s", name)
    return metadata


# Sentinel to distinguish "cache miss" from "cached None value"
_METADATA_CACHE_MISS: dict[str, Any] = {"__cache_miss__": True}


def is_cache_miss(value: dict[str, Any] | None) -> bool:
    """Check if the value represents a cache miss."""
    return value is _METADATA_CACHE_MISS


def set_collection_metadata(name: str, metadata: dict[str, Any] | None) -> None:
    """Cache metadata with TTL."""
    _evict_if_needed(_collection_metadata_cache)
    _collection_metadata_cache[name] = (time.monotonic(), metadata)
    logger.debug("Cached collection metadata for: %s", name)


def clear_cache() -> None:
    """Clear all caches (for testing)."""
    _collection_info_cache.clear()
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
