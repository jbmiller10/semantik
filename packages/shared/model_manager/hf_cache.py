"""HuggingFace cache utilities for the model manager.

Provides functions for scanning and analyzing the HuggingFace hub cache
to determine which models are installed locally.

IMPORTANT: The scan_hf_cache() function is synchronous. Callers in async
contexts must use asyncio.to_thread() to avoid blocking the event loop.
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

# Repo identity: HuggingFace Hub allows the same repo_id across repo types
# (model|dataset|space). We key installed repos by (repo_type, repo_id) to avoid
# collisions and incorrect install/size reporting.
RepoKey = tuple[str, str]

# Module-level cache state for TTL caching
_cache_result: "HFCacheInfo | None" = None
_cache_timestamp: float = 0.0
_cache_path: Path | None = None

# Default TTL from constants
DEFAULT_TTL_SECONDS = 15


def resolve_hf_cache_dir() -> Path:
    """Resolve the HuggingFace hub cache directory.

    Resolution priority:
    1. HF_HUB_CACHE environment variable
    2. HF_HOME environment variable + /hub
    3. Default ~/.cache/huggingface/hub

    Returns:
        Path to the HuggingFace hub cache directory.
    """
    # Check HF_HUB_CACHE first
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        return Path(hf_hub_cache)

    # Check HF_HOME
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"

    # Default location
    return Path.home() / ".cache" / "huggingface" / "hub"


@dataclass
class InstalledModel:
    """Information about a model installed in the HuggingFace cache."""

    repo_id: str
    size_on_disk_mb: int
    repo_type: str
    last_accessed: datetime | None
    revisions: list[str]


class CacheSizeBreakdown(TypedDict):
    """Breakdown of HuggingFace cache size."""

    total_cache_size_mb: int
    managed_cache_size_mb: int
    unmanaged_cache_size_mb: int
    unmanaged_repo_count: int


@dataclass
class HFCacheInfo:
    """Parsed HuggingFace cache information."""

    cache_dir: Path
    repos: dict[RepoKey, InstalledModel]
    total_size_mb: int
    scan_time: datetime


def scan_hf_cache(
    force_refresh: bool = False,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> HFCacheInfo:
    """Scan the HuggingFace cache directory.

    This is a SYNCHRONOUS function. In async contexts, use:
        info = await asyncio.to_thread(scan_hf_cache, force_refresh=True)

    Uses TTL-based caching to avoid excessive filesystem scans.

    Args:
        force_refresh: If True, bypass the TTL cache and force a fresh scan.
        ttl_seconds: Cache TTL in seconds (default: 15).

    Returns:
        HFCacheInfo with parsed cache information.
    """
    global _cache_result, _cache_timestamp, _cache_path

    cache_dir = resolve_hf_cache_dir()
    now = time.monotonic()

    # Check if we can use cached result
    if not force_refresh and _cache_result is not None:
        cache_age = now - _cache_timestamp
        if cache_age < ttl_seconds and _cache_path == cache_dir:
            return _cache_result

    # Scan the cache
    repos: dict[RepoKey, InstalledModel] = {}
    total_size_bytes = 0

    if cache_dir.exists():
        try:
            from huggingface_hub import scan_cache_dir

            scan_result = scan_cache_dir(cache_dir)
            total_size_bytes = scan_result.size_on_disk

            for repo in scan_result.repos:
                # Get the latest revision's last modified time
                last_accessed = None
                revisions = []
                for revision in repo.revisions:
                    revisions.append(revision.commit_hash[:8])
                    if revision.last_modified is not None:
                        modified_dt = datetime.fromtimestamp(revision.last_modified, tz=UTC)
                        if last_accessed is None or modified_dt > last_accessed:
                            last_accessed = modified_dt

                key: RepoKey = (repo.repo_type, repo.repo_id)
                repos[key] = InstalledModel(
                    repo_id=repo.repo_id,
                    size_on_disk_mb=repo.size_on_disk // (1024 * 1024),
                    repo_type=repo.repo_type,
                    last_accessed=last_accessed,
                    revisions=revisions,
                )
        except ImportError:
            # huggingface_hub not installed - expected in some deployments
            logger.debug("huggingface_hub not installed - HF cache scan unavailable")
        except PermissionError as e:
            logger.warning(
                "Permission denied scanning HF cache at %s: %s. Models may show as not installed.",
                cache_dir,
                e,
            )
        except Exception as e:
            logger.warning(
                "Failed to scan HF cache at %s: %s. Models may show as not installed.",
                cache_dir,
                e,
            )

    # Build result
    result = HFCacheInfo(
        cache_dir=cache_dir,
        repos=repos,
        total_size_mb=total_size_bytes // (1024 * 1024),
        scan_time=datetime.now(tz=UTC),
    )

    # Update cache
    _cache_result = result
    _cache_timestamp = now
    _cache_path = cache_dir

    return result


def get_installed_models(force_refresh: bool = False) -> dict[str, InstalledModel]:
    """Get dictionary of installed models from HuggingFace cache.

    This is a SYNCHRONOUS function. In async contexts, use:
        models = await asyncio.to_thread(get_installed_models, force_refresh=True)

    Args:
        force_refresh: If True, bypass the TTL cache and force a fresh scan.

    Returns:
        Dictionary mapping repo_id to InstalledModel for repos of type "model".
    """
    repos = scan_hf_cache(force_refresh=force_refresh).repos
    return {repo_id: repo for (repo_type, repo_id), repo in repos.items() if repo_type == "model"}


def is_model_installed(model_id: str) -> bool:
    """Check if a model is installed in the HuggingFace cache.

    This is a SYNCHRONOUS function. In async contexts, use:
        installed = await asyncio.to_thread(is_model_installed, model_id)

    Args:
        model_id: The HuggingFace model ID (e.g., "Qwen/Qwen3-Embedding-0.6B").

    Returns:
        True if the model is installed, False otherwise.
    """
    return model_id in get_installed_models()


def get_model_size_on_disk(model_id: str) -> int | None:
    """Get the size of a model on disk in MB.

    This is a SYNCHRONOUS function. In async contexts, use:
        size = await asyncio.to_thread(get_model_size_on_disk, model_id)

    Args:
        model_id: The HuggingFace model ID.

    Returns:
        Size in MB if installed, None otherwise.
    """
    models = get_installed_models()
    if model_id in models:
        return models[model_id].size_on_disk_mb
    return None


def get_cache_size_info(curated_model_ids: set[str]) -> CacheSizeBreakdown:
    """Get cache size breakdown for managed vs unmanaged models.

    This is a SYNCHRONOUS function. In async contexts, use:
        info = await asyncio.to_thread(get_cache_size_info, curated_ids)

    Args:
        curated_model_ids: Set of model IDs considered "managed" by the model manager.

    Returns:
        CacheSizeBreakdown with total, managed, and unmanaged sizes.
    """
    cache_info = scan_hf_cache()

    managed_size_mb = 0
    unmanaged_count = 0

    for (repo_type, repo_id), repo in cache_info.repos.items():
        is_managed_model = repo_type == "model" and repo_id in curated_model_ids
        if is_managed_model:
            managed_size_mb += repo.size_on_disk_mb
        else:
            unmanaged_count += 1

    return CacheSizeBreakdown(
        total_cache_size_mb=cache_info.total_size_mb,
        managed_cache_size_mb=managed_size_mb,
        unmanaged_cache_size_mb=max(cache_info.total_size_mb - managed_size_mb, 0),
        unmanaged_repo_count=unmanaged_count,
    )


def clear_cache() -> None:
    """Clear the module-level cache.

    Useful for testing or when you know the cache state has changed externally.
    """
    global _cache_result, _cache_timestamp, _cache_path
    _cache_result = None
    _cache_timestamp = 0.0
    _cache_path = None


__all__ = [
    "CacheSizeBreakdown",
    "HFCacheInfo",
    "InstalledModel",
    "clear_cache",
    "get_cache_size_info",
    "get_installed_models",
    "get_model_size_on_disk",
    "is_model_installed",
    "resolve_hf_cache_dir",
    "scan_hf_cache",
]
