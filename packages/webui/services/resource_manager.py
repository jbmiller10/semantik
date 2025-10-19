#!/usr/bin/env python3
"""
Resource Manager for Collection Operations.
Manages resource allocation, limits, and quotas for collection operations.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class ResourceEstimate:
    """Estimated resources for an operation."""

    def __init__(self, memory_mb: int = 0, storage_gb: float = 0.0, cpu_cores: float = 0.0, gpu_memory_mb: int = 0):
        self.memory_mb = memory_mb
        self.storage_gb = storage_gb
        self.cpu_cores = cpu_cores
        self.gpu_memory_mb = gpu_memory_mb

    def __str__(self) -> str:
        return (
            f"ResourceEstimate(memory={self.memory_mb}MB, storage={self.storage_gb}GB, "
            f"cpu={self.cpu_cores}, gpu={self.gpu_memory_mb}MB)"
        )


class ResourceManager:
    """Manages resource allocation for collection operations."""

    RESOURCE_USAGE_CACHE_TTL_SECONDS = 30

    def __init__(
        self,
        collection_repo: Any,
        operation_repo: Any,
        qdrant_manager: Any | None = None,
    ) -> None:
        self.collection_repo = collection_repo
        self.operation_repo = operation_repo
        self.qdrant_manager = qdrant_manager

        self._reserved_resources: dict[str, ResourceEstimate] = {}
        self._lock = asyncio.Lock()
        self._usage_cache: dict[str, tuple[dict[str, Any], datetime]] = {}
        self._usage_cache_lock = asyncio.Lock()
        self._usage_cache_ttl_seconds = self.RESOURCE_USAGE_CACHE_TTL_SECONDS

    async def can_create_collection(self, user_id: int) -> bool:
        """Check if user can create a new collection."""
        try:
            # Get user's current collection count
            collections, _ = await self.collection_repo.list_for_user(user_id)
            active_collections = [c for c in collections if c.status != "deleted"]

            # TODO: Get user's collection limit from user settings/subscription
            max_collections = 10  # Default limit

            return len(active_collections) < max_collections

        except Exception as e:
            logger.error(f"Failed to check collection creation limit: {e}")
            return False

    async def can_allocate(self, user_id: int, resources: ResourceEstimate) -> bool:
        """Check if resources can be allocated for an operation."""
        try:
            # Get system resources
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Check system resources (reserve 20% for system)
            available_memory_mb = (memory.available / 1024 / 1024) * 0.8
            available_storage_gb = (disk.free / 1024 / 1024 / 1024) * 0.8

            if resources.memory_mb > available_memory_mb:
                logger.warning(
                    f"Insufficient memory: requested={resources.memory_mb}MB, available={available_memory_mb}MB"
                )
                return False

            if resources.storage_gb > available_storage_gb:
                logger.warning(
                    f"Insufficient storage: requested={resources.storage_gb}GB, available={available_storage_gb}GB"
                )
                return False

            # Check user quotas
            user_usage = await self._get_user_resource_usage(user_id)

            # TODO: Get user's resource limits from settings/subscription
            max_storage_gb = 50.0  # Default 50GB per user
            # max_operations_per_hour = 10  # Not enforced - rate limiting disabled

            if user_usage["storage_gb"] + resources.storage_gb > max_storage_gb:
                logger.warning(f"User storage quota exceeded: current={user_usage['storage_gb']}GB")
                return False

            # Operation rate limit check disabled
            # recent_operations = await self._get_recent_operations_count(user_id, hours=1)
            # if recent_operations >= max_operations_per_hour:
            #     logger.warning(f"User operation rate limit exceeded: {recent_operations} in last hour")
            #     return False

            return True

        except Exception as e:
            logger.error(f"Failed to check resource allocation: {e}")
            return False

    async def estimate_resources(self, source_path: str, model_name: str) -> ResourceEstimate:
        """Estimate resources needed for processing a source path."""
        try:
            # Get total size of files

            total_size_bytes = 0
            file_count = 0

            path = Path(source_path)
            if path.is_file():
                total_size_bytes = path.stat().st_size
                file_count = 1
            elif path.is_dir():
                # Use rglob to recursively find all files
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        try:
                            total_size_bytes += file_path.stat().st_size
                            file_count += 1
                        except OSError:
                            pass

            # Estimate based on model and file size
            # These are rough estimates and should be tuned based on actual usage
            size_gb = total_size_bytes / 1024 / 1024 / 1024

            # Memory estimate: 2x file size + model size + overhead
            model_sizes = {
                "Qwen/Qwen3-Embedding-0.6B": 1200,  # 1.2GB
                "BAAI/bge-base-en-v1.5": 400,  # 400MB
                "sentence-transformers/all-MiniLM-L6-v2": 100,  # 100MB
            }
            model_memory = model_sizes.get(model_name, 1000)  # Default 1GB

            memory_mb = int((size_gb * 2048) + model_memory + 500)  # 500MB overhead

            # Storage estimate: original files + vectors + metadata
            # Vectors typically add 50-100% to storage needs
            storage_gb = size_gb * 1.75

            # CPU estimate: 1 core per 10 files or 1GB, whichever is higher
            cpu_cores = max(file_count / 10, size_gb, 1.0)

            # GPU memory if using GPU
            gpu_memory_mb = model_memory if self._is_gpu_model(model_name) else 0

            return ResourceEstimate(
                memory_mb=memory_mb, storage_gb=storage_gb, cpu_cores=cpu_cores, gpu_memory_mb=gpu_memory_mb
            )

        except Exception as e:
            logger.error(f"Failed to estimate resources: {e}")
            # Return conservative estimate
            return ResourceEstimate(memory_mb=2000, storage_gb=1.0, cpu_cores=1.0)

    async def reserve_for_reindex(self, collection_id: str) -> bool:
        """Reserve resources for reindexing operation."""
        async with self._lock:
            try:
                # Get collection info
                collection = await self.collection_repo.get_by_id(collection_id)
                if not collection:
                    return False

                # Estimate resources (2x current size for blue-green)
                size_gb = (collection.get("total_size_bytes", 0) / 1024 / 1024 / 1024) * 2
                memory_mb = int(size_gb * 1024 + 1000)  # Add 1GB overhead

                estimate = ResourceEstimate(memory_mb=memory_mb, storage_gb=size_gb, cpu_cores=2.0)

                # Check if resources available
                if not await self._check_system_resources(estimate):
                    return False

                # Reserve resources
                self._reserved_resources[f"reindex_{collection_id}"] = estimate
                return True

            except Exception as e:
                logger.error(f"Failed to reserve resources for reindex: {e}")
                return False

    async def release_reindex_reservation(self, collection_id: str) -> None:
        """Release reserved resources for reindex."""
        async with self._lock:
            key = f"reindex_{collection_id}"
            if key in self._reserved_resources:
                del self._reserved_resources[key]
                logger.info(f"Released resources for reindex of collection {collection_id}")

    async def get_resource_usage(self, collection_id: str) -> dict[str, Any]:
        """Get current resource usage for a collection."""
        try:
            now = datetime.now(UTC)

            async with self._usage_cache_lock:
                cached = self._usage_cache.get(collection_id)
                if cached is not None:
                    cached_usage, cached_at = cached
                    if (now - cached_at).total_seconds() < self._usage_cache_ttl_seconds:
                        return cached_usage
                    self._usage_cache.pop(collection_id, None)

            collection = await self.collection_repo.get_by_id(collection_id)
            if not collection:
                return {}

            vector_store_name = self._get_collection_value(collection, "vector_store_name")
            qdrant_collection_name = vector_store_name or f"collection_{collection_id}"

            if self.qdrant_manager is not None:
                try:
                    qdrant_usage = await self.qdrant_manager.get_collection_usage(qdrant_collection_name)
                    normalized = self._normalize_usage(qdrant_usage, collection)

                    async with self._usage_cache_lock:
                        self._usage_cache[collection_id] = (normalized, datetime.now(UTC))

                    return normalized
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning(
                        "Failed to retrieve Qdrant metrics for %s: %s -- using repository statistics",
                        qdrant_collection_name,
                        exc,
                    )

            return self._normalize_usage({}, collection)

        except Exception as e:
            logger.error(f"Failed to get resource usage: {e}")
            return {}

    async def _get_user_resource_usage(self, user_id: int) -> dict[str, Any]:
        """Get total resource usage for a user."""
        try:
            collections, _ = await self.collection_repo.list_for_user(user_id)

            totals = {
                "collections": len(collections),
                "storage_bytes": 0,
                "storage_gb": 0.0,
            }

            for collection in collections:
                collection_id = getattr(collection, "id", None) or getattr(collection, "uuid", None)
                if not collection_id:
                    continue

                usage = await self.get_resource_usage(str(collection_id))
                totals["storage_bytes"] += usage.get("storage_bytes", 0)

            totals["storage_gb"] = totals["storage_bytes"] / 1024 / 1024 / 1024

            return totals

        except Exception as e:
            logger.error(f"Failed to get user resource usage: {e}")
            return {"collections": 0, "storage_bytes": 0, "storage_gb": 0}

    async def _get_recent_operations_count(self, user_id: int, hours: int = 1) -> int:
        """Get count of recent operations for rate limiting."""
        try:
            since = datetime.now(UTC) - timedelta(hours=hours)
            operations = await self.operation_repo.list_by_user(user_id, since=since)
            return len(operations)

        except Exception as e:
            logger.error(f"Failed to get recent operations count: {e}")
            return 0

    @staticmethod
    def _get_collection_value(collection: Any, key: str, default: Any | None = None) -> Any:
        if isinstance(collection, dict):
            return collection.get(key, default)
        return getattr(collection, key, default)

    def _normalize_usage(self, usage: dict[str, Any], collection: Any) -> dict[str, Any]:
        documents = usage.get("documents") or usage.get("points")
        if documents is None:
            documents = self._get_collection_value(collection, "document_count", 0) or 0

        vectors = usage.get("vectors") or usage.get("points")
        if vectors is None:
            vectors = self._get_collection_value(collection, "vector_count", 0) or 0

        storage_bytes = (
            usage.get("storage_bytes")
            or usage.get("disk_usage_bytes")
            or self._get_collection_value(collection, "total_size_bytes", 0)
            or 0
        )

        storage_gb = storage_bytes / 1024 / 1024 / 1024

        return {
            "documents": documents,
            "vectors": vectors,
            "storage_bytes": storage_bytes,
            "storage_gb": storage_gb,
        }

    async def _check_system_resources(self, estimate: ResourceEstimate) -> bool:
        """Check if system has enough resources."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Include reserved resources in calculation
            total_reserved_memory = sum(r.memory_mb for r in self._reserved_resources.values())
            total_reserved_storage = sum(r.storage_gb for r in self._reserved_resources.values())

            available_memory_mb = (memory.available / 1024 / 1024) - total_reserved_memory
            available_storage_gb = (disk.free / 1024 / 1024 / 1024) - total_reserved_storage

            return bool(
                estimate.memory_mb <= available_memory_mb * 0.8 and estimate.storage_gb <= available_storage_gb * 0.8
            )

        except Exception as e:
            logger.error(f"Failed to check system resources: {e}")
            return False

    def _is_gpu_model(self, model_name: str) -> bool:  # noqa: ARG002
        """Check if model typically uses GPU."""
        # Most embedding models benefit from GPU
        return True
