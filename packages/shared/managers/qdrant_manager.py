"""Utilities for managing Qdrant collections used across Semantik services."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import CollectionInfo, Distance, PointStruct, VectorParams

from shared.metrics.collection_metrics import QdrantOperationTimer

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


class QdrantCollectionNotFoundError(RuntimeError):
    """Raised when the requested Qdrant collection cannot be located."""

    def __init__(self, collection_name: str, message: str | None = None) -> None:
        msg = message or f"Collection {collection_name} not found in Qdrant"
        super().__init__(msg)
        self.collection_name = collection_name


class QdrantManager:
    """
    Service for managing Qdrant resources with support for blue-green deployments.

    This manager handles:
    - Creation of staging collections for zero-downtime reindexing
    - Cleanup of orphaned collections after migrations
    - Collection health checks and metadata management
    """

    def __init__(self, qdrant_client: QdrantClient):
        """
        Initialize the Qdrant manager.

        Args:
            qdrant_client: Pre-configured Qdrant client instance
        """
        self.client = qdrant_client
        self._staging_prefix = "staging_"
        self._collection_prefix = "collection_"

    def create_staging_collection(
        self,
        base_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
        optimizers_config: dict[str, Any] | None = None,
    ) -> str:
        """
        Create a uniquely named staging collection for blue-green deployment.

        Args:
            base_name: Base name for the collection (e.g., "collection_uuid")
            vector_size: Dimension of vectors to be stored
            distance: Distance metric to use (default: COSINE)
            optimizers_config: Optional optimizer configuration

        Returns:
            str: The name of the created staging collection

        Raises:
            Exception: If collection creation fails
        """
        # Generate unique staging collection name with timestamp
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        staging_name = f"{self._staging_prefix}{base_name}_{timestamp}"

        logger.info(f"Creating staging collection: {staging_name}")

        try:
            # Set default optimizer config if not provided
            if optimizers_config is None:
                optimizers_config = {
                    "indexing_threshold": 20000,
                    "memmap_threshold": 0,
                }

            # Create the staging collection
            self.client.create_collection(
                collection_name=staging_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
                optimizers_config=optimizers_config,
            )

            # Verify collection was created
            collection_info = self.client.get_collection(staging_name)
            if collection_info:
                logger.info(
                    f"Successfully created staging collection {staging_name} "
                    f"with {collection_info.vectors_count} vectors"
                )
                return staging_name
            raise Exception(f"Failed to verify staging collection creation: {staging_name}")

        except Exception as e:
            logger.error(f"Failed to create staging collection {staging_name}: {str(e)}")
            # Attempt cleanup if partially created
            with contextlib.suppress(Exception):
                self.client.delete_collection(staging_name)
            raise

    def cleanup_orphaned_collections(self, active_collections: list[str], dry_run: bool = False) -> list[str]:
        """
        Safely delete collections that are no longer referenced.

        This method identifies and removes Qdrant collections that are not in the
        active collections list, helping to clean up after failed operations or
        completed migrations.

        Args:
            active_collections: List of collection names that should be kept
            dry_run: If True, only report what would be deleted without deletion

        Returns:
            List[str]: Names of collections that were deleted (or would be deleted in dry_run)
        """
        logger.info(f"Starting orphaned collection cleanup (dry_run={dry_run})")

        try:
            # Get all collections from Qdrant
            all_collections = self.list_collections()

            # Convert active collections to set for efficient lookup
            active_set = set(active_collections)

            # Identify orphaned collections
            orphaned = []
            for collection_name in all_collections:
                # Skip system collections
                if collection_name.startswith("_"):
                    continue

                # Check if it's a staging collection or not in active set
                is_staging = collection_name.startswith(self._staging_prefix)
                is_orphaned = collection_name not in active_set

                if is_orphaned:
                    # For staging collections, check age before deletion
                    if is_staging:
                        if self._is_staging_collection_old(collection_name):
                            orphaned.append(collection_name)
                        else:
                            logger.info(f"Skipping recent staging collection: {collection_name}")
                    else:
                        orphaned.append(collection_name)

            logger.info(f"Found {len(orphaned)} orphaned collections")

            # Delete orphaned collections
            deleted = []
            for collection_name in orphaned:
                try:
                    collection_info = self._get_collection_info_safe(collection_name)
                    vector_count = collection_info.vectors_count if collection_info else 0

                    if dry_run:
                        logger.info(f"[DRY RUN] Would delete collection {collection_name} with {vector_count} vectors")
                    else:
                        logger.info(f"Deleting orphaned collection {collection_name} with {vector_count} vectors")
                        self.client.delete_collection(collection_name)
                        # Add small delay to avoid overwhelming Qdrant
                        time.sleep(0.1)

                    deleted.append(collection_name)

                except Exception as e:
                    logger.error(f"Failed to delete collection {collection_name}: {str(e)}")

            logger.info(f"Cleanup complete. {'Would delete' if dry_run else 'Deleted'} {len(deleted)} collections")
            return deleted

        except Exception as e:
            logger.error(f"Error during orphaned collection cleanup: {str(e)}")
            raise

    def list_collections(self) -> list[str]:
        """
        List all collection names in Qdrant.

        Returns:
            List[str]: Names of all collections
        """
        try:
            collections = self.client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {str(e)}")
            raise

    def get_collection_info(self, collection_name: str) -> CollectionInfo:
        """
        Get detailed information about a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            CollectionInfo: Collection information including vector count

        Raises:
            Exception: If collection doesn't exist or info retrieval fails
        """
        try:
            return self.client.get_collection(collection_name)
        except UnexpectedResponse as e:
            if e.status_code == 404:
                raise ValueError(f"Collection {collection_name} not found") from e
            raise
        except Exception as e:
            logger.error(f"Failed to get collection info for {collection_name}: {str(e)}")
            raise

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of the collection to check

        Returns:
            bool: True if collection exists, False otherwise
        """
        try:
            self.client.get_collection(collection_name)
            return True
        except UnexpectedResponse as e:
            if e.status_code == 404:
                return False
            logger.error(f"Unexpected error checking collection {collection_name}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error checking collection existence: {str(e)}")
            return False

    async def get_collection_usage(self, collection_name: str) -> dict[str, int]:
        """Return document, vector, and storage usage metrics for a collection.

        The underlying Qdrant client is synchronous, so calls are executed in a
        thread executor to avoid blocking the event loop.
        """

        loop = asyncio.get_running_loop()

        def _fetch_usage() -> dict[str, int]:
            with QdrantOperationTimer("get_collection_usage"):
                try:
                    info = self.client.get_collection(collection_name)
                except UnexpectedResponse as exc:
                    if getattr(exc, "status_code", None) == 404:
                        raise QdrantCollectionNotFoundError(collection_name) from exc
                    raise

                stats: Any | None = None
                try:
                    stats = self.client.get_collection_stats(collection_name)
                except UnexpectedResponse as exc:
                    if getattr(exc, "status_code", None) == 404:
                        raise QdrantCollectionNotFoundError(collection_name) from exc
                    logger.warning("Failed to fetch stats for %s: %s", collection_name, exc)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning("Unexpected error fetching stats for %s: %s", collection_name, exc)

            return self._normalize_usage_payload(info, stats)

        return await loop.run_in_executor(None, _fetch_usage)

    async def rename_collection(self, old_name: str, new_name: str, batch_size: int = 256) -> None:
        """Rename a collection by cloning config + data to a new name then removing the old one."""

        loop = asyncio.get_running_loop()

        def _rename() -> None:
            self._rename_collection_sync(old_name, new_name, batch_size=batch_size)

        await loop.run_in_executor(None, _rename)

    def _is_staging_collection_old(self, collection_name: str, hours: int = 24) -> bool:
        """
        Check if a staging collection is older than specified hours.

        Args:
            collection_name: Name of the staging collection
            hours: Age threshold in hours (default: 24)

        Returns:
            bool: True if collection is older than threshold
        """
        try:
            # Extract timestamp from staging collection name
            # Format: staging_collection_uuid_YYYYMMDD_HHMMSS
            parts = collection_name.split("_")
            if len(parts) >= 4:
                date_str = parts[-2]
                time_str = parts[-1]
                timestamp_str = f"{date_str}_{time_str}"

                # Parse timestamp
                collection_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S").replace(tzinfo=UTC)
                age_hours = (datetime.now(UTC) - collection_time).total_seconds() / 3600

                return age_hours > hours
        except Exception as e:
            logger.warning(f"Could not parse timestamp from collection name {collection_name}: {str(e)}")
            # If we can't parse the timestamp, consider it old to be safe
            return True

        return True

    def _get_collection_info_safe(self, collection_name: str) -> CollectionInfo | None:
        """
        Safely get collection info, returning None if collection doesn't exist.

        Args:
            collection_name: Name of the collection

        Returns:
            Optional[CollectionInfo]: Collection info or None if not found
        """
        try:
            return self.get_collection_info(collection_name)
        except ValueError:
            return None
        except Exception as e:
            logger.warning(f"Error getting info for collection {collection_name}: {str(e)}")
            return None

    def validate_collection_health(self, collection_name: str) -> dict[str, Any]:
        """
        Perform health check on a collection.

        Args:
            collection_name: Name of the collection to check

        Returns:
            Dict containing health status and metrics
        """
        try:
            if not self.collection_exists(collection_name):
                return {"healthy": False, "exists": False, "error": "Collection does not exist"}

            info = self.get_collection_info(collection_name)

            # Basic health metrics
            health: dict[str, Any] = {
                "healthy": True,
                "exists": True,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": str(info.status),
                "optimizer_status": info.optimizer_status,
            }

            # Check for potential issues
            if str(info.status) != "green":
                health["healthy"] = False
                health["warning"] = f"Collection status is {info.status}"

            if info.optimizer_status and hasattr(info.optimizer_status, "error") and info.optimizer_status.error:
                health["healthy"] = False
                health["optimizer_error"] = str(info.optimizer_status.error)

            return health

        except Exception as e:
            return {"healthy": False, "exists": None, "error": str(e)}

    def _rename_collection_sync(self, old_name: str, new_name: str, batch_size: int) -> None:
        if old_name == new_name:
            logger.info("Requested rename for %s to identical name; skipping", old_name)
            return

        if not self.collection_exists(old_name):
            raise QdrantCollectionNotFoundError(old_name)

        if self.collection_exists(new_name):
            raise ValueError(f"Target collection {new_name} already exists")

        with QdrantOperationTimer("rename_collection_prepare"):
            old_info = self.get_collection_info(old_name)

        create_kwargs = self._build_collection_create_kwargs(old_info)

        logger.info("Creating replacement collection %s for %s", new_name, old_name)
        with QdrantOperationTimer("rename_collection_create"):
            self.client.create_collection(collection_name=new_name, **create_kwargs)

        try:
            copied = self._copy_collection_points(old_name, new_name, batch_size=batch_size)
            self._copy_payload_indexes(old_info, new_name)

            logger.info(
                "Deleting old collection %s after migrating %d points to %s",
                old_name,
                copied,
                new_name,
            )
            with QdrantOperationTimer("rename_collection_delete_old"):
                self.client.delete_collection(old_name)

        except Exception:
            logger.error("Rename failed; rolling back new collection %s", new_name)
            with contextlib.suppress(Exception):
                self.client.delete_collection(new_name)
            raise

    def _build_collection_create_kwargs(self, info: CollectionInfo) -> dict[str, Any]:
        config = getattr(info, "config", None)
        if config is None or getattr(config, "params", None) is None:
            raise RuntimeError("Collection config missing vector parameters; cannot rename")

        params = config.params
        vectors_config = getattr(params, "vectors", None)
        if vectors_config is None:
            raise RuntimeError("Collection vectors configuration unavailable")

        kwargs: dict[str, Any] = {"vectors_config": vectors_config}

        optional_mappings: list[tuple[str, Any]] = [
            ("sparse_vectors_config", getattr(config, "sparse_vectors_config", None) or getattr(params, "sparse_vectors", None)),
            ("optimizers_config", getattr(config, "optimizer_config", None)),
            ("hnsw_config", getattr(config, "hnsw_config", None)),
            ("wal_config", getattr(config, "wal_config", None)),
            ("quantization_config", getattr(config, "quantization_config", None)),
            ("shard_number", getattr(params, "shard_number", None)),
            ("replication_factor", getattr(params, "replication_factor", None)),
            ("write_consistency_factor", getattr(params, "write_consistency_factor", None)),
            ("on_disk_payload", getattr(params, "on_disk_payload", None)),
            ("on_disk_vector", getattr(params, "on_disk_vector", None)),
            ("sharding_method", getattr(params, "sharding_method", None)),
            ("disabled", getattr(params, "disabled", None)),
        ]

        for key, value in optional_mappings:
            if value is not None:
                kwargs[key] = value

        return kwargs

    def _copy_collection_points(self, source: str, destination: str, batch_size: int) -> int:
        logger.info("Copying points from %s to %s", source, destination)

        offset: Any | None = None
        copied = 0

        while True:
            with QdrantOperationTimer("rename_collection_scroll"):
                records, next_offset = self.client.scroll(
                    collection_name=source,
                    offset=offset,
                    limit=batch_size,
                    with_payload=True,
                    with_vectors=True,
                )

            if not records:
                break

            points = [
                PointStruct(id=record.id, vector=getattr(record, "vector", None), payload=getattr(record, "payload", None))
                for record in records
            ]

            with QdrantOperationTimer("rename_collection_upsert"):
                self.client.upsert(collection_name=destination, points=points, wait=True)

            copied += len(points)
            offset = next_offset
            if not next_offset:
                break

        logger.info("Copied %d points from %s to %s", copied, source, destination)
        return copied

    def _copy_payload_indexes(self, info: CollectionInfo, destination: str) -> None:
        schema = getattr(info, "payload_schema", None)
        if not schema:
            return

        for field_name, field_schema in schema.items():
            try:
                with QdrantOperationTimer("rename_collection_index"):
                    self.client.create_payload_index(
                        collection_name=destination,
                        field_name=field_name,
                        field_schema=field_schema,
                    )
            except Exception as exc:
                logger.error("Failed to recreate payload index %s on %s: %s", field_name, destination, exc)
                raise

    @staticmethod
    def _sum_vectors(value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, dict):
            return int(sum(v for v in value.values() if isinstance(v, int | float)))
        if isinstance(value, list | tuple):
            return int(sum(v for v in value if isinstance(v, int | float)))
        if isinstance(value, int | float):
            return int(value)
        return 0

    def _normalize_usage_payload(self, info: CollectionInfo, stats: Any | None) -> dict[str, int]:
        """Coerce Qdrant collection info/stat structures into simple counters."""

        documents = None
        if stats is not None:
            documents = getattr(stats, "points_count", None)
            if documents is None:
                documents = getattr(stats, "points", None)
        if documents is None:
            documents = getattr(info, "points_count", None)
        if documents is None:
            documents = getattr(info, "vectors_count", None)
        documents_count = self._sum_vectors(documents)

        vectors = None
        if stats is not None:
            vectors = getattr(stats, "vectors_count", None)
        if vectors is None:
            vectors = getattr(info, "vectors_count", None)
        vectors_count = self._sum_vectors(vectors)

        storage_bytes = None
        for candidate in (
            getattr(info, "disk_data_size", None),
            getattr(info, "storage_data_bytes", None),
            getattr(info, "data_size", None),
            getattr(stats, "disk_data_size", None) if stats is not None else None,
        ):
            if isinstance(candidate, int | float) and candidate >= 0:
                storage_bytes = int(candidate)
                break

        if storage_bytes is None:
            status = getattr(info, "status", None)
            if status is not None:
                disk_data = getattr(status, "disk_data", None)
                if isinstance(disk_data, dict):
                    storage_bytes = int(sum(v for v in disk_data.values() if isinstance(v, int | float)))
                else:
                    storage_bytes = getattr(disk_data, "data_files_size", None)
                    if isinstance(storage_bytes, int | float):
                        storage_bytes = int(storage_bytes)

        documents_count = documents_count or vectors_count
        vectors_count = vectors_count or documents_count
        storage_bytes = int(storage_bytes) if isinstance(storage_bytes, int | float) and storage_bytes >= 0 else 0

        return {
            "documents": int(documents_count or 0),
            "vectors": int(vectors_count or 0),
            "storage_bytes": storage_bytes,
        }
