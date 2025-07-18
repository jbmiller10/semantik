"""
Qdrant Management Service for managing vector storage resources.

This service provides methods for managing Qdrant collections with a focus on
supporting blue-green deployment strategies for zero-downtime reindexing.
"""

import contextlib
import logging
import time
from datetime import UTC, datetime
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import CollectionInfo, Distance, VectorParams

logger = logging.getLogger(__name__)


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

    def rename_collection(self, old_name: str, new_name: str) -> None:
        """
        Rename a collection by creating a new one and copying data.

        Note: Qdrant doesn't support direct renaming, so this creates a new
        collection and copies all data.

        Args:
            old_name: Current collection name
            new_name: New collection name

        Raises:
            Exception: If rename operation fails
        """
        logger.info(f"Renaming collection {old_name} to {new_name}")

        try:
            # Get old collection info
            old_info = self.get_collection_info(old_name)

            # Create new collection with same configuration
            self.client.create_collection(
                collection_name=new_name,
                vectors_config=old_info.config.params.vectors,
                optimizers_config=old_info.config.optimizer_config,
            )

            # Note: Actual data migration would require scrolling through points
            # and copying them. This is a placeholder for the structure.
            # In production, you might want to use Qdrant's snapshot feature
            # or implement point-by-point copying.

            logger.warning(
                f"Collection {new_name} created. Data migration from {old_name} "
                "needs to be implemented based on specific requirements."
            )

        except Exception as e:
            logger.error(f"Failed to rename collection: {str(e)}")
            # Cleanup new collection if it was created
            with contextlib.suppress(Exception):
                if self.collection_exists(new_name):
                    self.client.delete_collection(new_name)
            raise

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
