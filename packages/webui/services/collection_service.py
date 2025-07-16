"""Collection Service for managing collection operations."""

import logging
import uuid
from typing import Any

from shared.database.exceptions import InvalidStateError
from shared.database.models import CollectionStatus, OperationType
from shared.database.repositories.collection_repository import CollectionRepository
from shared.database.repositories.document_repository import DocumentRepository
from shared.database.repositories.operation_repository import OperationRepository
from sqlalchemy.ext.asyncio import AsyncSession
from webui.celery_app import celery_app
from webui.utils.qdrant_manager import qdrant_manager

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_VECTOR_DIMENSION = 768
QDRANT_COLLECTION_PREFIX = "collection_"


class CollectionService:
    """Service for managing collection operations."""

    def __init__(
        self,
        db_session: AsyncSession,
        collection_repo: CollectionRepository,
        operation_repo: OperationRepository,
        document_repo: DocumentRepository,
    ):
        """Initialize the collection service."""
        self.db_session = db_session
        self.collection_repo = collection_repo
        self.operation_repo = operation_repo
        self.document_repo = document_repo

    async def create_collection(
        self,
        user_id: int,
        name: str,
        description: str | None = None,
        config: dict[str, Any] | None = None,
        resource_limits: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Create a new collection and dispatch indexing operation.

        Args:
            user_id: ID of the user creating the collection
            name: Name of the collection
            description: Optional description
            config: Optional configuration (embedding model, chunk settings, etc.)
            resource_limits: Optional resource limits

        Returns:
            Tuple of (collection, operation) dictionaries

        Raises:
            ValueError: If validation fails
            AccessDeniedError: If user doesn't have permission
        """
        # Validate inputs
        if not name or not name.strip():
            raise ValueError("Collection name is required")

        # Use transaction for atomic operations
        async with self.db_session.begin():
            # Create collection in database
            try:
                collection = await self.collection_repo.create(
                    user_id=user_id,
                    name=name,
                    description=description,
                    config=config or {},
                    resource_limits=resource_limits,
                )
            except Exception as e:
                logger.error(f"Failed to create collection: {e}")
                raise

            # Create operation record
            operation = await self.operation_repo.create(
                collection_id=collection["id"],
                user_id=user_id,
                type=OperationType.INDEX,
                config={
                    "sources": [],  # Initial creation has no sources
                    "collection_config": config or {},
                },
            )

            # Dispatch Celery task
            task_result = celery_app.send_task(
                "webui.tasks.process_collection_operation",
                args=[operation["uuid"]],
                task_id=str(uuid.uuid4()),
            )

            # Update operation with task ID
            await self.operation_repo.set_task_id(operation["uuid"], task_result.id)

            # Transaction commits here automatically

        return dict(collection), dict(operation)

    async def add_source(
        self,
        collection_id: str,
        user_id: int,
        source_path: str,
        source_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a source to an existing collection.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user performing the operation
            source_path: Path to the source (file or directory)
            source_config: Optional source-specific configuration

        Returns:
            Operation dictionary

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have permission
            InvalidStateError: If collection is in invalid state
        """
        # Get collection with permission check
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id, user_id=user_id
        )

        # Validate collection state
        if collection["status"] not in [CollectionStatus.READY, CollectionStatus.PARTIALLY_READY]:
            raise InvalidStateError(
                f"Cannot add source to collection in {collection['status']} state. "
                f"Collection must be in {CollectionStatus.READY} or {CollectionStatus.PARTIALLY_READY} state."
            )

        # Check if there's already an active operation
        active_ops = await self.operation_repo.get_active_operations_count(collection["id"])
        if active_ops > 0:
            raise InvalidStateError(
                "Cannot add source while another operation is in progress. "
                "Please wait for the current operation to complete."
            )

        # Use transaction for atomic operations
        async with self.db_session.begin():
            # Create operation record
            operation = await self.operation_repo.create(
                collection_id=collection["id"],
                user_id=user_id,
                type=OperationType.APPEND,
                config={
                    "source_path": source_path,
                    "source_config": source_config or {},
                },
            )

            # Update collection status to indexing
            await self.collection_repo.update_status(collection["id"], CollectionStatus.INDEXING)

            # Dispatch Celery task
            task_result = celery_app.send_task(
                "webui.tasks.process_collection_operation",
                args=[operation["uuid"]],
                task_id=str(uuid.uuid4()),
            )

            # Update operation with task ID
            await self.operation_repo.set_task_id(operation["uuid"], task_result.id)

            # Transaction commits here automatically

        return dict(operation)

    async def reindex_collection(
        self, collection_id: str, user_id: int, config_updates: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Reindex a collection with optional configuration updates.

        Implements blue-green reindexing for zero downtime.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user performing the operation
            config_updates: Optional configuration updates to apply

        Returns:
            Operation dictionary

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have permission
            InvalidStateError: If collection is in invalid state
        """
        # Get collection with permission check
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id, user_id=user_id
        )

        # Validate collection state
        if collection["status"] == CollectionStatus.INDEXING:
            raise InvalidStateError(
                "Cannot reindex collection that is currently indexing. "
                "Please wait for the current operation to complete."
            )

        if collection["status"] == CollectionStatus.FAILED:
            raise InvalidStateError(
                "Cannot reindex failed collection. Please delete and recreate the collection instead."
            )

        # Check if there's already an active operation
        active_ops = await self.operation_repo.get_active_operations_count(collection["id"])
        if active_ops > 0:
            raise InvalidStateError(
                "Cannot reindex while another operation is in progress. "
                "Please wait for the current operation to complete."
            )

        # Merge config updates with existing config
        new_config = collection["config"].copy()
        if config_updates:
            new_config.update(config_updates)

        # Use transaction for atomic operations
        async with self.db_session.begin():
            # Create operation record
            operation = await self.operation_repo.create(
                collection_id=collection["id"],
                user_id=user_id,
                type=OperationType.REINDEX,
                config={
                    "previous_config": collection["config"],
                    "new_config": new_config,
                    "blue_green": True,  # Always use blue-green for zero downtime
                },
            )

            # Update collection status to indexing
            await self.collection_repo.update_status(collection["id"], CollectionStatus.INDEXING)

            # Dispatch Celery task
            task_result = celery_app.send_task(
                "webui.tasks.process_collection_operation",
                args=[operation["uuid"]],
                task_id=str(uuid.uuid4()),
            )

            # Update operation with task ID
            await self.operation_repo.set_task_id(operation["uuid"], task_result.id)

            # Transaction commits here automatically

        return dict(operation)

    async def delete_collection(self, collection_id: str, user_id: int) -> None:
        """Delete a collection and all associated data.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user performing the operation

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have permission
            InvalidStateError: If collection cannot be deleted
        """
        # Get collection with permission check
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id, user_id=user_id, require_owner=True  # Only owner can delete
        )

        # Check if there's an active operation
        active_ops = await self.operation_repo.get_active_operations_count(collection["id"])
        if active_ops > 0:
            raise InvalidStateError(
                "Cannot delete collection while operations are in progress. "
                "Please cancel or wait for operations to complete."
            )

        try:
            # Delete from Qdrant if collection exists
            if collection["qdrant_collection_name"]:
                try:
                    qdrant_client = qdrant_manager.get_client()
                    # Check if collection exists in Qdrant
                    collections = qdrant_client.get_collections().collections
                    collection_names = [c.name for c in collections]
                    if collection["qdrant_collection_name"] in collection_names:
                        qdrant_client.delete_collection(collection["qdrant_collection_name"])
                        logger.info(f"Deleted Qdrant collection: {collection['qdrant_collection_name']}")
                except Exception as e:
                    logger.error(f"Failed to delete Qdrant collection: {e}")
                    # Continue with database deletion even if Qdrant deletion fails

            # Delete from database (cascade will handle operations, documents, etc.)
            await self.collection_repo.delete(collection["id"])

            logger.info(f"Deleted collection {collection_id} and all associated data")

        except Exception as e:
            logger.error(f"Failed to delete collection {collection_id}: {e}")
            raise

    async def remove_source(self, collection_id: str, user_id: int, source_path: str) -> dict[str, Any]:
        """Remove documents from a specific source.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user performing the operation
            source_path: Path of the source to remove

        Returns:
            Operation dictionary

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have permission
            InvalidStateError: If collection is in invalid state
        """
        # Get collection with permission check
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id, user_id=user_id
        )

        # Validate collection state
        if collection["status"] not in [CollectionStatus.READY, CollectionStatus.PARTIALLY_READY]:
            raise InvalidStateError(
                f"Cannot remove source from collection in {collection['status']} state. "
                f"Collection must be in {CollectionStatus.READY} or {CollectionStatus.PARTIALLY_READY} state."
            )

        # Check if there's already an active operation
        active_ops = await self.operation_repo.get_active_operations_count(collection["id"])
        if active_ops > 0:
            raise InvalidStateError(
                "Cannot remove source while another operation is in progress. "
                "Please wait for the current operation to complete."
            )

        # Use transaction for atomic operations
        async with self.db_session.begin():
            # Create operation record
            operation = await self.operation_repo.create(
                collection_id=collection["id"],
                user_id=user_id,
                type=OperationType.REMOVE_SOURCE,
                config={
                    "source_path": source_path,
                },
            )

            # Update collection status
            await self.collection_repo.update_status(collection["id"], CollectionStatus.PROCESSING)

            # Dispatch Celery task
            task_result = celery_app.send_task(
                "webui.tasks.process_collection_operation",
                args=[operation["uuid"]],
                task_id=str(uuid.uuid4()),
            )

            # Update operation with task ID
            await self.operation_repo.set_task_id(operation["uuid"], task_result.id)

            # Transaction commits here automatically

        return dict(operation)
