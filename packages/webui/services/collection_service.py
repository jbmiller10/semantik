"""Collection Service for managing collection operations."""

import logging
import uuid
from typing import Any, cast

from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.exceptions import (
    AccessDeniedError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    InvalidStateError,
)
from packages.shared.database.models import Collection, CollectionStatus, OperationType
from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.shared.database.repositories.document_repository import DocumentRepository
from packages.shared.database.repositories.operation_repository import OperationRepository
from packages.webui.celery_app import celery_app
from packages.webui.utils.qdrant_manager import qdrant_manager

logger = logging.getLogger(__name__)

# Configuration constants
QDRANT_COLLECTION_PREFIX = "collection_"
DEFAULT_VECTOR_DIMENSION = 1536  # Default vector dimension for embeddings


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
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Create a new collection and dispatch indexing operation.

        Args:
            user_id: ID of the user creating the collection
            name: Name of the collection
            description: Optional description
            config: Optional configuration (embedding model, chunk settings, etc.)

        Returns:
            Tuple of (collection, operation) dictionaries

        Raises:
            ValueError: If validation fails
            AccessDeniedError: If user doesn't have permission
        """
        # Validate inputs
        if not name or not name.strip():
            raise ValueError("Collection name is required")

        # Create collection in database
        try:
            collection = await self.collection_repo.create(
                owner_id=user_id,
                name=name,
                description=description,
                embedding_model=(
                    config.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B")
                    if config
                    else "Qwen/Qwen3-Embedding-0.6B"
                ),
                quantization=config.get("quantization", "float16") if config else "float16",
                chunk_size=config.get("chunk_size") if config else None,
                chunk_overlap=config.get("chunk_overlap") if config else None,
                chunking_strategy=config.get("chunking_strategy") if config else None,
                chunking_config=config.get("chunking_config") if config else None,
                is_public=config.get("is_public", False) if config else False,
                meta=config.get("metadata") if config else None,
            )
        except EntityAlreadyExistsError:
            # Re-raise EntityAlreadyExistsError to be handled by the API endpoint
            raise
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

        # Create operation record
        operation = await self.operation_repo.create(
            collection_id=collection.id,
            user_id=user_id,
            operation_type=OperationType.INDEX,
            config={
                "sources": [],  # Initial creation has no sources
                "collection_config": config or {},
            },
        )

        # Commit the transaction BEFORE dispatching the task
        await self.db_session.commit()

        # Dispatch Celery task AFTER commit to avoid race condition
        celery_app.send_task(
            "webui.tasks.process_collection_operation",
            args=[operation.uuid],
            task_id=str(uuid.uuid4()),
        )

        # Convert ORM objects to dictionaries
        collection_dict = {
            "id": collection.id,
            "name": collection.name,
            "description": collection.description,
            "owner_id": collection.owner_id,
            "vector_store_name": collection.vector_store_name,
            "embedding_model": collection.embedding_model,
            "quantization": collection.quantization,
            "chunk_size": collection.chunk_size,
            "chunk_overlap": collection.chunk_overlap,
            "is_public": collection.is_public,
            "metadata": collection.meta,
            "created_at": collection.created_at,
            "updated_at": collection.updated_at,
            "document_count": 0,  # New collection has no documents
            "vector_count": 0,  # New collection has no vectors
            "status": collection.status.value if hasattr(collection.status, "value") else collection.status,
            "config": {
                "embedding_model": collection.embedding_model,
                "quantization": collection.quantization,
                "chunk_size": collection.chunk_size,
                "chunk_overlap": collection.chunk_overlap,
                "is_public": collection.is_public,
                "metadata": collection.meta,
            },
        }

        operation_dict = {
            "uuid": operation.uuid,
            "collection_id": operation.collection_id,
            "type": operation.type.value,
            "status": operation.status.value,
            "config": operation.config,
            "created_at": operation.created_at,
            "started_at": operation.started_at,
            "completed_at": operation.completed_at,
            "error_message": operation.error_message,
        }

        return collection_dict, operation_dict

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
        if collection.status not in [CollectionStatus.PENDING, CollectionStatus.READY, CollectionStatus.DEGRADED]:
            raise InvalidStateError(
                f"Cannot add source to collection in {collection.status} state. "
                f"Collection must be in {CollectionStatus.PENDING}, {CollectionStatus.READY} or {CollectionStatus.DEGRADED} state."
            )

        # Check if there's already an active operation
        active_operations = await self.operation_repo.get_active_operations(collection.id)
        if active_operations:
            raise InvalidStateError(
                "Cannot add source while another operation is in progress. "
                "Please wait for the current operation to complete."
            )

        # Create operation record
        operation = await self.operation_repo.create(
            collection_id=collection.id,
            user_id=user_id,
            operation_type=OperationType.APPEND,
            config={
                "source_path": source_path,
                "source_config": source_config or {},
            },
        )

        # Update collection status to processing
        await self.collection_repo.update_status(collection.id, CollectionStatus.PROCESSING)

        # Commit the transaction BEFORE dispatching the task
        await self.db_session.commit()

        # Dispatch Celery task AFTER commit to avoid race condition
        celery_app.send_task(
            "webui.tasks.process_collection_operation",
            args=[operation.uuid],
            task_id=str(uuid.uuid4()),
        )

        # Convert ORM object to dictionary
        return {
            "uuid": operation.uuid,
            "collection_id": operation.collection_id,
            "type": operation.type.value,
            "status": operation.status.value,
            "config": operation.config,
            "created_at": operation.created_at,
            "started_at": operation.started_at,
            "completed_at": operation.completed_at,
            "error_message": operation.error_message,
        }

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
        if collection.status == CollectionStatus.PROCESSING:
            raise InvalidStateError(
                "Cannot reindex collection that is currently processing. "
                "Please wait for the current operation to complete."
            )

        if collection.status == CollectionStatus.ERROR:
            raise InvalidStateError(
                "Cannot reindex failed collection. Please delete and recreate the collection instead."
            )

        # Check if there's already an active operation
        active_ops = await self.operation_repo.get_active_operations_count(collection.id)
        if active_ops > 0:
            raise InvalidStateError(
                "Cannot reindex while another operation is in progress. "
                "Please wait for the current operation to complete."
            )

        # Merge config updates with existing config
        new_config = {
            "embedding_model": collection.embedding_model,
            "quantization": collection.quantization,
            "chunk_size": collection.chunk_size,
            "chunk_overlap": collection.chunk_overlap,
            "is_public": collection.is_public,
            "metadata": collection.meta,
        }
        if config_updates:
            new_config.update(config_updates)

        # Create operation record
        operation = await self.operation_repo.create(
            collection_id=collection.id,
            user_id=user_id,
            operation_type=OperationType.REINDEX,
            config={
                "previous_config": {
                    "embedding_model": collection.embedding_model,
                    "quantization": collection.quantization,
                    "chunk_size": collection.chunk_size,
                    "chunk_overlap": collection.chunk_overlap,
                    "is_public": collection.is_public,
                    "metadata": collection.meta,
                },
                "new_config": new_config,
                "blue_green": True,  # Always use blue-green for zero downtime
            },
        )

        # Update collection status to processing
        await self.collection_repo.update_status(collection.id, CollectionStatus.PROCESSING)

        # Commit the transaction BEFORE dispatching the task
        await self.db_session.commit()

        # Dispatch Celery task AFTER commit to avoid race condition
        celery_app.send_task(
            "webui.tasks.process_collection_operation",
            args=[operation.uuid],
            task_id=str(uuid.uuid4()),
        )

        # Convert ORM object to dictionary
        return {
            "uuid": operation.uuid,
            "collection_id": operation.collection_id,
            "type": operation.type.value,
            "status": operation.status.value,
            "config": operation.config,
            "created_at": operation.created_at,
            "started_at": operation.started_at,
            "completed_at": operation.completed_at,
            "error_message": operation.error_message,
        }

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
            collection_uuid=collection_id, user_id=user_id
        )

        # Only owner can delete
        if collection.owner_id != user_id:
            raise AccessDeniedError(user_id=str(user_id), resource_type="Collection", resource_id=collection_id)

        # Check if there's an active operation
        active_ops = await self.operation_repo.get_active_operations_count(collection.id)
        if active_ops > 0:
            raise InvalidStateError(
                "Cannot delete collection while operations are in progress. "
                "Please cancel or wait for operations to complete."
            )

        try:
            # Delete from Qdrant if collection exists
            if collection.vector_store_name:
                try:
                    qdrant_client = qdrant_manager.get_client()
                    # Check if collection exists in Qdrant
                    collections = qdrant_client.get_collections().collections
                    collection_names = [c.name for c in collections]
                    if collection.vector_store_name in collection_names:
                        qdrant_client.delete_collection(collection.vector_store_name)
                        logger.info(f"Deleted Qdrant collection: {collection.vector_store_name}")
                except Exception as e:
                    logger.error(f"Failed to delete Qdrant collection: {e}")
                    # Continue with database deletion even if Qdrant deletion fails

            # Delete from database (cascade will handle operations, documents, etc.)
            await self.collection_repo.delete(collection_id, user_id)

            # Commit the transaction to persist the deletion
            await self.db_session.commit()

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
        if collection.status not in [CollectionStatus.READY, CollectionStatus.DEGRADED]:
            raise InvalidStateError(
                f"Cannot remove source from collection in {collection.status} state. "
                f"Collection must be in {CollectionStatus.READY} or {CollectionStatus.DEGRADED} state."
            )

        # Check if there's already an active operation
        active_ops = await self.operation_repo.get_active_operations_count(collection.id)
        if active_ops > 0:
            raise InvalidStateError(
                "Cannot remove source while another operation is in progress. "
                "Please wait for the current operation to complete."
            )

        # Create operation record
        operation = await self.operation_repo.create(
            collection_id=collection.id,
            user_id=user_id,
            operation_type=OperationType.REMOVE_SOURCE,
            config={
                "source_path": source_path,
            },
        )

        # Update collection status
        await self.collection_repo.update_status(collection.id, CollectionStatus.PROCESSING)

        # Commit the transaction BEFORE dispatching the task
        await self.db_session.commit()

        # Dispatch Celery task AFTER commit to avoid race condition
        celery_app.send_task(
            "webui.tasks.process_collection_operation",
            args=[operation.uuid],
            task_id=str(uuid.uuid4()),
        )

        # Convert ORM object to dictionary
        return {
            "uuid": operation.uuid,
            "collection_id": operation.collection_id,
            "type": operation.type.value,
            "status": operation.status.value,
            "config": operation.config,
            "created_at": operation.created_at,
            "started_at": operation.started_at,
            "completed_at": operation.completed_at,
            "error_message": operation.error_message,
        }

    async def list_for_user(
        self, user_id: int, offset: int = 0, limit: int = 50, include_public: bool = True
    ) -> tuple[list[Collection], int]:
        """List collections accessible to a user.

        Args:
            user_id: ID of the user
            offset: Pagination offset
            limit: Pagination limit
            include_public: Whether to include public collections

        Returns:
            Tuple of (collections list, total count)
        """
        return await self.collection_repo.list_for_user(
            user_id=user_id,
            offset=offset,
            limit=limit,
            include_public=include_public,
        )

    async def update(self, collection_id: str, user_id: int, updates: dict[str, Any]) -> Collection:
        """Update collection metadata.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user performing the update
            updates: Dictionary of fields to update

        Returns:
            Updated collection object

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have permission
            EntityAlreadyExistsError: If new name already exists
        """
        # Get collection with permission check (only owner can update)
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id, user_id=user_id
        )

        # Only the owner can update the collection
        if collection.owner_id != user_id:
            raise AccessDeniedError(user_id=str(user_id), resource_type="Collection", resource_id=collection_id)

        # Perform the update
        updated_collection = await self.collection_repo.update(str(collection.id), updates)

        # Commit the transaction
        await self.db_session.commit()

        return cast(Collection, updated_collection)

    async def list_documents(
        self, collection_id: str, user_id: int, offset: int = 0, limit: int = 50
    ) -> tuple[list[Any], int]:
        """List documents in a collection.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user requesting documents
            offset: Pagination offset
            limit: Pagination limit

        Returns:
            Tuple of (documents list, total count)

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have permission
        """
        # Get collection with permission check
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id, user_id=user_id
        )

        # Get documents for the collection
        documents, total = await self.document_repo.list_by_collection(
            collection_id=collection.id,
            offset=offset,
            limit=limit,
        )

        return documents, total

    async def list_operations(
        self, collection_id: str, user_id: int, offset: int = 0, limit: int = 50
    ) -> tuple[list[Any], int]:
        """List operations for a collection.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user requesting operations
            offset: Pagination offset
            limit: Pagination limit

        Returns:
            Tuple of (operations list, total count)

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have permission
        """
        # Get collection with permission check
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id, user_id=user_id
        )

        # Get operations for the collection
        operations, total = await self.operation_repo.list_for_collection(
            collection_id=collection.id,
            user_id=user_id,
            offset=offset,
            limit=limit,
        )

        return operations, total

    async def create_operation(
        self,
        collection_id: str,
        operation_type: str,
        config: dict[str, Any],
        user_id: int,
    ) -> dict[str, Any]:
        """Create a new operation for a collection.

        Args:
            collection_id: Collection UUID
            operation_type: Type of operation
            config: Operation configuration
            user_id: User ID

        Returns:
            Created operation data
        """
        from packages.shared.database.models import OperationStatus, OperationType

        # Get collection
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id,
            user_id=user_id,
        )

        if not collection:
            raise EntityNotFoundError(f"Collection {collection_id} not found")

        # Map operation type string to enum
        operation_type_enum = {
            "chunking": OperationType.CHUNKING,
            "rechunking": OperationType.CHUNKING,
            "index": OperationType.INDEX,
            "reindex": OperationType.REINDEX,
        }.get(operation_type, OperationType.INDEX)

        # Create operation
        operation = await self.operation_repo.create(
            collection_id=collection.id,
            type=operation_type_enum,
            status=OperationStatus.PENDING,
            meta=config,
        )

        await self.db_session.commit()

        return {
            "uuid": operation.uuid,
            "collection_id": collection_id,
            "type": operation.type.value,
            "status": operation.status.value,
            "meta": operation.meta,
            "created_at": operation.created_at.isoformat() if operation.created_at else None,
        }

    async def update_collection(
        self,
        collection_id: str,
        updates: dict[str, Any],
        user_id: int,
    ) -> None:
        """Update collection settings.

        Args:
            collection_id: Collection UUID
            updates: Fields to update
            user_id: User ID
        """
        # Get collection with permission check
        collection = await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id,
            user_id=user_id,
        )

        if not collection:
            raise EntityNotFoundError(f"Collection {collection_id} not found")

        # Update allowed fields
        allowed_fields = ["name", "description", "chunking_strategy", "chunking_config", "meta"]
        for field, value in updates.items():
            if field in allowed_fields:
                setattr(collection, field, value)

        await self.db_session.commit()
