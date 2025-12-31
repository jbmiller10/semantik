"""Repository implementation for Collection model."""

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, desc, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from shared.database.exceptions import (
    AccessDeniedError,
    DatabaseOperationError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    ValidationError,
)
from shared.database.models import Collection, CollectionStatus, Document

logger = logging.getLogger(__name__)


class CollectionRepository:
    """Repository for Collection model operations."""

    def __init__(self, session: AsyncSession):
        """Initialize with database session.

        Args:
            session: AsyncSession instance for database operations
        """
        self.session = session

    async def create(
        self,
        name: str,
        owner_id: int,
        description: str | None = None,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        quantization: str = "float16",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunking_strategy: str | None = None,
        chunking_config: dict[str, Any] | None = None,
        is_public: bool = False,
        meta: dict[str, Any] | None = None,
        sync_mode: str = "one_time",
        sync_interval_minutes: int | None = None,
        sync_paused_at: datetime | None = None,
        sync_next_run_at: datetime | None = None,
    ) -> Collection:
        """Create a new collection.

        Args:
            name: Unique collection name
            owner_id: ID of the user creating the collection
            description: Optional description
            embedding_model: Model to use for embeddings
            quantization: Model quantization level (float32, float16, or int8)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            chunking_strategy: Chunking strategy type (e.g., 'recursive', 'semantic')
            chunking_config: Strategy-specific configuration
            is_public: Whether collection is publicly accessible
            meta: Optional metadata
            sync_mode: Sync mode ('one_time' or 'continuous')
            sync_interval_minutes: Sync interval in minutes for continuous mode (min 15)
            sync_paused_at: Timestamp when sync was paused, if applicable
            sync_next_run_at: Timestamp for when next sync should run

        Returns:
            Created Collection instance

        Raises:
            EntityAlreadyExistsError: If collection name already exists
            ValidationError: If chunk configuration is invalid
            DatabaseOperationError: For database errors
        """
        try:
            # Validate chunk configuration
            if chunk_size <= 0:
                raise ValidationError("Chunk size must be positive", "chunk_size")
            if chunk_overlap < 0:
                raise ValidationError("Chunk overlap cannot be negative", "chunk_overlap")
            if chunk_overlap >= chunk_size:
                raise ValidationError("Chunk overlap must be less than chunk size", "chunk_overlap")
            # Check if collection name already exists
            existing = await self.session.execute(select(Collection).where(Collection.name == name))
            if existing.scalar_one_or_none():
                raise EntityAlreadyExistsError("collection", name)

            # Generate UUID for the collection
            collection_id = str(uuid4())

            # Generate unique vector store name
            vector_store_name = f"col_{collection_id.replace('-', '_')}"

            collection = Collection(
                id=collection_id,
                name=name,
                owner_id=owner_id,
                description=description,
                embedding_model=embedding_model,
                quantization=quantization,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunking_strategy=chunking_strategy,
                chunking_config=chunking_config,
                is_public=is_public,
                vector_store_name=vector_store_name,
                status=CollectionStatus.PENDING,  # Pass enum object, not value
                meta=meta or {},
                sync_mode=sync_mode,
                sync_interval_minutes=sync_interval_minutes,
                sync_paused_at=sync_paused_at,
                sync_next_run_at=sync_next_run_at,
            )

            self.session.add(collection)
            await self.session.flush()

            logger.info(f"Created collection {collection.id} with name '{name}' for user {owner_id}")
            return collection

        except IntegrityError as e:
            logger.error("Integrity error creating collection: %s", e, exc_info=True)
            raise EntityAlreadyExistsError("collection", name) from e
        except (EntityAlreadyExistsError, ValidationError):
            raise
        except Exception as e:
            logger.error("Failed to create collection: %s", e, exc_info=True)
            raise DatabaseOperationError("create", "collection", str(e)) from e

    async def get_by_uuid(self, collection_uuid: str) -> Collection | None:
        """Get a collection by UUID.

        Args:
            collection_uuid: UUID of the collection

        Returns:
            Collection instance or None if not found
        """
        try:
            result = await self.session.execute(
                select(Collection).where(Collection.id == collection_uuid).options(selectinload(Collection.documents))
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Failed to get collection %s: %s", collection_uuid, e, exc_info=True)
            raise DatabaseOperationError("get", "collection", str(e)) from e

    async def get_by_name(self, name: str) -> Collection | None:
        """Get a collection by name.

        Args:
            name: Name of the collection

        Returns:
            Collection instance or None if not found
        """
        try:
            result = await self.session.execute(
                select(Collection).where(Collection.name == name).options(selectinload(Collection.documents))
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Failed to get collection by name '%s': %s", name, e, exc_info=True)
            raise DatabaseOperationError("get", "collection", str(e)) from e

    async def get_by_uuid_with_permission_check(self, collection_uuid: str, user_id: int) -> Collection:
        """Get a collection by UUID with permission check.

        Args:
            collection_uuid: UUID of the collection
            user_id: ID of the user requesting access

        Returns:
            Collection instance

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have access
        """
        collection = await self.get_by_uuid(collection_uuid)

        if not collection:
            raise EntityNotFoundError("collection", collection_uuid)

        # Check if user owns the collection or it's public
        if collection.owner_id != user_id and not collection.is_public:
            # TODO: Check CollectionPermission table for shared access
            raise AccessDeniedError(str(user_id), "collection", collection_uuid)

        return collection

    async def list_for_user(
        self,
        user_id: int,
        offset: int = 0,
        limit: int = 50,
        include_public: bool = True,
    ) -> tuple[list[Collection], int]:
        """List collections accessible to a user.

        Args:
            user_id: ID of the user
            offset: Pagination offset
            limit: Maximum number of results
            include_public: Whether to include public collections

        Returns:
            Tuple of (collections list, total count)
        """
        try:
            # Build query for collections with OR conditions (more efficient than union)
            if include_public:
                query = select(Collection).where(or_(Collection.owner_id == user_id, Collection.is_public.is_(True)))
            else:
                query = select(Collection).where(Collection.owner_id == user_id)

            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total = await self.session.scalar(count_query)

            # Get paginated results
            query = query.order_by(desc(Collection.created_at)).offset(offset).limit(limit)
            result = await self.session.execute(query)
            collections = result.scalars().all()

            return list(collections), total or 0

        except Exception as e:
            logger.error("Failed to list collections for user %s: %s", user_id, e, exc_info=True)
            raise DatabaseOperationError("list", "collections", str(e)) from e

    async def update_status(
        self,
        collection_uuid: str,
        status: CollectionStatus,
        status_message: str | None = None,
    ) -> Collection:
        """Update collection status.

        Args:
            collection_uuid: UUID of the collection
            status: New status
            status_message: Optional status message

        Returns:
            Updated Collection instance

        Raises:
            EntityNotFoundError: If collection not found
        """
        try:
            collection = await self.get_by_uuid(collection_uuid)
            if not collection:
                raise EntityNotFoundError("collection", collection_uuid)

            collection.status = status  # Pass enum object directly
            collection.status_message = status_message
            collection.updated_at = datetime.now(UTC)

            await self.session.flush()

            logger.info(f"Updated collection {collection_uuid} status to {status}")
            return collection

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to update collection status: %s", e, exc_info=True)
            raise DatabaseOperationError("update", "collection", str(e)) from e

    async def update_stats(
        self,
        collection_uuid: str,
        document_count: int | None = None,
        vector_count: int | None = None,
        total_size_bytes: int | None = None,
    ) -> Collection:
        """Update collection statistics.

        Args:
            collection_uuid: UUID of the collection
            document_count: New document count
            vector_count: New vector count
            total_size_bytes: New total size

        Returns:
            Updated Collection instance

        Raises:
            EntityNotFoundError: If collection not found
            ValidationError: If any count is negative
        """
        try:
            # Validate non-negative counts
            if document_count is not None and document_count < 0:
                raise ValidationError("Document count cannot be negative", "document_count")
            if vector_count is not None and vector_count < 0:
                raise ValidationError("Vector count cannot be negative", "vector_count")
            if total_size_bytes is not None and total_size_bytes < 0:
                raise ValidationError("Total size cannot be negative", "total_size_bytes")

            collection = await self.get_by_uuid(collection_uuid)
            if not collection:
                raise EntityNotFoundError("collection", collection_uuid)

            if document_count is not None:
                collection.document_count = document_count
            if vector_count is not None:
                collection.vector_count = vector_count
            if total_size_bytes is not None:
                collection.total_size_bytes = total_size_bytes

            collection.updated_at = datetime.now(UTC)

            await self.session.flush()

            logger.debug(f"Updated collection {collection_uuid} stats")
            return collection

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to update collection stats: %s", e, exc_info=True)
            raise DatabaseOperationError("update", "collection", str(e)) from e

    async def rename(self, collection_uuid: str, new_name: str, user_id: int) -> Collection:
        """Rename a collection.

        Args:
            collection_uuid: UUID of the collection
            new_name: New collection name
            user_id: ID of the user performing the rename

        Returns:
            Updated Collection instance

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't own the collection
            EntityAlreadyExistsError: If new name already exists
            ValidationError: If new name is invalid
        """
        try:
            # Validate new name
            if not new_name or len(new_name.strip()) == 0:
                raise ValidationError("Collection name cannot be empty", "name")

            new_name = new_name.strip()

            # Check if new name already exists
            existing = await self.get_by_name(new_name)
            if existing:
                raise EntityAlreadyExistsError("collection", new_name)

            # Get collection with permission check
            collection = await self.get_by_uuid_with_permission_check(collection_uuid, user_id)

            # Only owner can rename
            if collection.owner_id != user_id:
                raise AccessDeniedError(str(user_id), "collection", collection_uuid)

            collection.name = new_name
            collection.updated_at = datetime.now(UTC)

            await self.session.flush()

            logger.info(f"Renamed collection {collection_uuid} to '{new_name}'")
            return collection

        except (EntityNotFoundError, AccessDeniedError, EntityAlreadyExistsError, ValidationError):
            raise
        except Exception as e:
            logger.error("Failed to rename collection: %s", e, exc_info=True)
            raise DatabaseOperationError("rename", "collection", str(e)) from e

    async def delete(self, collection_uuid: str, user_id: int) -> None:
        """Delete a collection.

        Args:
            collection_uuid: UUID of the collection
            user_id: ID of the user performing the deletion

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't own the collection
        """
        try:
            # Get collection without permission check first
            collection = await self.get_by_uuid(collection_uuid)

            if not collection:
                raise EntityNotFoundError("collection", collection_uuid)

            # Check if user is the owner (only owners can delete)
            if collection.owner_id != user_id:
                raise AccessDeniedError(str(user_id), "collection", collection_uuid)

            # Delete the collection using SQLAlchemy's delete statement for async
            # This will cascade delete all related records (operations, documents, etc.)
            await self.session.execute(delete(Collection).where(Collection.id == collection.id))
            await self.session.flush()

            logger.info(f"Deleted collection {collection_uuid}")

        except (EntityNotFoundError, AccessDeniedError):
            raise
        except Exception as e:
            logger.error("Failed to delete collection: %s", e, exc_info=True)
            raise DatabaseOperationError("delete", "collection", str(e)) from e

    async def update(self, collection_uuid: str, updates: dict[str, Any]) -> Collection:
        """Update a collection with multiple fields atomically.

        Args:
            collection_uuid: UUID of the collection
            updates: Dictionary of field names and values to update

        Returns:
            Updated Collection instance

        Raises:
            EntityNotFoundError: If collection not found
            ValidationError: If updates contain invalid fields
            DatabaseOperationError: For database errors
        """
        try:
            # Get the collection
            collection = await self.get_by_uuid(collection_uuid)
            if not collection:
                raise EntityNotFoundError("collection", collection_uuid)

            # List of allowed fields to update
            allowed_fields = {
                "name",
                "description",
                "embedding_model",
                "chunk_size",
                "chunk_overlap",
                "is_public",
                "status",
                "status_message",
                "document_count",
                "vector_count",
                "total_size_bytes",
                "qdrant_collections",
                "qdrant_staging",
                "config",
                "vector_store_name",
                "meta",
                # Sync policy fields
                "sync_mode",
                "sync_interval_minutes",
                "sync_paused_at",
                "sync_next_run_at",
                "sync_last_run_started_at",
                "sync_last_run_completed_at",
                "sync_last_run_status",
                "sync_last_error",
            }

            # Validate fields
            invalid_fields = set(updates.keys()) - allowed_fields
            if invalid_fields:
                raise ValidationError(f"Invalid fields: {invalid_fields}", "updates")

            # Apply updates
            for field, value in updates.items():
                # Perform field-specific validation
                if field == "chunk_size" and value is not None and value <= 0:
                    raise ValidationError("Chunk size must be positive", field)
                if field == "chunk_overlap" and value is not None:
                    if value < 0:
                        raise ValidationError("Chunk overlap cannot be negative", field)
                    chunk_size = updates.get("chunk_size", collection.chunk_size)
                    if value >= chunk_size:
                        raise ValidationError("Chunk overlap must be less than chunk size", field)
                if field in ["document_count", "vector_count", "total_size_bytes"] and value is not None and value < 0:
                    raise ValidationError(f"{field.replace('_', ' ').capitalize()} cannot be negative", field)

                setattr(collection, field, value)

            # Always update the timestamp
            collection.updated_at = datetime.now(UTC)

            await self.session.flush()

            logger.info(f"Updated collection {collection_uuid} with fields: {list(updates.keys())}")
            return collection

        except (EntityNotFoundError, ValidationError):
            raise
        except Exception as e:
            logger.error("Failed to update collection: %s", e, exc_info=True)
            raise DatabaseOperationError("update", "collection", str(e)) from e

    async def get_document_count(self, collection_uuid: str) -> int:
        """Get document count for a collection.

        Args:
            collection_uuid: UUID of the collection

        Returns:
            Number of documents in the collection
        """
        try:
            result = await self.session.scalar(
                select(func.count(Document.id)).where(Document.collection_id == collection_uuid)
            )
            return result or 0
        except Exception as e:
            logger.error("Failed to get document count: %s", e, exc_info=True)
            raise DatabaseOperationError("count", "documents", str(e)) from e

    # =========================================================================
    # Sync-related methods for collection-level continuous sync
    # =========================================================================

    async def get_due_for_sync(self, limit: int = 50) -> list[Collection]:
        """Get collections due for sync.

        Returns continuous sync collections where:
        - sync_mode = 'continuous'
        - sync_paused_at IS NULL
        - sync_next_run_at <= now()
        - status in (READY, DEGRADED)

        Args:
            limit: Maximum number of collections to return

        Returns:
            List of collections due for sync, ordered by sync_next_run_at

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            now = datetime.now(UTC)
            result = await self.session.execute(
                select(Collection)
                .where(
                    Collection.sync_mode == "continuous",
                    Collection.sync_paused_at.is_(None),
                    Collection.sync_next_run_at <= now,
                    Collection.status.in_([CollectionStatus.READY, CollectionStatus.DEGRADED]),
                )
                .order_by(Collection.sync_next_run_at.asc())
                .limit(limit)
            )
            return list(result.scalars().all())
        except Exception as e:
            logger.error("Failed to get collections due for sync: %s", e, exc_info=True)
            raise DatabaseOperationError("list", "collections", str(e)) from e

    async def update_sync_status(
        self,
        collection_uuid: str,
        status: str,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        error: str | None = None,
    ) -> Collection:
        """Update collection sync run status.

        Args:
            collection_uuid: UUID of the collection
            status: Sync status ('running', 'success', 'failed', 'partial')
            started_at: When the sync run started
            completed_at: When the sync run completed
            error: Error message if failed

        Returns:
            Updated Collection instance

        Raises:
            EntityNotFoundError: If collection not found
            DatabaseOperationError: For database errors
        """
        try:
            collection = await self.get_by_uuid(collection_uuid)
            if not collection:
                raise EntityNotFoundError("collection", collection_uuid)

            collection.sync_last_run_status = status

            if started_at is not None:
                collection.sync_last_run_started_at = started_at

            if completed_at is not None:
                collection.sync_last_run_completed_at = completed_at

            if error is not None:
                collection.sync_last_error = error

            collection.updated_at = datetime.now(UTC)

            await self.session.flush()

            logger.info(f"Updated collection {collection_uuid} sync status to {status}")
            return collection

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to update sync status: %s", e, exc_info=True)
            raise DatabaseOperationError("update", "collection", str(e)) from e

    async def set_next_sync_run(
        self,
        collection_uuid: str,
        next_run_at: datetime | None = None,
    ) -> Collection:
        """Schedule the next sync run.

        If next_run_at is not provided, calculates it from sync_interval_minutes.

        Args:
            collection_uuid: UUID of the collection
            next_run_at: Explicit next run time, or None to calculate

        Returns:
            Updated Collection instance

        Raises:
            EntityNotFoundError: If collection not found
            DatabaseOperationError: For database errors
        """
        try:
            from datetime import timedelta

            collection = await self.get_by_uuid(collection_uuid)
            if not collection:
                raise EntityNotFoundError("collection", collection_uuid)

            if next_run_at is None:
                # Calculate from interval
                interval = collection.sync_interval_minutes or 60
                next_run_at = datetime.now(UTC) + timedelta(minutes=interval)

            collection.sync_next_run_at = next_run_at
            collection.updated_at = datetime.now(UTC)

            await self.session.flush()

            logger.debug(f"Set collection {collection_uuid} next sync run at {next_run_at}")
            return collection

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to set next sync run: %s", e, exc_info=True)
            raise DatabaseOperationError("update", "collection", str(e)) from e

    async def pause_sync(self, collection_uuid: str) -> Collection:
        """Pause a collection's sync schedule.

        Sets sync_paused_at to now and clears sync_next_run_at.

        Args:
            collection_uuid: UUID of the collection

        Returns:
            Updated Collection instance

        Raises:
            EntityNotFoundError: If collection not found
            ValidationError: If collection is not in continuous sync mode
            DatabaseOperationError: For database errors
        """
        try:
            collection = await self.get_by_uuid(collection_uuid)
            if not collection:
                raise EntityNotFoundError("collection", collection_uuid)

            if collection.sync_mode != "continuous":
                raise ValidationError("Can only pause continuous sync collections", "sync_mode")

            collection.sync_paused_at = datetime.now(UTC)
            collection.sync_next_run_at = None
            collection.updated_at = datetime.now(UTC)

            await self.session.flush()

            logger.info(f"Paused sync for collection {collection_uuid}")
            return collection

        except (EntityNotFoundError, ValidationError):
            raise
        except Exception as e:
            logger.error("Failed to pause sync: %s", e, exc_info=True)
            raise DatabaseOperationError("update", "collection", str(e)) from e

    async def resume_sync(self, collection_uuid: str) -> Collection:
        """Resume a paused collection's sync schedule.

        Clears sync_paused_at and schedules next run immediately.
        If the collection is not paused, this is a no-op.

        Args:
            collection_uuid: UUID of the collection

        Returns:
            Updated Collection instance

        Raises:
            EntityNotFoundError: If collection not found
            ValidationError: If collection is not in continuous sync mode
            DatabaseOperationError: For database errors
        """
        try:
            collection = await self.get_by_uuid(collection_uuid)
            if not collection:
                raise EntityNotFoundError("collection", collection_uuid)

            if collection.sync_mode != "continuous":
                raise ValidationError("Can only resume continuous sync collections", "sync_mode")

            # If not paused, this is a no-op
            if collection.sync_paused_at is None:
                return collection

            collection.sync_paused_at = None
            collection.sync_next_run_at = datetime.now(UTC)  # Schedule immediately
            collection.updated_at = datetime.now(UTC)

            await self.session.flush()

            logger.info(f"Resumed sync for collection {collection_uuid}")
            return collection

        except (EntityNotFoundError, ValidationError):
            raise
        except Exception as e:
            logger.error("Failed to resume sync: %s", e, exc_info=True)
            raise DatabaseOperationError("update", "collection", str(e)) from e
