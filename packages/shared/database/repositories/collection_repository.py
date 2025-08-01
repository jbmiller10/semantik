"""Repository implementation for Collection model."""

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from shared.database.exceptions import (
    AccessDeniedError,
    DatabaseOperationError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    ValidationError,
)
from shared.database.models import Collection, CollectionStatus, Document
from sqlalchemy import delete, desc, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

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
        is_public: bool = False,
        meta: dict[str, Any] | None = None,
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
            is_public: Whether collection is publicly accessible
            meta: Optional metadata

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
                is_public=is_public,
                vector_store_name=vector_store_name,
                status=CollectionStatus.PENDING,  # Pass enum object, not value
                meta=meta or {},
            )

            self.session.add(collection)
            await self.session.flush()

            logger.info(f"Created collection {collection.id} with name '{name}' for user {owner_id}")
            return collection

        except IntegrityError as e:
            logger.error(f"Integrity error creating collection: {e}")
            raise EntityAlreadyExistsError("collection", name) from e
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
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
            logger.error(f"Failed to get collection {collection_uuid}: {e}")
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
            logger.error(f"Failed to get collection by name '{name}': {e}")
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
            logger.error(f"Failed to list collections for user {user_id}: {e}")
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
            logger.error(f"Failed to update collection status: {e}")
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
            logger.error(f"Failed to update collection stats: {e}")
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
            logger.error(f"Failed to rename collection: {e}")
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
            logger.error(f"Failed to delete collection: {e}")
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
            logger.error(f"Failed to update collection: {e}")
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
            logger.error(f"Failed to get document count: {e}")
            raise DatabaseOperationError("count", "documents", str(e)) from e
