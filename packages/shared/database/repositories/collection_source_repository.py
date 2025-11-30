"""Repository implementation for CollectionSource model."""

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.exceptions import (
    DatabaseOperationError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    ValidationError,
)
from shared.database.models import CollectionSource

logger = logging.getLogger(__name__)


class CollectionSourceRepository:
    """Repository for CollectionSource model operations."""

    def __init__(self, session: AsyncSession):
        """Initialize with database session.

        Args:
            session: AsyncSession instance for database operations
        """
        self.session = session

    async def create(
        self,
        collection_id: str,
        source_type: str,
        source_path: str,
        source_config: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> CollectionSource:
        """Create a new collection source.

        Args:
            collection_id: UUID of the parent collection
            source_type: Type of source (directory, web, slack, etc.)
            source_path: Display path or identifier for the source
            source_config: Connector-specific configuration
            meta: Optional metadata

        Returns:
            Created CollectionSource instance

        Raises:
            EntityAlreadyExistsError: If source with same path already exists
            ValidationError: If required fields are invalid
            DatabaseOperationError: For database errors
        """
        try:
            # Validate inputs
            if not collection_id:
                raise ValidationError("Collection ID is required", "collection_id")
            if not source_type:
                raise ValidationError("Source type is required", "source_type")
            if not source_path:
                raise ValidationError("Source path is required", "source_path")

            source = CollectionSource(
                collection_id=collection_id,
                source_type=source_type,
                source_path=source_path,
                source_config=source_config or {},
                meta=meta or {},
            )

            self.session.add(source)
            await self.session.flush()

            logger.info(
                f"Created collection source {source.id} for collection {collection_id} "
                f"(type={source_type}, path={source_path})"
            )
            return source

        except IntegrityError as e:
            logger.error(f"Integrity error creating collection source: {e}")
            raise EntityAlreadyExistsError("collection_source", source_path) from e
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to create collection source: {e}")
            raise DatabaseOperationError("create", "collection_source", str(e)) from e

    async def get_by_id(self, source_id: int) -> CollectionSource | None:
        """Get a collection source by ID.

        Args:
            source_id: Integer ID of the source

        Returns:
            CollectionSource instance or None if not found
        """
        try:
            result = await self.session.execute(
                select(CollectionSource).where(CollectionSource.id == source_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get collection source {source_id}: {e}")
            raise DatabaseOperationError("get", "collection_source", str(e)) from e

    async def get_by_collection_and_path(
        self,
        collection_id: str,
        source_path: str,
    ) -> CollectionSource | None:
        """Get a collection source by collection ID and source path.

        Args:
            collection_id: UUID of the parent collection
            source_path: Path/identifier of the source

        Returns:
            CollectionSource instance or None if not found
        """
        try:
            result = await self.session.execute(
                select(CollectionSource).where(
                    CollectionSource.collection_id == collection_id,
                    CollectionSource.source_path == source_path,
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(
                f"Failed to get collection source for collection {collection_id} "
                f"with path {source_path}: {e}"
            )
            raise DatabaseOperationError("get", "collection_source", str(e)) from e

    async def get_or_create(
        self,
        collection_id: str,
        source_type: str,
        source_path: str,
        source_config: dict[str, Any] | None = None,
    ) -> tuple[CollectionSource, bool]:
        """Get an existing source or create a new one.

        Uses (collection_id, source_path) as the unique lookup key.
        If found, updates source_type and source_config if different.

        Args:
            collection_id: UUID of the parent collection
            source_type: Type of source (directory, web, slack, etc.)
            source_path: Display path or identifier for the source
            source_config: Connector-specific configuration

        Returns:
            Tuple of (CollectionSource, is_new) where is_new indicates creation

        Raises:
            ValidationError: If required fields are invalid
            DatabaseOperationError: For database errors
        """
        try:
            # Try to find existing source
            existing = await self.get_by_collection_and_path(collection_id, source_path)

            if existing:
                # Update source_type and source_config if they've changed
                updated = False
                if existing.source_type != source_type:
                    existing.source_type = source_type
                    updated = True
                if source_config and existing.source_config != source_config:
                    existing.source_config = source_config
                    updated = True

                if updated:
                    existing.updated_at = datetime.now(UTC)
                    await self.session.flush()
                    logger.info(
                        f"Updated existing collection source {existing.id} "
                        f"for collection {collection_id}"
                    )

                return existing, False

            # Create new source
            source = await self.create(
                collection_id=collection_id,
                source_type=source_type,
                source_path=source_path,
                source_config=source_config,
            )
            return source, True

        except (ValidationError, EntityAlreadyExistsError):
            raise
        except Exception as e:
            logger.error(f"Failed to get_or_create collection source: {e}")
            raise DatabaseOperationError("get_or_create", "collection_source", str(e)) from e

    async def list_by_collection(
        self,
        collection_id: str,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[CollectionSource], int]:
        """List sources for a collection.

        Args:
            collection_id: UUID of the parent collection
            offset: Pagination offset
            limit: Maximum number of results

        Returns:
            Tuple of (sources list, total count)
        """
        try:
            # Build base query
            query = select(CollectionSource).where(
                CollectionSource.collection_id == collection_id
            )

            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total = await self.session.scalar(count_query)

            # Get paginated results
            query = query.order_by(CollectionSource.created_at.desc()).offset(offset).limit(limit)
            result = await self.session.execute(query)
            sources = result.scalars().all()

            return list(sources), total or 0

        except Exception as e:
            logger.error(f"Failed to list sources for collection {collection_id}: {e}")
            raise DatabaseOperationError("list", "collection_source", str(e)) from e

    async def update_stats(
        self,
        source_id: int,
        document_count: int | None = None,
        size_bytes: int | None = None,
        last_indexed_at: datetime | None = None,
    ) -> CollectionSource:
        """Update source statistics.

        Args:
            source_id: Integer ID of the source
            document_count: New document count
            size_bytes: New size in bytes
            last_indexed_at: Last indexing timestamp

        Returns:
            Updated CollectionSource instance

        Raises:
            EntityNotFoundError: If source not found
            ValidationError: If any count is negative
        """
        try:
            # Validate non-negative counts
            if document_count is not None and document_count < 0:
                raise ValidationError("Document count cannot be negative", "document_count")
            if size_bytes is not None and size_bytes < 0:
                raise ValidationError("Size bytes cannot be negative", "size_bytes")

            source = await self.get_by_id(source_id)
            if not source:
                raise EntityNotFoundError("collection_source", str(source_id))

            if document_count is not None:
                source.document_count = document_count
            if size_bytes is not None:
                source.size_bytes = size_bytes
            if last_indexed_at is not None:
                source.last_indexed_at = last_indexed_at

            source.updated_at = datetime.now(UTC)
            await self.session.flush()

            logger.debug(f"Updated collection source {source_id} stats")
            return source

        except (EntityNotFoundError, ValidationError):
            raise
        except Exception as e:
            logger.error(f"Failed to update collection source stats: {e}")
            raise DatabaseOperationError("update", "collection_source", str(e)) from e

    async def delete(self, source_id: int) -> None:
        """Delete a collection source.

        Args:
            source_id: Integer ID of the source to delete

        Raises:
            EntityNotFoundError: If source not found
        """
        try:
            source = await self.get_by_id(source_id)
            if not source:
                raise EntityNotFoundError("collection_source", str(source_id))

            await self.session.delete(source)
            await self.session.flush()

            logger.info(f"Deleted collection source {source_id}")

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete collection source: {e}")
            raise DatabaseOperationError("delete", "collection_source", str(e)) from e
