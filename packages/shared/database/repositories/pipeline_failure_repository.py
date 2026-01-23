"""Repository implementation for PipelineFailure model."""

import logging
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.db_retry import with_db_retry
from shared.database.exceptions import DatabaseOperationError, EntityNotFoundError
from shared.database.models import PipelineFailure

logger = logging.getLogger(__name__)


class PipelineFailureRepository:
    """Repository for PipelineFailure model operations.

    Manages pipeline failure records for tracking files that failed during
    pipeline execution. Supports retry tracking and bulk deletion operations.

    Example:
        ```python
        repo = PipelineFailureRepository(session)

        # Record a failure
        failure = await repo.create({
            "collection_id": "uuid",
            "file_uri": "/path/to/file.pdf",
            "stage_id": "parser_1",
            "stage_type": "parser",
            "error_type": "parse_error",
            "error_message": "Failed to parse PDF",
        })

        # Get failures for a collection
        failures = await repo.get_by_collection("uuid")

        # Increment retry count
        await repo.increment_retry(failure.id)
        ```
    """

    def __init__(self, session: AsyncSession):
        """Initialize with database session.

        Args:
            session: AsyncSession instance for database operations
        """
        self.session = session

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def create(self, failure_data: dict) -> PipelineFailure:
        """Create a new pipeline failure record.

        Args:
            failure_data: Dictionary with failure fields:
                - collection_id (required): Collection UUID
                - file_uri (required): URI of the failed file
                - stage_id (required): Pipeline node ID
                - stage_type (required): Pipeline node type
                - error_type (required): Error category
                - error_message (required): Error description
                - operation_id (optional): Operation UUID
                - file_metadata (optional): File metadata dict
                - error_traceback (optional): Stack trace

        Returns:
            Created PipelineFailure instance

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            failure = PipelineFailure(
                id=str(uuid4()),
                collection_id=failure_data["collection_id"],
                operation_id=failure_data.get("operation_id"),
                file_uri=failure_data["file_uri"],
                file_metadata=failure_data.get("file_metadata"),
                stage_id=failure_data["stage_id"],
                stage_type=failure_data["stage_type"],
                error_type=failure_data["error_type"],
                error_message=failure_data["error_message"],
                error_traceback=failure_data.get("error_traceback"),
                retry_count=failure_data.get("retry_count", 0),
            )

            self.session.add(failure)
            await self.session.flush()

            logger.debug(
                "Created pipeline failure %s for file %s at stage %s",
                failure.id,
                failure.file_uri,
                failure.stage_id,
            )

            return failure

        except Exception as e:
            logger.error(
                "Failed to create pipeline failure for file %s: %s",
                failure_data.get("file_uri"),
                e,
                exc_info=True,
            )
            raise DatabaseOperationError("create", "PipelineFailure", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get_by_id(self, failure_id: str) -> PipelineFailure | None:
        """Get a pipeline failure by ID.

        Args:
            failure_id: UUID of the failure record

        Returns:
            PipelineFailure instance or None if not found
        """
        try:
            result = await self.session.execute(
                select(PipelineFailure).where(PipelineFailure.id == failure_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Failed to get pipeline failure %s: %s", failure_id, e, exc_info=True)
            raise DatabaseOperationError("get", "PipelineFailure", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get_by_collection(
        self,
        collection_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PipelineFailure]:
        """Get pipeline failures for a collection.

        Args:
            collection_id: UUID of the collection
            limit: Maximum number of records to return (default: 100)
            offset: Number of records to skip (default: 0)

        Returns:
            List of PipelineFailure instances ordered by created_at descending
        """
        try:
            result = await self.session.execute(
                select(PipelineFailure)
                .where(PipelineFailure.collection_id == collection_id)
                .order_by(PipelineFailure.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())
        except Exception as e:
            logger.error(
                "Failed to get pipeline failures for collection %s: %s",
                collection_id,
                e,
                exc_info=True,
            )
            raise DatabaseOperationError("get", "PipelineFailure", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get_by_operation(self, operation_id: str) -> list[PipelineFailure]:
        """Get pipeline failures for an operation.

        Args:
            operation_id: UUID of the operation

        Returns:
            List of PipelineFailure instances for the operation
        """
        try:
            result = await self.session.execute(
                select(PipelineFailure)
                .where(PipelineFailure.operation_id == operation_id)
                .order_by(PipelineFailure.created_at.desc())
            )
            return list(result.scalars().all())
        except Exception as e:
            logger.error(
                "Failed to get pipeline failures for operation %s: %s",
                operation_id,
                e,
                exc_info=True,
            )
            raise DatabaseOperationError("get", "PipelineFailure", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def increment_retry(self, failure_id: str) -> PipelineFailure:
        """Increment retry count and update last_retry_at timestamp.

        Args:
            failure_id: UUID of the failure record

        Returns:
            Updated PipelineFailure instance

        Raises:
            EntityNotFoundError: If failure record not found
            DatabaseOperationError: For database errors
        """
        try:
            failure = await self.get_by_id(failure_id)
            if failure is None:
                raise EntityNotFoundError("PipelineFailure", failure_id)

            failure.retry_count += 1
            failure.last_retry_at = datetime.now(UTC)
            await self.session.flush()

            logger.debug(
                "Incremented retry count for failure %s to %d",
                failure_id,
                failure.retry_count,
            )

            return failure

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error(
                "Failed to increment retry for failure %s: %s",
                failure_id,
                e,
                exc_info=True,
            )
            raise DatabaseOperationError("update", "PipelineFailure", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def delete_by_collection(self, collection_id: str) -> int:
        """Delete all pipeline failures for a collection.

        Args:
            collection_id: UUID of the collection

        Returns:
            Number of records deleted
        """
        try:
            result = await self.session.execute(
                delete(PipelineFailure).where(PipelineFailure.collection_id == collection_id)
            )
            count = result.rowcount or 0
            logger.debug("Deleted %d pipeline failures for collection %s", count, collection_id)
            return count
        except Exception as e:
            logger.error(
                "Failed to delete pipeline failures for collection %s: %s",
                collection_id,
                e,
                exc_info=True,
            )
            raise DatabaseOperationError("delete", "PipelineFailure", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def delete_by_file_uri(self, collection_id: str, file_uri: str) -> int:
        """Delete pipeline failures for a specific file URI within a collection.

        Args:
            collection_id: UUID of the collection
            file_uri: URI of the file

        Returns:
            Number of records deleted
        """
        try:
            result = await self.session.execute(
                delete(PipelineFailure).where(
                    PipelineFailure.collection_id == collection_id,
                    PipelineFailure.file_uri == file_uri,
                )
            )
            count = result.rowcount or 0
            logger.debug(
                "Deleted %d pipeline failures for file %s in collection %s",
                count,
                file_uri,
                collection_id,
            )
            return count
        except Exception as e:
            logger.error(
                "Failed to delete pipeline failures for file %s: %s",
                file_uri,
                e,
                exc_info=True,
            )
            raise DatabaseOperationError("delete", "PipelineFailure", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def count_by_collection(self, collection_id: str) -> int:
        """Count pipeline failures for a collection.

        Args:
            collection_id: UUID of the collection

        Returns:
            Number of failure records
        """
        try:
            result = await self.session.execute(
                select(func.count(PipelineFailure.id)).where(
                    PipelineFailure.collection_id == collection_id
                )
            )
            return result.scalar() or 0
        except Exception as e:
            logger.error(
                "Failed to count pipeline failures for collection %s: %s",
                collection_id,
                e,
                exc_info=True,
            )
            raise DatabaseOperationError("count", "PipelineFailure", str(e)) from e
