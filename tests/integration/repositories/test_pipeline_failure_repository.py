"""Integration tests for PipelineFailureRepository."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from sqlalchemy import select

from shared.database.exceptions import EntityNotFoundError
from shared.database.models import PipelineFailure
from shared.database.repositories.pipeline_failure_repository import PipelineFailureRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestPipelineFailureRepositoryIntegration:
    """Exercise PipelineFailureRepository against the real database session."""

    @pytest.fixture()
    def repository(self, db_session: AsyncSession) -> PipelineFailureRepository:
        return PipelineFailureRepository(db_session)

    async def test_create_failure_persists_row(
        self,
        repository: PipelineFailureRepository,
        db_session: AsyncSession,
        collection_factory,
        test_user_db,
    ) -> None:
        """Creating a pipeline failure should persist all fields."""
        collection = await collection_factory(owner_id=test_user_db.id)

        failure = await repository.create(
            {
                "collection_id": collection.id,
                "file_uri": "/path/to/document.pdf",
                "stage_id": "parser_node_1",
                "stage_type": "parser",
                "error_type": "parse_error",
                "error_message": "Failed to parse PDF: corrupted file",
                "error_traceback": "Traceback (most recent call last):\n  ...",
                "file_metadata": {"size": 1024, "mime_type": "application/pdf"},
            }
        )
        await db_session.commit()

        result = await db_session.execute(select(PipelineFailure).where(PipelineFailure.id == failure.id))
        persisted = result.scalar_one()

        assert persisted.collection_id == collection.id
        assert persisted.file_uri == "/path/to/document.pdf"
        assert persisted.stage_id == "parser_node_1"
        assert persisted.stage_type == "parser"
        assert persisted.error_type == "parse_error"
        assert persisted.error_message == "Failed to parse PDF: corrupted file"
        assert persisted.error_traceback is not None
        assert persisted.file_metadata["size"] == 1024
        assert persisted.retry_count == 0
        assert persisted.last_retry_at is None
        assert persisted.created_at is not None

    async def test_create_failure_with_operation_link(
        self,
        repository: PipelineFailureRepository,
        db_session: AsyncSession,
        collection_factory,
        operation_factory,
        test_user_db,
    ) -> None:
        """Creating a failure with operation_id should link to the operation."""
        collection = await collection_factory(owner_id=test_user_db.id)
        operation = await operation_factory(collection_id=collection.id, user_id=test_user_db.id)

        failure = await repository.create(
            {
                "collection_id": collection.id,
                "operation_id": operation.uuid,
                "file_uri": "/path/to/file.txt",
                "stage_id": "chunker_1",
                "stage_type": "chunker",
                "error_type": "timeout",
                "error_message": "Operation timed out",
            }
        )
        await db_session.commit()

        result = await db_session.execute(select(PipelineFailure).where(PipelineFailure.id == failure.id))
        persisted = result.scalar_one()

        assert persisted.operation_id == operation.uuid

    async def test_get_by_collection_returns_failures(
        self,
        repository: PipelineFailureRepository,
        db_session: AsyncSession,
        collection_factory,
        test_user_db,
    ) -> None:
        """get_by_collection should return failures for the specified collection."""
        collection = await collection_factory(owner_id=test_user_db.id)
        other_collection = await collection_factory(owner_id=test_user_db.id)

        # Create failures for both collections
        await repository.create(
            {
                "collection_id": collection.id,
                "file_uri": "/path/to/file1.pdf",
                "stage_id": "parser_1",
                "stage_type": "parser",
                "error_type": "parse_error",
                "error_message": "Error 1",
            }
        )
        await repository.create(
            {
                "collection_id": collection.id,
                "file_uri": "/path/to/file2.pdf",
                "stage_id": "parser_1",
                "stage_type": "parser",
                "error_type": "parse_error",
                "error_message": "Error 2",
            }
        )
        await repository.create(
            {
                "collection_id": other_collection.id,
                "file_uri": "/path/to/other.pdf",
                "stage_id": "parser_1",
                "stage_type": "parser",
                "error_type": "parse_error",
                "error_message": "Other error",
            }
        )
        await db_session.commit()

        failures = await repository.get_by_collection(collection.id)

        assert len(failures) == 2
        assert all(f.collection_id == collection.id for f in failures)

    async def test_get_by_collection_respects_limit_offset(
        self,
        repository: PipelineFailureRepository,
        db_session: AsyncSession,
        collection_factory,
        test_user_db,
    ) -> None:
        """get_by_collection should respect limit and offset parameters."""
        collection = await collection_factory(owner_id=test_user_db.id)

        # Create 5 failures
        for i in range(5):
            await repository.create(
                {
                    "collection_id": collection.id,
                    "file_uri": f"/path/to/file{i}.pdf",
                    "stage_id": "parser_1",
                    "stage_type": "parser",
                    "error_type": "parse_error",
                    "error_message": f"Error {i}",
                }
            )
        await db_session.commit()

        # Test limit
        limited = await repository.get_by_collection(collection.id, limit=2)
        assert len(limited) == 2

        # Test offset
        offset = await repository.get_by_collection(collection.id, limit=2, offset=2)
        assert len(offset) == 2

        # Verify no overlap
        limited_uris = {f.file_uri for f in limited}
        offset_uris = {f.file_uri for f in offset}
        assert limited_uris.isdisjoint(offset_uris)

    async def test_get_by_operation_returns_failures(
        self,
        repository: PipelineFailureRepository,
        db_session: AsyncSession,
        collection_factory,
        operation_factory,
        test_user_db,
    ) -> None:
        """get_by_operation should return failures linked to the operation."""
        collection = await collection_factory(owner_id=test_user_db.id)
        operation1 = await operation_factory(collection_id=collection.id, user_id=test_user_db.id)
        operation2 = await operation_factory(collection_id=collection.id, user_id=test_user_db.id)

        # Create failures for different operations
        await repository.create(
            {
                "collection_id": collection.id,
                "operation_id": operation1.uuid,
                "file_uri": "/path/to/file1.pdf",
                "stage_id": "parser_1",
                "stage_type": "parser",
                "error_type": "parse_error",
                "error_message": "Error 1",
            }
        )
        await repository.create(
            {
                "collection_id": collection.id,
                "operation_id": operation2.uuid,
                "file_uri": "/path/to/file2.pdf",
                "stage_id": "parser_1",
                "stage_type": "parser",
                "error_type": "parse_error",
                "error_message": "Error 2",
            }
        )
        await db_session.commit()

        failures = await repository.get_by_operation(operation1.uuid)

        assert len(failures) == 1
        assert failures[0].operation_id == operation1.uuid

    async def test_increment_retry_updates_count_and_timestamp(
        self,
        repository: PipelineFailureRepository,
        db_session: AsyncSession,
        collection_factory,
        test_user_db,
    ) -> None:
        """increment_retry should bump retry_count and set last_retry_at."""
        collection = await collection_factory(owner_id=test_user_db.id)

        failure = await repository.create(
            {
                "collection_id": collection.id,
                "file_uri": "/path/to/file.pdf",
                "stage_id": "parser_1",
                "stage_type": "parser",
                "error_type": "parse_error",
                "error_message": "Error",
            }
        )
        await db_session.commit()

        assert failure.retry_count == 0
        assert failure.last_retry_at is None

        # First retry
        updated = await repository.increment_retry(failure.id)
        await db_session.commit()

        assert updated.retry_count == 1
        assert updated.last_retry_at is not None
        first_retry_time = updated.last_retry_at

        # Second retry
        updated = await repository.increment_retry(failure.id)
        await db_session.commit()

        assert updated.retry_count == 2
        assert updated.last_retry_at >= first_retry_time

    async def test_increment_retry_not_found_raises_error(
        self,
        repository: PipelineFailureRepository,
    ) -> None:
        """increment_retry should raise EntityNotFoundError for non-existent failure."""
        with pytest.raises(EntityNotFoundError):
            await repository.increment_retry(str(uuid4()))

    async def test_delete_by_collection_removes_all(
        self,
        repository: PipelineFailureRepository,
        db_session: AsyncSession,
        collection_factory,
        test_user_db,
    ) -> None:
        """delete_by_collection should remove all failures for the collection."""
        collection = await collection_factory(owner_id=test_user_db.id)
        other_collection = await collection_factory(owner_id=test_user_db.id)

        # Create failures for both collections
        for i in range(3):
            await repository.create(
                {
                    "collection_id": collection.id,
                    "file_uri": f"/path/to/file{i}.pdf",
                    "stage_id": "parser_1",
                    "stage_type": "parser",
                    "error_type": "parse_error",
                    "error_message": f"Error {i}",
                }
            )
        await repository.create(
            {
                "collection_id": other_collection.id,
                "file_uri": "/path/to/other.pdf",
                "stage_id": "parser_1",
                "stage_type": "parser",
                "error_type": "parse_error",
                "error_message": "Other error",
            }
        )
        await db_session.commit()

        deleted_count = await repository.delete_by_collection(collection.id)
        await db_session.commit()

        assert deleted_count == 3

        # Verify collection failures are deleted
        remaining = await repository.get_by_collection(collection.id)
        assert len(remaining) == 0

        # Verify other collection failures remain
        other_remaining = await repository.get_by_collection(other_collection.id)
        assert len(other_remaining) == 1

    async def test_delete_by_file_uri_removes_specific(
        self,
        repository: PipelineFailureRepository,
        db_session: AsyncSession,
        collection_factory,
        test_user_db,
    ) -> None:
        """delete_by_file_uri should remove failures for specific file only."""
        collection = await collection_factory(owner_id=test_user_db.id)

        # Create failures for different files
        await repository.create(
            {
                "collection_id": collection.id,
                "file_uri": "/path/to/file1.pdf",
                "stage_id": "parser_1",
                "stage_type": "parser",
                "error_type": "parse_error",
                "error_message": "Error 1",
            }
        )
        await repository.create(
            {
                "collection_id": collection.id,
                "file_uri": "/path/to/file1.pdf",
                "stage_id": "chunker_1",
                "stage_type": "chunker",
                "error_type": "oom",
                "error_message": "Error 2",
            }
        )
        await repository.create(
            {
                "collection_id": collection.id,
                "file_uri": "/path/to/file2.pdf",
                "stage_id": "parser_1",
                "stage_type": "parser",
                "error_type": "parse_error",
                "error_message": "Error 3",
            }
        )
        await db_session.commit()

        deleted_count = await repository.delete_by_file_uri(collection.id, "/path/to/file1.pdf")
        await db_session.commit()

        assert deleted_count == 2

        remaining = await repository.get_by_collection(collection.id)
        assert len(remaining) == 1
        assert remaining[0].file_uri == "/path/to/file2.pdf"

    async def test_cascade_delete_on_collection_removal(
        self,
        repository: PipelineFailureRepository,
        db_session: AsyncSession,
        collection_factory,
        test_user_db,
    ) -> None:
        """Deleting a collection should cascade delete pipeline failures."""
        collection = await collection_factory(owner_id=test_user_db.id)

        await repository.create(
            {
                "collection_id": collection.id,
                "file_uri": "/path/to/file.pdf",
                "stage_id": "parser_1",
                "stage_type": "parser",
                "error_type": "parse_error",
                "error_message": "Error",
            }
        )
        await db_session.commit()

        # Verify failure exists
        failures = await repository.get_by_collection(collection.id)
        assert len(failures) == 1
        failure_id = failures[0].id

        # Delete the collection
        await db_session.delete(collection)
        await db_session.commit()

        # Verify failure was cascade deleted
        result = await db_session.execute(select(PipelineFailure).where(PipelineFailure.id == failure_id))
        assert result.scalar_one_or_none() is None

    async def test_set_null_on_operation_removal(
        self,
        repository: PipelineFailureRepository,
        db_session: AsyncSession,
        collection_factory,
        operation_factory,
        test_user_db,
    ) -> None:
        """Deleting an operation should SET NULL on pipeline failure operation_id."""
        collection = await collection_factory(owner_id=test_user_db.id)
        operation = await operation_factory(collection_id=collection.id, user_id=test_user_db.id)

        failure = await repository.create(
            {
                "collection_id": collection.id,
                "operation_id": operation.uuid,
                "file_uri": "/path/to/file.pdf",
                "stage_id": "parser_1",
                "stage_type": "parser",
                "error_type": "parse_error",
                "error_message": "Error",
            }
        )
        await db_session.commit()

        # Verify operation_id is set
        assert failure.operation_id == operation.uuid

        # Delete the operation
        await db_session.delete(operation)
        await db_session.commit()

        # Refresh and verify operation_id is NULL but failure still exists
        await db_session.refresh(failure)
        assert failure.operation_id is None
        assert failure.collection_id == collection.id

    async def test_count_by_collection(
        self,
        repository: PipelineFailureRepository,
        db_session: AsyncSession,
        collection_factory,
        test_user_db,
    ) -> None:
        """count_by_collection should return accurate count."""
        collection = await collection_factory(owner_id=test_user_db.id)
        other_collection = await collection_factory(owner_id=test_user_db.id)

        # Create 3 failures for collection
        for i in range(3):
            await repository.create(
                {
                    "collection_id": collection.id,
                    "file_uri": f"/path/to/file{i}.pdf",
                    "stage_id": "parser_1",
                    "stage_type": "parser",
                    "error_type": "parse_error",
                    "error_message": f"Error {i}",
                }
            )

        # Create 1 failure for other collection
        await repository.create(
            {
                "collection_id": other_collection.id,
                "file_uri": "/path/to/other.pdf",
                "stage_id": "parser_1",
                "stage_type": "parser",
                "error_type": "parse_error",
                "error_message": "Other error",
            }
        )
        await db_session.commit()

        count = await repository.count_by_collection(collection.id)
        assert count == 3

        other_count = await repository.count_by_collection(other_collection.id)
        assert other_count == 1

        empty_count = await repository.count_by_collection(str(uuid4()))
        assert empty_count == 0
