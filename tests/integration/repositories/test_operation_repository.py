"""Integration coverage for OperationRepository."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from shared.database.exceptions import AccessDeniedError, EntityNotFoundError, ValidationError
from shared.database.models import Operation, OperationStatus, OperationType
from shared.database.repositories.operation_repository import OperationRepository
from sqlalchemy import select


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestOperationRepositoryIntegration:
    """Run repository calls against the actual database session."""

    @pytest.fixture()
    def repository(self, db_session):
        return OperationRepository(db_session)

    async def test_create_operation_persists_row(self, repository, db_session, collection_factory, test_user_db):
        """Creating an operation should persist it with pending status."""
        collection = await collection_factory(owner_id=test_user_db.id)
        operation = await repository.create(
            collection_id=collection.id,
            user_id=test_user_db.id,
            operation_type=OperationType.INDEX,
            config={"source": "integration"},
        )
        await db_session.commit()

        result = await db_session.execute(select(Operation).where(Operation.uuid == operation.uuid))
        persisted = result.scalar_one()
        assert persisted.collection_id == collection.id
        assert persisted.user_id == test_user_db.id
        assert persisted.status == OperationStatus.PENDING
        assert persisted.config == {"source": "integration"}

    async def test_create_operation_denies_non_owner_for_private_collection(
        self, repository, collection_factory, test_user_db, other_user_db
    ):
        """Non-owners should receive AccessDeniedError when collection is private."""
        collection = await collection_factory(owner_id=other_user_db.id, is_public=False)

        with pytest.raises(AccessDeniedError):
            await repository.create(
                collection_id=collection.id,
                user_id=test_user_db.id,
                operation_type=OperationType.DELETE,
                config={"reason": "cleanup"},
            )

    async def test_get_by_uuid_with_permission_check_allows_owner(
        self, repository, db_session, collection_factory, test_user_db
    ):
        """Permission check should succeed for the owning user."""
        collection = await collection_factory(owner_id=test_user_db.id)
        operation = await repository.create(
            collection.id,
            test_user_db.id,
            OperationType.INDEX,
            config={"source": "integration"},
        )
        await db_session.commit()

        fetched = await repository.get_by_uuid_with_permission_check(operation.uuid, test_user_db.id)
        assert fetched.uuid == operation.uuid
        assert fetched.collection_id == collection.id

    async def test_get_by_uuid_with_permission_check_rejects_other_user(
        self, repository, db_session, collection_factory, test_user_db, other_user_db
    ):
        """Permission check should raise for users without access."""
        collection = await collection_factory(owner_id=test_user_db.id, is_public=False)
        operation = await repository.create(
            collection.id,
            test_user_db.id,
            OperationType.DELETE,
            config={"source": "integration"},
        )
        await db_session.commit()

        with pytest.raises(AccessDeniedError):
            await repository.get_by_uuid_with_permission_check(operation.uuid, other_user_db.id)

    async def test_update_status_transitions_and_timestamps(
        self, repository, db_session, collection_factory, test_user_db
    ):
        """Updating status should set timestamps when not provided."""
        collection = await collection_factory(owner_id=test_user_db.id)
        operation = await repository.create(
            collection.id,
            test_user_db.id,
            OperationType.INDEX,
            config={"step": "initial"},
        )
        await db_session.commit()

        updated = await repository.update_status(operation.uuid, OperationStatus.PROCESSING)
        assert updated.status == OperationStatus.PROCESSING
        assert updated.started_at is not None

        completed = await repository.update_status(operation.uuid, OperationStatus.COMPLETED)
        await db_session.commit()
        assert completed.status == OperationStatus.COMPLETED
        assert completed.completed_at is not None

    async def test_cancel_operation_sets_status_and_completed_at(
        self, repository, db_session, collection_factory, test_user_db
    ):
        """Cancelling from processing state should set CANCELLED and completion timestamp."""
        collection = await collection_factory(owner_id=test_user_db.id)
        operation = await repository.create(
            collection.id,
            test_user_db.id,
            OperationType.DELETE,
            config={"source": "integration"},
        )
        await db_session.commit()
        await repository.update_status(operation.uuid, OperationStatus.PROCESSING, started_at=datetime.now(UTC))
        await db_session.commit()

        cancelled = await repository.cancel(operation.uuid, test_user_db.id)
        await db_session.commit()
        assert cancelled.status == OperationStatus.CANCELLED
        assert cancelled.completed_at is not None

    async def test_cancel_operation_invalid_state_raises(
        self, repository, db_session, collection_factory, test_user_db
    ):
        """Cancelling a completed operation should raise ValidationError."""
        collection = await collection_factory(owner_id=test_user_db.id)
        operation = await repository.create(
            collection.id,
            test_user_db.id,
            OperationType.INDEX,
            config={"source": "integration"},
        )
        await db_session.commit()
        await repository.update_status(operation.uuid, OperationStatus.COMPLETED, started_at=datetime.now(UTC))
        await db_session.commit()

        with pytest.raises(ValidationError):
            await repository.cancel(operation.uuid, test_user_db.id)

    async def test_list_for_user_returns_operations(self, repository, db_session, collection_factory, test_user_db):
        """Listing for user should return operations they created."""
        collection = await collection_factory(owner_id=test_user_db.id)
        op1 = await repository.create(collection.id, test_user_db.id, OperationType.INDEX, config={"idx": 1})
        op2 = await repository.create(collection.id, test_user_db.id, OperationType.DELETE, config={"idx": 2})
        await db_session.commit()

        operations, total = await repository.list_for_user(test_user_db.id)
        uuids = {op.uuid for op in operations}
        assert {op1.uuid, op2.uuid} <= uuids
        assert total >= 2

    async def test_set_task_id_updates_row(self, repository, db_session, collection_factory, test_user_db):
        """Setting a task id should persist to the database immediately."""
        collection = await collection_factory(owner_id=test_user_db.id)
        operation = await repository.create(
            collection.id,
            test_user_db.id,
            OperationType.INDEX,
            config={"source": "integration"},
        )
        await db_session.commit()

        await repository.set_task_id(operation.uuid, "celery-task-123")
        await db_session.commit()

        result = await db_session.execute(select(Operation).where(Operation.uuid == operation.uuid))
        persisted = result.scalar_one()
        assert persisted.task_id == "celery-task-123"

    async def test_get_by_uuid_missing_operation_raises(self, repository):
        """Missing operations should raise when fetching with permission check."""
        with pytest.raises(EntityNotFoundError):
            await repository.get_by_uuid_with_permission_check(str(uuid4()), 1)
