"""Integration coverage for OperationRepository."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from packages.shared.database.models import Operation, OperationStatus, OperationType
from packages.shared.database.repositories.operation_repository import OperationRepository
from shared.database.exceptions import AccessDeniedError, EntityNotFoundError, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

pytestmark = [pytest.mark.asyncio(), pytest.mark.usefixtures("_db_isolation")]


class TestOperationRepositoryIntegration:
    """Exercise operation lifecycles against the database."""

    @pytest.fixture()
    def repository(self, db_session: AsyncSession) -> OperationRepository:
        return OperationRepository(db_session)

    async def _fetch_operation(self, db_session: AsyncSession, operation_uuid: str) -> Operation | None:
        result = await db_session.execute(select(Operation).where(Operation.uuid == operation_uuid))
        return result.scalar_one_or_none()

    async def test_create_operation_persists_record(
        self,
        repository: OperationRepository,
        db_session: AsyncSession,
        collection_factory,
        test_user_db,
    ) -> None:
        collection = await collection_factory(owner_id=test_user_db.id)

        operation = await repository.create(
            collection_id=collection.id,
            user_id=test_user_db.id,
            operation_type=OperationType.APPEND,
            config={"source": "manual"},
        )

        assert operation.status == OperationStatus.PENDING
        assert operation.collection_id == collection.id

        persisted = await self._fetch_operation(db_session, operation.uuid)
        assert persisted is not None
        assert persisted.user_id == test_user_db.id

    async def test_create_operation_requires_access(
        self,
        repository: OperationRepository,
        collection_factory,
        test_user_db,
        other_user_db,
    ) -> None:
        collection = await collection_factory(owner_id=test_user_db.id, is_public=False)

        with pytest.raises(AccessDeniedError):
            await repository.create(
                collection_id=collection.id,
                user_id=other_user_db.id,
                operation_type=OperationType.REINDEX,
                config={"reason": "no access"},
            )

    async def test_get_by_uuid_with_permission_check(
        self,
        repository: OperationRepository,
        collection_factory,
        test_user_db,
    ) -> None:
        collection = await collection_factory(owner_id=test_user_db.id)
        operation = await repository.create(
            collection_id=collection.id,
            user_id=test_user_db.id,
            operation_type=OperationType.REINDEX,
            config={"request": "test"},
        )

        fetched = await repository.get_by_uuid_with_permission_check(operation.uuid, test_user_db.id)
        assert fetched.uuid == operation.uuid

        with pytest.raises(EntityNotFoundError):
            await repository.get_by_uuid_with_permission_check(str(uuid4()), test_user_db.id)

    async def test_set_task_id_and_update_status(
        self,
        repository: OperationRepository,
        collection_factory,
        test_user_db,
    ) -> None:
        collection = await collection_factory(owner_id=test_user_db.id)
        operation = await repository.create(
            collection_id=collection.id,
            user_id=test_user_db.id,
            operation_type=OperationType.REINDEX,
            config={"phase": "init"},
        )

        updated = await repository.set_task_id(operation.uuid, "celery-task-123")
        assert updated.task_id == "celery-task-123"

        processing = await repository.update_status(operation.uuid, OperationStatus.PROCESSING)
        assert processing.status == OperationStatus.PROCESSING
        assert processing.started_at is not None

        completed = await repository.update_status(
            operation.uuid,
            OperationStatus.COMPLETED,
            started_at=datetime.now(UTC) - timedelta(minutes=5),
            completed_at=datetime.now(UTC),
        )
        assert completed.status == OperationStatus.COMPLETED
        assert completed.completed_at is not None

    async def test_list_for_collection_filters(
        self,
        repository: OperationRepository,
        collection_factory,
        test_user_db,
    ) -> None:
        collection = await collection_factory(owner_id=test_user_db.id)
        other_collection = await collection_factory(owner_id=test_user_db.id)

        await repository.create(collection.id, test_user_db.id, OperationType.APPEND, {"idx": 1})
        op2 = await repository.create(collection.id, test_user_db.id, OperationType.REINDEX, {"idx": 2})
        await repository.update_status(op2.uuid, OperationStatus.COMPLETED)
        await repository.create(other_collection.id, test_user_db.id, OperationType.APPEND, {"idx": 3})

        operations, total = await repository.list_for_collection(collection.id, test_user_db.id)
        assert total == 2
        assert len(operations) == 2

        completed, _ = await repository.list_for_collection(
            collection.id, test_user_db.id, status=OperationStatus.COMPLETED
        )
        assert len(completed) == 1
        assert completed[0].status == OperationStatus.COMPLETED

    async def test_list_for_user_supports_filters(
        self,
        repository: OperationRepository,
        collection_factory,
        test_user_db,
    ) -> None:
        collection = await collection_factory(owner_id=test_user_db.id)
        other_collection = await collection_factory(owner_id=test_user_db.id)

        await repository.create(collection.id, test_user_db.id, OperationType.APPEND, {"idx": 1})
        reindex = await repository.create(other_collection.id, test_user_db.id, OperationType.REINDEX, {"idx": 2})
        await repository.update_status(reindex.uuid, OperationStatus.FAILED, error_message="boom")

        listings, total = await repository.list_for_user(test_user_db.id, status_list=[OperationStatus.FAILED])
        assert total == 1
        assert len(listings) == 1
        assert listings[0].status == OperationStatus.FAILED

    async def test_cancel_operation_validates_state(
        self,
        repository: OperationRepository,
        collection_factory,
        test_user_db,
    ) -> None:
        collection = await collection_factory(owner_id=test_user_db.id)
        operation = await repository.create(collection.id, test_user_db.id, OperationType.APPEND, {"idx": 1})

        cancelled = await repository.cancel(operation.uuid, test_user_db.id)
        assert cancelled.status == OperationStatus.CANCELLED
        assert cancelled.completed_at is not None

        with pytest.raises(ValidationError):
            await repository.cancel(operation.uuid, test_user_db.id)
