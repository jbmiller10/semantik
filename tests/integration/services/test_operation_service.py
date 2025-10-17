"""Integration tests for OperationService with real repositories."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from packages.shared.database.models import OperationStatus, OperationType
from packages.shared.database.repositories.operation_repository import OperationRepository
from packages.webui.services.operation_service import OperationService, celery_app


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestOperationServiceIntegration:
    """Validate OperationService behaviour against the database."""

    @pytest.fixture()
    def repository(self, db_session):
        return OperationRepository(db_session)

    @pytest.fixture()
    def service(self, db_session, repository):
        return OperationService(db_session, repository)

    async def _create_operation(self, repository, collection_factory, test_user_db, status=OperationStatus.PENDING):
        collection = await collection_factory(owner_id=test_user_db.id)
        operation = await repository.create(
            collection_id=collection.id,
            user_id=test_user_db.id,
            operation_type=OperationType.INDEX,
            config={"source": "integration"},
        )
        if status != OperationStatus.PENDING:
            await repository.update_status(operation.uuid, status, started_at=datetime.now(UTC))
        return operation

    async def test_get_operation_returns_entity(self, service, repository, collection_factory, test_user_db):
        operation = await self._create_operation(repository, collection_factory, test_user_db)
        fetched = await service.get_operation(operation.uuid, test_user_db.id)
        assert fetched.uuid == operation.uuid
        assert fetched.user_id == test_user_db.id

    async def test_cancel_operation_revokes_celery_task(
        self, service, repository, collection_factory, test_user_db, monkeypatch
    ):
        operation = await self._create_operation(repository, collection_factory, test_user_db)
        await repository.set_task_id(operation.uuid, "celery-task-xyz")
        revoke_calls: list[str] = []

        def fake_revoke(task_id, terminate=True):  # noqa: ARG001
            revoke_calls.append(task_id)

        mock_control = MagicMock()
        mock_control.revoke.side_effect = fake_revoke
        monkeypatch.setattr(celery_app, "control", mock_control)

        cancelled = await service.cancel_operation(operation.uuid, test_user_db.id)
        assert cancelled.status == OperationStatus.CANCELLED
        assert revoke_calls == ["celery-task-xyz"]

    async def test_list_operations_with_filters(self, service, repository, collection_factory, test_user_db):
        op_one = await self._create_operation(repository, collection_factory, test_user_db)
        op_two = await self._create_operation(
            repository,
            collection_factory,
            test_user_db,
            status=OperationStatus.COMPLETED,
        )

        operations, total = await service.list_operations_with_filters(
            user_id=test_user_db.id,
            status=OperationStatus.COMPLETED.value,
            operation_type=OperationType.INDEX.value,
        )
        uuids = {operation.uuid for operation in operations}
        assert op_two.uuid in uuids
        assert op_one.uuid not in uuids
        assert total >= 1

    async def test_parse_status_filter_invalid_value(self, service):
        with pytest.raises(ValueError):
            await service.parse_status_filter("does-not-exist")
