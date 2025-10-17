"""Integration tests for OperationService with the real repository."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.models import OperationStatus, OperationType
from packages.shared.database.repositories.operation_repository import OperationRepository
from packages.webui.services.operation_service import OperationService

pytestmark = [pytest.mark.asyncio(), pytest.mark.usefixtures("_db_isolation")]


@pytest.fixture()
def operation_service(db_session: AsyncSession) -> OperationService:
    repo = OperationRepository(db_session)
    return OperationService(db_session=db_session, operation_repo=repo)


async def _create_operation(
    service: OperationService,
    collection_factory,
    test_user_db,
    *,
    status: OperationStatus = OperationStatus.PENDING,
    operation_type: OperationType = OperationType.APPEND,
    task_id: str | None = None,
) -> str:
    repo = service.operation_repo
    collection = await collection_factory(owner_id=test_user_db.id)
    operation = await repo.create(
        collection_id=collection.id,
        user_id=test_user_db.id,
        operation_type=operation_type,
        config={"source": "integration"},
    )
    if task_id:
        await repo.set_task_id(operation.uuid, task_id)
    if status != OperationStatus.PENDING:
        await repo.update_status(operation.uuid, status)
    return operation.uuid


async def test_get_operation_returns_entity(
    operation_service: OperationService, collection_factory, test_user_db
) -> None:
    operation_uuid = await _create_operation(operation_service, collection_factory, test_user_db)

    fetched = await operation_service.get_operation(operation_uuid, test_user_db.id)
    assert fetched.uuid == operation_uuid


async def test_cancel_operation_revokes_task(
    operation_service: OperationService,
    collection_factory,
    test_user_db,
    monkeypatch,
) -> None:
    revoked: dict[str, Any] = {}

    class DummyControl:
        def revoke(self, task_id: str, terminate: bool) -> None:
            revoked["task_id"] = task_id
            revoked["terminate"] = terminate

    monkeypatch.setattr("packages.webui.services.operation_service.celery_app.control", DummyControl())

    operation_uuid = await _create_operation(
        operation_service,
        collection_factory,
        test_user_db,
        status=OperationStatus.PROCESSING,
        task_id="celery-123",
    )

    cancelled = await operation_service.cancel_operation(operation_uuid, test_user_db.id)
    assert cancelled.status == OperationStatus.CANCELLED
    assert revoked == {"task_id": "celery-123", "terminate": True}


async def test_parse_and_list_filters(operation_service: OperationService, collection_factory, test_user_db) -> None:
    await _create_operation(operation_service, collection_factory, test_user_db, status=OperationStatus.COMPLETED)
    await _create_operation(
        operation_service,
        collection_factory,
        test_user_db,
        status=OperationStatus.FAILED,
        operation_type=OperationType.REINDEX,
    )

    parsed_status = await operation_service.parse_status_filter("COMPLETED,FAILED")
    assert parsed_status == [OperationStatus.COMPLETED, OperationStatus.FAILED]

    parsed_type = await operation_service.parse_type_filter("REINDEX")
    assert parsed_type is OperationType.REINDEX

    operations, total = await operation_service.list_operations_with_filters(
        user_id=test_user_db.id,
        status="FAILED",
        operation_type="REINDEX",
    )
    assert total == 1
    assert operations[0].type is OperationType.REINDEX

    # Ensure invalid filters raise errors
    with pytest.raises(ValueError, match="INVALID"):
        await operation_service.parse_status_filter("INVALID")
    with pytest.raises(ValueError, match="UNKNOWN"):
        await operation_service.parse_type_filter("UNKNOWN")


async def test_list_operations_pagination(
    operation_service: OperationService,
    collection_factory,
    test_user_db,
) -> None:
    uuids = []
    for idx in range(3):
        uuid = await _create_operation(
            operation_service,
            collection_factory,
            test_user_db,
            status=OperationStatus.PROCESSING,
            task_id=f"task-{idx}",
        )
        uuids.append(uuid)

    operations, total = await operation_service.list_operations(test_user_db.id, offset=0, limit=2)
    assert total == 3
    assert len(operations) == 2
