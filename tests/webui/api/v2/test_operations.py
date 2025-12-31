"""Integration tests for the v2 operations API."""

from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient

from shared.database.models import OperationStatus
from webui.celery_app import celery_app


@pytest.mark.asyncio()
async def test_get_operation_returns_owned_operation(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    operation_factory,
) -> None:
    """Fetching an operation should surface persisted data."""

    collection = await collection_factory(owner_id=test_user_db.id)
    operation = await operation_factory(collection_id=collection.id, user_id=test_user_db.id)

    response = await api_client.get(f"/api/v2/operations/{operation.uuid}", headers=api_auth_headers)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["id"] == operation.uuid
    assert payload["collection_id"] == collection.id
    assert payload["status"] == operation.status.value


@pytest.mark.asyncio()
async def test_list_operations_filters_by_status(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    operation_factory,
    db_session,
) -> None:
    """Status filters should limit the results returned."""

    collection = await collection_factory(owner_id=test_user_db.id)
    completed = await operation_factory(
        collection_id=collection.id,
        user_id=test_user_db.id,
        status=OperationStatus.COMPLETED,
    )
    await operation_factory(
        collection_id=collection.id,
        user_id=test_user_db.id,
        status=OperationStatus.PROCESSING,
    )
    await db_session.commit()

    response = await api_client.get("/api/v2/operations?status=completed", headers=api_auth_headers)

    assert response.status_code == 200, response.text
    payload = response.json()
    identifiers = {item["id"] for item in payload["operations"]}
    assert completed.uuid in identifiers
    assert all(item["status"] == "completed" for item in payload["operations"])


@pytest.mark.asyncio()
async def test_cancel_operation_marks_operation_cancelled(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    operation_factory,
    db_session,
    monkeypatch,
) -> None:
    """Cancelling an operation should update status and revoke the Celery task."""

    collection = await collection_factory(owner_id=test_user_db.id)
    operation = await operation_factory(
        collection_id=collection.id,
        user_id=test_user_db.id,
        status=OperationStatus.PROCESSING,
    )
    operation.task_id = "celery-task-123"
    await db_session.commit()

    revoke_mock = MagicMock()
    monkeypatch.setattr(celery_app.control, "revoke", revoke_mock)

    response = await api_client.delete(f"/api/v2/operations/{operation.uuid}", headers=api_auth_headers)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == OperationStatus.CANCELLED.value
