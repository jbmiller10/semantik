"""Integration tests for collection-scoped operation endpoints."""

import pytest
from httpx import AsyncClient

from packages.shared.database.models import OperationStatus


@pytest.mark.asyncio()
async def test_list_collection_operations_returns_recent_operations(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    operation_factory,
) -> None:
    """Collection operations endpoint should return operations tied to the collection."""

    collection = await collection_factory(owner_id=test_user_db.id)
    operation = await operation_factory(collection_id=collection.id, user_id=test_user_db.id)

    response = await api_client.get(
        f"/api/v2/collections/{collection.id}/operations",
        headers=api_auth_headers,
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    ids = {item["id"] for item in payload}
    assert operation.uuid in ids


@pytest.mark.asyncio()
async def test_list_collection_operations_supports_status_filter(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    operation_factory,
) -> None:
    """Filtering by status should narrow the collection operation list."""

    collection = await collection_factory(owner_id=test_user_db.id)
    await operation_factory(
        collection_id=collection.id,
        user_id=test_user_db.id,
        status=OperationStatus.COMPLETED,
    )
    await operation_factory(
        collection_id=collection.id,
        user_id=test_user_db.id,
        status=OperationStatus.PROCESSING,
    )

    response = await api_client.get(
        f"/api/v2/collections/{collection.id}/operations?status=completed",
        headers=api_auth_headers,
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload
    assert all(item["status"] == OperationStatus.COMPLETED.value for item in payload)


@pytest.mark.asyncio()
async def test_collection_operations_forbidden_for_non_owner(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    other_user_db,
    collection_factory,
    operation_factory,
) -> None:
    """Non-owners should not be able to list another user's collection operations."""

    foreign_collection = await collection_factory(owner_id=other_user_db.id, is_public=False)
    await operation_factory(collection_id=foreign_collection.id, user_id=other_user_db.id)

    response = await api_client.get(
        f"/api/v2/collections/{foreign_collection.id}/operations",
        headers=api_auth_headers,
    )

    assert response.status_code == 403, response.text
