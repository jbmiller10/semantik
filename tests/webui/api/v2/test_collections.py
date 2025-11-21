"""Integration tests for the v2 collections API."""

from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from shared.database.models import Collection, Operation


@pytest.mark.asyncio()
async def test_create_collection_persists_and_dispatches_operation(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    stub_celery_send_task,
    db_session,
) -> None:
    """Creating a collection should persist data and enqueue the indexing task."""

    request_payload = {
        "name": f"Integration Collection {uuid4().hex[:8]}",
        "description": "Created via API integration test",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "quantization": "float16",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "is_public": False,
        "metadata": {"env": "test"},
    }

    response = await api_client.post("/api/v2/collections", json=request_payload, headers=api_auth_headers)

    assert response.status_code == 201, response.text
    body = response.json()

    result = await db_session.execute(select(Collection).where(Collection.id == body["id"]))
    collection = result.scalar_one()
    assert collection.name == request_payload["name"]
    assert collection.meta == request_payload["metadata"]

    op_result = await db_session.execute(select(Operation).where(Operation.collection_id == collection.id))
    operation = op_result.scalar_one()
    assert body["initial_operation_id"] == operation.uuid
    stub_celery_send_task.assert_called_once()


@pytest.mark.asyncio()
async def test_list_collections_returns_owned_and_public(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    other_user_db,
    collection_factory,
    db_session,
) -> None:
    """The list endpoint should include owned and public collections."""

    owned = await collection_factory(owner_id=test_user_db.id, is_public=False)
    public = await collection_factory(owner_id=other_user_db.id, is_public=True)
    await db_session.commit()

    response = await api_client.get("/api/v2/collections", headers=api_auth_headers)

    assert response.status_code == 200, response.text
    payload = response.json()
    identifiers = {item["id"] for item in payload["collections"]}
    assert owned.id in identifiers
    assert public.id in identifiers
    assert payload["total"] >= 2


@pytest.mark.asyncio()
async def test_get_collection_returns_owned_details(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
) -> None:
    """Owned collections should be retrievable via the detail endpoint."""

    collection = await collection_factory(owner_id=test_user_db.id, is_public=False)

    response = await api_client.get(f"/api/v2/collections/{collection.id}", headers=api_auth_headers)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == collection.id
    assert body["name"] == collection.name


@pytest.mark.asyncio()
async def test_get_collection_forbidden_for_non_owner(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    other_user_db,
    collection_factory,
) -> None:
    """Users should receive 403 when accessing collections they do not own."""

    foreign_collection = await collection_factory(owner_id=other_user_db.id, is_public=False)

    response = await api_client.get(f"/api/v2/collections/{foreign_collection.id}", headers=api_auth_headers)
    assert response.status_code == 403, response.text
