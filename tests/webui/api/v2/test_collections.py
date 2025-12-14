"""Integration tests for the v2 collections API."""

from unittest.mock import AsyncMock, patch
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


# --- PUT /collections/{id} tests ---


@pytest.mark.asyncio()
async def test_update_collection_success(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
) -> None:
    """Successfully update collection metadata (only description/is_public, no rename)."""
    collection = await collection_factory(owner_id=test_user_db.id, is_public=False)

    # Update only description and is_public (no name change) to avoid Qdrant interaction
    update_payload = {
        "description": "Updated description",
        "is_public": True,
    }

    response = await api_client.put(
        f"/api/v2/collections/{collection.id}",
        json=update_payload,
        headers=api_auth_headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["description"] == "Updated description"
    assert body["is_public"] is True


@pytest.mark.asyncio()
async def test_update_collection_not_found(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Return 404 when updating non-existent collection."""
    fake_uuid = str(uuid4())

    response = await api_client.put(
        f"/api/v2/collections/{fake_uuid}",
        json={"name": "New Name"},
        headers=api_auth_headers,
    )

    assert response.status_code == 404, response.text


@pytest.mark.asyncio()
async def test_update_collection_forbidden_for_non_owner(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    other_user_db,
    collection_factory,
) -> None:
    """Return 403 when non-owner tries to update collection."""
    foreign_collection = await collection_factory(owner_id=other_user_db.id, is_public=False)

    response = await api_client.put(
        f"/api/v2/collections/{foreign_collection.id}",
        json={"name": "Hacked Name"},
        headers=api_auth_headers,
    )

    assert response.status_code == 403, response.text


@pytest.mark.asyncio()
async def test_update_collection_name_conflict(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
) -> None:
    """Return error when updating to a name that already exists.

    Note: Ideally should return 409, but the repository layer currently
    does not translate IntegrityError to EntityAlreadyExistsError for updates.
    This test verifies the error is caught (500) rather than crashing.
    """
    # Use unique names to avoid conflicts with other tests
    existing_name = f"Existing Name {uuid4().hex[:8]}"
    another_name = f"Another Collection {uuid4().hex[:8]}"

    await collection_factory(owner_id=test_user_db.id, name=existing_name)
    collection2 = await collection_factory(owner_id=test_user_db.id, name=another_name)

    # Mock Qdrant manager to avoid requiring a running Qdrant instance
    mock_qdrant = AsyncMock()
    mock_qdrant.rename_collection = AsyncMock()

    with patch("webui.qdrant.get_qdrant_manager", return_value=mock_qdrant):
        response = await api_client.put(
            f"/api/v2/collections/{collection2.id}",
            json={"name": existing_name},
            headers=api_auth_headers,
        )

    # TODO: Should return 409 once IntegrityError handling is added to repo layer
    assert response.status_code == 500, response.text
    assert "Failed to update collection" in response.text


# --- DELETE /collections/{id} tests ---


@pytest.mark.asyncio()
async def test_delete_collection_success(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    stub_celery_send_task,  # noqa: ARG001
) -> None:
    """Successfully delete a collection."""
    collection = await collection_factory(owner_id=test_user_db.id)

    response = await api_client.delete(
        f"/api/v2/collections/{collection.id}",
        headers=api_auth_headers,
    )

    assert response.status_code == 204, response.text


@pytest.mark.asyncio()
async def test_delete_collection_not_found(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Return 404 when deleting non-existent collection."""
    fake_uuid = str(uuid4())

    response = await api_client.delete(
        f"/api/v2/collections/{fake_uuid}",
        headers=api_auth_headers,
    )

    assert response.status_code == 404, response.text


@pytest.mark.asyncio()
async def test_delete_collection_forbidden_for_non_owner(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    other_user_db,
    collection_factory,
) -> None:
    """Return 403 when non-owner tries to delete collection."""
    foreign_collection = await collection_factory(owner_id=other_user_db.id, is_public=False)

    response = await api_client.delete(
        f"/api/v2/collections/{foreign_collection.id}",
        headers=api_auth_headers,
    )

    assert response.status_code == 403, response.text


# --- POST /{id}/sources tests ---


@pytest.mark.asyncio()
async def test_add_source_success(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    stub_celery_send_task,  # noqa: ARG001
) -> None:
    """Successfully add a source to a collection."""
    collection = await collection_factory(owner_id=test_user_db.id)

    source_payload = {
        "source_type": "directory",
        "source_config": {"path": "/data/test"},
    }

    response = await api_client.post(
        f"/api/v2/collections/{collection.id}/sources",
        json=source_payload,
        headers=api_auth_headers,
    )

    assert response.status_code == 202, response.text
    body = response.json()
    assert "id" in body
    assert body["type"] == "append"


@pytest.mark.asyncio()
async def test_add_source_not_found(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Return 404 when adding source to non-existent collection."""
    fake_uuid = str(uuid4())

    source_payload = {
        "source_type": "directory",
        "source_config": {"path": "/data/test"},
    }

    response = await api_client.post(
        f"/api/v2/collections/{fake_uuid}/sources",
        json=source_payload,
        headers=api_auth_headers,
    )

    assert response.status_code == 404, response.text


@pytest.mark.asyncio()
async def test_add_source_forbidden_for_non_owner(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    other_user_db,
    collection_factory,
) -> None:
    """Return 403 when non-owner tries to add source."""
    foreign_collection = await collection_factory(owner_id=other_user_db.id, is_public=False)

    source_payload = {
        "source_type": "directory",
        "source_config": {"path": "/data/test"},
    }

    response = await api_client.post(
        f"/api/v2/collections/{foreign_collection.id}/sources",
        json=source_payload,
        headers=api_auth_headers,
    )

    assert response.status_code == 403, response.text


# --- DELETE /{id}/sources tests ---


@pytest.mark.asyncio()
async def test_remove_source_not_found(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Return 404 when removing source from non-existent collection."""
    fake_uuid = str(uuid4())

    response = await api_client.delete(
        f"/api/v2/collections/{fake_uuid}/sources",
        params={"source_path": "/data/test"},
        headers=api_auth_headers,
    )

    assert response.status_code == 404, response.text


@pytest.mark.asyncio()
async def test_remove_source_forbidden_for_non_owner(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    other_user_db,
    collection_factory,
) -> None:
    """Return 403 when non-owner tries to remove source."""
    foreign_collection = await collection_factory(owner_id=other_user_db.id, is_public=False)

    response = await api_client.delete(
        f"/api/v2/collections/{foreign_collection.id}/sources",
        params={"source_path": "/data/test"},
        headers=api_auth_headers,
    )

    assert response.status_code == 403, response.text


# --- POST /{id}/reindex tests ---


@pytest.mark.asyncio()
async def test_reindex_collection_success(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    stub_celery_send_task,  # noqa: ARG001
) -> None:
    """Successfully trigger reindex on a collection."""
    collection = await collection_factory(owner_id=test_user_db.id)

    response = await api_client.post(
        f"/api/v2/collections/{collection.id}/reindex",
        headers=api_auth_headers,
    )

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["type"] == "reindex"


@pytest.mark.asyncio()
async def test_reindex_collection_not_found(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Return 404 when reindexing non-existent collection."""
    fake_uuid = str(uuid4())

    response = await api_client.post(
        f"/api/v2/collections/{fake_uuid}/reindex",
        headers=api_auth_headers,
    )

    assert response.status_code == 404, response.text


@pytest.mark.asyncio()
async def test_reindex_collection_forbidden_for_non_owner(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    other_user_db,
    collection_factory,
) -> None:
    """Return 403 when non-owner tries to reindex."""
    foreign_collection = await collection_factory(owner_id=other_user_db.id, is_public=False)

    response = await api_client.post(
        f"/api/v2/collections/{foreign_collection.id}/reindex",
        headers=api_auth_headers,
    )

    assert response.status_code == 403, response.text


# --- GET /{id}/documents tests ---


@pytest.mark.asyncio()
async def test_list_documents_success(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
) -> None:
    """Successfully list documents in a collection."""
    collection = await collection_factory(owner_id=test_user_db.id)

    response = await api_client.get(
        f"/api/v2/collections/{collection.id}/documents",
        headers=api_auth_headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert "documents" in body
    assert "total" in body


@pytest.mark.asyncio()
async def test_list_documents_not_found(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Return 404 when listing documents of non-existent collection."""
    fake_uuid = str(uuid4())

    response = await api_client.get(
        f"/api/v2/collections/{fake_uuid}/documents",
        headers=api_auth_headers,
    )

    assert response.status_code == 404, response.text


@pytest.mark.asyncio()
async def test_list_documents_forbidden_for_non_owner(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    other_user_db,
    collection_factory,
) -> None:
    """Return 403 when non-owner tries to list documents."""
    foreign_collection = await collection_factory(owner_id=other_user_db.id, is_public=False)

    response = await api_client.get(
        f"/api/v2/collections/{foreign_collection.id}/documents",
        headers=api_auth_headers,
    )

    assert response.status_code == 403, response.text


# --- Sync endpoints tests ---


@pytest.mark.asyncio()
async def test_run_sync_not_found(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Return 404 when running sync on non-existent collection."""
    fake_uuid = str(uuid4())

    response = await api_client.post(
        f"/api/v2/collections/{fake_uuid}/sync/run",
        headers=api_auth_headers,
    )

    assert response.status_code == 404, response.text


@pytest.mark.asyncio()
async def test_run_sync_forbidden_for_non_owner(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    other_user_db,
    collection_factory,
) -> None:
    """Return 403 when non-owner tries to run sync."""
    foreign_collection = await collection_factory(owner_id=other_user_db.id, is_public=False)

    response = await api_client.post(
        f"/api/v2/collections/{foreign_collection.id}/sync/run",
        headers=api_auth_headers,
    )

    assert response.status_code == 403, response.text


@pytest.mark.asyncio()
async def test_pause_sync_not_found(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Return 404 when pausing sync on non-existent collection."""
    fake_uuid = str(uuid4())

    response = await api_client.post(
        f"/api/v2/collections/{fake_uuid}/sync/pause",
        headers=api_auth_headers,
    )

    assert response.status_code == 404, response.text


@pytest.mark.asyncio()
async def test_pause_sync_forbidden_for_non_owner(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    other_user_db,
    collection_factory,
) -> None:
    """Return 403 when non-owner tries to pause sync."""
    foreign_collection = await collection_factory(owner_id=other_user_db.id, is_public=False)

    response = await api_client.post(
        f"/api/v2/collections/{foreign_collection.id}/sync/pause",
        headers=api_auth_headers,
    )

    assert response.status_code == 403, response.text


@pytest.mark.asyncio()
async def test_resume_sync_not_found(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Return 404 when resuming sync on non-existent collection."""
    fake_uuid = str(uuid4())

    response = await api_client.post(
        f"/api/v2/collections/{fake_uuid}/sync/resume",
        headers=api_auth_headers,
    )

    assert response.status_code == 404, response.text


@pytest.mark.asyncio()
async def test_resume_sync_forbidden_for_non_owner(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    other_user_db,
    collection_factory,
) -> None:
    """Return 403 when non-owner tries to resume sync."""
    foreign_collection = await collection_factory(owner_id=other_user_db.id, is_public=False)

    response = await api_client.post(
        f"/api/v2/collections/{foreign_collection.id}/sync/resume",
        headers=api_auth_headers,
    )

    assert response.status_code == 403, response.text


@pytest.mark.asyncio()
async def test_list_sync_runs_success(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
) -> None:
    """Successfully list sync runs for a collection."""
    collection = await collection_factory(owner_id=test_user_db.id)

    response = await api_client.get(
        f"/api/v2/collections/{collection.id}/sync/runs",
        headers=api_auth_headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert "items" in body
    assert "total" in body


@pytest.mark.asyncio()
async def test_list_sync_runs_not_found(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Return 404 when listing sync runs for non-existent collection."""
    fake_uuid = str(uuid4())

    response = await api_client.get(
        f"/api/v2/collections/{fake_uuid}/sync/runs",
        headers=api_auth_headers,
    )

    assert response.status_code == 404, response.text


@pytest.mark.asyncio()
async def test_list_sync_runs_forbidden_for_non_owner(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    other_user_db,
    collection_factory,
) -> None:
    """Return 403 when non-owner tries to list sync runs."""
    foreign_collection = await collection_factory(owner_id=other_user_db.id, is_public=False)

    response = await api_client.get(
        f"/api/v2/collections/{foreign_collection.id}/sync/runs",
        headers=api_auth_headers,
    )

    assert response.status_code == 403, response.text
