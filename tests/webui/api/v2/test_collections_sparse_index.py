"""Unit tests for v2 sparse index management endpoints.

These endpoints are thin wrappers around CollectionService; override the
dependency to avoid needing a live DB/Qdrant.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from shared.database.exceptions import EntityNotFoundError, InvalidStateError, ValidationError
from webui.auth import get_current_user
from webui.main import app
from webui.services.factory import get_collection_service


@pytest_asyncio.fixture()
async def sparse_api_client() -> AsyncGenerator[AsyncClient, None]:
    async def override_get_current_user():
        return {"id": 1, "username": "tester"}

    app.dependency_overrides[get_current_user] = override_get_current_user
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


@pytest.mark.asyncio()
async def test_get_sparse_index_status_returns_disabled_when_none(sparse_api_client: AsyncClient) -> None:
    service = AsyncMock()
    service.get_sparse_index_config.return_value = None

    async def override():
        return service

    app.dependency_overrides[get_collection_service] = override
    try:
        response = await sparse_api_client.get("/api/v2/collections/col-1/sparse-index")
    finally:
        app.dependency_overrides.pop(get_collection_service, None)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["enabled"] is False
    assert body["plugin_id"] is None
    assert body["sparse_collection_name"] is None
    assert body["model_config"] is None


@pytest.mark.asyncio()
async def test_get_sparse_index_status_returns_config(sparse_api_client: AsyncClient) -> None:
    service = AsyncMock()
    service.get_sparse_index_config.return_value = {
        "enabled": True,
        "plugin_id": "bm25-local",
        "sparse_collection_name": "dense_sparse_bm25",
        "model_config": {"k1": 1.2},
        "document_count": 3,
        "created_at": "2026-01-01T00:00:00Z",
        "last_indexed_at": None,
    }

    async def override():
        return service

    app.dependency_overrides[get_collection_service] = override
    try:
        response = await sparse_api_client.get("/api/v2/collections/col-1/sparse-index")
    finally:
        app.dependency_overrides.pop(get_collection_service, None)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["enabled"] is True
    assert body["plugin_id"] == "bm25-local"
    assert body["sparse_collection_name"] == "dense_sparse_bm25"
    assert body["model_config"] == {"k1": 1.2}


@pytest.mark.asyncio()
async def test_enable_sparse_index_maps_validation_error_to_400(sparse_api_client: AsyncClient) -> None:
    service = AsyncMock()
    service.enable_sparse_index.side_effect = ValidationError("Sparse indexer plugin 'x' not found", "plugin_id")

    async def override():
        return service

    app.dependency_overrides[get_collection_service] = override
    try:
        response = await sparse_api_client.post(
            "/api/v2/collections/col-1/sparse-index",
            json={"plugin_id": "x", "model_config_data": {}, "reindex_existing": False},
        )
    finally:
        app.dependency_overrides.pop(get_collection_service, None)

    assert response.status_code == 400, response.text


@pytest.mark.asyncio()
async def test_enable_sparse_index_maps_invalid_state_to_409(sparse_api_client: AsyncClient) -> None:
    service = AsyncMock()
    service.enable_sparse_index.side_effect = InvalidStateError("already enabled")

    async def override():
        return service

    app.dependency_overrides[get_collection_service] = override
    try:
        response = await sparse_api_client.post(
            "/api/v2/collections/col-1/sparse-index",
            json={"plugin_id": "bm25-local", "model_config_data": {}, "reindex_existing": False},
        )
    finally:
        app.dependency_overrides.pop(get_collection_service, None)

    assert response.status_code == 409, response.text


@pytest.mark.asyncio()
async def test_disable_sparse_index_returns_204(sparse_api_client: AsyncClient) -> None:
    service = AsyncMock()
    service.disable_sparse_index.return_value = None

    async def override():
        return service

    app.dependency_overrides[get_collection_service] = override
    try:
        response = await sparse_api_client.delete("/api/v2/collections/col-1/sparse-index")
    finally:
        app.dependency_overrides.pop(get_collection_service, None)

    assert response.status_code == 204, response.text
    service.disable_sparse_index.assert_awaited_once()


@pytest.mark.asyncio()
async def test_trigger_sparse_reindex_maps_not_found_to_404(sparse_api_client: AsyncClient) -> None:
    service = AsyncMock()
    service.trigger_sparse_reindex.side_effect = EntityNotFoundError("Sparse indexing not enabled", entity_id="col-1")

    async def override():
        return service

    app.dependency_overrides[get_collection_service] = override
    try:
        response = await sparse_api_client.post(
            "/api/v2/collections/col-1/sparse-index/reindex",
        )
    finally:
        app.dependency_overrides.pop(get_collection_service, None)

    assert response.status_code == 404, response.text


@pytest.mark.asyncio()
async def test_get_sparse_reindex_progress_returns_status(sparse_api_client: AsyncClient) -> None:
    service = AsyncMock()
    service.get_sparse_reindex_progress.return_value = {"status": "PROGRESS", "progress": 12.3}

    async def override():
        return service

    app.dependency_overrides[get_collection_service] = override
    try:
        response = await sparse_api_client.get(
            "/api/v2/collections/col-1/sparse-index/reindex/job-1",
        )
    finally:
        app.dependency_overrides.pop(get_collection_service, None)

    assert response.status_code == 200, response.text
    assert response.json()["job_id"] == "job-1"
    assert response.json()["status"] == "PROGRESS"
