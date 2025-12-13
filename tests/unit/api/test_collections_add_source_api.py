"""Unit tests for Collections API v2 add-source endpoint."""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from shared.database.exceptions import ValidationError


@pytest.fixture()
def test_client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, AsyncMock]:
    """
    Create a test FastAPI app with the collections router and mocked deps.

    Import order matters because the SlowAPI limiter is applied at import time.
    """
    monkeypatch.setenv("DISABLE_RATE_LIMITING", "true")

    import webui.rate_limiter as rate_limiter

    importlib.reload(rate_limiter)

    import webui.api.v2.collections as collections

    importlib.reload(collections)

    app = FastAPI()
    app.include_router(collections.router)

    mock_service: AsyncMock = AsyncMock()
    app.dependency_overrides[collections.get_current_user] = lambda: {"id": "1", "username": "testuser"}
    app.dependency_overrides[collections.get_collection_service] = lambda: mock_service

    client = TestClient(app)
    try:
        yield client, mock_service
    finally:
        app.dependency_overrides.clear()


def test_add_source_accepts_camelcase_payload(test_client: tuple[TestClient, AsyncMock]) -> None:
    client, mock_service = test_client
    collection_id = str(uuid4())
    op_id = str(uuid4())

    mock_service.add_source.return_value = {
        "uuid": op_id,
        "collection_id": collection_id,
        "type": "append",
        "status": "pending",
        "config": {
            "source_type": "directory",
            "source_path": "/data/docs",
            "source_config": {"path": "/data/docs"},
            "additional_config": {},
        },
        "created_at": "2025-12-13T00:00:00Z",
        "started_at": None,
        "completed_at": None,
        "error_message": None,
    }

    response = client.post(
        f"/api/v2/collections/{collection_id}/sources",
        json={
            "sourceType": "directory",
            "sourcePath": "/data/docs",
            "config": {},
        },
    )

    assert response.status_code == 202, response.text
    mock_service.add_source.assert_awaited_once()

    _, kwargs = mock_service.add_source.call_args
    assert kwargs["collection_id"] == collection_id
    assert kwargs["user_id"] == 1
    assert kwargs["source_type"] == "directory"
    assert kwargs["source_config"] == {"path": "/data/docs"}
    assert kwargs["legacy_source_path"] == "/data/docs"


def test_add_source_returns_400_on_validation_error(test_client: tuple[TestClient, AsyncMock]) -> None:
    client, mock_service = test_client
    collection_id = str(uuid4())

    mock_service.add_source.side_effect = ValidationError("Source path is required", "source_path")

    response = client.post(
        f"/api/v2/collections/{collection_id}/sources",
        json={
            "sourceType": "directory",
            "sourcePath": "",
        },
    )

    assert response.status_code == 400
    assert "Source path is required" in response.json()["detail"]

