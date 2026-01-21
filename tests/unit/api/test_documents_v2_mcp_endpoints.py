from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from shared.database import get_db
from webui.api.v2 import documents as documents_module
from webui.api.v2.documents import router
from webui.auth import get_current_user
from webui.dependencies import get_collection_for_user
from webui.middleware.exception_handlers import register_global_exception_handlers


@pytest.fixture()
def app() -> FastAPI:
    app = FastAPI()
    register_global_exception_handlers(app)  # Register global exception handlers
    app.include_router(router)

    app.dependency_overrides[get_current_user] = lambda: {"id": "1"}
    app.dependency_overrides[get_collection_for_user] = lambda: MagicMock()
    app.dependency_overrides[get_db] = lambda: MagicMock()

    yield app
    app.dependency_overrides.clear()


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def _mock_document(*, collection_id: str) -> MagicMock:
    doc = MagicMock()
    doc.id = str(uuid4())
    doc.collection_id = collection_id
    doc.file_name = "file.txt"
    doc.file_path = "/tmp/file.txt"
    doc.file_size = 123
    doc.mime_type = "text/plain"
    doc.content_hash = "x" * 64
    doc.status = MagicMock(value="completed")
    doc.error_message = None
    doc.chunk_count = 2
    doc.meta = {"k": "v"}
    doc.created_at = datetime.now(UTC)
    doc.updated_at = datetime.now(UTC)
    # Retry tracking fields
    doc.retry_count = 0
    doc.last_retry_at = None
    doc.error_category = None
    return doc


def test_get_document_200(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    collection_id = str(uuid4())
    document = _mock_document(collection_id=collection_id)

    repo = MagicMock()
    repo.get_by_id = AsyncMock(return_value=document)
    monkeypatch.setattr(documents_module, "create_document_repository", lambda _db: repo)

    response = client.get(f"/api/v2/collections/{collection_id}/documents/{document.id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == document.id
    assert body["collection_id"] == collection_id


def test_get_document_404(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    collection_id = str(uuid4())
    doc_id = str(uuid4())

    repo = MagicMock()
    repo.get_by_id = AsyncMock(return_value=None)
    monkeypatch.setattr(documents_module, "create_document_repository", lambda _db: repo)

    response = client.get(f"/api/v2/collections/{collection_id}/documents/{doc_id}")
    assert response.status_code == 404


def test_get_document_403_cross_collection(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    url_collection_id = str(uuid4())
    actual_collection_id = str(uuid4())
    document = _mock_document(collection_id=actual_collection_id)

    repo = MagicMock()
    repo.get_by_id = AsyncMock(return_value=document)
    monkeypatch.setattr(documents_module, "create_document_repository", lambda _db: repo)

    response = client.get(f"/api/v2/collections/{url_collection_id}/documents/{document.id}")
    assert response.status_code == 403


def test_get_document_500_on_repo_error(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    collection_id = str(uuid4())
    doc_id = str(uuid4())

    repo = MagicMock()
    repo.get_by_id = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(documents_module, "create_document_repository", lambda _db: repo)

    response = client.get(f"/api/v2/collections/{collection_id}/documents/{doc_id}")
    assert response.status_code == 500
