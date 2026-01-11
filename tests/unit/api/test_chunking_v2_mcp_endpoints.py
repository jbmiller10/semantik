from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from shared.database import get_db
from webui.api.v2 import chunking as chunking_module
from webui.api.v2.chunking import router
from webui.auth import get_current_user
from webui.dependencies import get_collection_for_user_safe


@pytest.fixture()
def app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    app = FastAPI()
    app.include_router(router)

    app.dependency_overrides[get_current_user] = lambda: {"id": "1"}
    app.dependency_overrides[get_collection_for_user_safe] = lambda: {"id": "collection"}
    app.dependency_overrides[get_db] = lambda: MagicMock()

    # Avoid rate limiter circuit breaker behavior in this unit test.
    monkeypatch.setattr(chunking_module, "check_circuit_breaker", lambda _req: None)

    yield app
    app.dependency_overrides.clear()


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _mock_chunk(*, collection_id: str) -> MagicMock:
    chunk = MagicMock()
    chunk.id = 123
    chunk.collection_id = collection_id
    chunk.document_id = str(uuid4())
    chunk.chunk_index = 7
    chunk.content = "hello"
    chunk.token_count = 5
    chunk.start_offset = 0
    chunk.end_offset = 5
    chunk.meta = {"chunk_id": "doc_0007"}
    chunk.created_at = datetime.now(UTC)
    chunk.updated_at = datetime.now(UTC)
    return chunk


def test_get_chunk_by_id_numeric_path(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    collection_id = str(uuid4())
    chunk = _mock_chunk(collection_id=collection_id)

    class FakeRepo:
        def __init__(self, _db):
            self.meta_called = False

        async def get_chunk_by_id(self, _chunk_id: int, _collection_id: str):
            return chunk

        async def get_chunk_by_metadata_chunk_id(self, _chunk_id: str, _collection_id: str):
            self.meta_called = True
            return

    fake_repo = FakeRepo(None)
    monkeypatch.setattr(chunking_module, "ChunkRepository", lambda _db: fake_repo)

    response = client.get(f"/api/v2/chunking/collections/{collection_id}/chunks/123")
    assert response.status_code == 200
    assert response.json()["id"] == 123
    assert fake_repo.meta_called is False


def test_get_chunk_by_id_metadata_fallback(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    collection_id = str(uuid4())
    chunk = _mock_chunk(collection_id=collection_id)

    class FakeRepo:
        def __init__(self, _db):
            self.id_called = False

        async def get_chunk_by_id(self, _chunk_id: int, _collection_id: str):
            self.id_called = True
            return

        async def get_chunk_by_metadata_chunk_id(self, _chunk_id: str, _collection_id: str):
            return chunk

    fake_repo = FakeRepo(None)
    monkeypatch.setattr(chunking_module, "ChunkRepository", lambda _db: fake_repo)

    response = client.get(f"/api/v2/chunking/collections/{collection_id}/chunks/doc_0007")
    assert response.status_code == 200
    assert response.json()["id"] == 123
    assert fake_repo.id_called is False


def test_get_chunk_by_id_404(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    collection_id = str(uuid4())

    class FakeRepo:
        def __init__(self, _db):
            pass

        async def get_chunk_by_id(self, _chunk_id: int, _collection_id: str):
            return None

        async def get_chunk_by_metadata_chunk_id(self, _chunk_id: str, _collection_id: str):
            return None

    monkeypatch.setattr(chunking_module, "ChunkRepository", lambda _db: FakeRepo(None))

    response = client.get(f"/api/v2/chunking/collections/{collection_id}/chunks/999")
    assert response.status_code == 404
