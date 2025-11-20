"""Regression tests for the chunking operation progress endpoint."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from packages.webui.api.v2.chunking import get_chunking_orchestrator_dependency
from packages.webui.auth import get_current_user
from packages.webui.main import app


class _FakeProgressService:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls: list[tuple[str, int | None]] = []

    async def get_chunking_progress(self, operation_id: str, user_id: int | None = None):
        self.calls.append((operation_id, user_id))
        return self.payload


@pytest.fixture(autouse=True)
def clear_overrides() -> Generator[None, None, None]:
    """Ensure dependency overrides do not leak between tests."""

    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def stub_current_user() -> Generator[None, None, None]:
    """Provide a deterministic authenticated user for tests."""

    async def _fake_user():
        return {"id": 0, "username": "test", "email": "test@example.com"}

    app.dependency_overrides[get_current_user] = _fake_user
    yield


def test_progress_endpoint_uses_service_response() -> None:
    payload = {
        "status": "in_progress",
        "progress_percentage": 42.5,
        "documents_processed": 5,
        "total_documents": 12,
        "chunks_created": 48,
        "current_document": "doc-5.pdf",
        "errors": [],
    }

    service = _FakeProgressService(payload)

    async def override_service():
        return service

    app.dependency_overrides[get_chunking_orchestrator_dependency] = override_service

    client = TestClient(app)
    response = client.get("/api/v2/chunking/operations/op-123/progress")

    assert response.status_code == 200
    body = response.json()
    assert body["operation_id"] == "op-123"
    assert body["progress_percentage"] == pytest.approx(42.5)
    assert body["documents_processed"] == 5
    assert body["total_documents"] == 12
    assert service.calls == [("op-123", 0)]  # dev user injected when auth disabled


def test_progress_endpoint_returns_404_when_missing() -> None:
    class _NullService:
        async def get_chunking_progress(self, operation_id: str, user_id: int | None = None):  # noqa: ARG002
            return None

    async def override_service():
        return _NullService()

    app.dependency_overrides[get_chunking_orchestrator_dependency] = override_service

    client = TestClient(app)
    response = client.get("/api/v2/chunking/operations/missing/progress")

    assert response.status_code == 404
