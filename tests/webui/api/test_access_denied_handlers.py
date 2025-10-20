"""Ensure AccessDeniedError variants map to HTTP 403 responses."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from packages.shared.database.exceptions import AccessDeniedError as PackagesAccessDeniedError

try:
    from shared.database.exceptions import AccessDeniedError as SharedAccessDeniedError  # type: ignore[import]
except Exception:  # pragma: no cover - shared may be unavailable in some environments
    SharedAccessDeniedError = None  # type: ignore[assignment]

from packages.webui.middleware.exception_handlers import register_global_exception_handlers


def _app_with_handlers() -> TestClient:
    app = FastAPI()
    register_global_exception_handlers(app)
    return TestClient(app)


def test_packages_access_denied_maps_to_403() -> None:
    client = _app_with_handlers()

    @client.app.get("/packages-denied")
    def _endpoint() -> None:  # pragma: no cover - invoked via TestClient
        raise PackagesAccessDeniedError("user", "collection", "123")

    response = client.get("/packages-denied")
    assert response.status_code == 403
    assert response.json()["detail"]


def test_shared_access_denied_maps_to_403_when_available() -> None:
    if SharedAccessDeniedError is None:
        return  # Shared module not importable in this environment

    client = _app_with_handlers()

    @client.app.get("/shared-denied")
    def _endpoint() -> None:  # pragma: no cover - invoked via TestClient
        raise SharedAccessDeniedError("user", "collection", "456")

    response = client.get("/shared-denied")
    assert response.status_code == 403
    assert response.json()["detail"]
