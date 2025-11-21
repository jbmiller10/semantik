"""Integration tests for the metrics API endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def test_metrics_endpoint_returns_prometheus_payload(
    test_client: TestClient, auth_headers: dict[str, str], test_user: dict[str, Any], monkeypatch
) -> None:
    """Ensure the /api/metrics endpoint returns Prometheus-formatted data."""
    from webui.api import metrics as metrics_module
    from webui.main import app

    expected_metrics = "# HELP embedding_operations_created_total\nembedding_operations_created_total 3"

    def fake_generate_latest(_registry: Any) -> bytes:
        return expected_metrics.encode("utf-8")

    async def override_current_user() -> dict[str, Any]:
        return test_user

    monkeypatch.setattr(metrics_module, "generate_latest", fake_generate_latest, raising=False)
    monkeypatch.setattr(metrics_module, "METRICS_AVAILABLE", True, raising=False)

    app.dependency_overrides[metrics_module.get_current_user] = override_current_user
    try:
        response = test_client.get("/api/metrics", headers=auth_headers)
    finally:
        app.dependency_overrides.pop(metrics_module.get_current_user, None)

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["metrics_port"] == metrics_module.METRICS_PORT
    assert expected_metrics in body["data"]


def test_metrics_endpoint_reports_error_when_remote_unhealthy(
    test_client: TestClient, auth_headers: dict[str, str], test_user: dict[str, Any], monkeypatch
) -> None:
    """Ensure the endpoint surfaces a clear error if remote metrics are unavailable."""
    from webui.api import metrics as metrics_module
    from webui.main import app

    fallback_port = 9876

    monkeypatch.setattr(metrics_module, "METRICS_AVAILABLE", False, raising=False)
    monkeypatch.setattr(metrics_module, "METRICS_PORT", fallback_port, raising=False)

    async def override_current_user() -> dict[str, Any]:
        return test_user

    class DummyResponse:
        status_code = 503
        text = "service unavailable"

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> DummyAsyncClient:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

        async def get(self, url: str) -> DummyResponse:
            assert url == f"http://localhost:{fallback_port}/metrics"
            return DummyResponse()

    monkeypatch.setattr(httpx, "AsyncClient", DummyAsyncClient)

    app.dependency_overrides[metrics_module.get_current_user] = override_current_user
    try:
        response = test_client.get("/api/metrics", headers=auth_headers)
    finally:
        app.dependency_overrides.pop(metrics_module.get_current_user, None)

    assert response.status_code == 200
    body = response.json()
    assert body["error"] == "Metrics server not responding"
    assert body["metrics_port"] == fallback_port
    assert "available" not in body
