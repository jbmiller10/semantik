"""Tests for v2 reranker endpoints."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from shared.plugins.manifest import PluginManifest
from shared.plugins.registry import PluginRecord, PluginSource, plugin_registry
from webui.api.v2 import rerankers as rerankers_module
from webui.auth import get_current_user
from webui.main import app


@pytest.fixture(autouse=True)
def _reset_registry(monkeypatch):
    plugin_registry.reset()
    monkeypatch.setattr(rerankers_module, "load_plugins", lambda **_: plugin_registry)
    yield
    plugin_registry.reset()


@pytest_asyncio.fixture()
async def api_client() -> AsyncClient:
    async def override_get_current_user() -> dict[str, Any]:
        return {"id": 1, "username": "tester"}

    app.dependency_overrides[get_current_user] = override_get_current_user
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


def _register_reranker(plugin_id: str) -> None:
    manifest = PluginManifest(
        id=plugin_id,
        type="reranker",
        version="1.0.0",
        display_name="Test Reranker",
        description="Test reranker.",
        capabilities={
            "max_documents": 10,
            "max_query_length": 128,
            "max_doc_length": 256,
            "supports_batching": True,
            "models": ["test-model"],
        },
    )
    record = PluginRecord(
        plugin_type="reranker",
        plugin_id=plugin_id,
        plugin_version="1.0.0",
        manifest=manifest,
        plugin_class=MagicMock,
        source=PluginSource.BUILTIN,
    )
    plugin_registry.register(record)


@pytest.mark.asyncio()
async def test_list_rerankers(api_client: AsyncClient) -> None:
    _register_reranker("test-reranker")

    response = await api_client.get("/api/v2/rerankers")
    assert response.status_code == 200

    payload = response.json()
    assert payload["total"] == 1
    assert payload["rerankers"][0]["id"] == "test-reranker"
    assert payload["rerankers"][0]["capabilities"]["models"] == ["test-model"]


@pytest.mark.asyncio()
async def test_get_reranker_not_found(api_client: AsyncClient) -> None:
    response = await api_client.get("/api/v2/rerankers/missing")
    assert response.status_code == 404


@pytest.mark.asyncio()
async def test_get_reranker_manifest(api_client: AsyncClient) -> None:
    _register_reranker("test-reranker")

    response = await api_client.get("/api/v2/rerankers/test-reranker/manifest")
    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "test-reranker"
    assert payload["capabilities"]["max_documents"] == 10
