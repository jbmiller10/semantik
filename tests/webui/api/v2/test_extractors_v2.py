"""Tests for v2 extractor endpoints."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from shared.plugins.manifest import PluginManifest
from shared.plugins.registry import PluginRecord, PluginSource, plugin_registry
from shared.plugins.types.extractor import ExtractionResult
from webui.api.v2 import extractors as extractors_module
from webui.auth import get_current_user
from webui.main import app


@pytest.fixture(autouse=True)
def _reset_registry(monkeypatch):
    plugin_registry.reset()
    monkeypatch.setattr(extractors_module, "load_plugins", lambda **_: plugin_registry)
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


def _register_extractor(plugin_id: str, supported: list[str] | None = None) -> None:
    manifest = PluginManifest(
        id=plugin_id,
        type="extractor",
        version="1.0.0",
        display_name="Test Extractor",
        description="Test extractor.",
        capabilities={"supported_extractions": supported or []},
    )
    record = PluginRecord(
        plugin_type="extractor",
        plugin_id=plugin_id,
        plugin_version="1.0.0",
        manifest=manifest,
        plugin_class=MagicMock,
        source=PluginSource.BUILTIN,
    )
    plugin_registry.register(record)


@pytest.mark.asyncio()
async def test_list_extractors(api_client: AsyncClient) -> None:
    _register_extractor("keyword-extractor", supported=["keywords"])

    response = await api_client.get("/api/v2/extractors")
    assert response.status_code == 200

    payload = response.json()
    assert payload["total"] == 1
    assert payload["extractors"][0]["id"] == "keyword-extractor"
    assert payload["extractors"][0]["supported_extractions"] == ["keywords"]


@pytest.mark.asyncio()
async def test_get_extractor_not_found(api_client: AsyncClient) -> None:
    response = await api_client.get("/api/v2/extractors/missing")
    assert response.status_code == 404


@pytest.mark.asyncio()
async def test_test_extraction_uses_service(monkeypatch, api_client: AsyncClient) -> None:
    _register_extractor("keyword-extractor")

    class StubService:
        async def run_extractors(self, **kwargs):
            return ExtractionResult(keywords=["alpha"], custom={"flag": True})

    monkeypatch.setattr(extractors_module, "get_extractor_service", lambda: StubService())

    response = await api_client.post(
        "/api/v2/extractors/test",
        json={"text": "hello", "extractor_ids": ["keyword-extractor"], "extraction_types": ["keywords"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["keywords"] == ["alpha"]
    assert payload["custom"] == {"flag": True}


@pytest.mark.asyncio()
async def test_test_extraction_rejects_unknown_extractor(api_client: AsyncClient) -> None:
    response = await api_client.post(
        "/api/v2/extractors/test",
        json={"text": "hello", "extractor_ids": ["missing"], "extraction_types": ["keywords"]},
    )
    assert response.status_code == 400
