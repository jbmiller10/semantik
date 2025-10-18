"""Tests for chunking composition container."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from packages.webui.services.chunking import container
from packages.webui.services.chunking.adapter import ChunkingServiceAdapter


@pytest.mark.asyncio()
async def test_resolve_api_dependency_prefers_adapter(monkeypatch):
    """Adapter is returned when orchestrator flag enabled."""

    monkeypatch.setattr(container.settings, "USE_CHUNKING_ORCHESTRATOR", True)

    fake_orchestrator = MagicMock()
    fake_orchestrator.collection_repo = MagicMock()
    fake_orchestrator.document_repo = MagicMock()
    monkeypatch.setattr(
        container,
        "build_chunking_orchestrator",
        AsyncMock(return_value=fake_orchestrator),
    )

    result = await container.resolve_api_chunking_dependency(
        AsyncMock(),
        prefer_adapter=True,
    )

    assert isinstance(result, ChunkingServiceAdapter)


@pytest.mark.asyncio()
async def test_resolve_api_dependency_legacy_when_flag_disabled(monkeypatch):
    """Legacy service is used when orchestrator flag disabled."""

    monkeypatch.setattr(container.settings, "USE_CHUNKING_ORCHESTRATOR", False)

    legacy_service = MagicMock()
    monkeypatch.setattr(
        container,
        "get_legacy_chunking_service",
        AsyncMock(return_value=legacy_service),
    )

    result = await container.resolve_api_chunking_dependency(AsyncMock())

    assert result is legacy_service


@pytest.mark.asyncio()
async def test_resolve_celery_dependency_returns_adapter(monkeypatch):
    """Celery resolver returns adapter when orchestrator enabled."""

    monkeypatch.setattr(container.settings, "USE_CHUNKING_ORCHESTRATOR", True)

    fake_orchestrator = MagicMock()
    fake_orchestrator.collection_repo = MagicMock()
    fake_orchestrator.document_repo = MagicMock()
    monkeypatch.setattr(
        container,
        "build_chunking_orchestrator",
        AsyncMock(return_value=fake_orchestrator),
    )

    result = await container.resolve_celery_chunking_service(AsyncMock())

    assert isinstance(result, ChunkingServiceAdapter)
