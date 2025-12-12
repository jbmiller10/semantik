"""Integration tests for the /api/models endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:  # pragma: no cover
    from httpx import AsyncClient


@pytest.mark.asyncio()
async def test_models_endpoint_returns_expected_shape(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Verify /api/models returns the expected response structure."""

    response = await api_client.get("/api/models", headers=api_auth_headers)

    assert response.status_code == 200, response.text

    data = response.json()

    # Verify legacy response keys are present
    assert "models" in data
    assert "current_device" in data
    assert "using_real_embeddings" in data

    # Verify types
    assert isinstance(data["models"], dict)
    assert isinstance(data["current_device"], str)
    assert isinstance(data["using_real_embeddings"], bool)


@pytest.mark.asyncio()
async def test_models_endpoint_includes_plugin_models(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Verify plugin models are returned from /api/models."""

    # Mock get_all_supported_models to include a plugin model
    mock_models = [
        {
            "model_name": "Qwen/Qwen3-Embedding-0.6B",
            "dimension": 1024,
            "description": "Default model",
            "provider": "dense_local",
        },
        {
            "model_name": "test-plugin/custom-model",
            "dimension": 768,
            "description": "Test Plugin Model",
            "provider": "test_plugin",
            "supports_quantization": False,
        },
    ]

    with patch("webui.api.models.get_all_supported_models", return_value=mock_models):
        response = await api_client.get("/api/models", headers=api_auth_headers)

    assert response.status_code == 200, response.text
    data = response.json()

    # Verify both models are included
    assert "Qwen/Qwen3-Embedding-0.6B" in data["models"]
    assert "test-plugin/custom-model" in data["models"]

    # Verify plugin model has correct provider
    plugin_model = data["models"]["test-plugin/custom-model"]
    assert plugin_model["provider"] == "test_plugin"
    assert plugin_model["dimension"] == 768


@pytest.mark.asyncio()
async def test_models_endpoint_models_keyed_by_name(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Verify models dict is keyed by model_name for backward compatibility."""

    mock_models = [
        {"model_name": "org/model-a", "dimension": 512, "provider": "test"},
        {"model_name": "org/model-b", "dimension": 768, "provider": "test"},
    ]

    with patch("webui.api.models.get_all_supported_models", return_value=mock_models):
        response = await api_client.get("/api/models", headers=api_auth_headers)

    assert response.status_code == 200, response.text
    data = response.json()

    # Keys should be model names, not numeric indices
    assert "org/model-a" in data["models"]
    assert "org/model-b" in data["models"]

    # Each model should be a dict, not a list item
    assert isinstance(data["models"]["org/model-a"], dict)


@pytest.mark.asyncio()
async def test_models_endpoint_requires_auth(
    api_client_unauthenticated: AsyncClient,
) -> None:
    """Verify /api/models requires authentication."""
    from shared.config import settings

    # Temporarily disable the DISABLE_AUTH setting to test real authentication
    original_disable_auth = settings.DISABLE_AUTH
    settings.DISABLE_AUTH = False
    try:
        response = await api_client_unauthenticated.get("/api/models")

        # Should return 401 without auth header
        assert response.status_code == 401
    finally:
        settings.DISABLE_AUTH = original_disable_auth


@pytest.mark.asyncio()
async def test_models_endpoint_empty_models_handled(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Verify empty model list is handled gracefully."""

    with patch("webui.api.models.get_all_supported_models", return_value=[]):
        response = await api_client.get("/api/models", headers=api_auth_headers)

    assert response.status_code == 200, response.text
    data = response.json()

    # Should return empty dict, not error
    assert data["models"] == {}
    assert "current_device" in data
    assert "using_real_embeddings" in data
