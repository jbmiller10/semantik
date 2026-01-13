"""Integration tests for LLM settings API endpoints."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from shared.database.repositories.llm_provider_config_repository import LLMProviderConfigRepository
from shared.database.repositories.llm_usage_repository import LLMUsageRepository, UsageSummary
from shared.llm.exceptions import LLMAuthenticationError
from webui.api.v2.llm_settings import _get_config_repo, _get_usage_repo
from webui.auth import get_current_user
from webui.main import app


@pytest.fixture()
def mock_llm_config():
    """Create a mock LLM config object."""
    config = MagicMock()
    config.id = 1
    config.user_id = 1
    config.high_quality_provider = "anthropic"
    config.high_quality_model = "claude-opus-4-5-20251101"
    config.low_quality_provider = "anthropic"
    config.low_quality_model = "claude-sonnet-4-5-20250929"
    config.default_temperature = 0.7
    config.default_max_tokens = None
    config.created_at = "2024-01-15T10:30:00+00:00"
    config.updated_at = "2024-01-15T10:30:00+00:00"
    return config


@pytest.fixture()
def mock_usage_summary():
    """Create a mock usage summary."""
    return UsageSummary(
        total_input_tokens=15000,
        total_output_tokens=5000,
        total_tokens=20000,
        by_feature={
            "hyde": {"input_tokens": 10000, "output_tokens": 3000, "total_tokens": 13000, "count": 100},
        },
        by_provider={
            "anthropic": {"input_tokens": 15000, "output_tokens": 5000, "total_tokens": 20000, "count": 150},
        },
        event_count=150,
        period_days=30,
    )


@pytest_asyncio.fixture
async def mock_config_repo(mock_llm_config):
    """Create a mock LLM config repository."""
    repo = MagicMock(spec=LLMProviderConfigRepository)
    repo.get_by_user_id = AsyncMock(return_value=mock_llm_config)
    repo.get_or_create = AsyncMock(return_value=mock_llm_config)
    repo.update = AsyncMock(return_value=mock_llm_config)
    repo.has_api_key = AsyncMock(return_value=True)
    repo.set_api_key = AsyncMock(return_value=MagicMock())
    return repo


@pytest_asyncio.fixture
async def mock_usage_repo(mock_usage_summary):
    """Create a mock LLM usage repository."""
    repo = MagicMock(spec=LLMUsageRepository)
    repo.get_user_usage_summary = AsyncMock(return_value=mock_usage_summary)
    return repo


@pytest_asyncio.fixture
async def llm_api_client(mock_config_repo, mock_usage_repo):
    """Provide an AsyncClient with LLM dependencies mocked."""
    mock_user = {
        "id": 1,
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
    }

    async def override_get_current_user() -> dict[str, Any]:
        return mock_user

    async def override_get_config_repo() -> LLMProviderConfigRepository:
        return mock_config_repo

    async def override_get_usage_repo() -> LLMUsageRepository:
        return mock_usage_repo

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[_get_config_repo] = override_get_config_repo
    app.dependency_overrides[_get_usage_repo] = override_get_usage_repo

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, mock_config_repo, mock_usage_repo

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def llm_api_client_not_configured():
    """Provide an AsyncClient for testing when LLM is not configured."""
    mock_user = {
        "id": 1,
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
    }

    mock_repo = MagicMock(spec=LLMProviderConfigRepository)
    mock_repo.get_by_user_id = AsyncMock(return_value=None)  # Not configured

    async def override_get_current_user() -> dict[str, Any]:
        return mock_user

    async def override_get_config_repo() -> LLMProviderConfigRepository:
        return mock_repo

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[_get_config_repo] = override_get_config_repo

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


class TestGetLLMSettings:
    """Tests for GET /api/v2/llm/settings endpoint."""

    @pytest.mark.asyncio()
    async def test_get_settings_returns_config(self, llm_api_client):
        """Test that settings are returned correctly."""
        client, mock_repo, _ = llm_api_client

        response = await client.get("/api/v2/llm/settings")

        assert response.status_code == 200
        data = response.json()
        assert data["high_quality_provider"] == "anthropic"
        assert data["high_quality_model"] == "claude-opus-4-5-20251101"
        assert data["anthropic_has_key"] is True
        assert data["openai_has_key"] is True

    @pytest.mark.asyncio()
    async def test_get_settings_not_configured_returns_404(self, llm_api_client_not_configured):
        """Test that 404 is returned when LLM is not configured."""
        client = llm_api_client_not_configured

        response = await client.get("/api/v2/llm/settings")

        assert response.status_code == 404


class TestUpdateLLMSettings:
    """Tests for PUT /api/v2/llm/settings endpoint."""

    @pytest.mark.asyncio()
    async def test_put_settings_creates_config(self, llm_api_client):
        """Test that settings can be created/updated."""
        client, mock_repo, _ = llm_api_client

        response = await client.put(
            "/api/v2/llm/settings",
            json={
                "high_quality_provider": "anthropic",
                "high_quality_model": "claude-opus-4-5-20251101",
            },
        )

        assert response.status_code == 200
        mock_repo.get_or_create.assert_called_once()
        mock_repo.update.assert_called_once()

    @pytest.mark.asyncio()
    async def test_put_settings_stores_api_keys(self, llm_api_client):
        """Test that API keys are stored when provided."""
        client, mock_repo, _ = llm_api_client

        response = await client.put(
            "/api/v2/llm/settings",
            json={
                "anthropic_api_key": "sk-ant-test-key",
                "openai_api_key": "sk-test-key",
            },
        )

        assert response.status_code == 200
        # Verify set_api_key was called for both providers
        assert mock_repo.set_api_key.call_count == 2

    @pytest.mark.asyncio()
    async def test_put_settings_validates_temperature(self, llm_api_client):
        """Test that temperature is validated."""
        client, _, _ = llm_api_client

        # Temperature above 2.0 should fail
        response = await client.put(
            "/api/v2/llm/settings",
            json={"default_temperature": 2.5},
        )

        assert response.status_code == 422  # Validation error


class TestListModels:
    """Tests for GET /api/v2/llm/models endpoint."""

    @pytest.mark.asyncio()
    async def test_get_models_returns_registry(self, llm_api_client):
        """Test that models are returned from the registry."""
        client, _, _ = llm_api_client

        response = await client.get("/api/v2/llm/models")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) > 0
        # Check that required fields are present
        model = data["models"][0]
        assert "id" in model
        assert "name" in model
        assert "display_name" in model
        assert "provider" in model
        assert "tier_recommendation" in model
        assert "context_window" in model


class TestTestApiKey:
    """Tests for POST /api/v2/llm/test endpoint."""

    @pytest.mark.asyncio()
    async def test_test_endpoint_validates_key_success(self, llm_api_client):
        """Test that valid API key returns success."""
        client, _, _ = llm_api_client

        with patch("webui.api.v2.llm_settings.AnthropicLLMProvider") as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
            mock_provider.__aexit__ = AsyncMock(return_value=None)
            mock_provider.initialize = AsyncMock()
            mock_provider.generate = AsyncMock()
            mock_provider_class.return_value = mock_provider

            response = await client.post(
                "/api/v2/llm/test",
                json={
                    "provider": "anthropic",
                    "api_key": "sk-ant-test-key",
                },
            )

        # Debug: print response if not 200
        if response.status_code != 200:
            print(f"Response: {response.status_code} - {response.text}")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "API key is valid"
        assert data["model_tested"] is not None

    @pytest.mark.asyncio()
    async def test_test_endpoint_validates_key_failure(self, llm_api_client):
        """Test that invalid API key returns failure."""
        client, _, _ = llm_api_client

        with patch("webui.api.v2.llm_settings.AnthropicLLMProvider") as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.initialize = AsyncMock(side_effect=LLMAuthenticationError("anthropic", "Invalid key"))
            mock_provider_class.return_value = mock_provider

            response = await client.post(
                "/api/v2/llm/test",
                json={
                    "provider": "anthropic",
                    "api_key": "invalid-key",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "Authentication failed" in data["message"]


class TestGetUsage:
    """Tests for GET /api/v2/llm/usage endpoint."""

    @pytest.mark.asyncio()
    async def test_get_usage_returns_summary(self, llm_api_client):
        """Test that usage summary is returned."""
        client, _, mock_usage_repo = llm_api_client

        response = await client.get("/api/v2/llm/usage")

        assert response.status_code == 200
        data = response.json()
        assert data["total_tokens"] == 20000
        assert data["total_input_tokens"] == 15000
        assert data["total_output_tokens"] == 5000
        assert data["event_count"] == 150
        assert data["period_days"] == 30
        assert "by_feature" in data
        assert "by_provider" in data

    @pytest.mark.asyncio()
    async def test_get_usage_with_date_filter(self, llm_api_client):
        """Test that usage can be filtered by date range."""
        client, _, mock_usage_repo = llm_api_client

        response = await client.get("/api/v2/llm/usage?days=7")

        assert response.status_code == 200
        mock_usage_repo.get_user_usage_summary.assert_called_once()
        # Check that days parameter was passed
        call_args = mock_usage_repo.get_user_usage_summary.call_args
        assert call_args[0][1] == 7  # Second positional arg is days
