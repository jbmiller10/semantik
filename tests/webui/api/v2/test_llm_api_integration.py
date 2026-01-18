"""Integration tests for LLM API endpoints with real database."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from shared.database.models import LLMProviderApiKey, LLMProviderConfig
from shared.database.repositories.llm_usage_repository import LLMUsageRepository

if TYPE_CHECKING:
    from httpx import AsyncClient
    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.database.models import User


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestLLMSettingsAPIIntegration:
    """Integration tests for LLM settings API with real database."""

    async def test_put_settings_creates_new_config(
        self,
        api_client: AsyncClient,
        api_auth_headers: dict[str, str],
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """PUT settings creates new config for user."""
        response = await api_client.put(
            "/api/v2/llm/settings",
            headers=api_auth_headers,
            json={
                "high_quality_provider": "anthropic",
                "high_quality_model": "claude-opus-4-5-20251101",
                "low_quality_provider": "anthropic",
                "low_quality_model": "claude-sonnet-4-5-20250929",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["high_quality_provider"] == "anthropic"
        assert data["high_quality_model"] == "claude-opus-4-5-20251101"

        # Verify persisted in database
        result = await db_session.execute(select(LLMProviderConfig).where(LLMProviderConfig.user_id == test_user_db.id))
        config = result.scalar_one()
        assert config.high_quality_provider == "anthropic"

    async def test_put_settings_updates_existing_config(
        self,
        api_client: AsyncClient,
        api_auth_headers: dict[str, str],
        db_session: AsyncSession,
        llm_config: LLMProviderConfig,
    ) -> None:
        """PUT settings updates existing config."""
        # Update to OpenAI
        response = await api_client.put(
            "/api/v2/llm/settings",
            headers=api_auth_headers,
            json={
                "high_quality_provider": "openai",
                "high_quality_model": "gpt-4o",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["high_quality_provider"] == "openai"
        assert data["high_quality_model"] == "gpt-4o"
        # Low quality should remain unchanged
        assert data["low_quality_provider"] == "anthropic"

    async def test_get_settings_returns_persisted_data(
        self,
        api_client: AsyncClient,
        api_auth_headers: dict[str, str],
        llm_config_with_key: LLMProviderConfig,
    ) -> None:
        """GET settings returns persisted configuration."""
        response = await api_client.get(
            "/api/v2/llm/settings",
            headers=api_auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["high_quality_provider"] == "anthropic"
        assert data["high_quality_model"] == "claude-opus-4-5-20251101"
        assert data["anthropic_has_key"] is True

    async def test_put_settings_encrypts_api_key(
        self,
        api_client: AsyncClient,
        api_auth_headers: dict[str, str],
        db_session: AsyncSession,
        llm_config: LLMProviderConfig,
    ) -> None:
        """PUT settings with API key encrypts and stores it."""
        test_key = "sk-ant-test-key-from-api"

        response = await api_client.put(
            "/api/v2/llm/settings",
            headers=api_auth_headers,
            json={
                "anthropic_api_key": test_key,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["anthropic_has_key"] is True

        # Verify key is encrypted in database
        config_id = cast(int, llm_config.id)
        result = await db_session.execute(
            select(LLMProviderApiKey).where(
                LLMProviderApiKey.config_id == config_id,
                LLMProviderApiKey.provider == "anthropic",
            )
        )
        api_key_row = result.scalar_one()
        # Ciphertext should not contain plaintext
        assert test_key.encode() not in api_key_row.ciphertext

    async def test_get_settings_never_returns_api_key(
        self,
        api_client: AsyncClient,
        api_auth_headers: dict[str, str],
        llm_config_with_key: LLMProviderConfig,
    ) -> None:
        """GET settings never returns the actual API key."""
        response = await api_client.get(
            "/api/v2/llm/settings",
            headers=api_auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        # Should have has_key flag, not actual key
        assert data["anthropic_has_key"] is True
        assert "anthropic_api_key" not in data
        assert "sk-ant" not in str(data)

    async def test_user_cannot_access_other_user_settings(
        self,
        api_client_other_user: AsyncClient,
        other_user_auth_headers: dict[str, str],
        llm_config_with_key: LLMProviderConfig,
    ) -> None:
        """User cannot see another user's LLM settings."""
        # other_user should not see test_user's config
        response = await api_client_other_user.get(
            "/api/v2/llm/settings",
            headers=other_user_auth_headers,
        )

        # Should return 404 (other user has no config)
        assert response.status_code == 404

    async def test_models_endpoint_returns_curated_list(
        self,
        api_client: AsyncClient,
        api_auth_headers: dict[str, str],
    ) -> None:
        """GET models returns curated model list."""
        response = await api_client.get(
            "/api/v2/llm/models",
            headers=api_auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert "models" in data
        models = data["models"]
        assert len(models) > 0

        # Check structure of first model
        first_model = models[0]
        assert "id" in first_model
        assert "name" in first_model
        assert "provider" in first_model
        assert "tier_recommendation" in first_model

    async def test_get_usage_returns_aggregated_stats(
        self,
        api_client: AsyncClient,
        api_auth_headers: dict[str, str],
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """GET usage returns aggregated token statistics."""
        # Create some usage events first
        usage_repo = LLMUsageRepository(db_session)
        user_id = cast(int, test_user_db.id)

        await usage_repo.record_usage(
            user_id=user_id,
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            quality_tier="low",
            feature="hyde",
            input_tokens=100,
            output_tokens=200,
        )
        await db_session.commit()

        response = await api_client.get(
            "/api/v2/llm/usage",
            headers=api_auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["total_input_tokens"] == 100
        assert data["total_output_tokens"] == 200
        assert data["total_tokens"] == 300
        assert "by_feature" in data
        assert "hyde" in data["by_feature"]

    async def test_usage_filters_by_days_parameter(
        self,
        api_client: AsyncClient,
        api_auth_headers: dict[str, str],
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """GET usage respects days query parameter."""
        # Create usage event
        usage_repo = LLMUsageRepository(db_session)
        user_id = cast(int, test_user_db.id)

        await usage_repo.record_usage(
            user_id=user_id,
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            quality_tier="low",
            feature="hyde",
            input_tokens=100,
            output_tokens=200,
        )
        await db_session.commit()

        # Query with specific days parameter
        response = await api_client.get(
            "/api/v2/llm/usage?days=7",
            headers=api_auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["period_days"] == 7

    async def test_test_endpoint_validates_key(
        self,
        api_client: AsyncClient,
        api_auth_headers: dict[str, str],
        llm_config: LLMProviderConfig,
    ) -> None:
        """POST test validates API key format."""
        # Test with a properly formatted but invalid key
        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic") as mock_class:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(side_effect=Exception("Invalid API key"))
            mock_class.return_value = mock_client

            response = await api_client.post(
                "/api/v2/llm/test",
                headers=api_auth_headers,
                json={
                    "provider": "anthropic",
                    "api_key": "sk-ant-invalid-test-key",
                },
            )

            # Should fail the test
            assert response.status_code in (200, 400)
            data = response.json()
            if response.status_code == 200:
                assert data.get("success") is False

    async def test_get_settings_returns_404_when_not_configured(
        self,
        api_client: AsyncClient,
        api_auth_headers: dict[str, str],
    ) -> None:
        """GET settings returns 404 when user has no config."""
        response = await api_client.get(
            "/api/v2/llm/settings",
            headers=api_auth_headers,
        )

        assert response.status_code == 404
