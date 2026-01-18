"""Integration tests for LLM factory with real database."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, patch

import pytest

from shared.llm.exceptions import LLMAuthenticationError, LLMNotConfiguredError
from shared.llm.factory import LLMServiceFactory
from shared.llm.types import LLMQualityTier

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.database.models import LLMProviderConfig, User


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestLLMServiceFactoryIntegration:
    """Integration tests for LLMServiceFactory with real database."""

    async def test_create_provider_for_high_tier_with_anthropic(
        self,
        db_session: AsyncSession,
        llm_config_with_key: LLMProviderConfig,
        test_user_db: User,
        mock_anthropic_client: AsyncMock,
    ) -> None:
        """Creates Anthropic provider for HIGH tier with real config."""
        factory = LLMServiceFactory(db_session)
        user_id = cast(int, test_user_db.id)

        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic") as mock_class:
            mock_class.return_value = mock_anthropic_client

            provider = await factory.create_provider_for_tier(
                user_id=user_id,
                quality_tier=LLMQualityTier.HIGH,
            )

            # Verify provider was initialized with correct model
            assert provider.is_initialized
            model_info = provider.get_model_info()
            assert model_info["model"] == "claude-opus-4-5-20251101"

    async def test_create_provider_for_low_tier_with_anthropic(
        self,
        db_session: AsyncSession,
        llm_config_with_key: LLMProviderConfig,
        test_user_db: User,
        mock_anthropic_client: AsyncMock,
    ) -> None:
        """Creates Anthropic provider for LOW tier with real config."""
        factory = LLMServiceFactory(db_session)
        user_id = cast(int, test_user_db.id)

        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic") as mock_class:
            mock_class.return_value = mock_anthropic_client

            provider = await factory.create_provider_for_tier(
                user_id=user_id,
                quality_tier=LLMQualityTier.LOW,
            )

            assert provider.is_initialized
            model_info = provider.get_model_info()
            assert model_info["model"] == "claude-sonnet-4-5-20250929"

    async def test_create_provider_fails_without_config(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Raises LLMNotConfiguredError when user has no config."""
        factory = LLMServiceFactory(db_session)
        # Use a non-existent user ID
        nonexistent_user_id = 999999

        with pytest.raises(LLMNotConfiguredError) as exc_info:
            await factory.create_provider_for_tier(
                user_id=nonexistent_user_id,
                quality_tier=LLMQualityTier.LOW,
            )

        assert exc_info.value.user_id == nonexistent_user_id

    async def test_create_provider_fails_without_api_key(
        self,
        db_session: AsyncSession,
        llm_config: LLMProviderConfig,  # Config without API key
        test_user_db: User,
    ) -> None:
        """Raises LLMAuthenticationError when no API key configured."""
        factory = LLMServiceFactory(db_session)
        user_id = cast(int, test_user_db.id)

        with pytest.raises(LLMAuthenticationError) as exc_info:
            await factory.create_provider_for_tier(
                user_id=user_id,
                quality_tier=LLMQualityTier.LOW,
            )

        assert exc_info.value.provider == "anthropic"
        assert "No API key configured" in str(exc_info.value)

    async def test_multi_tier_different_providers(
        self,
        db_session: AsyncSession,
        llm_config_multi_provider: LLMProviderConfig,
        test_user_db: User,
        mock_anthropic_client: AsyncMock,
        mock_openai_client: AsyncMock,
    ) -> None:
        """Supports different providers per tier."""
        factory = LLMServiceFactory(db_session)
        user_id = cast(int, test_user_db.id)

        # HIGH tier uses OpenAI
        with patch("shared.llm.providers.openai_provider.AsyncOpenAI") as mock_openai_class:
            mock_openai_class.return_value = mock_openai_client

            high_provider = await factory.create_provider_for_tier(
                user_id=user_id,
                quality_tier=LLMQualityTier.HIGH,
            )

            assert high_provider.__class__.__name__ == "OpenAILLMProvider"
            high_model_info = high_provider.get_model_info()
            assert high_model_info["model"] == "gpt-4o"

        # LOW tier uses Anthropic
        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic") as mock_anthropic_class:
            mock_anthropic_class.return_value = mock_anthropic_client

            low_provider = await factory.create_provider_for_tier(
                user_id=user_id,
                quality_tier=LLMQualityTier.LOW,
            )

            assert low_provider.__class__.__name__ == "AnthropicLLMProvider"
            low_model_info = low_provider.get_model_info()
            assert low_model_info["model"] == "claude-sonnet-4-5-20250929"

    async def test_factory_uses_decrypted_key(
        self,
        db_session: AsyncSession,
        llm_config_with_key: LLMProviderConfig,
        test_user_db: User,
    ) -> None:
        """Factory correctly decrypts and passes API key to provider."""
        factory = LLMServiceFactory(db_session)
        user_id = cast(int, test_user_db.id)

        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic") as mock_class:
            mock_instance = AsyncMock()
            mock_class.return_value = mock_instance

            await factory.create_provider_for_tier(
                user_id=user_id,
                quality_tier=LLMQualityTier.LOW,
            )

            # Verify AsyncAnthropic was called with decrypted key
            mock_class.assert_called_once()
            call_kwargs = mock_class.call_args.kwargs
            assert call_kwargs["api_key"] == "sk-ant-test-key-for-integration-tests"

    async def test_has_provider_configured_true(
        self,
        db_session: AsyncSession,
        llm_config_with_key: LLMProviderConfig,
        test_user_db: User,
    ) -> None:
        """has_provider_configured returns True when key exists."""
        factory = LLMServiceFactory(db_session)
        user_id = cast(int, test_user_db.id)

        result = await factory.has_provider_configured(user_id)

        assert result is True

    async def test_has_provider_configured_false_no_config(
        self,
        db_session: AsyncSession,
    ) -> None:
        """has_provider_configured returns False when no config."""
        factory = LLMServiceFactory(db_session)

        result = await factory.has_provider_configured(user_id=999999)

        assert result is False

    async def test_has_provider_configured_false_no_key(
        self,
        db_session: AsyncSession,
        llm_config: LLMProviderConfig,  # Config without key
        test_user_db: User,
    ) -> None:
        """has_provider_configured returns False when no API key."""
        factory = LLMServiceFactory(db_session)
        user_id = cast(int, test_user_db.id)

        result = await factory.has_provider_configured(user_id)

        assert result is False

    async def test_has_provider_configured_tier_specific(
        self,
        db_session: AsyncSession,
        llm_config_multi_provider: LLMProviderConfig,
        test_user_db: User,
    ) -> None:
        """has_provider_configured checks specific tier when provided."""
        factory = LLMServiceFactory(db_session)
        user_id = cast(int, test_user_db.id)

        # Both tiers should be configured
        high_configured = await factory.has_provider_configured(user_id, quality_tier=LLMQualityTier.HIGH)
        low_configured = await factory.has_provider_configured(user_id, quality_tier=LLMQualityTier.LOW)

        assert high_configured is True
        assert low_configured is True
