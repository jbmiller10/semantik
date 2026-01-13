"""End-to-end tests for LLM integration.

These tests verify the complete LLM flow using mocked LLM SDK responses
for faster iteration without actual API calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.database.repositories.llm_provider_config_repository import LLMProviderConfigRepository
from shared.database.repositories.llm_usage_repository import LLMUsageRepository
from shared.llm.factory import LLMServiceFactory
from shared.llm.types import LLMQualityTier
from shared.llm.usage_tracking import record_llm_usage
from shared.utils.encryption import EncryptionNotConfiguredError, SecretEncryption, generate_fernet_key

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.database.models import User


@pytest.fixture(autouse=True)
def _initialize_encryption():
    """Initialize encryption for E2E tests."""
    test_key = generate_fernet_key()
    SecretEncryption.initialize(test_key)
    yield
    SecretEncryption.reset()


def _create_mock_anthropic_response(
    content: str = "Generated response text",
    input_tokens: int = 50,
    output_tokens: int = 100,
) -> MagicMock:
    """Create a mock Anthropic SDK response."""
    response = MagicMock()
    response.content = [MagicMock(text=content)]
    response.model = "claude-sonnet-4-5-20250929"
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    response.stop_reason = "end_turn"
    return response


def _create_mock_openai_response(
    content: str = "Generated response text",
    input_tokens: int = 50,
    output_tokens: int = 100,
) -> MagicMock:
    """Create a mock OpenAI SDK response."""
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    response.model = "gpt-4o"
    response.usage = MagicMock(prompt_tokens=input_tokens, completion_tokens=output_tokens)
    response.choices[0].finish_reason = "stop"
    return response


@pytest.mark.e2e()
class TestLLMConfigurationWorkflow:
    """End-to-end tests for LLM configuration workflow."""

    async def test_complete_configuration_workflow(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Complete flow: create config → set key → verify → use provider."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        # Step 1: Create configuration
        config = await config_repo.get_or_create(user_id)
        config_id = cast(int, config.id)

        # Step 2: Update with provider preferences
        await config_repo.update(
            user_id,
            high_quality_provider="anthropic",
            high_quality_model="claude-opus-4-5-20251101",
            low_quality_provider="anthropic",
            low_quality_model="claude-sonnet-4-5-20250929",
        )

        # Step 3: Set API key
        await config_repo.set_api_key(config_id, "anthropic", "sk-ant-test-key")
        await db_session.commit()

        # Step 4: Verify configuration is complete
        has_key = await config_repo.has_api_key(config_id, "anthropic")
        assert has_key is True

        # Step 5: Create provider and verify it works
        factory = LLMServiceFactory(db_session)
        mock_response = _create_mock_anthropic_response()

        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic") as mock_class:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            provider = await factory.create_provider_for_tier(
                user_id=user_id,
                quality_tier=LLMQualityTier.LOW,
            )

            assert provider.is_initialized

    async def test_usage_tracking_records_llm_calls(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """LLM calls should record usage to database."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)
        usage_repo = LLMUsageRepository(db_session)

        # Setup: Create config with API key
        config = await config_repo.get_or_create(user_id)
        config_id = cast(int, config.id)
        await config_repo.update(
            user_id,
            low_quality_provider="anthropic",
            low_quality_model="claude-sonnet-4-5-20250929",
        )
        await config_repo.set_api_key(config_id, "anthropic", "sk-ant-test-key")
        await db_session.commit()

        # Create provider and make a call
        factory = LLMServiceFactory(db_session)
        mock_response = _create_mock_anthropic_response(
            input_tokens=150,
            output_tokens=300,
        )

        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic") as mock_class:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            provider = await factory.create_provider_for_tier(
                user_id=user_id,
                quality_tier=LLMQualityTier.LOW,
            )

            # Make LLM call
            result = await provider.generate(
                prompt="Test prompt",
                max_tokens=256,
            )

            # Record usage (simulating what HyDE would do)
            await record_llm_usage(
                session=db_session,
                user_id=user_id,
                response=result,
                feature="hyde",
                quality_tier="low",
            )
            await db_session.commit()

        # Verify usage was recorded
        summary = await usage_repo.get_user_usage_summary(user_id, days=30)
        assert summary.event_count >= 1
        assert summary.total_input_tokens >= 150
        assert summary.total_output_tokens >= 300

    async def test_settings_persist_across_queries(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Settings should persist and be retrievable."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        # Create and configure
        await config_repo.update(
            user_id,
            high_quality_provider="openai",
            high_quality_model="gpt-4o",
            low_quality_provider="anthropic",
            low_quality_model="claude-sonnet-4-5-20250929",
            default_temperature=0.7,
        )
        await db_session.commit()

        # Retrieve and verify
        config = await config_repo.get_by_user_id(user_id)
        assert config is not None
        assert config.high_quality_provider == "openai"
        assert config.high_quality_model == "gpt-4o"
        assert config.low_quality_provider == "anthropic"
        assert config.default_temperature == 0.7

    async def test_invalid_api_key_returns_clear_error(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Invalid API key should return clear authentication error."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        # Setup with invalid key
        config = await config_repo.get_or_create(user_id)
        config_id = cast(int, config.id)
        await config_repo.set_api_key(config_id, "anthropic", "invalid-key")
        await db_session.commit()

        factory = LLMServiceFactory(db_session)

        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic") as mock_class:
            mock_client = AsyncMock()
            # Simulate authentication error
            from anthropic import AuthenticationError

            mock_client.messages.create = AsyncMock(
                side_effect=AuthenticationError(
                    message="Invalid API Key",
                    response=MagicMock(status_code=401),
                    body=None,
                )
            )
            mock_class.return_value = mock_client

            provider = await factory.create_provider_for_tier(
                user_id=user_id,
                quality_tier=LLMQualityTier.LOW,
            )

            # The error should be raised when trying to generate
            from shared.llm.exceptions import LLMAuthenticationError

            with pytest.raises(LLMAuthenticationError):
                await provider.generate(prompt="Test")

    async def test_missing_encryption_key_surfaces_error(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Missing encryption key should surface clear error."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        # Reset encryption to simulate missing key
        SecretEncryption.reset()

        # Trying to set API key should fail
        config = await config_repo.get_or_create(user_id)
        config_id = cast(int, config.id)

        with pytest.raises(EncryptionNotConfiguredError):
            await config_repo.set_api_key(config_id, "anthropic", "sk-ant-test")

        # Re-initialize for cleanup
        SecretEncryption.initialize(generate_fernet_key())


@pytest.mark.e2e()
class TestLLMProviderBehavior:
    """End-to-end tests for LLM provider behavior."""

    async def test_sequential_provider_creation(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Multiple sequential provider creations should work correctly."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        # Setup
        config = await config_repo.get_or_create(user_id)
        config_id = cast(int, config.id)
        await config_repo.set_api_key(config_id, "anthropic", "sk-ant-test")
        await db_session.commit()

        factory = LLMServiceFactory(db_session)
        providers = []

        # Create multiple providers sequentially (SQLAlchemy sessions can't be used concurrently)
        for _ in range(3):
            with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic") as mock_class:
                mock_class.return_value = AsyncMock()
                provider = await factory.create_provider_for_tier(
                    user_id=user_id,
                    quality_tier=LLMQualityTier.LOW,
                )
                providers.append(provider)

        assert len(providers) == 3
        assert all(p.is_initialized for p in providers)

    async def test_provider_graceful_degradation(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Provider errors should be catchable for graceful degradation."""
        import anthropic

        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        # Setup
        config = await config_repo.get_or_create(user_id)
        config_id = cast(int, config.id)
        await config_repo.set_api_key(config_id, "anthropic", "sk-ant-test")
        await db_session.commit()

        factory = LLMServiceFactory(db_session)

        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic") as mock_class:
            mock_client = AsyncMock()
            # Simulate a provider error using Anthropic's exception type
            mock_client.messages.create = AsyncMock(
                side_effect=anthropic.APIError(
                    message="Service unavailable",
                    request=MagicMock(),
                    body=None,
                )
            )
            mock_class.return_value = mock_client

            provider = await factory.create_provider_for_tier(
                user_id=user_id,
                quality_tier=LLMQualityTier.LOW,
            )

            # Error should be catchable
            from shared.llm.exceptions import LLMProviderError

            with pytest.raises(LLMProviderError):
                await provider.generate(prompt="Test")

    async def test_usage_tracking_by_feature(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Usage tracking should correctly categorize by feature."""
        user_id = cast(int, test_user_db.id)
        usage_repo = LLMUsageRepository(db_session)

        # Record usage for different features
        await usage_repo.record_usage(
            user_id=user_id,
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            quality_tier="low",
            feature="hyde",
            input_tokens=100,
            output_tokens=200,
        )
        await usage_repo.record_usage(
            user_id=user_id,
            provider="anthropic",
            model="claude-opus-4-5-20251101",
            quality_tier="high",
            feature="summary",
            input_tokens=500,
            output_tokens=300,
        )
        await db_session.commit()

        # Verify by-feature breakdown
        summary = await usage_repo.get_user_usage_summary(user_id)

        assert "hyde" in summary.by_feature
        assert summary.by_feature["hyde"]["input_tokens"] == 100
        assert summary.by_feature["hyde"]["output_tokens"] == 200

        assert "summary" in summary.by_feature
        assert summary.by_feature["summary"]["input_tokens"] == 500
        assert summary.by_feature["summary"]["output_tokens"] == 300

    async def test_multi_provider_configuration(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """User can configure different providers for different tiers."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        # Setup: OpenAI for high, Anthropic for low
        config = await config_repo.get_or_create(user_id)
        config_id = cast(int, config.id)
        await config_repo.update(
            user_id,
            high_quality_provider="openai",
            high_quality_model="gpt-4o",
            low_quality_provider="anthropic",
            low_quality_model="claude-sonnet-4-5-20250929",
        )
        await config_repo.set_api_key(config_id, "anthropic", "sk-ant-test")
        await config_repo.set_api_key(config_id, "openai", "sk-openai-test")
        await db_session.commit()

        factory = LLMServiceFactory(db_session)

        # Create HIGH tier provider (OpenAI)
        with patch("shared.llm.providers.openai_provider.AsyncOpenAI") as mock_openai:
            mock_openai.return_value = AsyncMock()
            high_provider = await factory.create_provider_for_tier(
                user_id=user_id,
                quality_tier=LLMQualityTier.HIGH,
            )
            assert high_provider.__class__.__name__ == "OpenAILLMProvider"

        # Create LOW tier provider (Anthropic)
        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic") as mock_anthropic:
            mock_anthropic.return_value = AsyncMock()
            low_provider = await factory.create_provider_for_tier(
                user_id=user_id,
                quality_tier=LLMQualityTier.LOW,
            )
            assert low_provider.__class__.__name__ == "AnthropicLLMProvider"
