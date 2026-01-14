"""End-to-end tests for local LLM feature.

Phase 5 E2E tests verify the complete local LLM flow:
- Configuration workflow (no API key required)
- Generation with mocked VecPipe
- HyDE search integration
- Usage tracking
- Error handling
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from shared.database.repositories.llm_provider_config_repository import LLMProviderConfigRepository
from shared.database.repositories.llm_usage_repository import LLMUsageRepository
from shared.llm.exceptions import LLMProviderError, LLMTimeoutError
from shared.llm.factory import LLMServiceFactory
from shared.llm.providers.local_provider import LocalLLMProvider
from shared.llm.types import LLMQualityTier
from shared.llm.usage_tracking import record_llm_usage
from shared.utils.encryption import SecretEncryption, generate_fernet_key

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.database.models import User


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _initialize_encryption():  # type: ignore[misc]
    """Initialize encryption for E2E tests."""
    test_key = generate_fernet_key()
    SecretEncryption.initialize(test_key)
    yield
    SecretEncryption.reset()


def _create_mock_vecpipe_response(
    content: str = "Generated hypothetical passage about the topic.",
    prompt_tokens: int = 50,
    completion_tokens: int = 100,
) -> MagicMock:
    """Create a mock VecPipe LLM response."""
    response = MagicMock()
    response.json.return_value = {
        "contents": [content],
        "prompt_tokens": [prompt_tokens],
        "completion_tokens": [completion_tokens],
    }
    response.raise_for_status = MagicMock()
    return response


async def _create_local_provider(
    factory: LLMServiceFactory,
    user_id: int,
    quality_tier: LLMQualityTier = LLMQualityTier.LOW,
) -> LocalLLMProvider:
    """Create and return a LocalLLMProvider with mocked internal API key.

    This helper handles the patching and casting for type safety.
    """
    with patch("shared.config.internal_api_key.ensure_internal_api_key", return_value="test-key"):
        base_provider = await factory.create_provider_for_tier(
            user_id=user_id,
            quality_tier=quality_tier,
        )
    # Safe cast since we configure local provider
    return cast(LocalLLMProvider, base_provider)


# =============================================================================
# Test Classes
# =============================================================================


@pytest.mark.e2e()
class TestLocalLLMConfigurationWorkflow:
    """E2E tests for local LLM configuration workflow."""

    async def test_configure_local_provider_for_low_tier(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Configure local provider for LOW tier without API key."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        # Step 1: Create configuration
        await config_repo.get_or_create(user_id)

        # Step 2: Update with local provider for LOW tier
        await config_repo.update(
            user_id,
            low_quality_provider="local",
            low_quality_model="Qwen/Qwen2.5-1.5B-Instruct",
        )
        await db_session.commit()

        # Step 3: Verify configuration
        config = await config_repo.get_by_user_id(user_id)
        assert config is not None
        assert config.low_quality_provider == "local"
        assert config.low_quality_model == "Qwen/Qwen2.5-1.5B-Instruct"

    async def test_configure_local_provider_for_high_tier(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Configure local provider for HIGH tier."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        await config_repo.get_or_create(user_id)
        await config_repo.update(
            user_id,
            high_quality_provider="local",
            high_quality_model="Qwen/Qwen2.5-7B-Instruct",
        )
        await db_session.commit()

        config = await config_repo.get_by_user_id(user_id)
        assert config is not None
        assert config.high_quality_provider == "local"
        assert config.high_quality_model == "Qwen/Qwen2.5-7B-Instruct"

    async def test_quantization_persists_in_provider_config(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Quantization settings persist in provider_config JSON."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        await config_repo.get_or_create(user_id)
        await config_repo.update(
            user_id,
            low_quality_provider="local",
            low_quality_model="Qwen/Qwen2.5-1.5B-Instruct",
            provider_config={
                "local": {
                    "low_quantization": "int4",
                    "high_quantization": "int8",
                }
            },
        )
        await db_session.commit()

        config = await config_repo.get_by_user_id(user_id)
        assert config is not None
        assert config.provider_config is not None
        local_config = config.provider_config.get("local", {})
        assert local_config.get("low_quantization") == "int4"
        assert local_config.get("high_quantization") == "int8"

    async def test_local_provider_requires_no_api_key(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Local provider creation succeeds without API key."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        # Setup with local provider (no API key set)
        await config_repo.get_or_create(user_id)
        await config_repo.update(
            user_id,
            low_quality_provider="local",
            low_quality_model="Qwen/Qwen2.5-1.5B-Instruct",
        )
        await db_session.commit()

        # Create provider - should NOT require API key
        factory = LLMServiceFactory(db_session)
        provider = await _create_local_provider(factory, user_id)

        assert provider.is_initialized
        assert provider.__class__.__name__ == "LocalLLMProvider"

    async def test_has_provider_configured_returns_true_for_local(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """has_provider_configured returns True for local without API key."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        await config_repo.get_or_create(user_id)
        await config_repo.update(
            user_id,
            low_quality_provider="local",
            low_quality_model="Qwen/Qwen2.5-1.5B-Instruct",
        )
        await db_session.commit()

        factory = LLMServiceFactory(db_session)
        is_configured = await factory.has_provider_configured(
            user_id=user_id,
            quality_tier=LLMQualityTier.LOW,
        )

        assert is_configured is True


@pytest.mark.e2e()
class TestLocalLLMGeneration:
    """E2E tests for local LLM generation flow."""

    async def test_complete_local_llm_generation_flow(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Complete flow: configure → create provider → generate."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        # Setup
        await config_repo.get_or_create(user_id)
        await config_repo.update(
            user_id,
            low_quality_provider="local",
            low_quality_model="Qwen/Qwen2.5-1.5B-Instruct",
            provider_config={"local": {"low_quantization": "int8"}},
        )
        await db_session.commit()

        # Create provider
        factory = LLMServiceFactory(db_session)
        mock_response = _create_mock_vecpipe_response(
            content="Machine learning is a field of artificial intelligence...",
            prompt_tokens=45,
            completion_tokens=120,
        )

        provider = await _create_local_provider(factory, user_id)

        # Mock the HTTP client and generate
        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await provider.generate(
                prompt="Write a passage about machine learning",
                max_tokens=256,
            )

            assert result.content == "Machine learning is a field of artificial intelligence..."
            assert result.provider == "local"
            assert result.model == "Qwen/Qwen2.5-1.5B-Instruct"
            assert result.input_tokens == 45
            assert result.output_tokens == 120

    async def test_generation_with_system_prompt(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Generation includes system prompt in request."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        await config_repo.get_or_create(user_id)
        await config_repo.update(
            user_id,
            low_quality_provider="local",
            low_quality_model="Qwen/Qwen2.5-1.5B-Instruct",
        )
        await db_session.commit()

        factory = LLMServiceFactory(db_session)
        mock_response = _create_mock_vecpipe_response()

        provider = await _create_local_provider(factory, user_id)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await provider.generate(
                prompt="What is Python?",
                system_prompt="You are a helpful programming assistant.",
                max_tokens=128,
            )

            # Verify system_prompt was sent
            call_args = mock_post.call_args
            payload = call_args.kwargs["json"]
            assert payload["system_prompt"] == "You are a helpful programming assistant."

    async def test_generation_records_usage_tracking(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Generation records usage to database."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)
        usage_repo = LLMUsageRepository(db_session)

        await config_repo.get_or_create(user_id)
        await config_repo.update(
            user_id,
            low_quality_provider="local",
            low_quality_model="Qwen/Qwen2.5-1.5B-Instruct",
        )
        await db_session.commit()

        factory = LLMServiceFactory(db_session)
        mock_response = _create_mock_vecpipe_response(
            prompt_tokens=60,
            completion_tokens=150,
        )

        provider = await _create_local_provider(factory, user_id)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await provider.generate(prompt="Test prompt")

            # Record usage
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
        assert summary.total_input_tokens >= 60
        assert summary.total_output_tokens >= 150
        assert "hyde" in summary.by_feature

    async def test_generation_timeout_handling(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Timeout errors are converted to LLMTimeoutError."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        await config_repo.get_or_create(user_id)
        await config_repo.update(
            user_id,
            low_quality_provider="local",
            low_quality_model="Qwen/Qwen2.5-1.5B-Instruct",
        )
        await db_session.commit()

        factory = LLMServiceFactory(db_session)
        provider = await _create_local_provider(factory, user_id)

        with patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("Request timed out"),
        ):
            with pytest.raises(LLMTimeoutError) as exc_info:
                await provider.generate(prompt="Test")

            assert exc_info.value.provider == "local"

    async def test_generation_oom_handling(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """OOM errors (HTTP 507) are converted to LLMProviderError."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        await config_repo.get_or_create(user_id)
        await config_repo.update(
            user_id,
            low_quality_provider="local",
            low_quality_model="Qwen/Qwen2.5-7B-Instruct",
        )
        await db_session.commit()

        factory = LLMServiceFactory(db_session)
        provider = await _create_local_provider(factory, user_id)

        mock_http_response = MagicMock()
        mock_http_response.status_code = 507
        mock_http_response.text = "Insufficient GPU memory for model"

        with patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
            side_effect=httpx.HTTPStatusError("error", request=MagicMock(), response=mock_http_response),
        ):
            with pytest.raises(LLMProviderError) as exc_info:
                await provider.generate(prompt="Test")

            assert exc_info.value.provider == "local"
            assert exc_info.value.status_code == 507
            assert "GPU memory" in str(exc_info.value)

    async def test_generation_service_unavailable_handling(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Service unavailable (HTTP 503) is converted to LLMProviderError."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        await config_repo.get_or_create(user_id)
        await config_repo.update(
            user_id,
            low_quality_provider="local",
            low_quality_model="Qwen/Qwen2.5-1.5B-Instruct",
        )
        await db_session.commit()

        factory = LLMServiceFactory(db_session)
        provider = await _create_local_provider(factory, user_id)

        mock_http_response = MagicMock()
        mock_http_response.status_code = 503
        mock_http_response.text = "LLM service is disabled"

        with patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
            side_effect=httpx.HTTPStatusError("error", request=MagicMock(), response=mock_http_response),
        ):
            with pytest.raises(LLMProviderError) as exc_info:
                await provider.generate(prompt="Test")

            assert exc_info.value.provider == "local"
            assert exc_info.value.status_code == 503


@pytest.mark.e2e()
class TestLocalLLMHyDEIntegration:
    """E2E tests for HyDE search with local LLM."""

    async def test_hyde_generation_with_local_provider(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """HyDE uses local provider for LOW tier query expansion."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        # Configure local provider for LOW tier (HyDE uses LOW)
        await config_repo.get_or_create(user_id)
        await config_repo.update(
            user_id,
            low_quality_provider="local",
            low_quality_model="Qwen/Qwen2.5-1.5B-Instruct",
            provider_config={"local": {"low_quantization": "int8"}},
        )
        await db_session.commit()

        factory = LLMServiceFactory(db_session)

        # HyDE prompt generates hypothetical passages
        hyde_prompt = (
            "Write a passage that would be relevant to the following search query: 'How does machine learning work?'"
        )

        mock_response = _create_mock_vecpipe_response(
            content=(
                "Machine learning works by using algorithms that learn patterns from data. "
                "Neural networks process input through layers of interconnected nodes..."
            ),
            prompt_tokens=30,
            completion_tokens=80,
        )

        provider = await _create_local_provider(factory, user_id)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await provider.generate(
                prompt=hyde_prompt,
                max_tokens=256,
                temperature=0.7,
            )

            assert "Machine learning" in result.content
            assert result.provider == "local"

    async def test_hyde_usage_tracking_records_feature(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """HyDE usage is recorded with feature='hyde' and quality_tier='low'."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)
        usage_repo = LLMUsageRepository(db_session)

        await config_repo.get_or_create(user_id)
        await config_repo.update(
            user_id,
            low_quality_provider="local",
            low_quality_model="Qwen/Qwen2.5-1.5B-Instruct",
        )
        await db_session.commit()

        factory = LLMServiceFactory(db_session)
        mock_response = _create_mock_vecpipe_response(
            prompt_tokens=40,
            completion_tokens=90,
        )

        provider = await _create_local_provider(factory, user_id)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await provider.generate(prompt="HyDE expansion prompt")

            # Record as HyDE usage
            await record_llm_usage(
                session=db_session,
                user_id=user_id,
                response=result,
                feature="hyde",
                quality_tier="low",
            )
            await db_session.commit()

        # Verify HyDE feature tracking
        summary = await usage_repo.get_user_usage_summary(user_id)
        assert "hyde" in summary.by_feature
        hyde_usage = summary.by_feature["hyde"]
        assert hyde_usage["input_tokens"] == 40
        assert hyde_usage["output_tokens"] == 90

    async def test_hyde_with_different_quantizations(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """HyDE works with both int4 and int8 quantization."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        for quantization in ["int4", "int8"]:
            await config_repo.update(
                user_id,
                low_quality_provider="local",
                low_quality_model="Qwen/Qwen2.5-1.5B-Instruct",
                provider_config={"local": {"low_quantization": quantization}},
            )
            await db_session.commit()

            factory = LLMServiceFactory(db_session)
            mock_response = _create_mock_vecpipe_response()

            provider = await _create_local_provider(factory, user_id)

            # Verify quantization was set
            assert provider._quantization == quantization

            with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = mock_response

                await provider.generate(prompt="Test prompt")

                # Verify quantization in payload
                payload = mock_post.call_args.kwargs["json"]
                assert payload["quantization"] == quantization


@pytest.mark.e2e()
class TestLocalLLMMixedProviders:
    """E2E tests for mixed local and cloud provider configurations."""

    async def test_local_for_low_cloud_for_high(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """User can use local for LOW tier and cloud for HIGH tier."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        # Configure: HIGH = anthropic, LOW = local
        config = await config_repo.get_or_create(user_id)
        config_id = cast(int, config.id)
        await config_repo.update(
            user_id,
            high_quality_provider="anthropic",
            high_quality_model="claude-sonnet-4-5-20250929",
            low_quality_provider="local",
            low_quality_model="Qwen/Qwen2.5-1.5B-Instruct",
        )
        await config_repo.set_api_key(config_id, "anthropic", "sk-ant-test-key")
        await db_session.commit()

        factory = LLMServiceFactory(db_session)

        # Verify LOW tier uses local provider
        low_provider = await _create_local_provider(factory, user_id, LLMQualityTier.LOW)
        assert low_provider.__class__.__name__ == "LocalLLMProvider"

        # Verify HIGH tier uses anthropic provider
        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic") as mock_anthropic:
            mock_anthropic.return_value = AsyncMock()
            high_provider = await factory.create_provider_for_tier(
                user_id=user_id,
                quality_tier=LLMQualityTier.HIGH,
            )
            assert high_provider.__class__.__name__ == "AnthropicLLMProvider"

    async def test_both_tiers_use_local(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """User can use local provider for both HIGH and LOW tiers."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        await config_repo.get_or_create(user_id)
        await config_repo.update(
            user_id,
            high_quality_provider="local",
            high_quality_model="Qwen/Qwen2.5-7B-Instruct",
            low_quality_provider="local",
            low_quality_model="Qwen/Qwen2.5-1.5B-Instruct",
            provider_config={
                "local": {
                    "high_quantization": "int8",
                    "low_quantization": "int4",
                }
            },
        )
        await db_session.commit()

        factory = LLMServiceFactory(db_session)

        low_provider = await _create_local_provider(factory, user_id, LLMQualityTier.LOW)
        assert low_provider.__class__.__name__ == "LocalLLMProvider"
        assert low_provider._model == "Qwen/Qwen2.5-1.5B-Instruct"
        assert low_provider._quantization == "int4"

        high_provider = await _create_local_provider(factory, user_id, LLMQualityTier.HIGH)
        assert high_provider.__class__.__name__ == "LocalLLMProvider"
        assert high_provider._model == "Qwen/Qwen2.5-7B-Instruct"
        assert high_provider._quantization == "int8"
