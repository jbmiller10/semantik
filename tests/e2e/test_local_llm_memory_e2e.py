"""End-to-end tests for local LLM memory management.

Phase 5 E2E tests for GPU memory governance, eviction, and model lifecycle.
Tests are mocked to run without GPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.database.repositories.llm_provider_config_repository import LLMProviderConfigRepository
from shared.llm.factory import LLMServiceFactory
from shared.llm.providers.local_provider import LocalLLMProvider
from shared.llm.types import LLMQualityTier
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
    content: str = "Generated text",
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
    """Create a LocalLLMProvider with mocked internal API key."""
    with patch("shared.config.internal_api_key.ensure_internal_api_key", return_value="test-key"):
        base_provider = await factory.create_provider_for_tier(
            user_id=user_id,
            quality_tier=quality_tier,
        )
    return cast(LocalLLMProvider, base_provider)


# =============================================================================
# Test Classes
# =============================================================================


@pytest.mark.e2e()
class TestLocalLLMMemoryManagement:
    """E2E tests for GPU memory management."""

    async def test_provider_sends_quantization_to_vecpipe(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Provider includes quantization in VecPipe request for memory management."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        await config_repo.get_or_create(user_id)
        await config_repo.update(
            user_id,
            low_quality_provider="local",
            low_quality_model="Qwen/Qwen2.5-1.5B-Instruct",
            provider_config={"local": {"low_quantization": "int4"}},
        )
        await db_session.commit()

        factory = LLMServiceFactory(db_session)
        provider = await _create_local_provider(factory, user_id)
        mock_response = _create_mock_vecpipe_response()

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await provider.generate(prompt="Test")

            # Verify quantization is passed to VecPipe for memory estimation
            payload = mock_post.call_args.kwargs["json"]
            assert payload["quantization"] == "int4"

    async def test_memory_constrained_model_selection(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Users can select smaller models and int4 quantization for memory-constrained GPUs."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        # Configure smallest model with most aggressive quantization
        await config_repo.get_or_create(user_id)
        await config_repo.update(
            user_id,
            low_quality_provider="local",
            low_quality_model="Qwen/Qwen2.5-0.5B-Instruct",
            provider_config={"local": {"low_quantization": "int4"}},
        )
        await db_session.commit()

        factory = LLMServiceFactory(db_session)
        provider = await _create_local_provider(factory, user_id)
        mock_response = _create_mock_vecpipe_response()

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await provider.generate(prompt="Test")

            assert result.model == "Qwen/Qwen2.5-0.5B-Instruct"
            payload = mock_post.call_args.kwargs["json"]
            assert payload["model_name"] == "Qwen/Qwen2.5-0.5B-Instruct"
            assert payload["quantization"] == "int4"


@pytest.mark.e2e()
class TestLocalLLMModelSwitching:
    """E2E tests for switching between models."""

    async def test_switch_between_quantizations(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Switching quantization creates new provider with different settings."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        # Start with int8
        await config_repo.get_or_create(user_id)
        await config_repo.update(
            user_id,
            low_quality_provider="local",
            low_quality_model="Qwen/Qwen2.5-1.5B-Instruct",
            provider_config={"local": {"low_quantization": "int8"}},
        )
        await db_session.commit()

        factory = LLMServiceFactory(db_session)
        provider1 = await _create_local_provider(factory, user_id)
        assert provider1._quantization == "int8"

        # Switch to int4
        await config_repo.update(
            user_id,
            provider_config={"local": {"low_quantization": "int4"}},
        )
        await db_session.commit()

        # Create new provider (simulating new request after settings change)
        factory2 = LLMServiceFactory(db_session)
        provider2 = await _create_local_provider(factory2, user_id)
        assert provider2._quantization == "int4"

    async def test_switch_between_models(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Switching models creates provider for different model."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        # Start with 1.5B model
        await config_repo.get_or_create(user_id)
        await config_repo.update(
            user_id,
            low_quality_provider="local",
            low_quality_model="Qwen/Qwen2.5-1.5B-Instruct",
        )
        await db_session.commit()

        factory = LLMServiceFactory(db_session)
        provider1 = await _create_local_provider(factory, user_id)
        assert provider1._model == "Qwen/Qwen2.5-1.5B-Instruct"

        # Switch to 3B model
        await config_repo.update(
            user_id,
            low_quality_model="Qwen/Qwen2.5-3B-Instruct",
        )
        await db_session.commit()

        factory2 = LLMServiceFactory(db_session)
        provider2 = await _create_local_provider(factory2, user_id)
        assert provider2._model == "Qwen/Qwen2.5-3B-Instruct"

    async def test_switch_model_and_quantization(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Can switch both model and quantization in one update."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        await config_repo.get_or_create(user_id)
        await config_repo.update(
            user_id,
            low_quality_provider="local",
            low_quality_model="Qwen/Qwen2.5-1.5B-Instruct",
            provider_config={"local": {"low_quantization": "int8"}},
        )
        await db_session.commit()

        factory = LLMServiceFactory(db_session)
        provider1 = await _create_local_provider(factory, user_id)
        assert provider1._model == "Qwen/Qwen2.5-1.5B-Instruct"
        assert provider1._quantization == "int8"

        # Switch to larger model with more aggressive quantization to fit memory
        await config_repo.update(
            user_id,
            low_quality_model="Qwen/Qwen2.5-7B-Instruct",
            provider_config={"local": {"low_quantization": "int4"}},
        )
        await db_session.commit()

        factory2 = LLMServiceFactory(db_session)
        provider2 = await _create_local_provider(factory2, user_id)
        assert provider2._model == "Qwen/Qwen2.5-7B-Instruct"
        assert provider2._quantization == "int4"


@pytest.mark.e2e()
class TestLocalLLMModelRequests:
    """E2E tests for concurrent model requests."""

    async def test_sequential_requests_use_same_config(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Sequential requests use consistent model configuration."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        await config_repo.get_or_create(user_id)
        await config_repo.update(
            user_id,
            low_quality_provider="local",
            low_quality_model="Qwen/Qwen2.5-1.5B-Instruct",
            provider_config={"local": {"low_quantization": "int8"}},
        )
        await db_session.commit()

        mock_response = _create_mock_vecpipe_response()

        # Make multiple sequential requests
        for i in range(3):
            factory = LLMServiceFactory(db_session)
            provider = await _create_local_provider(factory, user_id)

            with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = mock_response
                await provider.generate(prompt=f"Request {i}")

                # All requests use same model/quantization
                payload = mock_post.call_args.kwargs["json"]
                assert payload["model_name"] == "Qwen/Qwen2.5-1.5B-Instruct"
                assert payload["quantization"] == "int8"

    async def test_provider_reuse_pattern(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Single provider can be reused for multiple generations."""
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
        mock_response = _create_mock_vecpipe_response()

        # Reuse same provider for multiple generations
        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result1 = await provider.generate(prompt="First prompt")
            result2 = await provider.generate(prompt="Second prompt")
            result3 = await provider.generate(prompt="Third prompt")

            # All succeeded
            assert result1.provider == "local"
            assert result2.provider == "local"
            assert result3.provider == "local"

            # Verify 3 calls made
            assert mock_post.call_count == 3


@pytest.mark.e2e()
class TestLocalLLMTierDifferentiation:
    """E2E tests for different model/quantization per quality tier."""

    async def test_high_and_low_tiers_different_quantization(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """HIGH tier can use int8 while LOW tier uses int4 for memory efficiency."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        await config_repo.get_or_create(user_id)
        await config_repo.update(
            user_id,
            high_quality_provider="local",
            high_quality_model="Qwen/Qwen2.5-3B-Instruct",
            low_quality_provider="local",
            low_quality_model="Qwen/Qwen2.5-1.5B-Instruct",
            provider_config={
                "local": {
                    "high_quantization": "int8",  # Better quality for HIGH
                    "low_quantization": "int4",  # More memory efficient for LOW
                }
            },
        )
        await db_session.commit()

        factory = LLMServiceFactory(db_session)

        high_provider = await _create_local_provider(factory, user_id, LLMQualityTier.HIGH)
        assert high_provider._model == "Qwen/Qwen2.5-3B-Instruct"
        assert high_provider._quantization == "int8"

        low_provider = await _create_local_provider(factory, user_id, LLMQualityTier.LOW)
        assert low_provider._model == "Qwen/Qwen2.5-1.5B-Instruct"
        assert low_provider._quantization == "int4"

    async def test_model_size_for_tier_usage(
        self,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        """Users can configure larger models for HIGH tier and smaller for LOW tier."""
        user_id = cast(int, test_user_db.id)
        config_repo = LLMProviderConfigRepository(db_session)

        await config_repo.get_or_create(user_id)
        await config_repo.update(
            user_id,
            high_quality_provider="local",
            high_quality_model="Qwen/Qwen2.5-7B-Instruct",  # Largest for quality
            low_quality_provider="local",
            low_quality_model="Qwen/Qwen2.5-0.5B-Instruct",  # Smallest for speed
            provider_config={
                "local": {
                    "high_quantization": "int4",  # Needed to fit 7B
                    "low_quantization": "int8",  # 0.5B fits in int8
                }
            },
        )
        await db_session.commit()

        factory = LLMServiceFactory(db_session)
        mock_response = _create_mock_vecpipe_response()

        # Verify HIGH tier uses large model
        high_provider = await _create_local_provider(factory, user_id, LLMQualityTier.HIGH)
        with patch.object(high_provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await high_provider.generate(prompt="Complex task")
            payload = mock_post.call_args.kwargs["json"]
            assert payload["model_name"] == "Qwen/Qwen2.5-7B-Instruct"

        # Verify LOW tier uses small model
        low_provider = await _create_local_provider(factory, user_id, LLMQualityTier.LOW)
        with patch.object(low_provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await low_provider.generate(prompt="Simple task")
            payload = mock_post.call_args.kwargs["json"]
            assert payload["model_name"] == "Qwen/Qwen2.5-0.5B-Instruct"
