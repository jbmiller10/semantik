"""Factory for creating LLM providers based on user configuration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from shared.config import settings
from shared.database.repositories.llm_provider_config_repository import LLMProviderConfigRepository
from shared.llm.exceptions import LLMAuthenticationError, LLMNotConfiguredError
from shared.llm.model_registry import get_default_model
from shared.llm.providers.anthropic_provider import AnthropicLLMProvider
from shared.llm.providers.local_provider import LocalLLMProvider
from shared.llm.providers.openai_provider import OpenAILLMProvider
from shared.llm.types import LLMQualityTier

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.llm.base import BaseLLMService

logger = logging.getLogger(__name__)


# Provider class mapping
_PROVIDER_CLASSES: dict[str, type[BaseLLMService]] = {
    "anthropic": AnthropicLLMProvider,
    "openai": OpenAILLMProvider,
    "local": LocalLLMProvider,
}

# Default provider when user has no preference
DEFAULT_PROVIDER = "anthropic"


def _create_provider_instance(provider_type: str) -> BaseLLMService:
    """Instantiate the correct provider class.

    Args:
        provider_type: Provider name ('anthropic', 'openai')

    Returns:
        Uninitialized provider instance

    Raises:
        ValueError: If provider type is unknown
    """
    cls = _PROVIDER_CLASSES.get(provider_type)
    if cls is None:
        raise ValueError(f"Unknown provider type: {provider_type}")
    return cls()


class LLMServiceFactory:
    """Factory for creating initialized LLM providers.

    This factory handles:
    1. Loading user's LLM configuration from database
    2. Selecting the correct provider and model for the requested tier
    3. Decrypting the API key
    4. Initializing and returning a ready-to-use provider

    Example:
        ```python
        from shared.llm.factory import LLMServiceFactory
        from shared.llm.types import LLMQualityTier

        factory = LLMServiceFactory(session)

        # Create provider for user's low-quality tier (HyDE)
        provider = await factory.create_provider_for_tier(
            user_id=123,
            quality_tier=LLMQualityTier.LOW,
        )

        async with provider:
            response = await provider.generate(
                prompt="Write a passage about machine learning",
                max_tokens=256,
            )
            print(response.content)
        ```
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize factory with database session.

        Args:
            session: AsyncSession for database operations
        """
        self.session = session
        self._config_repo = LLMProviderConfigRepository(session)

    async def create_provider_for_tier(
        self,
        user_id: int,
        quality_tier: LLMQualityTier,
        **kwargs: Any,
    ) -> BaseLLMService:
        """Create an initialized provider for the user's configured tier.

        This is the main entry point for getting an LLM provider.
        The provider is fully initialized and ready to use.

        Flow:
        1. Load user's LLMProviderConfig from database
        2. Get provider + model for the requested tier (or use defaults)
        3. Decrypt API key for that provider
        4. Instantiate the correct provider class
        5. Call provider.initialize(api_key, model)
        6. Return the ready-to-use provider

        Args:
            user_id: The user's ID
            quality_tier: Which tier to use (HIGH or LOW)
            **kwargs: Additional arguments passed to provider.initialize()

        Returns:
            Initialized BaseLLMService ready for generate() calls

        Raises:
            LLMNotConfiguredError: User hasn't set up LLM configuration
            LLMAuthenticationError: No API key configured for the provider
            ValueError: Unknown provider type

        Example:
            # For HyDE search (cost-effective, fast)
            provider = await factory.create_provider_for_tier(
                user_id=user.id,
                quality_tier=LLMQualityTier.LOW,
            )

            # For important summaries (best quality)
            provider = await factory.create_provider_for_tier(
                user_id=user.id,
                quality_tier=LLMQualityTier.HIGH,
            )
        """
        # Load user's config
        config = await self._config_repo.get_by_user_id(user_id)
        if config is None:
            logger.debug(f"LLM not configured for user {user_id}")
            raise LLMNotConfiguredError(user_id)

        # Get provider and model for the requested tier
        if quality_tier == LLMQualityTier.HIGH:
            provider_type = config.high_quality_provider
            model = config.high_quality_model
        else:
            provider_type = config.low_quality_provider
            model = config.low_quality_model

        # Use defaults if not configured
        if provider_type is None:
            provider_type = DEFAULT_PROVIDER
        if model is None:
            model = get_default_model(provider_type, quality_tier.value)

        # Get API key for the provider (skip for local)
        if provider_type == "local":
            api_key = ""  # Local models don't need authentication
            # Get quantization from provider_config JSON
            local_cfg = (config.provider_config or {}).get("local", {})
            default_quantization = settings.DEFAULT_LLM_QUANTIZATION or "int8"
            quantization = local_cfg.get(f"{quality_tier.value}_quantization") or default_quantization
        else:
            api_key = await self._config_repo.get_api_key(config.id, provider_type)
            if api_key is None:
                logger.warning(f"No API key for {provider_type} (user {user_id})")
                raise LLMAuthenticationError(
                    provider_type,
                    f"No API key configured for {provider_type}. Please add your API key in settings.",
                )
            # Update last_used timestamp (only for providers with API keys)
            # This is telemetry - not critical for operation
            try:
                await self._config_repo.update_key_last_used(config.id, provider_type)
            except Exception as e:
                logger.warning(f"Failed to update last_used timestamp for {provider_type}: {e}")
                # Rollback to clear the failed transaction state so subsequent operations can proceed
                await self.session.rollback()
            quantization = None

        # Create and initialize provider
        provider = _create_provider_instance(provider_type)
        if quantization is not None:
            await provider.initialize(api_key=api_key, model=model, quantization=quantization, **kwargs)
        else:
            await provider.initialize(api_key=api_key, model=model, **kwargs)

        logger.info(f"Created {provider_type} provider for user {user_id}, tier={quality_tier.value}, model={model}")

        return provider

    async def has_provider_configured(
        self,
        user_id: int,
        quality_tier: LLMQualityTier | None = None,
    ) -> bool:
        """Check if user has a provider configured (with API key).

        Useful for conditional UI display or feature gating.

        Args:
            user_id: The user's ID
            quality_tier: Optional tier to check. If None, checks if any
                         provider has an API key configured.

        Returns:
            True if the user can use LLM features for the tier

        Raises:
            DatabaseOperationError: For database errors
        """
        config = await self._config_repo.get_by_user_id(user_id)
        if config is None:
            return False

        if quality_tier is None:
            # Check if any provider has a key OR if local is configured
            providers = await self._config_repo.get_configured_providers(config.id)
            if len(providers) > 0:
                return True
            # Also check if either tier is configured for local (no API key needed)
            if config.high_quality_provider == "local" or config.low_quality_provider == "local":
                return True
            return False

        # Check specific tier
        if quality_tier == LLMQualityTier.HIGH:
            provider_type = config.high_quality_provider or DEFAULT_PROVIDER
        else:
            provider_type = config.low_quality_provider or DEFAULT_PROVIDER

        # Local provider doesn't need an API key
        if provider_type == "local":
            return True

        has_key = await self._config_repo.has_api_key(config.id, provider_type)
        return bool(has_key)


__all__ = ["LLMServiceFactory", "DEFAULT_PROVIDER"]
