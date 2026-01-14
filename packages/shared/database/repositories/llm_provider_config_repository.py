"""Repository implementation for LLM provider configuration."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, cast

from sqlalchemy import delete, select, update

from shared.database.db_retry import with_db_retry
from shared.database.exceptions import DatabaseOperationError
from shared.database.models import LLMProviderApiKey, LLMProviderConfig
from shared.utils.encryption import DecryptionError, EncryptionNotConfiguredError, SecretEncryption

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.llm.types import LLMQualityTier

logger = logging.getLogger(__name__)


class _UnsetType:
    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


UNSET: Final[_UnsetType] = _UnsetType()


class LLMProviderConfigRepository:
    """Repository for LLM provider configuration and API key management.

    This repository manages per-user LLM configuration including:
    - Quality tier settings (high/low provider and model selection)
    - Encrypted API keys per provider (shared across tiers)
    - Default temperature and max_tokens settings

    IMPORTANT: SecretEncryption.initialize() must be called before using
    API key operations. If encryption is not configured, those operations
    will raise EncryptionNotConfiguredError.

    Example:
        ```python
        repo = LLMProviderConfigRepository(session)

        # Create or get user's config
        config = await repo.get_or_create(user_id=123)

        # Update tier configuration
        await repo.update_tier_config(
            user_id=123,
            tier=LLMQualityTier.HIGH,
            provider="anthropic",
            model="claude-opus-4-5-20251101",
        )

        # Set API key (encrypts automatically)
        await repo.set_api_key(config.id, "anthropic", "sk-ant-...")

        # Get API key (decrypts automatically)
        key = await repo.get_api_key(config.id, "anthropic")
        ```
    """

    # Valid provider types (validated in application layer, not DB)
    VALID_PROVIDERS = frozenset({"anthropic", "openai"})

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with database session.

        Args:
            session: AsyncSession instance for database operations
        """
        self.session = session

    def _validate_provider(self, provider: str) -> None:
        """Validate that provider is supported.

        Args:
            provider: The provider name to validate

        Raises:
            ValueError: If provider is not valid
        """
        if provider not in self.VALID_PROVIDERS:
            raise ValueError(
                f"Invalid provider '{provider}'. Must be one of: {', '.join(sorted(self.VALID_PROVIDERS))}"
            )

    # =========================================================================
    # Config CRUD Operations
    # =========================================================================

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get_by_user_id(self, user_id: int) -> LLMProviderConfig | None:
        """Get LLM configuration for a user.

        Args:
            user_id: The user's ID

        Returns:
            LLMProviderConfig instance or None if not configured

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            result = await self.session.execute(select(LLMProviderConfig).where(LLMProviderConfig.user_id == user_id))
            return result.scalar_one_or_none()

        except Exception as e:
            logger.error("Failed to get LLM config for user %s: %s", user_id, e, exc_info=True)
            raise DatabaseOperationError("get", "LLMProviderConfig", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get_or_create(self, user_id: int) -> LLMProviderConfig:
        """Get existing config or create a new one with defaults.

        The new config is created with NULL provider/model values,
        which means the factory will use application defaults from
        the model registry.

        Args:
            user_id: The user's ID

        Returns:
            LLMProviderConfig instance (existing or newly created)

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            config = await self.get_by_user_id(user_id)
            if config is not None:
                return config

            # Create new config with NULL values (use registry defaults)
            config = LLMProviderConfig(user_id=user_id)
            self.session.add(config)
            await self.session.flush()

            logger.info(f"Created LLM config for user_id={user_id}")
            return config

        except DatabaseOperationError:
            raise
        except Exception as e:
            logger.error("Failed to get/create LLM config for user %s: %s", user_id, e, exc_info=True)
            raise DatabaseOperationError("get_or_create", "LLMProviderConfig", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def update(
        self,
        user_id: int,
        *,
        high_quality_provider: str | None | _UnsetType = UNSET,
        high_quality_model: str | None | _UnsetType = UNSET,
        low_quality_provider: str | None | _UnsetType = UNSET,
        low_quality_model: str | None | _UnsetType = UNSET,
        default_temperature: float | None | _UnsetType = UNSET,
        default_max_tokens: int | None | _UnsetType = UNSET,
        provider_config: dict[str, Any] | None | _UnsetType = UNSET,
    ) -> LLMProviderConfig:
        """Update LLM configuration for a user.

        Only updates fields that are explicitly provided.
        Pass None to clear a field back to defaults.
        Creates config if it doesn't exist.

        Args:
            user_id: The user's ID
            high_quality_provider: Provider for high-quality tier
            high_quality_model: Model for high-quality tier
            low_quality_provider: Provider for low-quality tier
            low_quality_model: Model for low-quality tier
            default_temperature: Default temperature (0.0-2.0)
            default_max_tokens: Default max tokens
            provider_config: Provider-specific configuration

        Returns:
            Updated LLMProviderConfig instance

        Raises:
            ValueError: If temperature is out of range or provider is invalid
            DatabaseOperationError: For database errors
        """
        # Validate temperature if provided
        # Use isinstance for mypy type narrowing (mypy doesn't narrow custom sentinels with `is not`)
        if not isinstance(default_temperature, _UnsetType) and default_temperature is not None:  # noqa: SIM102
            if not (0.0 <= default_temperature <= 2.0):
                raise ValueError("default_temperature must be between 0.0 and 2.0")

        # Validate providers if provided
        # Use isinstance for mypy type narrowing (mypy doesn't narrow custom sentinels with `is not`)
        if not isinstance(high_quality_provider, _UnsetType) and high_quality_provider is not None:
            self._validate_provider(high_quality_provider)
        if not isinstance(low_quality_provider, _UnsetType) and low_quality_provider is not None:
            self._validate_provider(low_quality_provider)

        try:
            config = await self.get_or_create(user_id)

            # Update only provided fields
            if high_quality_provider is not UNSET:
                config.high_quality_provider = high_quality_provider
            if high_quality_model is not UNSET:
                config.high_quality_model = high_quality_model
            if low_quality_provider is not UNSET:
                config.low_quality_provider = low_quality_provider
            if low_quality_model is not UNSET:
                config.low_quality_model = low_quality_model
            if default_temperature is not UNSET:
                config.default_temperature = default_temperature
            if default_max_tokens is not UNSET:
                config.default_max_tokens = default_max_tokens
            if provider_config is not UNSET:
                config.provider_config = provider_config

            config.updated_at = datetime.now(UTC)

            await self.session.flush()
            logger.debug(f"Updated LLM config for user_id={user_id}")

            return config

        except ValueError:
            raise
        except DatabaseOperationError:
            raise
        except Exception as e:
            logger.error("Failed to update LLM config for user %s: %s", user_id, e, exc_info=True)
            raise DatabaseOperationError("update", "LLMProviderConfig", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def update_tier_config(
        self,
        user_id: int,
        tier: LLMQualityTier,
        provider: str,
        model: str,
    ) -> LLMProviderConfig:
        """Update configuration for a specific quality tier.

        Convenience method for updating a single tier's provider and model.

        Args:
            user_id: The user's ID
            tier: Quality tier to update (HIGH or LOW)
            provider: Provider name
            model: Model identifier

        Returns:
            Updated LLMProviderConfig instance

        Raises:
            ValueError: If provider is invalid
            DatabaseOperationError: For database errors
        """
        from shared.llm.types import LLMQualityTier

        self._validate_provider(provider)

        if tier == LLMQualityTier.HIGH:
            return await self.update(
                user_id,
                high_quality_provider=provider,
                high_quality_model=model,
            )
        return await self.update(
            user_id,
            low_quality_provider=provider,
            low_quality_model=model,
        )

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def delete(self, user_id: int) -> bool:
        """Delete LLM configuration for a user.

        This also cascades to delete all associated API keys.

        Args:
            user_id: The user's ID

        Returns:
            True if config was deleted, False if it didn't exist

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            result = await self.session.execute(delete(LLMProviderConfig).where(LLMProviderConfig.user_id == user_id))
            deleted = (result.rowcount or 0) > 0

            if deleted:
                logger.info(f"Deleted LLM config for user_id={user_id}")

            return deleted

        except Exception as e:
            logger.error("Failed to delete LLM config for user %s: %s", user_id, e, exc_info=True)
            raise DatabaseOperationError("delete", "LLMProviderConfig", str(e)) from e

    # =========================================================================
    # API Key Operations (Encrypted)
    # =========================================================================

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def set_api_key(
        self,
        config_id: int,
        provider: str,
        plaintext_key: str,
    ) -> LLMProviderApiKey:
        """Store or update an encrypted API key for a provider.

        If a key already exists for the provider, it is replaced.
        Keys are stored per-provider (not per-tier), so if both
        tiers use the same provider, they share the key.

        Args:
            config_id: ID of the LLMProviderConfig
            provider: Provider name ('anthropic', 'openai')
            plaintext_key: The API key to encrypt and store

        Returns:
            Created/updated LLMProviderApiKey instance

        Raises:
            ValueError: If provider is invalid
            EncryptionNotConfiguredError: If encryption key is not configured
            DatabaseOperationError: For database errors
        """
        self._validate_provider(provider)

        try:
            # Encrypt the key
            ciphertext = SecretEncryption.encrypt(plaintext_key)
            key_id = SecretEncryption.get_key_id()

            # Delete existing key (upsert pattern)
            await self.session.execute(
                delete(LLMProviderApiKey).where(
                    LLMProviderApiKey.config_id == config_id,
                    LLMProviderApiKey.provider == provider,
                )
            )

            # Create new key
            api_key = LLMProviderApiKey(
                config_id=config_id,
                provider=provider,
                ciphertext=ciphertext,
                key_id=key_id,
            )

            self.session.add(api_key)
            await self.session.flush()

            logger.debug(f"Stored encrypted API key for provider={provider} (key_id={key_id})")

            return api_key

        except (ValueError, EncryptionNotConfiguredError):
            raise
        except Exception as e:
            logger.error("Failed to store API key for provider %s: %s", provider, e, exc_info=True)
            raise DatabaseOperationError("store", "LLMProviderApiKey", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get_api_key(
        self,
        config_id: int,
        provider: str,
    ) -> str | None:
        """Retrieve and decrypt an API key for a provider.

        Args:
            config_id: ID of the LLMProviderConfig
            provider: Provider name ('anthropic', 'openai')

        Returns:
            Decrypted API key string, or None if not found

        Raises:
            ValueError: If provider is invalid
            EncryptionNotConfiguredError: If encryption key is not configured
            DecryptionError: If decryption fails (key may have changed)
            DatabaseOperationError: For database errors
        """
        self._validate_provider(provider)

        try:
            result = await self.session.execute(
                select(LLMProviderApiKey).where(
                    LLMProviderApiKey.config_id == config_id,
                    LLMProviderApiKey.provider == provider,
                )
            )
            api_key = result.scalar_one_or_none()

            if api_key is None:
                return None

            # Decrypt and return
            plaintext: str = SecretEncryption.decrypt(cast(bytes, api_key.ciphertext))

            logger.debug(f"Retrieved API key for provider={provider}")

            return plaintext

        except (ValueError, EncryptionNotConfiguredError, DecryptionError):
            raise
        except Exception as e:
            logger.error("Failed to retrieve API key for provider %s: %s", provider, e, exc_info=True)
            raise DatabaseOperationError("retrieve", "LLMProviderApiKey", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def has_api_key(
        self,
        config_id: int,
        provider: str,
    ) -> bool:
        """Check if an API key exists without decrypting it.

        This is a lightweight check that doesn't require the encryption
        key to be configured.

        Args:
            config_id: ID of the LLMProviderConfig
            provider: Provider name ('anthropic', 'openai')

        Returns:
            True if key exists, False otherwise

        Raises:
            ValueError: If provider is invalid
            DatabaseOperationError: For database errors
        """
        self._validate_provider(provider)

        try:
            result = await self.session.execute(
                select(LLMProviderApiKey.id).where(
                    LLMProviderApiKey.config_id == config_id,
                    LLMProviderApiKey.provider == provider,
                )
            )
            return result.scalar_one_or_none() is not None

        except ValueError:
            raise
        except Exception as e:
            logger.error("Failed to check API key existence for provider %s: %s", provider, e, exc_info=True)
            raise DatabaseOperationError("check", "LLMProviderApiKey", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def delete_api_key(
        self,
        config_id: int,
        provider: str,
    ) -> bool:
        """Delete an API key for a provider.

        Args:
            config_id: ID of the LLMProviderConfig
            provider: Provider name ('anthropic', 'openai')

        Returns:
            True if key was deleted, False if it didn't exist

        Raises:
            ValueError: If provider is invalid
            DatabaseOperationError: For database errors
        """
        self._validate_provider(provider)

        try:
            result = await self.session.execute(
                delete(LLMProviderApiKey).where(
                    LLMProviderApiKey.config_id == config_id,
                    LLMProviderApiKey.provider == provider,
                )
            )
            deleted = (result.rowcount or 0) > 0

            if deleted:
                logger.debug(f"Deleted API key for provider={provider}")

            return deleted

        except ValueError:
            raise
        except Exception as e:
            logger.error("Failed to delete API key for provider %s: %s", provider, e, exc_info=True)
            raise DatabaseOperationError("delete", "LLMProviderApiKey", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def update_key_last_used(
        self,
        config_id: int,
        provider: str,
    ) -> None:
        """Update the last_used_at timestamp for an API key.

        This is called by the factory when a key is used to create
        a provider instance.

        Args:
            config_id: ID of the LLMProviderConfig
            provider: Provider name ('anthropic', 'openai')

        Raises:
            ValueError: If provider is invalid
            DatabaseOperationError: For database errors
        """
        self._validate_provider(provider)

        try:
            await self.session.execute(
                update(LLMProviderApiKey)
                .where(
                    LLMProviderApiKey.config_id == config_id,
                    LLMProviderApiKey.provider == provider,
                )
                .values(last_used_at=datetime.now(UTC))
            )

        except ValueError:
            raise
        except Exception as e:
            logger.error("Failed to update last_used_at for provider %s: %s", provider, e, exc_info=True)
            raise DatabaseOperationError("update_last_used", "LLMProviderApiKey", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get_configured_providers(self, config_id: int) -> list[str]:
        """Get list of providers that have API keys configured.

        Useful for building API responses that indicate which
        providers are configured without revealing the keys.

        Args:
            config_id: ID of the LLMProviderConfig

        Returns:
            List of provider names with configured keys

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            result = await self.session.execute(
                select(LLMProviderApiKey.provider).where(LLMProviderApiKey.config_id == config_id)
            )
            return [row[0] for row in result.all()]

        except Exception as e:
            logger.error("Failed to get configured providers: %s", e, exc_info=True)
            raise DatabaseOperationError("get_providers", "LLMProviderApiKey", str(e)) from e
