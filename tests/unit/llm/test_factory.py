"""Tests for LLM service factory."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.llm.exceptions import LLMAuthenticationError, LLMNotConfiguredError
from shared.llm.factory import DEFAULT_PROVIDER, LLMServiceFactory, _create_provider_instance
from shared.llm.types import LLMQualityTier


class TestCreateProviderInstance:
    """Tests for _create_provider_instance helper."""

    def test_creates_anthropic_provider(self):
        """Creates AnthropicLLMProvider for 'anthropic'."""
        provider = _create_provider_instance("anthropic")
        assert provider.__class__.__name__ == "AnthropicLLMProvider"

    def test_creates_openai_provider(self):
        """Creates OpenAILLMProvider for 'openai'."""
        provider = _create_provider_instance("openai")
        assert provider.__class__.__name__ == "OpenAILLMProvider"

    def test_creates_local_provider(self):
        """Creates LocalLLMProvider for 'local'."""
        provider = _create_provider_instance("local")
        assert provider.__class__.__name__ == "LocalLLMProvider"

    def test_raises_for_unknown_provider(self):
        """Raises ValueError for unknown provider type."""
        with pytest.raises(ValueError, match="Unknown provider type"):
            _create_provider_instance("unknown")


class TestLLMServiceFactory:
    """Tests for LLMServiceFactory."""

    @pytest.fixture()
    def mock_session(self):
        """Create a mock database session."""
        return AsyncMock()

    @pytest.fixture()
    def factory(self, mock_session):
        """Create factory with mocked session."""
        return LLMServiceFactory(mock_session)

    @pytest.fixture()
    def mock_config(self):
        """Create a mock LLMProviderConfig."""
        config = MagicMock()
        config.id = 1
        config.user_id = 123
        config.high_quality_provider = "anthropic"
        config.high_quality_model = "claude-opus-4-5-20251101"
        config.low_quality_provider = "anthropic"
        config.low_quality_model = "claude-sonnet-4-5-20250929"
        return config

    async def test_create_provider_for_tier_not_configured(self, factory):
        """Raises LLMNotConfiguredError when user has no config."""
        with patch.object(factory._config_repo, "get_by_user_id", return_value=None):
            with pytest.raises(LLMNotConfiguredError) as exc_info:
                await factory.create_provider_for_tier(
                    user_id=123,
                    quality_tier=LLMQualityTier.LOW,
                )
            assert exc_info.value.user_id == 123

    async def test_create_provider_for_tier_no_api_key(self, factory, mock_config):
        """Raises LLMAuthenticationError when no API key configured."""
        with (
            patch.object(factory._config_repo, "get_by_user_id", return_value=mock_config),
            patch.object(factory._config_repo, "get_api_key", return_value=None),
        ):
            with pytest.raises(LLMAuthenticationError) as exc_info:
                await factory.create_provider_for_tier(
                    user_id=123,
                    quality_tier=LLMQualityTier.LOW,
                )
            assert exc_info.value.provider == "anthropic"

    async def test_create_provider_for_high_tier(self, factory, mock_config):
        """Creates provider with high tier configuration."""
        mock_provider = MagicMock()
        mock_provider.initialize = AsyncMock()

        with (
            patch.object(factory._config_repo, "get_by_user_id", return_value=mock_config),
            patch.object(factory._config_repo, "get_api_key", return_value="sk-ant-test"),
            patch.object(factory._config_repo, "update_key_last_used", return_value=None),
            patch("shared.llm.factory._create_provider_instance", return_value=mock_provider),
        ):
            result = await factory.create_provider_for_tier(
                user_id=123,
                quality_tier=LLMQualityTier.HIGH,
            )

            # Should use high tier config
            mock_provider.initialize.assert_called_once_with(
                api_key="sk-ant-test",
                model="claude-opus-4-5-20251101",
            )
            assert result == mock_provider

    async def test_create_provider_for_low_tier(self, factory, mock_config):
        """Creates provider with low tier configuration."""
        mock_provider = MagicMock()
        mock_provider.initialize = AsyncMock()

        with (
            patch.object(factory._config_repo, "get_by_user_id", return_value=mock_config),
            patch.object(factory._config_repo, "get_api_key", return_value="sk-ant-test"),
            patch.object(factory._config_repo, "update_key_last_used", return_value=None),
            patch("shared.llm.factory._create_provider_instance", return_value=mock_provider),
        ):
            result = await factory.create_provider_for_tier(
                user_id=123,
                quality_tier=LLMQualityTier.LOW,
            )

            # Should use low tier config
            mock_provider.initialize.assert_called_once_with(
                api_key="sk-ant-test",
                model="claude-sonnet-4-5-20250929",
            )
            assert result == mock_provider

    async def test_create_provider_uses_defaults_when_not_configured(self, factory):
        """Uses default provider and model when config has NULL values."""
        # Config with NULL provider/model
        mock_config = MagicMock()
        mock_config.id = 1
        mock_config.user_id = 123
        mock_config.high_quality_provider = None
        mock_config.high_quality_model = None
        mock_config.low_quality_provider = None
        mock_config.low_quality_model = None

        mock_provider = MagicMock()
        mock_provider.initialize = AsyncMock()

        with (
            patch.object(factory._config_repo, "get_by_user_id", return_value=mock_config),
            patch.object(factory._config_repo, "get_api_key", return_value="sk-ant-test"),
            patch.object(factory._config_repo, "update_key_last_used", return_value=None),
            patch("shared.llm.factory._create_provider_instance", return_value=mock_provider),
            patch("shared.llm.factory.get_default_model", return_value="claude-sonnet-4-5-20250929"),
        ):
            await factory.create_provider_for_tier(
                user_id=123,
                quality_tier=LLMQualityTier.LOW,
            )

            # Should use default provider (anthropic) and model from registry
            mock_provider.initialize.assert_called_once_with(
                api_key="sk-ant-test",
                model="claude-sonnet-4-5-20250929",
            )

    async def test_create_provider_updates_last_used(self, factory, mock_config):
        """Updates key last_used timestamp when creating provider."""
        with (
            patch.object(factory._config_repo, "get_by_user_id", return_value=mock_config),
            patch.object(factory._config_repo, "get_api_key", return_value="sk-ant-test"),
            patch.object(factory._config_repo, "update_key_last_used") as mock_update,
            patch("shared.llm.factory.AnthropicLLMProvider") as mock_provider_class,
        ):
            mock_provider = AsyncMock()
            mock_provider_class.return_value = mock_provider

            await factory.create_provider_for_tier(
                user_id=123,
                quality_tier=LLMQualityTier.LOW,
            )

            mock_update.assert_called_once_with(mock_config.id, "anthropic")

    async def test_create_provider_with_openai(self, factory):
        """Creates OpenAI provider when configured."""
        mock_config = MagicMock()
        mock_config.id = 1
        mock_config.user_id = 123
        mock_config.high_quality_provider = "openai"
        mock_config.high_quality_model = "gpt-4o"
        mock_config.low_quality_provider = "openai"
        mock_config.low_quality_model = "gpt-4o-mini"

        mock_provider = MagicMock()
        mock_provider.initialize = AsyncMock()

        with (
            patch.object(factory._config_repo, "get_by_user_id", return_value=mock_config),
            patch.object(factory._config_repo, "get_api_key", return_value="sk-test"),
            patch.object(factory._config_repo, "update_key_last_used", return_value=None),
            patch("shared.llm.factory._create_provider_instance", return_value=mock_provider),
        ):
            await factory.create_provider_for_tier(
                user_id=123,
                quality_tier=LLMQualityTier.HIGH,
            )

            mock_provider.initialize.assert_called_once_with(
                api_key="sk-test",
                model="gpt-4o",
            )


class TestHasProviderConfigured:
    """Tests for has_provider_configured method."""

    @pytest.fixture()
    def mock_session(self):
        """Create a mock database session."""
        return AsyncMock()

    @pytest.fixture()
    def factory(self, mock_session):
        """Create factory with mocked session."""
        return LLMServiceFactory(mock_session)

    async def test_no_config_returns_false(self, factory):
        """Returns False when user has no config."""
        with patch.object(factory._config_repo, "get_by_user_id", return_value=None):
            result = await factory.has_provider_configured(user_id=123)
            assert result is False

    async def test_no_keys_returns_false(self, factory):
        """Returns False when user has config but no keys."""
        mock_config = MagicMock()
        mock_config.id = 1

        with (
            patch.object(factory._config_repo, "get_by_user_id", return_value=mock_config),
            patch.object(factory._config_repo, "get_configured_providers", return_value=[]),
        ):
            result = await factory.has_provider_configured(user_id=123)
            assert result is False

    async def test_with_keys_returns_true(self, factory):
        """Returns True when user has at least one key configured."""
        mock_config = MagicMock()
        mock_config.id = 1

        with (
            patch.object(factory._config_repo, "get_by_user_id", return_value=mock_config),
            patch.object(factory._config_repo, "get_configured_providers", return_value=["anthropic"]),
        ):
            result = await factory.has_provider_configured(user_id=123)
            assert result is True

    async def test_tier_specific_check(self, factory):
        """Checks specific tier when quality_tier provided."""
        mock_config = MagicMock()
        mock_config.id = 1
        mock_config.low_quality_provider = "anthropic"

        with (
            patch.object(factory._config_repo, "get_by_user_id", return_value=mock_config),
            patch.object(factory._config_repo, "has_api_key", return_value=True),
        ):
            result = await factory.has_provider_configured(
                user_id=123,
                quality_tier=LLMQualityTier.LOW,
            )
            assert result is True

    async def test_local_provider_returns_true_without_key(self, factory):
        """Returns True for local provider without API key."""
        mock_config = MagicMock()
        mock_config.id = 1
        mock_config.low_quality_provider = "local"
        mock_config.high_quality_provider = None

        with (patch.object(factory._config_repo, "get_by_user_id", return_value=mock_config),):
            result = await factory.has_provider_configured(
                user_id=123,
                quality_tier=LLMQualityTier.LOW,
            )
            # Local provider doesn't need API key
            assert result is True

    async def test_local_provider_counted_in_any_check(self, factory):
        """Returns True when local is configured even without API keys."""
        mock_config = MagicMock()
        mock_config.id = 1
        mock_config.low_quality_provider = "local"
        mock_config.high_quality_provider = None

        with (
            patch.object(factory._config_repo, "get_by_user_id", return_value=mock_config),
            patch.object(factory._config_repo, "get_configured_providers", return_value=[]),
        ):
            # No tier specified, should check for any configured provider
            result = await factory.has_provider_configured(user_id=123)
            # Local provider counts as configured
            assert result is True


class TestDefaultProvider:
    """Tests for DEFAULT_PROVIDER constant."""

    def test_default_provider_is_anthropic(self):
        """Default provider is anthropic."""
        assert DEFAULT_PROVIDER == "anthropic"
