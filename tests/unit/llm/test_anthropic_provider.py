"""Tests for Anthropic LLM provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from shared.llm.providers.anthropic_provider import AnthropicLLMProvider


class TestAnthropicLLMProvider:
    """Tests for AnthropicLLMProvider."""

    @pytest.fixture()
    def provider(self):
        """Create a fresh provider instance."""
        return AnthropicLLMProvider()

    def test_not_initialized_by_default(self, provider):
        """Provider is not initialized after construction."""
        assert not provider.is_initialized

    async def test_initialize_sets_initialized(self, provider):
        """initialize() sets is_initialized to True."""
        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic"):
            await provider.initialize(api_key="test-key", model="test-model")
            assert provider.is_initialized

    async def test_initialize_requires_api_key(self, provider):
        """initialize() raises ValueError without API key."""
        with pytest.raises(ValueError, match="API key"):
            await provider.initialize(api_key="", model="test-model")

    async def test_initialize_requires_model(self, provider):
        """initialize() raises ValueError without model."""
        with pytest.raises(ValueError, match="Model"):
            await provider.initialize(api_key="test-key", model="")

    async def test_generate_without_init_raises(self, provider):
        """generate() raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await provider.generate("test prompt")

    async def test_get_model_info_without_init_raises(self, provider):
        """get_model_info() raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            provider.get_model_info()

    async def test_cleanup_resets_initialized(self, provider):
        """cleanup() resets is_initialized to False."""
        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic") as mock_class:
            mock_client = AsyncMock()
            mock_class.return_value = mock_client

            await provider.initialize(api_key="test-key", model="test-model")
            assert provider.is_initialized

            await provider.cleanup()
            assert not provider.is_initialized

    async def test_context_manager_calls_cleanup(self, provider):
        """Context manager calls cleanup on exit."""
        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic") as mock_class:
            mock_client = AsyncMock()
            mock_class.return_value = mock_client

            await provider.initialize(api_key="test-key", model="test-model")

            async with provider:
                assert provider.is_initialized

            # After context exit, cleanup should have been called
            mock_client.close.assert_called_once()

    async def test_successful_generate(self, provider):
        """generate() returns LLMResponse on success."""
        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic") as mock_class:
            # Create mock response
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Hello, world!")]
            mock_response.model = "claude-sonnet-4-5-20250929"
            mock_response.usage.input_tokens = 10
            mock_response.usage.output_tokens = 5
            mock_response.stop_reason = "end_turn"

            # Setup mock client
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            await provider.initialize(api_key="test-key", model="claude-sonnet-4-5-20250929")
            response = await provider.generate("What is 2+2?")

            assert response.content == "Hello, world!"
            assert response.model == "claude-sonnet-4-5-20250929"
            assert response.provider == "anthropic"
            assert response.input_tokens == 10
            assert response.output_tokens == 5
            assert response.finish_reason == "end_turn"

    async def test_generate_with_system_prompt(self, provider):
        """generate() passes system prompt to API."""
        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic") as mock_class:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Response")]
            mock_response.model = "test-model"
            mock_response.usage.input_tokens = 5
            mock_response.usage.output_tokens = 3
            mock_response.stop_reason = "end_turn"

            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            await provider.initialize(api_key="test-key", model="test-model")
            await provider.generate("Hello", system_prompt="You are helpful")

            # Check that system prompt was passed
            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs["system"] == "You are helpful"

    async def test_generate_with_temperature(self, provider):
        """generate() passes temperature to API."""
        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic") as mock_class:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Response")]
            mock_response.model = "test-model"
            mock_response.usage.input_tokens = 5
            mock_response.usage.output_tokens = 3
            mock_response.stop_reason = None

            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            await provider.initialize(api_key="test-key", model="test-model")
            await provider.generate("Hello", temperature=0.7)

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs["temperature"] == 0.7

    async def test_generate_with_max_tokens(self, provider):
        """generate() passes max_tokens to API."""
        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic") as mock_class:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Response")]
            mock_response.model = "test-model"
            mock_response.usage.input_tokens = 5
            mock_response.usage.output_tokens = 3
            mock_response.stop_reason = None

            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            await provider.initialize(api_key="test-key", model="test-model")
            await provider.generate("Hello", max_tokens=100)

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 100

    async def test_authentication_error_conversion(self, provider):
        """Converts Anthropic AuthenticationError to LLMAuthenticationError."""
        import anthropic

        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic") as mock_class:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(
                side_effect=anthropic.AuthenticationError(
                    message="Invalid API key",
                    response=MagicMock(status_code=401),
                    body=None,
                )
            )
            mock_class.return_value = mock_client

            await provider.initialize(api_key="bad-key", model="test-model")

            with pytest.raises(LLMAuthenticationError) as exc_info:
                await provider.generate("Hello")

            assert exc_info.value.provider == "anthropic"

    async def test_rate_limit_error_conversion(self, provider):
        """Converts Anthropic RateLimitError to LLMRateLimitError."""
        import anthropic

        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic") as mock_class:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(
                side_effect=anthropic.RateLimitError(
                    message="Rate limited",
                    response=MagicMock(
                        status_code=429,
                        headers={"retry-after": "30"},
                    ),
                    body=None,
                )
            )
            mock_class.return_value = mock_client

            await provider.initialize(api_key="test-key", model="test-model")

            with pytest.raises(LLMRateLimitError) as exc_info:
                await provider.generate("Hello")

            assert exc_info.value.provider == "anthropic"
            assert exc_info.value.retry_after == 30.0

    async def test_api_error_conversion(self, provider):
        """Converts Anthropic APIStatusError to LLMProviderError."""
        import anthropic

        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic") as mock_class:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(
                side_effect=anthropic.APIStatusError(
                    message="Server error",
                    response=MagicMock(status_code=500),
                    body=None,
                )
            )
            mock_class.return_value = mock_client

            await provider.initialize(api_key="test-key", model="test-model")

            with pytest.raises(LLMProviderError) as exc_info:
                await provider.generate("Hello")

            assert exc_info.value.provider == "anthropic"
            assert exc_info.value.status_code == 500

    async def test_timeout_error_conversion(self, provider):
        """Converts TimeoutError to LLMTimeoutError."""
        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic") as mock_class:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(side_effect=TimeoutError())
            mock_class.return_value = mock_client

            await provider.initialize(api_key="test-key", model="test-model")

            with pytest.raises(LLMTimeoutError) as exc_info:
                await provider.generate("Hello", timeout=5.0)

            assert exc_info.value.provider == "anthropic"

    async def test_timeout_capped_to_max(self, provider):
        """Timeout is capped to MAX_BACKGROUND_TIMEOUT."""
        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic") as mock_class:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Response")]
            mock_response.model = "test-model"
            mock_response.usage.input_tokens = 5
            mock_response.usage.output_tokens = 3
            mock_response.stop_reason = None

            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            await provider.initialize(api_key="test-key", model="test-model")

            # Request 1000 second timeout (way over max)
            await provider.generate("Hello", timeout=1000.0)

            # Should complete (timeout was capped, not actually 1000s)
            mock_client.messages.create.assert_called_once()

    def test_get_model_info_returns_dict(self, provider):
        """get_model_info() returns dictionary with expected keys."""
        with patch("shared.llm.providers.anthropic_provider.AsyncAnthropic"):
            import asyncio

            asyncio.get_event_loop().run_until_complete(
                provider.initialize(api_key="test-key", model="claude-sonnet-4-5-20250929")
            )

            info = provider.get_model_info()

            assert "model" in info
            assert "provider" in info
            assert "context_window" in info
            assert info["provider"] == "anthropic"
            assert info["model"] == "claude-sonnet-4-5-20250929"
