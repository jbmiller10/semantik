"""Tests for OpenAI LLM provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from shared.llm.providers.openai_provider import OpenAILLMProvider


class TestOpenAILLMProvider:
    """Tests for OpenAILLMProvider."""

    @pytest.fixture()
    def provider(self):
        """Create a fresh provider instance."""
        return OpenAILLMProvider()

    def test_not_initialized_by_default(self, provider):
        """Provider is not initialized after construction."""
        assert not provider.is_initialized

    async def test_initialize_sets_initialized(self, provider):
        """initialize() sets is_initialized to True."""
        with patch("shared.llm.providers.openai_provider.AsyncOpenAI"):
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
        with patch("shared.llm.providers.openai_provider.AsyncOpenAI") as mock_class:
            mock_client = AsyncMock()
            mock_class.return_value = mock_client

            await provider.initialize(api_key="test-key", model="test-model")
            assert provider.is_initialized

            await provider.cleanup()
            assert not provider.is_initialized

    async def test_context_manager_calls_cleanup(self, provider):
        """Context manager calls cleanup on exit."""
        with patch("shared.llm.providers.openai_provider.AsyncOpenAI") as mock_class:
            mock_client = AsyncMock()
            mock_class.return_value = mock_client

            await provider.initialize(api_key="test-key", model="test-model")

            async with provider:
                assert provider.is_initialized

            # After context exit, cleanup should have been called
            mock_client.close.assert_called_once()

    async def test_successful_generate(self, provider):
        """generate() returns LLMResponse on success."""
        with patch("shared.llm.providers.openai_provider.AsyncOpenAI") as mock_class:
            # Create mock response
            mock_choice = MagicMock()
            mock_choice.message.content = "Hello, world!"
            mock_choice.finish_reason = "stop"

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.model = "gpt-4o-mini"
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5

            # Setup mock client
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            await provider.initialize(api_key="test-key", model="gpt-4o-mini")
            response = await provider.generate("What is 2+2?")

            assert response.content == "Hello, world!"
            assert response.model == "gpt-4o-mini"
            assert response.provider == "openai"
            assert response.input_tokens == 10
            assert response.output_tokens == 5
            assert response.finish_reason == "stop"

    async def test_generate_with_system_prompt(self, provider):
        """generate() includes system message when provided."""
        with patch("shared.llm.providers.openai_provider.AsyncOpenAI") as mock_class:
            mock_choice = MagicMock()
            mock_choice.message.content = "Response"
            mock_choice.finish_reason = "stop"

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.model = "test-model"
            mock_response.usage.prompt_tokens = 5
            mock_response.usage.completion_tokens = 3

            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            await provider.initialize(api_key="test-key", model="test-model")
            await provider.generate("Hello", system_prompt="You are helpful")

            # Check that messages include system prompt
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            messages = call_kwargs["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are helpful"
            assert messages[1]["role"] == "user"

    async def test_generate_without_system_prompt(self, provider):
        """generate() excludes system message when not provided."""
        with patch("shared.llm.providers.openai_provider.AsyncOpenAI") as mock_class:
            mock_choice = MagicMock()
            mock_choice.message.content = "Response"
            mock_choice.finish_reason = "stop"

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.model = "test-model"
            mock_response.usage.prompt_tokens = 5
            mock_response.usage.completion_tokens = 3

            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            await provider.initialize(api_key="test-key", model="test-model")
            await provider.generate("Hello")

            # Check that messages only include user message
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            messages = call_kwargs["messages"]
            assert len(messages) == 1
            assert messages[0]["role"] == "user"

    async def test_generate_with_temperature(self, provider):
        """generate() passes temperature to API."""
        with patch("shared.llm.providers.openai_provider.AsyncOpenAI") as mock_class:
            mock_choice = MagicMock()
            mock_choice.message.content = "Response"
            mock_choice.finish_reason = "stop"

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.model = "test-model"
            mock_response.usage.prompt_tokens = 5
            mock_response.usage.completion_tokens = 3

            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            await provider.initialize(api_key="test-key", model="test-model")
            await provider.generate("Hello", temperature=0.7)

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["temperature"] == 0.7

    async def test_generate_with_max_tokens(self, provider):
        """generate() passes max_tokens to API."""
        with patch("shared.llm.providers.openai_provider.AsyncOpenAI") as mock_class:
            mock_choice = MagicMock()
            mock_choice.message.content = "Response"
            mock_choice.finish_reason = "stop"

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.model = "test-model"
            mock_response.usage.prompt_tokens = 5
            mock_response.usage.completion_tokens = 3

            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            await provider.initialize(api_key="test-key", model="test-model")
            await provider.generate("Hello", max_tokens=100)

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 100

    async def test_authentication_error_conversion(self, provider):
        """Converts OpenAI AuthenticationError to LLMAuthenticationError."""
        import openai

        with patch("shared.llm.providers.openai_provider.AsyncOpenAI") as mock_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=openai.AuthenticationError(
                    message="Invalid API key",
                    response=MagicMock(status_code=401),
                    body=None,
                )
            )
            mock_class.return_value = mock_client

            await provider.initialize(api_key="bad-key", model="test-model")

            with pytest.raises(LLMAuthenticationError) as exc_info:
                await provider.generate("Hello")

            assert exc_info.value.provider == "openai"

    async def test_rate_limit_error_conversion(self, provider):
        """Converts OpenAI RateLimitError to LLMRateLimitError."""
        import openai

        with patch("shared.llm.providers.openai_provider.AsyncOpenAI") as mock_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=openai.RateLimitError(
                    message="Rate limited",
                    response=MagicMock(
                        status_code=429,
                        headers={"retry-after": "60"},
                    ),
                    body=None,
                )
            )
            mock_class.return_value = mock_client

            await provider.initialize(api_key="test-key", model="test-model")

            with pytest.raises(LLMRateLimitError) as exc_info:
                await provider.generate("Hello")

            assert exc_info.value.provider == "openai"
            assert exc_info.value.retry_after == 60.0

    async def test_api_error_conversion(self, provider):
        """Converts OpenAI APIStatusError to LLMProviderError."""
        import openai

        with patch("shared.llm.providers.openai_provider.AsyncOpenAI") as mock_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=openai.APIStatusError(
                    message="Server error",
                    response=MagicMock(status_code=500),
                    body=None,
                )
            )
            mock_class.return_value = mock_client

            await provider.initialize(api_key="test-key", model="test-model")

            with pytest.raises(LLMProviderError) as exc_info:
                await provider.generate("Hello")

            assert exc_info.value.provider == "openai"
            assert exc_info.value.status_code == 500

    async def test_timeout_error_conversion(self, provider):
        """Converts TimeoutError to LLMTimeoutError."""
        with patch("shared.llm.providers.openai_provider.AsyncOpenAI") as mock_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=TimeoutError())
            mock_class.return_value = mock_client

            await provider.initialize(api_key="test-key", model="test-model")

            with pytest.raises(LLMTimeoutError) as exc_info:
                await provider.generate("Hello", timeout=5.0)

            assert exc_info.value.provider == "openai"

    async def test_empty_response_handling(self, provider):
        """Handles response with no content gracefully."""
        with patch("shared.llm.providers.openai_provider.AsyncOpenAI") as mock_class:
            mock_choice = MagicMock()
            mock_choice.message.content = None
            mock_choice.finish_reason = "stop"

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.model = "test-model"
            mock_response.usage.prompt_tokens = 5
            mock_response.usage.completion_tokens = 0

            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_client

            await provider.initialize(api_key="test-key", model="test-model")
            response = await provider.generate("Hello")

            assert response.content == ""
            assert response.output_tokens == 0

    def test_get_model_info_returns_dict(self, provider):
        """get_model_info() returns dictionary with expected keys."""
        with patch("shared.llm.providers.openai_provider.AsyncOpenAI"):
            import asyncio

            asyncio.get_event_loop().run_until_complete(provider.initialize(api_key="test-key", model="gpt-4o-mini"))

            info = provider.get_model_info()

            assert "model" in info
            assert "provider" in info
            assert "context_window" in info
            assert info["provider"] == "openai"
            assert info["model"] == "gpt-4o-mini"
