"""Tests for Local LLM provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from shared.llm.exceptions import LLMProviderError, LLMTimeoutError
from shared.llm.providers.local_provider import LocalLLMProvider


class TestLocalLLMProvider:
    """Tests for LocalLLMProvider."""

    @pytest.fixture()
    def provider(self):
        """Create a fresh provider instance."""
        return LocalLLMProvider(search_api_url="http://test-vecpipe:8000")

    def test_not_initialized_by_default(self, provider):
        """Provider is not initialized after construction."""
        assert not provider.is_initialized

    async def test_initialize_sets_initialized(self, provider):
        """initialize() sets is_initialized to True."""
        with patch("shared.config.internal_api_key.ensure_internal_api_key", return_value="test-key"):
            await provider.initialize(api_key="", model="Qwen/Qwen2.5-1.5B-Instruct")
            assert provider.is_initialized

    async def test_initialize_requires_model(self, provider):
        """initialize() raises ValueError without model."""
        with pytest.raises(ValueError, match="Model"):
            await provider.initialize(api_key="", model="")

    async def test_initialize_ignores_api_key(self, provider):
        """initialize() does not require API key for local models."""
        with patch("shared.config.internal_api_key.ensure_internal_api_key", return_value="test-key"):
            # Empty API key should be fine for local models
            await provider.initialize(api_key="", model="Qwen/Qwen2.5-1.5B-Instruct")
            assert provider.is_initialized

    async def test_initialize_with_quantization(self, provider):
        """initialize() accepts quantization parameter."""
        with patch("shared.config.internal_api_key.ensure_internal_api_key", return_value="test-key"):
            await provider.initialize(api_key="", model="Qwen/Qwen2.5-1.5B-Instruct", quantization="int4")
            assert provider._quantization == "int4"

    async def test_initialize_default_quantization(self, provider):
        """initialize() uses int8 as default quantization."""
        with patch("shared.config.internal_api_key.ensure_internal_api_key", return_value="test-key"):
            await provider.initialize(api_key="", model="Qwen/Qwen2.5-1.5B-Instruct")
            assert provider._quantization == "int8"

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
        with patch("shared.config.internal_api_key.ensure_internal_api_key", return_value="test-key"):
            await provider.initialize(api_key="", model="Qwen/Qwen2.5-1.5B-Instruct")
            assert provider.is_initialized

            await provider.cleanup()
            assert not provider.is_initialized

    async def test_context_manager_calls_cleanup(self, provider):
        """Context manager calls cleanup on exit."""
        with patch("shared.config.internal_api_key.ensure_internal_api_key", return_value="test-key"):
            await provider.initialize(api_key="", model="Qwen/Qwen2.5-1.5B-Instruct")

            async with provider:
                assert provider.is_initialized

            # After context exit, should be cleaned up
            assert not provider.is_initialized

    async def test_successful_generate(self, provider):
        """generate() returns LLMResponse on success."""
        with patch("shared.config.internal_api_key.ensure_internal_api_key", return_value="test-key"):
            await provider.initialize(api_key="", model="Qwen/Qwen2.5-1.5B-Instruct")

            # Mock the HTTP response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "contents": ["Hello, world!"],
                "prompt_tokens": [10],
                "completion_tokens": [5],
            }
            mock_response.raise_for_status = MagicMock()

            with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = mock_response

                response = await provider.generate("What is 2+2?")

                assert response.content == "Hello, world!"
                assert response.model == "Qwen/Qwen2.5-1.5B-Instruct"
                assert response.provider == "local"
                assert response.input_tokens == 10
                assert response.output_tokens == 5

    async def test_generate_sends_correct_payload(self, provider):
        """generate() sends correct payload to VecPipe."""
        with patch("shared.config.internal_api_key.ensure_internal_api_key", return_value="test-key"):
            await provider.initialize(api_key="", model="Qwen/Qwen2.5-1.5B-Instruct", quantization="int4")

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "contents": ["Response"],
                "prompt_tokens": [5],
                "completion_tokens": [3],
            }
            mock_response.raise_for_status = MagicMock()

            with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = mock_response

                await provider.generate(
                    "Hello",
                    system_prompt="You are helpful",
                    temperature=0.5,
                    max_tokens=100,
                )

                # Check the request payload
                call_args = mock_post.call_args
                assert call_args.args[0] == "/llm/generate"
                payload = call_args.kwargs["json"]

                assert payload["model_name"] == "Qwen/Qwen2.5-1.5B-Instruct"
                assert payload["quantization"] == "int4"
                assert payload["prompts"] == ["Hello"]
                assert payload["system_prompt"] == "You are helpful"
                assert payload["temperature"] == 0.5
                assert payload["max_tokens"] == 100

    async def test_timeout_error_conversion(self, provider):
        """Converts httpx.TimeoutException to LLMTimeoutError."""
        with patch("shared.config.internal_api_key.ensure_internal_api_key", return_value="test-key"):
            await provider.initialize(api_key="", model="Qwen/Qwen2.5-1.5B-Instruct")

            with patch.object(
                provider._client, "post", new_callable=AsyncMock, side_effect=httpx.TimeoutException("timeout")
            ):
                with pytest.raises(LLMTimeoutError) as exc_info:
                    await provider.generate("Hello")

                assert exc_info.value.provider == "local"

    async def test_507_error_conversion(self, provider):
        """Converts HTTP 507 (Insufficient Storage) to LLMProviderError."""
        with patch("shared.config.internal_api_key.ensure_internal_api_key", return_value="test-key"):
            await provider.initialize(api_key="", model="Qwen/Qwen2.5-1.5B-Instruct")

            mock_response = MagicMock()
            mock_response.status_code = 507
            mock_response.text = "Insufficient GPU memory"

            with patch.object(
                provider._client,
                "post",
                new_callable=AsyncMock,
                side_effect=httpx.HTTPStatusError("error", request=MagicMock(), response=mock_response),
            ):
                with pytest.raises(LLMProviderError) as exc_info:
                    await provider.generate("Hello")

                assert exc_info.value.provider == "local"
                assert exc_info.value.status_code == 507
                assert "GPU memory" in str(exc_info.value)

    async def test_503_error_conversion(self, provider):
        """Converts HTTP 503 (Service Unavailable) to LLMProviderError."""
        with patch("shared.config.internal_api_key.ensure_internal_api_key", return_value="test-key"):
            await provider.initialize(api_key="", model="Qwen/Qwen2.5-1.5B-Instruct")

            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_response.text = "LLM service unavailable"

            with patch.object(
                provider._client,
                "post",
                new_callable=AsyncMock,
                side_effect=httpx.HTTPStatusError("error", request=MagicMock(), response=mock_response),
            ):
                with pytest.raises(LLMProviderError) as exc_info:
                    await provider.generate("Hello")

                assert exc_info.value.provider == "local"
                assert exc_info.value.status_code == 503
                assert "unavailable" in str(exc_info.value)

    async def test_connection_error_conversion(self, provider):
        """Converts httpx.RequestError to LLMProviderError."""
        with patch("shared.config.internal_api_key.ensure_internal_api_key", return_value="test-key"):
            await provider.initialize(api_key="", model="Qwen/Qwen2.5-1.5B-Instruct")

            with patch.object(
                provider._client,
                "post",
                new_callable=AsyncMock,
                side_effect=httpx.ConnectError("Connection refused"),
            ):
                with pytest.raises(LLMProviderError) as exc_info:
                    await provider.generate("Hello")

                assert exc_info.value.provider == "local"
                assert "Connection error" in str(exc_info.value)

    def test_get_model_info_returns_dict(self, provider):
        """get_model_info() returns dictionary with expected keys."""
        import asyncio

        with patch("shared.config.internal_api_key.ensure_internal_api_key", return_value="test-key"):
            asyncio.get_event_loop().run_until_complete(
                provider.initialize(api_key="", model="Qwen/Qwen2.5-1.5B-Instruct", quantization="int8")
            )

            info = provider.get_model_info()

            assert "model" in info
            assert "provider" in info
            assert "context_window" in info
            assert "quantization" in info
            assert info["provider"] == "local"
            assert info["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
            assert info["quantization"] == "int8"


class TestLocalProviderFactoryIntegration:
    """Tests for LocalLLMProvider integration with factory."""

    def test_creates_local_provider(self):
        """_create_provider_instance creates LocalLLMProvider for 'local'."""
        from shared.llm.factory import _create_provider_instance

        provider = _create_provider_instance("local")
        assert provider.__class__.__name__ == "LocalLLMProvider"

    async def test_factory_creates_local_provider_without_api_key(self):
        """Factory creates local provider without requiring API key."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from shared.llm.factory import LLMServiceFactory
        from shared.llm.types import LLMQualityTier

        mock_session = AsyncMock()
        factory = LLMServiceFactory(mock_session)

        # Mock config with local provider
        mock_config = MagicMock()
        mock_config.id = 1
        mock_config.user_id = 123
        mock_config.low_quality_provider = "local"
        mock_config.low_quality_model = "Qwen/Qwen2.5-1.5B-Instruct"
        mock_config.provider_config = {"local": {"low_quantization": "int4"}}

        with (
            patch.object(factory._config_repo, "get_by_user_id", return_value=mock_config),
            patch(
                "shared.config.internal_api_key.ensure_internal_api_key",
                return_value="test-key",
            ),
        ):
            result = await factory.create_provider_for_tier(
                user_id=123,
                quality_tier=LLMQualityTier.LOW,
            )

            # Should have created LocalLLMProvider
            assert result.__class__.__name__ == "LocalLLMProvider"
            assert result._model == "Qwen/Qwen2.5-1.5B-Instruct"
            assert result._quantization == "int4"

    async def test_has_provider_configured_returns_true_for_local(self):
        """has_provider_configured returns True for local provider without API key."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from shared.llm.factory import LLMServiceFactory
        from shared.llm.types import LLMQualityTier

        mock_session = AsyncMock()
        factory = LLMServiceFactory(mock_session)

        # Mock config with local provider
        mock_config = MagicMock()
        mock_config.id = 1
        mock_config.low_quality_provider = "local"

        with (
            patch.object(factory._config_repo, "get_by_user_id", return_value=mock_config),
            patch.object(factory._config_repo, "has_api_key", return_value=False),
        ):
            result = await factory.has_provider_configured(
                user_id=123,
                quality_tier=LLMQualityTier.LOW,
            )

            # Should return True even without API key for local provider
            assert result is True
