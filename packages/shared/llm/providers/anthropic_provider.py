"""Anthropic Claude LLM provider implementation."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from shared.llm.base import BaseLLMService
from shared.llm.exceptions import LLMAuthenticationError, LLMProviderError, LLMRateLimitError, LLMTimeoutError
from shared.llm.model_registry import get_model_by_id
from shared.llm.types import LLMResponse

logger = logging.getLogger(__name__)


class AnthropicLLMProvider(BaseLLMService):
    """Anthropic Claude provider implementation.

    Uses the Anthropic SDK's AsyncAnthropic client for async operations.
    Converts Anthropic-specific exceptions to our exception types.

    Example:
        provider = AnthropicLLMProvider()
        await provider.initialize(api_key="sk-ant-...", model="claude-sonnet-4-5-20250929")

        async with provider:
            response = await provider.generate(
                prompt="What is the capital of France?",
                system_prompt="You are a helpful assistant.",
            )
            print(response.content)  # "Paris"
            print(response.total_tokens)  # Token usage
    """

    PROVIDER_NAME = "anthropic"
    DEFAULT_MAX_TOKENS = 4096
    DEFAULT_TIMEOUT = 60.0

    def __init__(self) -> None:
        """Initialize the provider (not ready until initialize() is called)."""
        self._client: AsyncAnthropic | None = None
        self._model: str | None = None
        self._initialized = False

    async def initialize(self, api_key: str, model: str, **kwargs: Any) -> None:
        """Initialize with API key and model.

        Args:
            api_key: Anthropic API key
            model: Model identifier (e.g., "claude-sonnet-4-5-20250929")
            **kwargs: Additional options passed to AsyncAnthropic client

        Raises:
            ValueError: If api_key or model is empty
        """
        if not api_key:
            raise ValueError("API key is required")
        if not model:
            raise ValueError("Model is required")

        self._client = AsyncAnthropic(api_key=api_key, **kwargs)
        self._model = model
        self._initialized = True
        logger.debug(f"Initialized Anthropic provider with model {model}")

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion using Claude.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 to 1.0 for Anthropic)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Additional parameters passed to messages.create()

        Returns:
            LLMResponse with content and token usage

        Raises:
            RuntimeError: If not initialized
            LLMAuthenticationError: If API key is invalid
            LLMRateLimitError: If rate limit exceeded
            LLMTimeoutError: If request times out
            LLMProviderError: For other API errors
        """
        if not self._initialized or self._client is None or self._model is None:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        # Cap timeout to prevent runaway requests
        effective_timeout = min(
            timeout if timeout is not None else self.DEFAULT_TIMEOUT,
            self.MAX_BACKGROUND_TIMEOUT,
        )

        # Build message parameters
        params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens or self.DEFAULT_MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            params["system"] = system_prompt

        if temperature is not None:
            params["temperature"] = temperature

        # Merge any extra kwargs
        params.update(kwargs)

        try:
            async with asyncio.timeout(effective_timeout):
                response = await self._client.messages.create(**params)
        except TimeoutError as e:
            raise LLMTimeoutError(self.PROVIDER_NAME, effective_timeout) from e
        except anthropic.AuthenticationError as e:
            raise LLMAuthenticationError(self.PROVIDER_NAME, str(e)) from e
        except anthropic.RateLimitError as e:
            # Try to extract retry-after header if available
            retry_after = None
            if hasattr(e, "response") and e.response is not None:
                retry_after_str = e.response.headers.get("retry-after")
                if retry_after_str:
                    with contextlib.suppress(ValueError):
                        retry_after = float(retry_after_str)
            raise LLMRateLimitError(self.PROVIDER_NAME, retry_after) from e
        except anthropic.APIStatusError as e:
            raise LLMProviderError(self.PROVIDER_NAME, str(e), e.status_code) from e
        except anthropic.APIError as e:
            raise LLMProviderError(self.PROVIDER_NAME, str(e)) from e

        # Extract content from response
        content = ""
        if response.content:
            # Anthropic returns a list of content blocks
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

        return LLMResponse(
            content=content,
            model=response.model,
            provider=self.PROVIDER_NAME,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            finish_reason=response.stop_reason,
        )

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary with model, provider, and context_window

        Raises:
            RuntimeError: If not initialized
        """
        if not self._initialized or self._model is None:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        # Try to get context window from registry
        context_window = 200000  # Default for Claude models
        model_info = get_model_by_id(self._model)
        if model_info:
            context_window = model_info.context_window

        return {
            "model": self._model,
            "provider": self.PROVIDER_NAME,
            "context_window": context_window,
        }

    async def cleanup(self) -> None:
        """Clean up the HTTP client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
        self._initialized = False
        logger.debug("Anthropic provider cleaned up")

    @property
    def is_initialized(self) -> bool:
        """Check if the provider is ready for requests."""
        return self._initialized and self._client is not None

    @staticmethod
    async def list_models(api_key: str) -> list[dict[str, Any]]:
        """List available models from the Anthropic API.

        Args:
            api_key: Anthropic API key

        Returns:
            List of model dictionaries with id, name, display_name, provider,
            tier_recommendation, context_window, and description.

        Raises:
            LLMAuthenticationError: If API key is invalid
            LLMProviderError: For other API errors
        """
        client = AsyncAnthropic(api_key=api_key)
        try:
            # Anthropic models.list() returns available models
            # Note: This API was added in anthropic SDK 0.35.0+
            response = await client.models.list()  # type: ignore[attr-defined]
            models = []
            for model in response.data:
                model_id = model.id
                # Extract a friendly name from the model ID
                # e.g., "claude-sonnet-4-5-20250929" -> "Sonnet 4.5"
                name = _format_anthropic_model_name(model_id)
                display_name = f"Claude - {name}"

                # Determine tier recommendation based on model name
                tier = "high" if "opus" in model_id.lower() else "low"

                # Context window - default to 200k for Claude models
                context_window = 200000

                models.append(
                    {
                        "id": model_id,
                        "name": name,
                        "display_name": display_name,
                        "provider": "anthropic",
                        "tier_recommendation": tier,
                        "context_window": context_window,
                        "description": f"Claude model: {model_id}",
                        "is_curated": False,
                    }
                )
            return models
        except anthropic.AuthenticationError as e:
            raise LLMAuthenticationError("anthropic", str(e)) from e
        except anthropic.APIError as e:
            raise LLMProviderError("anthropic", str(e)) from e
        except AttributeError as e:
            # Handle case where models API isn't available in SDK version
            raise LLMProviderError(
                "anthropic",
                "Models API not available. Please update the anthropic SDK.",
            ) from e
        finally:
            await client.close()


def _format_anthropic_model_name(model_id: str) -> str:
    """Format Anthropic model ID to a friendly name.

    Examples:
        claude-opus-4-5-20251101 -> Opus 4.5
        claude-sonnet-4-5-20250929 -> Sonnet 4.5
        claude-3-5-sonnet-20241022 -> 3.5 Sonnet
    """
    lower_id = model_id.lower()

    # New naming convention: claude-{variant}-{major}-{minor}-{date}
    if "opus-4" in lower_id:
        return "Opus 4.5"
    if "sonnet-4" in lower_id:
        return "Sonnet 4.5"
    if "haiku-4" in lower_id:
        return "Haiku 4"
    # Old naming convention: claude-{major}-{minor}-{variant}-{date}
    if "3-5-opus" in lower_id:
        return "3.5 Opus"
    if "3-5-sonnet" in lower_id:
        return "3.5 Sonnet"
    if "3-5-haiku" in lower_id:
        return "3.5 Haiku"
    if "opus" in lower_id:
        return "Opus"
    if "sonnet" in lower_id:
        return "Sonnet"
    if "haiku" in lower_id:
        return "Haiku"
    # Fallback: capitalize the model ID
    return model_id.replace("-", " ").title()


__all__ = ["AnthropicLLMProvider"]
