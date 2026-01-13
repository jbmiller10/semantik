"""OpenAI GPT LLM provider implementation."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import openai
from openai import AsyncOpenAI

from shared.llm.base import BaseLLMService
from shared.llm.exceptions import LLMAuthenticationError, LLMProviderError, LLMRateLimitError, LLMTimeoutError
from shared.llm.model_registry import get_model_by_id
from shared.llm.types import LLMResponse

logger = logging.getLogger(__name__)


class OpenAILLMProvider(BaseLLMService):
    """OpenAI GPT provider implementation.

    Uses the OpenAI SDK's AsyncOpenAI client for async operations.
    Converts OpenAI-specific exceptions to our exception types.

    Example:
        provider = OpenAILLMProvider()
        await provider.initialize(api_key="sk-...", model="gpt-4o-mini")

        async with provider:
            response = await provider.generate(
                prompt="What is the capital of France?",
                system_prompt="You are a helpful assistant.",
            )
            print(response.content)  # "Paris"
            print(response.total_tokens)  # Token usage
    """

    PROVIDER_NAME = "openai"
    DEFAULT_MAX_TOKENS = 4096
    DEFAULT_TIMEOUT = 60.0

    def __init__(self) -> None:
        """Initialize the provider (not ready until initialize() is called)."""
        self._client: AsyncOpenAI | None = None
        self._model: str | None = None
        self._initialized = False

    async def initialize(self, api_key: str, model: str, **kwargs: Any) -> None:
        """Initialize with API key and model.

        Args:
            api_key: OpenAI API key
            model: Model identifier (e.g., "gpt-4o-mini")
            **kwargs: Additional options passed to AsyncOpenAI client

        Raises:
            ValueError: If api_key or model is empty
        """
        if not api_key:
            raise ValueError("API key is required")
        if not model:
            raise ValueError("Model is required")

        self._client = AsyncOpenAI(api_key=api_key, **kwargs)
        self._model = model
        self._initialized = True
        logger.debug(f"Initialized OpenAI provider with model {model}")

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
        """Generate a completion using GPT.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Additional parameters passed to chat.completions.create()

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

        # Build messages list
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build parameters
        params: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }

        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        if temperature is not None:
            params["temperature"] = temperature

        # Merge any extra kwargs
        params.update(kwargs)

        try:
            async with asyncio.timeout(effective_timeout):
                response = await self._client.chat.completions.create(**params)
        except TimeoutError as e:
            raise LLMTimeoutError(self.PROVIDER_NAME, effective_timeout) from e
        except openai.AuthenticationError as e:
            raise LLMAuthenticationError(self.PROVIDER_NAME, str(e)) from e
        except openai.RateLimitError as e:
            # Try to extract retry-after header if available
            retry_after = None
            if hasattr(e, "response") and e.response is not None:
                retry_after_str = e.response.headers.get("retry-after")
                if retry_after_str:
                    with contextlib.suppress(ValueError):
                        retry_after = float(retry_after_str)
            raise LLMRateLimitError(self.PROVIDER_NAME, retry_after) from e
        except openai.APIStatusError as e:
            raise LLMProviderError(self.PROVIDER_NAME, str(e), e.status_code) from e
        except openai.APIError as e:
            raise LLMProviderError(self.PROVIDER_NAME, str(e)) from e

        # Extract content and usage from response
        content = ""
        if response.choices and response.choices[0].message.content:
            content = response.choices[0].message.content

        finish_reason = None
        if response.choices and response.choices[0].finish_reason:
            finish_reason = response.choices[0].finish_reason

        # Token usage
        input_tokens = 0
        output_tokens = 0
        if response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

        return LLMResponse(
            content=content,
            model=response.model,
            provider=self.PROVIDER_NAME,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
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
        context_window = 128000  # Default for GPT-4 models
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
        logger.debug("OpenAI provider cleaned up")

    @property
    def is_initialized(self) -> bool:
        """Check if the provider is ready for requests."""
        return self._initialized and self._client is not None

    @staticmethod
    async def list_models(api_key: str) -> list[dict[str, Any]]:
        """List available chat models from the OpenAI API.

        Args:
            api_key: OpenAI API key

        Returns:
            List of model dictionaries with id, name, display_name, provider,
            tier_recommendation, context_window, and description.
            Only includes chat-capable models (gpt-* models).

        Raises:
            LLMAuthenticationError: If API key is invalid
            LLMProviderError: For other API errors
        """
        client = AsyncOpenAI(api_key=api_key)
        try:
            response = await client.models.list()
            models = []
            for model in response.data:
                model_id = model.id
                # Only include chat-capable models
                if not _is_chat_model(model_id):
                    continue

                name = _format_openai_model_name(model_id)
                display_name = f"OpenAI - {name}"
                tier = _get_openai_tier(model_id)
                context_window = _get_openai_context_window(model_id)

                models.append(
                    {
                        "id": model_id,
                        "name": name,
                        "display_name": display_name,
                        "provider": "openai",
                        "tier_recommendation": tier,
                        "context_window": context_window,
                        "description": f"OpenAI model: {model_id}",
                        "is_curated": False,
                    }
                )

            # Sort by name for consistent ordering
            models.sort(key=lambda m: str(m["name"]))
            return models
        except openai.AuthenticationError as e:
            raise LLMAuthenticationError("openai", str(e)) from e
        except openai.APIError as e:
            raise LLMProviderError("openai", str(e)) from e
        finally:
            await client.close()


def _is_chat_model(model_id: str) -> bool:
    """Check if a model is a chat-capable model."""
    lower_id = model_id.lower()
    # Include GPT models that support chat completions
    chat_prefixes = ["gpt-4", "gpt-3.5", "chatgpt", "o1", "o3"]
    # Exclude non-chat models
    exclude_suffixes = ["-instruct", "-base", "-vision-preview"]

    if any(prefix in lower_id for prefix in chat_prefixes) and not any(
        suffix in lower_id for suffix in exclude_suffixes
    ):
        return True
    return False


def _format_openai_model_name(model_id: str) -> str:
    """Format OpenAI model ID to a friendly name."""
    lower_id = model_id.lower()

    if "gpt-4o-mini" in lower_id:
        return "GPT-4o Mini"
    if "gpt-4o" in lower_id:
        return "GPT-4o"
    if "gpt-4-turbo" in lower_id:
        return "GPT-4 Turbo"
    if "gpt-4" in lower_id:
        return "GPT-4"
    if "gpt-3.5-turbo" in lower_id:
        return "GPT-3.5 Turbo"
    if "o1-mini" in lower_id:
        return "o1 Mini"
    if "o1-preview" in lower_id:
        return "o1 Preview"
    if "o1" in lower_id:
        return "o1"
    if "o3-mini" in lower_id:
        return "o3 Mini"
    if "o3" in lower_id:
        return "o3"
    return model_id


def _get_openai_tier(model_id: str) -> str:
    """Determine tier recommendation for OpenAI model."""
    lower_id = model_id.lower()
    # High tier: larger, more capable models
    high_tier_patterns = ["gpt-4o", "gpt-4-turbo", "o1", "o3"]
    # Low tier (fast/cheap): mini models, 3.5
    low_tier_patterns = ["mini", "gpt-3.5"]

    if any(p in lower_id for p in low_tier_patterns):
        return "low"
    if any(p in lower_id for p in high_tier_patterns):
        return "high"
    return "high"  # Default to high for unknown models


def _get_openai_context_window(model_id: str) -> int:
    """Get context window size for OpenAI model."""
    lower_id = model_id.lower()

    if "gpt-4o" in lower_id or "gpt-4-turbo" in lower_id:
        return 128000
    if "gpt-4" in lower_id:
        return 8192
    if "gpt-3.5-turbo" in lower_id:
        return 16385
    if "o1" in lower_id or "o3" in lower_id:
        return 128000
    return 128000  # Default for newer models


__all__ = ["OpenAILLMProvider"]
