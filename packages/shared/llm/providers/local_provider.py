"""Local LLM provider implementation.

HTTP client that calls VecPipe's /llm/generate endpoint for local model inference.
This mirrors the WebUI→VecPipe pattern used for embeddings.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from shared.llm.base import BaseLLMService
from shared.llm.exceptions import LLMProviderError, LLMTimeoutError
from shared.llm.model_registry import get_model_by_id
from shared.llm.types import LLMResponse

logger = logging.getLogger(__name__)


class LocalLLMProvider(BaseLLMService):
    """Local LLM provider - HTTP client to VecPipe.

    Unlike Anthropic/OpenAI providers that call external APIs,
    this calls VecPipe's /llm/generate endpoint for local inference.
    GPU memory governance happens in VecPipe via the GPUMemoryGovernor.

    Example:
        provider = LocalLLMProvider()
        await provider.initialize(api_key="", model="Qwen/Qwen2.5-1.5B-Instruct", quantization="int8")

        async with provider:
            response = await provider.generate(
                prompt="What is the capital of France?",
                system_prompt="You are a helpful assistant.",
            )
            print(response.content)
    """

    PROVIDER_NAME = "local"
    DEFAULT_MAX_TOKENS = 256
    DEFAULT_TIMEOUT = 120.0  # Local models may need more time to load

    def __init__(self, search_api_url: str | None = None) -> None:
        """Initialize the provider.

        Args:
            search_api_url: Optional override for VecPipe URL.
                            Defaults to settings.SEARCH_API_URL.
        """
        self._search_api_url = search_api_url
        self._headers: dict[str, str] | None = None
        self._model: str | None = None
        self._quantization: str = "int8"
        self._client: httpx.AsyncClient | None = None
        self._initialized = False

    async def initialize(self, api_key: str, model: str, **kwargs: Any) -> None:
        """Initialize with model and quantization.

        Args:
            api_key: Ignored for local models (required by interface)
            model: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-1.5B-Instruct")
            **kwargs: Additional options:
                - quantization: "int4", "int8", or "float16" (default: "int8")

        Raises:
            ValueError: If model is empty
            RuntimeError: If internal API key cannot be resolved
        """
        # api_key is required by the BaseLLMService interface, but ignored for local models.
        _ = api_key

        if not model:
            raise ValueError("Model is required")

        # Resolve VecPipe URL and internal API key
        from shared.config import settings
        from shared.config.internal_api_key import ensure_internal_api_key

        base_url = (self._search_api_url or settings.SEARCH_API_URL).rstrip("/")
        internal_key = ensure_internal_api_key(settings)

        self._headers = {"X-Internal-Api-Key": internal_key}
        self._model = model
        self._quantization = kwargs.get("quantization", "int8")

        # Use configurable timeout, capped by MAX_BACKGROUND_TIMEOUT
        default_timeout = min(self.DEFAULT_TIMEOUT, self.MAX_BACKGROUND_TIMEOUT)
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=self._headers,
            timeout=default_timeout,
        )
        self._initialized = True
        logger.debug(
            "Initialized local LLM provider with model %s (quantization=%s)",
            model,
            self._quantization,
        )

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        **_kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion using local LLM via VecPipe.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **_kwargs: Additional parameters (ignored)

        Returns:
            LLMResponse with content and token usage

        Raises:
            RuntimeError: If not initialized
            LLMProviderError: For VecPipe errors (OOM, unavailable, etc.)
            LLMTimeoutError: If request times out
        """
        if not self._initialized or self._client is None or self._model is None:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        effective_temperature = temperature if temperature is not None else 0.7
        effective_max_tokens = max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS
        # Cap timeout per BaseLLMService contract
        effective_timeout = min(timeout or self.DEFAULT_TIMEOUT, self.MAX_BACKGROUND_TIMEOUT)

        # VecPipe API is batch-shaped; provider sends a single prompt as a 1-item batch.
        try:
            response = await self._client.post(
                "/llm/generate",
                json={
                    "model_name": self._model,
                    "quantization": self._quantization,
                    "prompts": [prompt],
                    "system_prompt": system_prompt,
                    "temperature": effective_temperature,
                    "max_tokens": effective_max_tokens,
                },
                timeout=effective_timeout,
            )
            response.raise_for_status()
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(self.PROVIDER_NAME, effective_timeout) from e
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            detail = e.response.text
            if status == 507:
                raise LLMProviderError(
                    self.PROVIDER_NAME, f"Insufficient GPU memory: {detail}", status
                ) from e
            if status == 503:
                raise LLMProviderError(
                    self.PROVIDER_NAME, f"LLM service unavailable: {detail}", status
                ) from e
            raise LLMProviderError(
                self.PROVIDER_NAME, f"VecPipe error ({status}): {detail}", status
            ) from e
        except httpx.RequestError as e:
            raise LLMProviderError(
                self.PROVIDER_NAME, f"Connection error to VecPipe: {e}"
            ) from e

        data = response.json()
        return LLMResponse(
            content=data["contents"][0],
            model=self._model,
            provider=self.PROVIDER_NAME,
            input_tokens=data["prompt_tokens"][0],
            output_tokens=data["completion_tokens"][0],
        )

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary with model, provider, context_window, and quantization

        Raises:
            RuntimeError: If not initialized
        """
        if not self._initialized or self._model is None:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        # Get context window from the shared model registry
        context_window = 0
        model_info = get_model_by_id(self._model, provider="local")
        if model_info:
            context_window = model_info.context_window

        return {
            "model": self._model,
            "provider": self.PROVIDER_NAME,
            "context_window": context_window,
            "quantization": self._quantization,
        }

    async def cleanup(self) -> None:
        """Clean up the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._initialized = False
        logger.debug("Local LLM provider cleaned up")

    @property
    def is_initialized(self) -> bool:
        """Check if the provider is ready for requests."""
        return self._initialized and self._client is not None


__all__ = ["LocalLLMProvider"]
