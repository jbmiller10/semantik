"""LLM provider integration for Semantik.

This module provides a unified interface for interacting with LLM providers
(Anthropic Claude and OpenAI GPT). It follows the same patterns as the
embedding module.

Basic Usage:
    from shared.llm import AnthropicLLMProvider, LLMResponse

    provider = AnthropicLLMProvider()
    await provider.initialize(api_key="sk-ant-...", model="claude-sonnet-4-5-20250929")

    async with provider:
        response = await provider.generate(
            prompt="What is the capital of France?",
            system_prompt="You are a helpful assistant.",
        )
        print(response.content)
        print(f"Tokens used: {response.total_tokens}")

Model Registry:
    from shared.llm import get_all_models, get_default_model

    # Get all curated models
    models = get_all_models()

    # Get default model for a tier
    model_id = get_default_model("anthropic", "low")  # claude-sonnet-4-5-20250929

Exception Handling:
    from shared.llm import (
        LLMNotConfiguredError,
        LLMAuthenticationError,
        LLMRateLimitError,
        LLMProviderError,
    )

    try:
        response = await provider.generate(prompt)
    except LLMRateLimitError as e:
        # Retryable - wait and retry
        if e.retry_after:
            await asyncio.sleep(e.retry_after)
    except LLMAuthenticationError:
        # Invalid API key
        pass
"""

from .base import BaseLLMService
from .exceptions import (
    LLMAuthenticationError,
    LLMContextLengthError,
    LLMError,
    LLMNotConfiguredError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from .model_registry import (
    ModelInfo,
    get_all_models,
    get_default_model,
    get_model_by_id,
    load_model_registry,
)
from .providers import AnthropicLLMProvider, OpenAILLMProvider
from .types import LLMProviderType, LLMQualityTier, LLMResponse

__all__ = [
    # Base class
    "BaseLLMService",
    # Types
    "LLMQualityTier",
    "LLMProviderType",
    "LLMResponse",
    # Exceptions
    "LLMError",
    "LLMNotConfiguredError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMProviderError",
    "LLMTimeoutError",
    "LLMContextLengthError",
    # Model registry
    "ModelInfo",
    "load_model_registry",
    "get_default_model",
    "get_all_models",
    "get_model_by_id",
    # Providers
    "AnthropicLLMProvider",
    "OpenAILLMProvider",
]
