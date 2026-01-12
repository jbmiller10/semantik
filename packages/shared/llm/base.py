"""Base abstraction for LLM services."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import LLMResponse


class BaseLLMService(ABC):
    """Abstract base class for LLM services.

    This defines the minimal interface that all LLM services must implement.
    Provider is bound to ONE model at creation time - factory handles model
    selection based on quality tier.

    Note: Streaming deferred - initial features (HyDE, background summarization)
    don't need it. Can add stream_generate() when UI requires real-time output.

    Example:
        provider = AnthropicLLMProvider()
        await provider.initialize(api_key="...", model="claude-sonnet-4-5-20250929")

        async with provider:
            response = await provider.generate(
                prompt="What is the capital of France?",
                system_prompt="You are a helpful assistant.",
            )
            print(response.content)
    """

    # Timeout caps (seconds)
    MAX_INTERACTIVE_TIMEOUT: float = 30.0  # HyDE, real-time requests
    MAX_BACKGROUND_TIMEOUT: float = 120.0  # Celery tasks

    @abstractmethod
    async def initialize(self, api_key: str, model: str, **kwargs: Any) -> None:
        """Initialize the LLM provider with API key and model.

        Args:
            api_key: The API key for the provider
            model: The model identifier (e.g., "claude-sonnet-4-5-20250929")
            **kwargs: Implementation-specific configuration options

        Raises:
            LLMAuthenticationError: If the API key is invalid
            RuntimeError: If initialization fails
        """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> "LLMResponse":
        """Generate a completion for the given prompt.

        MUST check is_initialized and raise RuntimeError if False.

        Args:
            prompt: The user prompt to generate a response for
            system_prompt: Optional system prompt to set context
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds (capped by MAX_*_TIMEOUT)
            **kwargs: Implementation-specific options

        Returns:
            LLMResponse with content and token usage

        Raises:
            RuntimeError: If called before initialization
            LLMAuthenticationError: If API key is invalid
            LLMRateLimitError: If rate limit exceeded
            LLMTimeoutError: If request times out
            LLMProviderError: For other API errors
        """

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary containing at least:
            - model: str (model identifier)
            - provider: str (provider name)
            - context_window: int (max context length)

        Raises:
            RuntimeError: If called before initialization
        """

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources (HTTP clients, connections, etc.).

        This should be called when the service is no longer needed.
        After cleanup, the service must be re-initialized before use.
        """

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the service is initialized and ready to use."""

    async def __aenter__(self) -> "BaseLLMService":
        """Async context manager entry.

        Returns the service instance for use in async with statements.

        Example:
            async with provider:
                response = await provider.generate("Hello")
        """
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> bool:
        """Async context manager exit.

        Ensures cleanup is called when exiting the context, even if an
        exception occurred. This provides automatic resource management.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred

        Returns:
            False to propagate any exception that occurred
        """
        import logging

        try:
            await self.cleanup()
        except Exception as e:
            # Log but don't raise cleanup errors to avoid masking original exception
            logging.getLogger(__name__).error(f"Error during LLM cleanup: {e}")
        return False  # Don't suppress exceptions


__all__ = ["BaseLLMService"]
