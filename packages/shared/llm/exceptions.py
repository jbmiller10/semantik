"""LLM-specific exceptions.

Exception hierarchy follows the pattern from shared.database.exceptions.
"""


class LLMError(Exception):
    """Base exception for all LLM errors."""


class LLMNotConfiguredError(LLMError):
    """User has not configured LLM settings."""

    def __init__(self, user_id: int) -> None:
        self.user_id = user_id
        super().__init__(f"LLM not configured for user {user_id}")


class LLMAuthenticationError(LLMError):
    """Invalid or missing API key."""

    def __init__(self, provider: str, message: str | None = None) -> None:
        self.provider = provider
        msg = message or f"Authentication failed for {provider}"
        super().__init__(msg)


class LLMRateLimitError(LLMError):
    """Provider rate limit exceeded (retryable)."""

    def __init__(self, provider: str, retry_after: float | None = None) -> None:
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded for {provider}")


class LLMProviderError(LLMError):
    """General provider error (API error, network issue)."""

    def __init__(self, provider: str, message: str, status_code: int | None = None) -> None:
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"{provider} error: {message}")


class LLMTimeoutError(LLMError):
    """Request timed out."""

    def __init__(self, provider: str, timeout: float) -> None:
        self.provider = provider
        self.timeout = timeout
        super().__init__(f"{provider} request timed out after {timeout}s")


class LLMContextLengthError(LLMError):
    """Input exceeds model context window."""

    def __init__(self, provider: str, max_tokens: int, requested: int) -> None:
        self.provider = provider
        self.max_tokens = max_tokens
        self.requested = requested
        super().__init__(f"Context length {requested} exceeds {provider} limit of {max_tokens}")


__all__ = [
    "LLMError",
    "LLMNotConfiguredError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMProviderError",
    "LLMTimeoutError",
    "LLMContextLengthError",
]
