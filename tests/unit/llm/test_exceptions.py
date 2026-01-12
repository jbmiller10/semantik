"""Tests for LLM exceptions."""

import pytest

from shared.llm.exceptions import (
    LLMAuthenticationError,
    LLMContextLengthError,
    LLMError,
    LLMNotConfiguredError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)


class TestLLMError:
    """Tests for base LLMError."""

    def test_is_exception(self):
        """LLMError is an Exception."""
        assert issubclass(LLMError, Exception)

    def test_can_raise(self):
        """Can raise LLMError."""
        with pytest.raises(LLMError, match="test error"):
            raise LLMError("test error")


class TestLLMNotConfiguredError:
    """Tests for LLMNotConfiguredError."""

    def test_inherits_from_llm_error(self):
        """Inherits from LLMError."""
        assert issubclass(LLMNotConfiguredError, LLMError)

    def test_message_includes_user_id(self):
        """Error message includes user ID."""
        err = LLMNotConfiguredError(user_id=42)
        assert "42" in str(err)
        assert "not configured" in str(err).lower()

    def test_has_user_id_attribute(self):
        """Has user_id attribute."""
        err = LLMNotConfiguredError(user_id=123)
        assert err.user_id == 123


class TestLLMAuthenticationError:
    """Tests for LLMAuthenticationError."""

    def test_inherits_from_llm_error(self):
        """Inherits from LLMError."""
        assert issubclass(LLMAuthenticationError, LLMError)

    def test_default_message(self):
        """Default message mentions provider."""
        err = LLMAuthenticationError(provider="anthropic")
        assert "anthropic" in str(err).lower()
        assert "authentication" in str(err).lower()

    def test_custom_message(self):
        """Can provide custom message."""
        err = LLMAuthenticationError(provider="openai", message="Invalid key format")
        assert str(err) == "Invalid key format"

    def test_has_provider_attribute(self):
        """Has provider attribute."""
        err = LLMAuthenticationError(provider="anthropic")
        assert err.provider == "anthropic"


class TestLLMRateLimitError:
    """Tests for LLMRateLimitError."""

    def test_inherits_from_llm_error(self):
        """Inherits from LLMError."""
        assert issubclass(LLMRateLimitError, LLMError)

    def test_message_mentions_rate_limit(self):
        """Message mentions rate limit."""
        err = LLMRateLimitError(provider="openai")
        assert "rate limit" in str(err).lower()
        assert "openai" in str(err).lower()

    def test_retry_after_default_none(self):
        """retry_after defaults to None."""
        err = LLMRateLimitError(provider="anthropic")
        assert err.retry_after is None

    def test_retry_after_with_value(self):
        """Can set retry_after."""
        err = LLMRateLimitError(provider="anthropic", retry_after=30.0)
        assert err.retry_after == 30.0

    def test_has_provider_attribute(self):
        """Has provider attribute."""
        err = LLMRateLimitError(provider="openai")
        assert err.provider == "openai"


class TestLLMProviderError:
    """Tests for LLMProviderError."""

    def test_inherits_from_llm_error(self):
        """Inherits from LLMError."""
        assert issubclass(LLMProviderError, LLMError)

    def test_message_format(self):
        """Message includes provider and error message."""
        err = LLMProviderError(provider="anthropic", message="Server error")
        assert "anthropic" in str(err).lower()
        assert "server error" in str(err).lower()

    def test_status_code_default_none(self):
        """status_code defaults to None."""
        err = LLMProviderError(provider="openai", message="Error")
        assert err.status_code is None

    def test_status_code_with_value(self):
        """Can set status_code."""
        err = LLMProviderError(provider="openai", message="Error", status_code=500)
        assert err.status_code == 500

    def test_has_provider_attribute(self):
        """Has provider attribute."""
        err = LLMProviderError(provider="anthropic", message="Error")
        assert err.provider == "anthropic"


class TestLLMTimeoutError:
    """Tests for LLMTimeoutError."""

    def test_inherits_from_llm_error(self):
        """Inherits from LLMError."""
        assert issubclass(LLMTimeoutError, LLMError)

    def test_message_includes_timeout(self):
        """Message includes timeout value."""
        err = LLMTimeoutError(provider="anthropic", timeout=30.0)
        assert "30" in str(err)
        assert "timed out" in str(err).lower()

    def test_has_timeout_attribute(self):
        """Has timeout attribute."""
        err = LLMTimeoutError(provider="openai", timeout=60.0)
        assert err.timeout == 60.0

    def test_has_provider_attribute(self):
        """Has provider attribute."""
        err = LLMTimeoutError(provider="anthropic", timeout=10.0)
        assert err.provider == "anthropic"


class TestLLMContextLengthError:
    """Tests for LLMContextLengthError."""

    def test_inherits_from_llm_error(self):
        """Inherits from LLMError."""
        assert issubclass(LLMContextLengthError, LLMError)

    def test_message_includes_limits(self):
        """Message includes max and requested tokens."""
        err = LLMContextLengthError(
            provider="anthropic", max_tokens=200000, requested=250000
        )
        assert "200000" in str(err)
        assert "250000" in str(err)
        assert "anthropic" in str(err).lower()

    def test_has_all_attributes(self):
        """Has provider, max_tokens, and requested attributes."""
        err = LLMContextLengthError(
            provider="openai", max_tokens=128000, requested=150000
        )
        assert err.provider == "openai"
        assert err.max_tokens == 128000
        assert err.requested == 150000


class TestExceptionHierarchy:
    """Tests for exception inheritance hierarchy."""

    def test_all_exceptions_inherit_from_llm_error(self):
        """All custom exceptions inherit from LLMError."""
        exceptions = [
            LLMNotConfiguredError,
            LLMAuthenticationError,
            LLMRateLimitError,
            LLMProviderError,
            LLMTimeoutError,
            LLMContextLengthError,
        ]

        for exc_class in exceptions:
            assert issubclass(exc_class, LLMError), f"{exc_class.__name__} should inherit from LLMError"

    def test_can_catch_with_base_class(self):
        """Can catch all LLM exceptions with LLMError."""
        with pytest.raises(LLMError):
            raise LLMRateLimitError(provider="test")

        with pytest.raises(LLMError):
            raise LLMAuthenticationError(provider="test")

        with pytest.raises(LLMError):
            raise LLMTimeoutError(provider="test", timeout=10.0)
