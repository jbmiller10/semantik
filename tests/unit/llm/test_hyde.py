"""Unit tests for HyDE (Hypothetical Document Embeddings) generation.

Tests the core HyDE functionality in packages/shared/llm/hyde.py.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock

import pytest

from shared.llm.hyde import (
    HYDE_SYSTEM_PROMPT,
    HYDE_USER_PROMPT_TEMPLATE,
    HyDEConfig,
    HyDEResult,
    generate_hyde_expansion,
)
from shared.llm.types import LLMResponse


class TestHyDEConfig:
    """Test HyDEConfig dataclass."""

    def test_default_values(self) -> None:
        """Default configuration has sensible values."""
        config = HyDEConfig()
        assert config.timeout_seconds == 10
        assert config.max_tokens == 256
        assert config.temperature == 0.7

    def test_custom_values(self) -> None:
        """Custom values are properly set."""
        config = HyDEConfig(timeout_seconds=20, max_tokens=512, temperature=0.5)
        assert config.timeout_seconds == 20
        assert config.max_tokens == 512
        assert config.temperature == 0.5

    def test_frozen_dataclass(self) -> None:
        """HyDEConfig is immutable (frozen)."""
        config = HyDEConfig()
        with pytest.raises(FrozenInstanceError):
            config.timeout_seconds = 30  # type: ignore[misc]


class TestHyDEResult:
    """Test HyDEResult dataclass."""

    def test_success_result_structure(self) -> None:
        """Successful HyDE result has expected structure."""
        result = HyDEResult(
            expanded_query="This is a hypothetical document about machine learning...",
            original_query="machine learning",
            success=True,
        )
        assert result.success is True
        assert result.warning is None
        assert result.expanded_query != result.original_query

    def test_failure_result_with_warning(self) -> None:
        """Failed HyDE result includes warning message."""
        result = HyDEResult(
            expanded_query="machine learning",  # Falls back to original
            original_query="machine learning",
            success=False,
            warning="HyDE generation timed out",
        )
        assert result.success is False
        assert result.warning == "HyDE generation timed out"
        assert result.expanded_query == result.original_query

    def test_fallback_to_original_query(self) -> None:
        """On failure, expanded_query equals original_query."""
        result = HyDEResult(
            expanded_query="test query",
            original_query="test query",
            success=False,
            warning="LLM error",
        )
        assert result.expanded_query == result.original_query

    def test_frozen_dataclass(self) -> None:
        """HyDEResult is immutable (frozen)."""
        result = HyDEResult(
            expanded_query="test",
            original_query="test",
            success=True,
        )
        with pytest.raises(FrozenInstanceError):
            result.success = False  # type: ignore[misc]


class TestGenerateHyDEExpansion:
    """Test generate_hyde_expansion function."""

    @pytest.fixture()
    def mock_provider(self) -> AsyncMock:
        """Create a mock LLM provider."""
        provider = AsyncMock()
        provider.generate = AsyncMock()
        return provider

    @pytest.fixture()
    def mock_llm_response(self) -> LLMResponse:
        """Create a mock successful LLM response."""
        return LLMResponse(
            content="Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed. It involves algorithms that identify patterns in data and make predictions or decisions.",
            model="test-model",
            provider="mock",
            input_tokens=50,
            output_tokens=100,
        )

    @pytest.mark.asyncio()
    async def test_successful_generation(self, mock_provider: AsyncMock, mock_llm_response: LLMResponse) -> None:
        """Successful HyDE generation returns expanded query."""
        mock_provider.generate.return_value = mock_llm_response

        result, response = await generate_hyde_expansion(mock_provider, "machine learning")

        assert result.success is True
        assert result.expanded_query == mock_llm_response.content
        assert result.original_query == "machine learning"
        assert result.warning is None
        assert response == mock_llm_response

    @pytest.mark.asyncio()
    async def test_returns_both_result_and_response(
        self, mock_provider: AsyncMock, mock_llm_response: LLMResponse
    ) -> None:
        """Returns tuple of (HyDEResult, LLMResponse)."""
        mock_provider.generate.return_value = mock_llm_response

        result, response = await generate_hyde_expansion(mock_provider, "test query")

        assert isinstance(result, HyDEResult)
        assert isinstance(response, LLMResponse)

    @pytest.mark.asyncio()
    async def test_uses_custom_config(self, mock_provider: AsyncMock, mock_llm_response: LLMResponse) -> None:
        """Custom config values are passed to provider."""
        mock_provider.generate.return_value = mock_llm_response
        config = HyDEConfig(timeout_seconds=20, max_tokens=512, temperature=0.5)

        await generate_hyde_expansion(mock_provider, "test query", config=config)

        call_kwargs = mock_provider.generate.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 512
        assert call_kwargs["timeout"] == 20.0

    @pytest.mark.asyncio()
    async def test_uses_default_config_when_none(
        self, mock_provider: AsyncMock, mock_llm_response: LLMResponse
    ) -> None:
        """Default config is used when config=None."""
        mock_provider.generate.return_value = mock_llm_response

        await generate_hyde_expansion(mock_provider, "test query", config=None)

        call_kwargs = mock_provider.generate.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7  # Default
        assert call_kwargs["max_tokens"] == 256  # Default
        assert call_kwargs["timeout"] == 10.0  # Default

    @pytest.mark.asyncio()
    async def test_uses_hyde_system_prompt(self, mock_provider: AsyncMock, mock_llm_response: LLMResponse) -> None:
        """HYDE_SYSTEM_PROMPT is passed to provider."""
        mock_provider.generate.return_value = mock_llm_response

        await generate_hyde_expansion(mock_provider, "test")

        call_kwargs = mock_provider.generate.call_args.kwargs
        assert call_kwargs["system_prompt"] == HYDE_SYSTEM_PROMPT

    @pytest.mark.asyncio()
    async def test_prompt_formatting(self, mock_provider: AsyncMock, mock_llm_response: LLMResponse) -> None:
        """Query is inserted into HYDE_USER_PROMPT_TEMPLATE."""
        mock_provider.generate.return_value = mock_llm_response
        query = "how does semantic search work"

        await generate_hyde_expansion(mock_provider, query)

        call_kwargs = mock_provider.generate.call_args.kwargs
        expected_prompt = HYDE_USER_PROMPT_TEMPLATE.format(query=query)
        assert call_kwargs["prompt"] == expected_prompt

    @pytest.mark.asyncio()
    async def test_empty_response_falls_back(self, mock_provider: AsyncMock) -> None:
        """Empty/whitespace response falls back to original query."""
        mock_provider.generate.return_value = LLMResponse(
            content="   ",  # Empty/whitespace only
            model="test-model",
            provider="mock",
            input_tokens=50,
            output_tokens=0,
        )

        result, response = await generate_hyde_expansion(mock_provider, "test query")

        assert result.success is False
        assert result.expanded_query == "test query"  # Falls back to original
        assert result.warning is not None
        assert "empty" in result.warning.lower()
        assert response is not None  # Response is still returned

    @pytest.mark.asyncio()
    async def test_exception_falls_back_gracefully(self, mock_provider: AsyncMock) -> None:
        """Exception during generation falls back to original query."""
        mock_provider.generate.side_effect = Exception("Network error")

        result, response = await generate_hyde_expansion(mock_provider, "test query")

        assert result.success is False
        assert result.expanded_query == "test query"  # Falls back to original
        assert result.warning is not None
        assert "Exception" in result.warning
        assert response is None  # No response on exception

    @pytest.mark.asyncio()
    async def test_timeout_error_falls_back(self, mock_provider: AsyncMock) -> None:
        """Timeout error falls back to original query."""
        from shared.llm.exceptions import LLMTimeoutError

        mock_provider.generate.side_effect = LLMTimeoutError("mock", timeout=10.0)

        result, response = await generate_hyde_expansion(mock_provider, "test query")

        assert result.success is False
        assert result.expanded_query == "test query"
        assert result.warning is not None
        assert "LLMTimeoutError" in result.warning
        assert response is None

    @pytest.mark.asyncio()
    async def test_provider_error_falls_back(self, mock_provider: AsyncMock) -> None:
        """Provider error falls back to original query."""
        from shared.llm.exceptions import LLMProviderError

        mock_provider.generate.side_effect = LLMProviderError("mock", "GPU out of memory")

        result, response = await generate_hyde_expansion(mock_provider, "test query")

        assert result.success is False
        assert result.expanded_query == "test query"
        assert result.warning is not None
        assert "LLMProviderError" in result.warning
        assert response is None

    @pytest.mark.asyncio()
    async def test_strips_whitespace_from_response(self, mock_provider: AsyncMock) -> None:
        """Response content is stripped of leading/trailing whitespace."""
        mock_provider.generate.return_value = LLMResponse(
            content="\n\n  Generated hypothetical document.  \n\n",
            model="test-model",
            provider="mock",
            input_tokens=50,
            output_tokens=20,
        )

        result, _ = await generate_hyde_expansion(mock_provider, "test query")

        assert result.success is True
        assert result.expanded_query == "Generated hypothetical document."

    @pytest.mark.asyncio()
    async def test_preserves_original_query_always(
        self, mock_provider: AsyncMock, mock_llm_response: LLMResponse
    ) -> None:
        """Original query is always preserved in result."""
        mock_provider.generate.return_value = mock_llm_response
        original = "what is machine learning"

        result, _ = await generate_hyde_expansion(mock_provider, original)

        assert result.original_query == original
        # Even when expanded, original is preserved
        assert result.expanded_query != result.original_query
        assert result.original_query == original
