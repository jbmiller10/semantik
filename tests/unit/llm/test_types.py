"""Tests for LLM types."""

import pytest

from shared.llm.types import LLMProviderType, LLMQualityTier, LLMResponse


class TestLLMQualityTier:
    """Tests for LLMQualityTier enum."""

    def test_high_tier_value(self):
        """HIGH tier has correct string value."""
        assert LLMQualityTier.HIGH == "high"
        assert LLMQualityTier.HIGH.value == "high"

    def test_low_tier_value(self):
        """LOW tier has correct string value."""
        assert LLMQualityTier.LOW == "low"
        assert LLMQualityTier.LOW.value == "low"

    def test_tier_is_str_enum(self):
        """Tiers can be used as strings via .value."""
        assert LLMQualityTier.HIGH.value == "high"
        assert LLMQualityTier.LOW.value == "low"


class TestLLMProviderType:
    """Tests for LLMProviderType enum."""

    def test_anthropic_value(self):
        """ANTHROPIC has correct string value."""
        assert LLMProviderType.ANTHROPIC == "anthropic"

    def test_openai_value(self):
        """OPENAI has correct string value."""
        assert LLMProviderType.OPENAI == "openai"

    def test_provider_is_str_enum(self):
        """Providers can be used as strings via .value."""
        assert LLMProviderType.ANTHROPIC.value == "anthropic"


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_create_response(self):
        """Can create a response with all fields."""
        response = LLMResponse(
            content="Hello, world!",
            model="claude-sonnet-4-5-20250929",
            provider="anthropic",
            input_tokens=10,
            output_tokens=5,
            finish_reason="end_turn",
        )

        assert response.content == "Hello, world!"
        assert response.model == "claude-sonnet-4-5-20250929"
        assert response.provider == "anthropic"
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert response.finish_reason == "end_turn"

    def test_total_tokens_property(self):
        """total_tokens returns sum of input and output tokens."""
        response = LLMResponse(
            content="test",
            model="gpt-4o",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
        )

        assert response.total_tokens == 150

    def test_total_tokens_zero(self):
        """total_tokens works with zero tokens."""
        response = LLMResponse(
            content="",
            model="test",
            provider="test",
            input_tokens=0,
            output_tokens=0,
        )

        assert response.total_tokens == 0

    def test_optional_finish_reason(self):
        """finish_reason defaults to None."""
        response = LLMResponse(
            content="test",
            model="test",
            provider="test",
            input_tokens=1,
            output_tokens=1,
        )

        assert response.finish_reason is None

    def test_optional_metadata(self):
        """metadata defaults to empty dict."""
        response = LLMResponse(
            content="test",
            model="test",
            provider="test",
            input_tokens=1,
            output_tokens=1,
        )

        assert response.metadata == {}

    def test_metadata_with_values(self):
        """Can set custom metadata."""
        response = LLMResponse(
            content="test",
            model="test",
            provider="test",
            input_tokens=1,
            output_tokens=1,
            metadata={"request_id": "abc123"},
        )

        assert response.metadata == {"request_id": "abc123"}

    def test_response_is_frozen(self):
        """Response is immutable (frozen dataclass)."""
        response = LLMResponse(
            content="test",
            model="test",
            provider="test",
            input_tokens=1,
            output_tokens=1,
        )

        with pytest.raises(AttributeError):
            response.content = "modified"  # type: ignore[misc]
