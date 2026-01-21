"""Type definitions for the LLM module."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LLMQualityTier(str, Enum):
    """Quality tier for LLM model selection.

    Features explicitly specify which tier to use when calling LLM:
    - HIGH: Best quality, higher cost (summaries, entity extraction)
    - LOW: Good quality, lower cost (HyDE, keywords)
    """

    HIGH = "high"
    LOW = "low"


class LLMProviderType(str, Enum):
    """Supported LLM provider types."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    LOCAL = "local"


@dataclass(frozen=True)
class LLMResponse:
    """Response from an LLM generation request.

    Contains the generated content along with token usage metrics
    reported by the provider.
    """

    content: str
    model: str
    provider: str  # "anthropic" or "openai"
    input_tokens: int
    output_tokens: int
    finish_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens


__all__ = ["LLMQualityTier", "LLMProviderType", "LLMResponse"]
