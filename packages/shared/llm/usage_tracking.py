"""Convenience functions for LLM usage tracking.

This module provides a simple interface for recording LLM usage
after generate() calls. It wraps the LLMUsageRepository for
common usage patterns.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from shared.database.repositories.llm_usage_repository import LLMUsageRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.database.models import LLMUsageEvent
    from shared.llm.types import LLMResponse

logger = logging.getLogger(__name__)


async def record_llm_usage(
    session: AsyncSession,
    user_id: int,
    response: LLMResponse,
    *,
    feature: str,
    quality_tier: str,
    operation_id: int | None = None,
    collection_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> LLMUsageEvent:
    """Record LLM usage from a response to the database.

    This is the primary function for tracking LLM usage. Call it
    after every successful generate() call.

    Uses provider-reported token counts from the response.

    Args:
        session: Database session for recording
        user_id: The user who made the request
        response: LLMResponse from provider.generate()
        feature: Feature name that used the LLM (e.g., 'hyde', 'summary', 'extraction')
        quality_tier: Tier used ('high' or 'low')
        operation_id: Optional operation ID for background tasks (NULL for interactive)
        collection_id: Optional collection context
        metadata: Optional additional metadata

    Returns:
        Created LLMUsageEvent instance

    Example:
        ```python
        response = await provider.generate(prompt="...")

        await record_llm_usage(
            session,
            user_id=123,
            response=response,
            feature="hyde",
            quality_tier="low",
        )
        ```
    """
    repo = LLMUsageRepository(session)
    return await repo.record_usage(
        user_id=user_id,
        provider=response.provider,
        model=response.model,
        quality_tier=quality_tier,
        feature=feature,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        operation_id=operation_id,
        collection_id=collection_id,
        request_metadata=metadata,
    )


async def record_usage_simple(
    session: AsyncSession,
    user_id: int,
    provider: str,
    model: str,
    quality_tier: str,
    feature: str,
    input_tokens: int,
    output_tokens: int,
) -> LLMUsageEvent:
    """Record LLM usage with explicit token counts.

    Use this when you have raw token counts instead of an LLMResponse.
    For example, when aggregating multiple requests or handling
    streaming responses where you calculate tokens separately.

    Args:
        session: Database session for recording
        user_id: The user who made the request
        provider: Provider name ('anthropic', 'openai')
        model: Model identifier
        quality_tier: Tier used ('high' or 'low')
        feature: Feature name ('hyde', 'summary', etc.)
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Created LLMUsageEvent instance
    """
    repo = LLMUsageRepository(session)
    return await repo.record_usage(
        user_id=user_id,
        provider=provider,
        model=model,
        quality_tier=quality_tier,
        feature=feature,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


__all__ = ["record_llm_usage", "record_usage_simple"]
