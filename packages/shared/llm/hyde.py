"""HyDE (Hypothetical Document Embeddings) query expansion.

Generates a hypothetical document that would answer the query,
which is then embedded for improved semantic search matching.

This module provides the core HyDE generation logic. It does NOT
handle provider initialization or cleanup - callers must use
the LLMServiceFactory pattern.

Example:
    ```python
    from shared.llm.factory import LLMServiceFactory
    from shared.llm.hyde import HyDEConfig, generate_hyde_expansion
    from shared.llm.types import LLMQualityTier

    factory = LLMServiceFactory(session)
    provider = await factory.create_provider_for_tier(user_id, LLMQualityTier.LOW)

    async with provider:
        config = HyDEConfig(timeout_seconds=30)
        result, response = await generate_hyde_expansion(provider, "machine learning", config=config)
        print(result.expanded_query)  # Hypothetical document
    ```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.llm.base import BaseLLMService
    from shared.llm.types import LLMResponse

logger = logging.getLogger(__name__)


# Default HyDE prompt - generic and works well for most search scenarios
HYDE_SYSTEM_PROMPT = """You are a helpful assistant that generates hypothetical document passages.
Given a search query, write a short passage (1-2 paragraphs) that would be a perfect match for this query.
The passage should be informative and directly answer or address the query.
Do not include any preamble like "Here is a passage..." - just write the content directly."""

HYDE_USER_PROMPT_TEMPLATE = """Write a hypothetical document passage that would be a perfect search result for this query:

Query: {query}

Passage:"""


@dataclass(frozen=True)
class HyDEResult:
    """Result of HyDE generation.

    This is an internal result type used by the search service.
    The API response uses HyDEInfo (defined in webui schemas).

    Attributes:
        expanded_query: The generated hypothetical document (used for dense embedding)
        original_query: The original query (preserved for sparse/hybrid search)
        success: Whether generation succeeded
        warning: Warning message if generation failed gracefully
    """

    expanded_query: str
    original_query: str
    success: bool = True
    warning: str | None = None


@dataclass(frozen=True)
class HyDEConfig:
    """Configuration for HyDE generation.

    Attributes:
        timeout_seconds: Generation timeout (capped by provider limits)
        max_tokens: Max tokens for generation (default 256 for passages)
        temperature: Sampling temperature (default 0.7 for creativity)
    """

    timeout_seconds: int = 30
    max_tokens: int = 256
    temperature: float = 0.7


async def generate_hyde_expansion(
    provider: BaseLLMService,
    query: str,
    *,
    config: HyDEConfig | None = None,
) -> tuple[HyDEResult, LLMResponse | None]:
    """Generate a HyDE expansion for the given query.

    This function generates a hypothetical document that would be a good
    answer to the query. The generated text is then used for dense embedding
    instead of the original query, improving semantic matching.

    Args:
        provider: Initialized LLM provider (caller manages lifecycle)
        query: Original search query
        config: HyDE configuration (uses defaults if None)

    Returns:
        Tuple of (HyDEResult, LLMResponse or None if failed)

    Note:
        This function does NOT handle provider initialization or cleanup.
        The caller is responsible for using the factory and context manager:

            factory = LLMServiceFactory(session)
            provider = await factory.create_provider_for_tier(user_id, tier)
            async with provider:
                result, response = await generate_hyde_expansion(provider, query)
    """
    cfg = config or HyDEConfig()

    try:
        prompt = HYDE_USER_PROMPT_TEMPLATE.format(query=query)

        response = await provider.generate(
            prompt=prompt,
            system_prompt=HYDE_SYSTEM_PROMPT,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=float(cfg.timeout_seconds),
        )

        expanded = response.content.strip()

        if not expanded:
            logger.warning("HyDE generation returned empty content for query: %s", query[:50])
            return (
                HyDEResult(
                    expanded_query=query,  # Fallback to original
                    original_query=query,
                    success=False,
                    warning="HyDE generation returned empty content",
                ),
                response,
            )

        logger.debug(
            "HyDE expansion generated: query=%s..., expansion=%s..., tokens=%d",
            query[:30],
            expanded[:50],
            response.total_tokens,
        )

        return (
            HyDEResult(
                expanded_query=expanded,
                original_query=query,
                success=True,
            ),
            response,
        )

    except Exception as e:
        # Log but don't fail the search - return original query
        logger.warning("HyDE generation failed for query '%s': %s", query[:50], e)
        return (
            HyDEResult(
                expanded_query=query,  # Fallback to original
                original_query=query,
                success=False,
                warning=f"HyDE generation failed: {type(e).__name__}",
            ),
            None,
        )


__all__ = ["HyDEResult", "HyDEConfig", "generate_hyde_expansion", "HYDE_SYSTEM_PROMPT"]
