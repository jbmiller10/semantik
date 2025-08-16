#!/usr/bin/env python3
"""
Chunking strategies for different text processing approaches.

This module now uses the unified chunking strategies to eliminate duplication.
Each strategy implements a specific algorithm for breaking text into chunks.
"""

from packages.shared.chunking.domain.services.chunking_strategies.base import (
    ChunkingStrategy,
)
from packages.shared.chunking.unified.factory import (
    DomainStrategyAdapter,
    UnifiedChunkingFactory,
)

# Create adapted unified strategies (all strategies now unified)
CharacterChunkingStrategy = lambda: DomainStrategyAdapter(
    UnifiedChunkingFactory.create_strategy("character", use_llama_index=False)
)
RecursiveChunkingStrategy = lambda: DomainStrategyAdapter(
    UnifiedChunkingFactory.create_strategy("recursive", use_llama_index=False)
)
SemanticChunkingStrategy = lambda: DomainStrategyAdapter(
    UnifiedChunkingFactory.create_strategy("semantic", use_llama_index=False)
)
MarkdownChunkingStrategy = lambda: DomainStrategyAdapter(
    UnifiedChunkingFactory.create_strategy("markdown", use_llama_index=False)
)
HierarchicalChunkingStrategy = lambda: DomainStrategyAdapter(
    UnifiedChunkingFactory.create_strategy("hierarchical", use_llama_index=False)
)
HybridChunkingStrategy = lambda: DomainStrategyAdapter(
    UnifiedChunkingFactory.create_strategy("hybrid", use_llama_index=False)
)

__all__ = [
    "ChunkingStrategy",
    "CharacterChunkingStrategy",
    "RecursiveChunkingStrategy",
    "SemanticChunkingStrategy",
    "MarkdownChunkingStrategy",
    "HierarchicalChunkingStrategy",
    "HybridChunkingStrategy",
]

# Strategy registry for easy lookup
# Using lambda functions to create instances on demand
STRATEGY_REGISTRY: dict[str, type[ChunkingStrategy]] = {
    "character": CharacterChunkingStrategy,      # Unified
    "recursive": RecursiveChunkingStrategy,      # Unified
    "semantic": SemanticChunkingStrategy,        # Unified
    "markdown": MarkdownChunkingStrategy,        # Unified
    "hierarchical": HierarchicalChunkingStrategy,  # Unified
    "hybrid": HybridChunkingStrategy,            # Unified
}


def get_strategy(name: str) -> ChunkingStrategy:
    """
    Get a chunking strategy by name.

    Args:
        name: Name of the strategy

    Returns:
        Instance of the requested strategy

    Raises:
        ValueError: If strategy name is unknown
    """
    strategy_class = STRATEGY_REGISTRY.get(name)
    if not strategy_class:
        raise ValueError(f"Unknown chunking strategy: {name}. Available strategies: {list(STRATEGY_REGISTRY.keys())}")
    return strategy_class()  # type: ignore[call-arg]
