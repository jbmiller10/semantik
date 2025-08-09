#!/usr/bin/env python3
"""
Chunking strategies for different text processing approaches.

Each strategy implements a specific algorithm for breaking text into chunks.
"""

from packages.shared.chunking.domain.services.chunking_strategies.base import (
    ChunkingStrategy,
)
from packages.shared.chunking.domain.services.chunking_strategies.character import (
    CharacterChunkingStrategy,
)
from packages.shared.chunking.domain.services.chunking_strategies.hierarchical import (
    HierarchicalChunkingStrategy,
)
from packages.shared.chunking.domain.services.chunking_strategies.hybrid import (
    HybridChunkingStrategy,
)
from packages.shared.chunking.domain.services.chunking_strategies.markdown import (
    MarkdownChunkingStrategy,
)
from packages.shared.chunking.domain.services.chunking_strategies.recursive import (
    RecursiveChunkingStrategy,
)
from packages.shared.chunking.domain.services.chunking_strategies.semantic import (
    SemanticChunkingStrategy,
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
STRATEGY_REGISTRY = {
    "character": CharacterChunkingStrategy,
    "recursive": RecursiveChunkingStrategy,
    "semantic": SemanticChunkingStrategy,
    "markdown": MarkdownChunkingStrategy,
    "hierarchical": HierarchicalChunkingStrategy,
    "hybrid": HybridChunkingStrategy,
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
        raise ValueError(
            f"Unknown chunking strategy: {name}. "
            f"Available strategies: {list(STRATEGY_REGISTRY.keys())}"
        )
    return strategy_class()
