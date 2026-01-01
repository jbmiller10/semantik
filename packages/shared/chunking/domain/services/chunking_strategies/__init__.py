#!/usr/bin/env python3
"""
Chunking strategies for different text processing approaches.

This module now uses the unified chunking strategies to eliminate duplication.
Each strategy implements a specific algorithm for breaking text into chunks.
"""

import logging
from threading import RLock
from typing import cast

from shared.chunking.domain.services.chunking_strategies.base import ChunkingStrategy
from shared.chunking.domain.services.chunking_strategies.character import CharacterChunkingStrategy
from shared.chunking.domain.services.chunking_strategies.hierarchical import HierarchicalChunkingStrategy
from shared.chunking.domain.services.chunking_strategies.hybrid import HybridChunkingStrategy
from shared.chunking.domain.services.chunking_strategies.markdown import MarkdownChunkingStrategy
from shared.chunking.domain.services.chunking_strategies.recursive import RecursiveChunkingStrategy
from shared.chunking.domain.services.chunking_strategies.semantic import SemanticChunkingStrategy

__all__ = [
    "ChunkingStrategy",
    "CharacterChunkingStrategy",
    "RecursiveChunkingStrategy",
    "SemanticChunkingStrategy",
    "MarkdownChunkingStrategy",
    "HierarchicalChunkingStrategy",
    "HybridChunkingStrategy",
]

logger = logging.getLogger(__name__)

_STRATEGY_REGISTRY_LOCK = RLock()

# Strategy registry for easy lookup
STRATEGY_REGISTRY: dict[str, type[ChunkingStrategy]] = cast(
    dict[str, type[ChunkingStrategy]],
    {
        "character": CharacterChunkingStrategy,
        "recursive": RecursiveChunkingStrategy,
        "semantic": SemanticChunkingStrategy,
        "markdown": MarkdownChunkingStrategy,
        "hierarchical": HierarchicalChunkingStrategy,
        "hybrid": HybridChunkingStrategy,
    },
)


def register_strategy(name: str, strategy_class: type[ChunkingStrategy]) -> bool:
    """Register a chunking strategy class by name.

    Returns True if registered, False if already present.
    """
    with _STRATEGY_REGISTRY_LOCK:
        if name in STRATEGY_REGISTRY:
            logger.warning("Chunking strategy '%s' already registered, skipping", name)
            return False
        STRATEGY_REGISTRY[name] = strategy_class
        return True


def unregister_strategy(name: str) -> type[ChunkingStrategy] | None:
    """Remove a chunking strategy by name."""
    with _STRATEGY_REGISTRY_LOCK:
        return STRATEGY_REGISTRY.pop(name, None)


def _snapshot_strategy_registry() -> dict[str, type[ChunkingStrategy]]:
    """Return a snapshot of the strategy registry (testing only)."""
    with _STRATEGY_REGISTRY_LOCK:
        return dict(STRATEGY_REGISTRY)


def _restore_strategy_registry(snapshot: dict[str, type[ChunkingStrategy]]) -> None:
    """Restore the strategy registry from a snapshot (testing only)."""
    with _STRATEGY_REGISTRY_LOCK:
        STRATEGY_REGISTRY.clear()
        STRATEGY_REGISTRY.update(snapshot)


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
    with _STRATEGY_REGISTRY_LOCK:
        strategy_class = STRATEGY_REGISTRY.get(name)
        available = list(STRATEGY_REGISTRY.keys())
    if not strategy_class:
        raise ValueError(f"Unknown chunking strategy: {name}. Available strategies: {available}")
    return strategy_class()
