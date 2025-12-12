#!/usr/bin/env python3
"""
Unified chunking strategies module.

This module provides a single, consolidated implementation of all chunking strategies,
eliminating duplication between domain-based and LlamaIndex-based implementations.
"""

from shared.chunking.unified.base import UnifiedChunkingStrategy, UnifiedChunkResult
from shared.chunking.unified.character_strategy import CharacterChunkingStrategy
from shared.chunking.unified.hierarchical_strategy import HierarchicalChunkingStrategy
from shared.chunking.unified.hybrid_strategy import HybridChunkingStrategy
from shared.chunking.unified.markdown_strategy import MarkdownChunkingStrategy
from shared.chunking.unified.recursive_strategy import RecursiveChunkingStrategy
from shared.chunking.unified.semantic_strategy import SemanticChunkingStrategy

__all__ = [
    "UnifiedChunkingStrategy",
    "UnifiedChunkResult",
    "CharacterChunkingStrategy",
    "RecursiveChunkingStrategy",
    "SemanticChunkingStrategy",
    "HierarchicalChunkingStrategy",
    "MarkdownChunkingStrategy",
    "HybridChunkingStrategy",
]
