#!/usr/bin/env python3
"""
Recursive chunking strategy wrapper for domain interface.

This module provides a wrapper around the unified recursive chunking strategy
to make it compatible with domain tests.
"""

from packages.shared.chunking.unified.factory import (
    DomainStrategyAdapter,
    UnifiedChunkingFactory,
)


class RecursiveChunkingStrategy(DomainStrategyAdapter):
    """
    Recursive chunking strategy that splits on multiple separators.
    
    This wrapper provides compatibility with domain tests while delegating
    actual chunking to the unified strategy.
    """
    
    def __init__(self) -> None:
        """Initialize the recursive chunking strategy."""
        unified_strategy = UnifiedChunkingFactory.create_strategy("recursive", use_llama_index=False)
        super().__init__(unified_strategy)