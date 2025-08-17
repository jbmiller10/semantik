#!/usr/bin/env python3
"""
Compatibility wrapper for MarkdownChunker.

This module provides backward compatibility for tests that import MarkdownChunker directly.
"""

from packages.shared.chunking.unified.factory import TextProcessingStrategyAdapter, UnifiedChunkingFactory


class MarkdownChunker:
    """Wrapper class for backward compatibility."""

    def __init__(self, **kwargs):
        """Initialize using the unified strategy directly."""
        # Create unified strategy directly
        unified_strategy = UnifiedChunkingFactory.create_strategy("markdown", use_llama_index=True)
        self._chunker = TextProcessingStrategyAdapter(unified_strategy, **kwargs)

    def __getattr__(self, name):
        """Delegate all attributes to the actual chunker."""
        return getattr(self._chunker, name)
