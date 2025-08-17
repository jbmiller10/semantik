#!/usr/bin/env python3
"""
Compatibility wrapper for RecursiveChunker.

This module provides backward compatibility for tests that import RecursiveChunker directly.
"""

from packages.shared.chunking.unified.factory import TextProcessingStrategyAdapter, UnifiedChunkingFactory


class RecursiveChunker:
    """Wrapper class for backward compatibility."""

    def __init__(self, chunk_size=1000, chunk_overlap=200, **kwargs):
        """Initialize using the factory."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Use smaller defaults for the unified implementation
        # Use more aggressive conversion to ensure multiple chunks
        params = {
            "max_tokens": max(10, chunk_size // 5),  # More aggressive conversion
            "min_tokens": min(5, chunk_size // 10),
            "overlap_tokens": min(2, chunk_overlap // 5),
            **kwargs
        }

        # Create unified strategy directly
        unified_strategy = UnifiedChunkingFactory.create_strategy("recursive", use_llama_index=True)
        self._chunker = TextProcessingStrategyAdapter(unified_strategy, **params)

    def __getattr__(self, name):
        """Delegate all attributes to the actual chunker."""
        return getattr(self._chunker, name)

    def validate_config(self, config):
        """Validate configuration."""
        # Check for invalid overlap
        chunk_size = config.get('chunk_size', self.chunk_size)
        chunk_overlap = config.get('chunk_overlap', self.chunk_overlap)

        if chunk_overlap >= chunk_size:
            return False

        # Delegate to underlying chunker for other validation
        return self._chunker.validate_config(config)
