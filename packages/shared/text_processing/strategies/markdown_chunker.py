#!/usr/bin/env python3
"""
Compatibility wrapper for MarkdownChunker.

This module provides backward compatibility for tests that import MarkdownChunker directly.
"""

from packages.shared.text_processing.chunking_factory import ChunkingFactory


class MarkdownChunker:
    """Wrapper class for backward compatibility."""
    
    def __init__(self, **kwargs):
        """Initialize using the factory."""
        self._chunker = ChunkingFactory.create_chunker({
            "strategy": "markdown",
            "params": kwargs
        })
        
    def __getattr__(self, name):
        """Delegate all attributes to the actual chunker."""
        return getattr(self._chunker, name)