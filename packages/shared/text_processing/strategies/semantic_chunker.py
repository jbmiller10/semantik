#!/usr/bin/env python3
"""
Compatibility wrapper for SemanticChunker.

This module provides backward compatibility for tests that import SemanticChunker directly.
"""

from packages.shared.text_processing.chunking_factory import ChunkingFactory


class SemanticChunker:
    """Wrapper class for backward compatibility."""
    
    def __init__(self, embed_model=None, **kwargs):
        """Initialize using the factory."""
        self._chunker = ChunkingFactory.create_chunker({
            "strategy": "semantic",
            "params": {"embed_model": embed_model, **kwargs}
        })
        
    def __getattr__(self, name):
        """Delegate all attributes to the actual chunker."""
        return getattr(self._chunker, name)