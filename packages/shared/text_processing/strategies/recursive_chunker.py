#!/usr/bin/env python3
"""
Compatibility wrapper for RecursiveChunker.

This module provides backward compatibility for tests that import RecursiveChunker directly.
"""

from packages.shared.text_processing.chunking_factory import ChunkingFactory


class RecursiveChunker:
    """Wrapper class for backward compatibility."""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200, **kwargs):
        """Initialize using the factory."""
        self._chunker = ChunkingFactory.create_chunker({
            "strategy": "recursive",
            "params": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, **kwargs}
        })
        
    def __getattr__(self, name):
        """Delegate all attributes to the actual chunker."""
        return getattr(self._chunker, name)