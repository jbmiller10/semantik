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
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Use smaller defaults for the unified implementation
        params = {
            "max_tokens": chunk_size // 4,  # Approximate tokens from characters
            "min_tokens": min(50, chunk_size // 8),
            "overlap_tokens": chunk_overlap // 4,
            **kwargs
        }
        
        self._chunker = ChunkingFactory.create_chunker({
            "strategy": "recursive",
            "params": params
        })
        
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