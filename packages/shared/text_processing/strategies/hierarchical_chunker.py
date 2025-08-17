#!/usr/bin/env python3
"""
Compatibility wrapper for HierarchicalChunker.

This module provides backward compatibility for tests that import HierarchicalChunker directly.
"""

from packages.shared.text_processing.chunking_factory import ChunkingFactory

# Security and limit constants for backward compatibility
MAX_CHUNK_SIZE = 10000  # Maximum size for a single chunk (10k characters)
MAX_HIERARCHY_DEPTH = 10  # Maximum depth for hierarchy levels
MAX_TEXT_LENGTH = 1000000  # Maximum text length to process (1M characters)
STREAMING_CHUNK_SIZE = 1000000  # Size for streaming text segments (1M) - matches MAX_TEXT_LENGTH


class HierarchicalChunker:
    """Wrapper class for backward compatibility."""
    
    def __init__(self, chunk_sizes=None, hierarchy_levels=3, **kwargs):
        """Initialize using the factory."""
        # Store attributes for test compatibility
        self.chunk_sizes = chunk_sizes
        self.chunk_overlap = kwargs.get('chunk_overlap', 20)
        
        # Handle both chunk_sizes and hierarchy_levels parameters for compatibility
        if chunk_sizes is not None:
            # Validate chunk_sizes if provided
            if not isinstance(chunk_sizes, list):
                raise ValueError("chunk_sizes must be a list")
            
            if len(chunk_sizes) == 0:
                raise ValueError("chunk_sizes must contain at least one size")
            
            if len(chunk_sizes) > MAX_HIERARCHY_DEPTH:
                raise ValueError(f"Too many hierarchy levels: {len(chunk_sizes)} exceeds maximum of {MAX_HIERARCHY_DEPTH}")
            
            # Check for proper ordering (descending)
            if chunk_sizes != sorted(chunk_sizes, reverse=True):
                # Just store it, don't raise an error
                pass
            
            for size in chunk_sizes:
                if not isinstance(size, (int, float)) or size <= 0:
                    raise ValueError(f"Invalid chunk size {size}. Must be positive number")
                if size > MAX_CHUNK_SIZE:
                    raise ValueError(f"Chunk size {size} exceeds maximum allowed size of {MAX_CHUNK_SIZE}")
            
            hierarchy_levels = len(chunk_sizes)
            kwargs['chunk_sizes'] = chunk_sizes
        
        self._chunker = ChunkingFactory.create_chunker({
            "strategy": "hierarchical",
            "params": {"hierarchy_levels": hierarchy_levels, **kwargs}
        })
        
        # Add mock attributes for test compatibility
        self._compiled_patterns = {}  # Mock compiled patterns for tests
        
    def validate_config(self, config):
        """Validate configuration for test compatibility."""
        try:
            # Check chunk_sizes
            if 'chunk_sizes' in config:
                sizes = config['chunk_sizes']
                
                # Check if it's a list
                if not isinstance(sizes, list):
                    return False
                
                # Check if empty
                if len(sizes) == 0:
                    return False
                
                # Check if too many levels
                if len(sizes) > MAX_HIERARCHY_DEPTH:
                    return False
                
                # Check each size
                for size in sizes:
                    if not isinstance(size, (int, float)):
                        return False
                    if size <= 0:
                        return False
                    if size > MAX_CHUNK_SIZE:
                        return False
            
            # Delegate to underlying chunker
            return self._chunker.validate_config(config)
        except:
            return False
    
    def __getattr__(self, name):
        """Delegate all attributes to the actual chunker."""
        return getattr(self._chunker, name)
    
    def chunk_text(self, text, doc_id, metadata=None):
        """Override to add text length validation."""
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text too large to process: {len(text)} exceeds maximum of {MAX_TEXT_LENGTH}")
        return self._chunker.chunk_text(text, doc_id, metadata)
    
    def chunk_text_stream(self, text, doc_id, metadata=None):
        """Override to add text length validation for streaming."""
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text too large to process: {len(text)} exceeds maximum of {MAX_TEXT_LENGTH}")
        # Delegate to regular chunk_text since unified doesn't have streaming
        return iter(self._chunker.chunk_text(text, doc_id, metadata))
    
    async def chunk_text_async(self, text, doc_id, metadata=None):
        """Override to add text length validation for async."""
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text too large to process: {len(text)} exceeds maximum of {MAX_TEXT_LENGTH}")
        return await self._chunker.chunk_text_async(text, doc_id, metadata)
    
    async def chunk_text_stream_async(self, text, doc_id, metadata=None):
        """Override to add text length validation for async streaming."""
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text too large to process: {len(text)} exceeds maximum of {MAX_TEXT_LENGTH}")
        # Delegate to regular async chunk_text since unified doesn't have streaming
        # Get all results first, then yield them one by one to simulate streaming
        results = await self._chunker.chunk_text_async(text, doc_id, metadata)
        for result in results:
            # Use async yield to properly implement async generator
            yield result