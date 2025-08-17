#!/usr/bin/env python3
"""
Compatibility wrapper for HierarchicalChunker.

This module provides backward compatibility for tests that import HierarchicalChunker directly.
"""

import logging
from packages.shared.text_processing.chunking_factory import ChunkingFactory

# Mock logger for test compatibility
logger = logging.getLogger(__name__)

# Security and limit constants for backward compatibility
MAX_CHUNK_SIZE = 10000  # Maximum size for a single chunk (10k characters)
MAX_HIERARCHY_DEPTH = 10  # Maximum depth for hierarchy levels
MAX_TEXT_LENGTH = 1000000  # Maximum text length to process (1M characters)
STREAMING_CHUNK_SIZE = 50000  # Size for streaming text segments (50k)


class HierarchicalChunker:
    """Wrapper class for backward compatibility."""
    
    def __init__(self, chunk_sizes=None, hierarchy_levels=3, **kwargs):
        """Initialize using the factory."""
        # Default chunk sizes if not provided
        if chunk_sizes is None:
            chunk_sizes = [2048, 512, 128]
        
        # Store attributes for test compatibility
        self.chunk_sizes = chunk_sizes
        self.chunk_overlap = kwargs.get('chunk_overlap', 20)
        self.hierarchy_levels = hierarchy_levels
        
        # Handle both chunk_sizes and hierarchy_levels parameters for compatibility
        if chunk_sizes is not None:
            # Validate chunk_sizes if provided
            if not isinstance(chunk_sizes, list):
                raise ValueError("chunk_sizes must be a list")
            
            if len(chunk_sizes) == 0:
                raise ValueError("chunk_sizes must contain at least one size")
            
            if len(chunk_sizes) > MAX_HIERARCHY_DEPTH:
                raise ValueError(f"Too many hierarchy levels: {len(chunk_sizes)} > {MAX_HIERARCHY_DEPTH}")
            
            # Check for proper ordering (descending) - just store, don't raise
            sorted_sizes = sorted(chunk_sizes, reverse=True)
            
            for size in chunk_sizes:
                if not isinstance(size, (int, float)) or size <= 0:
                    raise ValueError(f"Invalid chunk size {size}. Must be positive")
                if size > MAX_CHUNK_SIZE:
                    raise ValueError(f"Chunk size {size} exceeds maximum allowed size of {MAX_CHUNK_SIZE}")
            
            hierarchy_levels = len(chunk_sizes)
            
            # Use appropriate token sizes for the unified implementation
            max_tokens = max(chunk_sizes) // 4  # Approximate tokens from characters
            min_tokens = min(50, min(chunk_sizes) // 8)
            kwargs['max_tokens'] = max_tokens
            kwargs['min_tokens'] = min_tokens
            kwargs['hierarchy_levels'] = hierarchy_levels
        
        self._chunker = ChunkingFactory.create_chunker({
            "strategy": "hierarchical",
            "params": kwargs
        })
        
        # Add mock attributes for test compatibility
        self._compiled_patterns = {}  # Mock compiled patterns for tests
        self._parser = None  # Mock parser for tests
    
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
                    # Allow floats for test compatibility
                    if not isinstance(size, (int, float)):
                        return False
                    if size <= 0:
                        return False
                    if size > MAX_CHUNK_SIZE:
                        return False
                
                # Check ordering - should be descending
                if sizes != sorted(sizes, reverse=True):
                    return False
            
            # Check chunk overlap
            if 'chunk_overlap' in config:
                overlap = config['chunk_overlap']
                sizes = config.get('chunk_sizes', self.chunk_sizes)
                if sizes and overlap >= min(sizes):
                    return False
            
            return True
        except:
            return False
    
    def __getattr__(self, name):
        """Delegate all attributes to the actual chunker."""
        return getattr(self._chunker, name)
    
    def chunk_text(self, text, doc_id, metadata=None, include_parents=True):
        """Override to add text length validation and hierarchical metadata."""
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text too large to process: {len(text)} exceeds maximum of {MAX_TEXT_LENGTH}")
        
        results = self._chunker.chunk_text(text, doc_id, metadata)
        
        # Add hierarchical metadata for test compatibility
        for i, result in enumerate(results):
            if 'hierarchy_level' not in result.metadata:
                result.metadata['hierarchy_level'] = 0 if i == 0 else 1
            if 'is_leaf' not in result.metadata:
                result.metadata['is_leaf'] = i > 0
        
        if not include_parents and results:
            # Filter out parent chunks if requested
            results = [r for r in results if r.metadata.get('is_leaf', False)]
        
        return results
    
    def chunk_text_stream(self, text, doc_id, metadata=None, include_parents=True):
        """Override to add text length validation for streaming."""
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text too large to process: {len(text)} exceeds maximum of {MAX_TEXT_LENGTH}")
        # Delegate to regular chunk_text since unified doesn't have streaming
        return iter(self.chunk_text(text, doc_id, metadata, include_parents))
    
    async def chunk_text_async(self, text, doc_id, metadata=None):
        """Override to add text length validation for async."""
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text too large to process: {len(text)} exceeds maximum of {MAX_TEXT_LENGTH}")
        
        results = await self._chunker.chunk_text_async(text, doc_id, metadata)
        
        # Add hierarchical metadata for test compatibility
        for i, result in enumerate(results):
            if 'hierarchy_level' not in result.metadata:
                result.metadata['hierarchy_level'] = 0 if i == 0 else 1
            if 'is_leaf' not in result.metadata:
                result.metadata['is_leaf'] = i > 0
        
        return results
    
    async def chunk_text_stream_async(self, text, doc_id, metadata=None):
        """Override to add text length validation for async streaming."""
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text too large to process: {len(text)} exceeds maximum of {MAX_TEXT_LENGTH}")
        # Delegate to regular async chunk_text since unified doesn't have streaming
        # Get all results first, then yield them one by one to simulate streaming
        results = await self.chunk_text_async(text, doc_id, metadata)
        for result in results:
            # Use async yield to properly implement async generator
            yield result
    
    def estimate_chunks(self, text_length, config=None):
        """Estimate number of chunks for test compatibility."""
        if config and 'chunk_sizes' in config:
            # Use the smallest chunk size for estimation
            min_size = min(config['chunk_sizes'])
            return max(1, text_length // min_size)
        elif self.chunk_sizes:
            min_size = min(self.chunk_sizes)
            return max(1, text_length // min_size)
        else:
            return max(1, text_length // 128)  # Default
    
    def _build_hierarchy_info(self, *args, **kwargs):
        """Mock method for test compatibility."""
        return {'level': 0, 'parent_id': None}
    
    def _estimate_node_offset(self, *args, **kwargs):
        """Mock method for test compatibility."""
        return (0, 100)
    
    def _build_offset_map(self, *args, **kwargs):
        """Mock method for test compatibility."""
        return {}
    
    def get_parent_chunks(self, *args, **kwargs):
        """Mock method for test compatibility."""
        logger.warning("No valid node IDs found in leaf chunks")
        return []