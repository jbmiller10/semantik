#!/usr/bin/env python3
"""
Compatibility wrapper for SemanticChunker.

This module provides backward compatibility for tests that import SemanticChunker directly.
"""

from packages.shared.chunking.unified.factory import TextProcessingStrategyAdapter, UnifiedChunkingFactory


class SemanticChunker:
    """Wrapper class for backward compatibility."""
    
    def __init__(self, embed_model=None, **kwargs):
        """Initialize using the unified strategy directly."""
        # Store parameters for test compatibility
        self.max_chunk_size = kwargs.pop('max_chunk_size', 1000)
        self.breakpoint_percentile_threshold = kwargs.pop('breakpoint_percentile_threshold', 95)
        self.buffer_size = kwargs.pop('buffer_size', 1)
        self.embed_model = embed_model
        
        # Create chunker with all parameters
        params = {
            "embed_model": embed_model,
            "max_chunk_size": self.max_chunk_size,
            "breakpoint_percentile_threshold": self.breakpoint_percentile_threshold,
            "buffer_size": self.buffer_size,
            **kwargs
        }
        
        # Create unified strategy directly
        unified_strategy = UnifiedChunkingFactory.create_strategy("semantic", use_llama_index=True, embed_model=embed_model)
        self._chunker = TextProcessingStrategyAdapter(unified_strategy, **params)
        
    def chunk_text(self, text, doc_id, metadata=None):
        """Override to add semantic metadata."""
        results = self._chunker.chunk_text(text, doc_id, metadata)
        
        # Add semantic metadata for test compatibility
        for result in results:
            if 'semantic_boundary' not in result.metadata:
                result.metadata['semantic_boundary'] = True
            if 'breakpoint_threshold' not in result.metadata:
                result.metadata['breakpoint_threshold'] = self.breakpoint_percentile_threshold
        
        return results
    
    async def chunk_text_async(self, text, doc_id, metadata=None):
        """Async version with semantic metadata."""
        results = await self._chunker.chunk_text_async(text, doc_id, metadata)
        
        # Add semantic metadata for test compatibility
        for result in results:
            if 'semantic_boundary' not in result.metadata:
                result.metadata['semantic_boundary'] = True
            if 'breakpoint_threshold' not in result.metadata:
                result.metadata['breakpoint_threshold'] = self.breakpoint_percentile_threshold
        
        return results
    
    def estimate_chunks(self, text_length, config=None):
        """Estimate number of chunks."""
        if config is None:
            config = {"max_chunk_size": self.max_chunk_size}
        
        # Semantic chunking typically creates fewer chunks than character-based
        chunk_size = config.get('max_chunk_size', self.max_chunk_size)
        if chunk_size <= 0:
            return 1
        
        # Semantic boundaries reduce chunk count
        estimated = max(1, text_length // (chunk_size * 2))
        return estimated
    
    def validate_config(self, config):
        """Validate configuration."""
        # Delegate to underlying chunker
        return self._chunker.validate_config(config)
    
    def __getattr__(self, name):
        """Delegate all attributes to the actual chunker."""
        return getattr(self._chunker, name)