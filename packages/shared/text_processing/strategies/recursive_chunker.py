#!/usr/bin/env python3
"""
Compatibility wrapper for RecursiveChunker.

This module provides backward compatibility for tests that import RecursiveChunker directly.
"""

from typing import Any

from shared.chunking.unified.factory import TextProcessingStrategyAdapter, UnifiedChunkingFactory
from shared.text_processing.base_chunker import BaseChunker, ChunkResult


class RecursiveChunker(BaseChunker):
    """Wrapper class for backward compatibility."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs: Any) -> None:
        """Initialize using the factory."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Use smaller defaults for the unified implementation
        # Use more aggressive conversion to ensure multiple chunks
        params = {
            "max_tokens": max(10, chunk_size // 5),  # More aggressive conversion
            "min_tokens": min(5, chunk_size // 10),
            "overlap_tokens": min(2, chunk_overlap // 5),
            **kwargs,
        }

        # Create unified strategy directly
        unified_strategy = UnifiedChunkingFactory.create_strategy("recursive", use_llama_index=True)
        self._chunker = TextProcessingStrategyAdapter(unified_strategy, **params)

        # Initialize parent
        super().__init__(**kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate all attributes to the actual chunker."""
        return getattr(self._chunker, name)

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate configuration."""
        # Check for invalid overlap
        chunk_size = config.get("chunk_size", self.chunk_size)
        chunk_overlap = config.get("chunk_overlap", self.chunk_overlap)

        if chunk_overlap >= chunk_size:
            return False

        # Delegate to underlying chunker for other validation
        return self._chunker.validate_config(config)

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Synchronous chunking method."""
        return self._chunker.chunk_text(text, doc_id, metadata)

    async def chunk_text_async(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Asynchronous chunking method."""
        return await self._chunker.chunk_text_async(text, doc_id, metadata)

    def estimate_chunks(self, text_length: int, config: dict[str, Any]) -> int:
        """Estimate number of chunks for capacity planning."""
        return self._chunker.estimate_chunks(text_length, config)
