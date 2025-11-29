#!/usr/bin/env python3
"""
Compatibility wrapper for MarkdownChunker.

This module provides backward compatibility for tests that import MarkdownChunker directly.
"""

from typing import Any

from shared.chunking.unified.factory import TextProcessingStrategyAdapter, UnifiedChunkingFactory
from shared.text_processing.base_chunker import BaseChunker, ChunkResult


class MarkdownChunker(BaseChunker):
    """Wrapper class for backward compatibility."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize using the unified strategy directly."""
        # Create unified strategy directly
        unified_strategy = UnifiedChunkingFactory.create_strategy("markdown", use_llama_index=True)
        self._chunker = TextProcessingStrategyAdapter(unified_strategy, **kwargs)

        # Initialize parent
        super().__init__(**kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate all attributes to the actual chunker."""
        return getattr(self._chunker, name)

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Synchronous chunking method."""
        result: list[ChunkResult] = self._chunker.chunk_text(text, doc_id, metadata)
        return result

    async def chunk_text_async(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Asynchronous chunking method."""
        result: list[ChunkResult] = await self._chunker.chunk_text_async(text, doc_id, metadata)
        return result

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate configuration."""
        result: bool = self._chunker.validate_config(config)
        return result

    def estimate_chunks(self, text_length: int, config: dict[str, Any]) -> int:
        """Estimate number of chunks for capacity planning."""
        result: int = self._chunker.estimate_chunks(text_length, config)
        return result
