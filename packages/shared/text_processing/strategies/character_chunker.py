#!/usr/bin/env python3
"""
Character/token-based chunking strategy using LlamaIndex.

This module implements fixed-size token-based text splitting using LlamaIndex's
TokenTextSplitter.
"""

import asyncio
import logging
from typing import Any

from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter

from packages.shared.text_processing.base_chunker import BaseChunker, ChunkResult

logger = logging.getLogger(__name__)


class CharacterChunker(BaseChunker):
    """Character/token-based chunking using LlamaIndex TokenTextSplitter."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs: Any) -> None:
        """Initialize CharacterChunker.

        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)

        # Validate parameters
        if chunk_overlap >= chunk_size:
            logger.warning(
                f"chunk_overlap ({chunk_overlap}) >= chunk_size ({chunk_size}), "
                f"setting overlap to chunk_size/2"
            )
            chunk_overlap = chunk_size // 2

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize LlamaIndex splitter
        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" ",
        )

        logger.info(
            f"Initialized CharacterChunker with chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Synchronous chunking using token-based splitting.

        Args:
            text: The text to chunk
            doc_id: Unique identifier for the document
            metadata: Optional metadata to include with chunks

        Returns:
            List of ChunkResult objects
        """
        if not text.strip():
            return []

        # Create a temporary document
        doc = Document(text=text, metadata=metadata or {})

        # Get nodes using token splitter
        nodes = self.splitter.get_nodes_from_documents([doc])

        # Convert to ChunkResult
        results: list[ChunkResult] = []

        # LlamaIndex doesn't always provide accurate character indices, especially with overlaps
        # So we'll calculate them ourselves based on a simple approach
        if len(nodes) == 0:
            return results

        # For single chunk, use the whole text
        if len(nodes) == 1:
            result = self._create_chunk_result(
                doc_id=doc_id,
                chunk_index=0,
                text=nodes[0].get_content(),
                start_offset=0,
                end_offset=len(nodes[0].get_content()),
                metadata=metadata,
            )
            return [result]

        # For multiple chunks, we need to ensure monotonic offsets
        # Estimate chars per token (typically 3-4 for English text)
        chars_per_token = len(text) / self._estimate_tokens(text)
        chunk_size_chars = int(self.chunk_size * chars_per_token)
        overlap_chars = int(self.chunk_overlap * chars_per_token)

        for idx, node in enumerate(nodes):
            # Calculate expected offsets
            if idx == 0:
                start_offset = 0
            else:
                # Start where previous chunk started + (chunk_size - overlap)
                start_offset = results[-1].start_offset + chunk_size_chars - overlap_chars
                # Ensure we don't go backwards
                start_offset = max(start_offset, results[-1].start_offset + 1)

            # End offset is start + actual chunk text length
            end_offset = min(start_offset + len(node.get_content()), len(text))

            # For the last chunk, ensure it reaches the end if needed
            if idx == len(nodes) - 1 and end_offset < len(text) - 10:
                end_offset = len(text)

            result = self._create_chunk_result(
                doc_id=doc_id,
                chunk_index=idx,
                text=node.get_content(),
                start_offset=start_offset,
                end_offset=end_offset,
                metadata=metadata,
            )
            results.append(result)

        logger.debug(f"Created {len(results)} chunks from {len(text)} characters")
        return results

    async def chunk_text_async(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Asynchronous chunking using token-based splitting.

        Args:
            text: The text to chunk
            doc_id: Unique identifier for the document
            metadata: Optional metadata to include with chunks

        Returns:
            List of ChunkResult objects
        """
        if not text.strip():
            return []

        # Run synchronous method in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.chunk_text,
            text,
            doc_id,
            metadata,
        )

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate character chunker configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            chunk_size = config.get("chunk_size", self.chunk_size)
            chunk_overlap = config.get("chunk_overlap", self.chunk_overlap)

            # Validate chunk size
            if not isinstance(chunk_size, int) or chunk_size <= 0:
                logger.error(f"Invalid chunk_size: {chunk_size}")
                return False

            # Validate chunk overlap
            if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
                logger.error(f"Invalid chunk_overlap: {chunk_overlap}")
                return False

            # Overlap should be less than chunk size
            if chunk_overlap >= chunk_size:
                logger.error(
                    f"chunk_overlap ({chunk_overlap}) must be less than "
                    f"chunk_size ({chunk_size})"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating config: {e}")
            return False

    def estimate_chunks(self, text_length: int, config: dict[str, Any]) -> int:
        """Estimate number of chunks for capacity planning.

        Args:
            text_length: Length of text in characters
            config: Configuration parameters

        Returns:
            Estimated number of chunks
        """
        chunk_size = config.get("chunk_size", self.chunk_size)
        chunk_overlap = config.get("chunk_overlap", self.chunk_overlap)

        # Rough estimate: assume ~4 characters per token
        estimated_tokens = text_length / 4

        if estimated_tokens <= chunk_size:
            return 1

        # Calculate with overlap
        effective_chunk_size = chunk_size - chunk_overlap
        return 1 + max(0, int((estimated_tokens - chunk_size) / effective_chunk_size))

    def _estimate_tokens(self, text: str) -> int:
        """Estimate number of tokens in text.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Rule of thumb: ~4 characters per token for English text
        # This is a rough estimate since we don't want to tokenize during chunking
        return max(1, len(text) // 4)
