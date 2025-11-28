#!/usr/bin/env python3
"""
Unified character/token-based chunking strategy.

This module merges the domain-based and LlamaIndex-based character chunking
implementations into a single unified strategy.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from shared.chunking.domain.entities.chunk import Chunk
from shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata
from shared.chunking.unified.base import UnifiedChunkingStrategy

logger = logging.getLogger(__name__)


class CharacterChunkingStrategy(UnifiedChunkingStrategy):
    """
    Unified character/token-based chunking strategy.

    This strategy creates chunks of approximately equal size based on character
    or token count, with optional overlap to maintain context between chunks.
    Can optionally use LlamaIndex for enhanced tokenization.
    """

    def __init__(self, use_llama_index: bool = False) -> None:
        """
        Initialize the character chunking strategy.

        Args:
            use_llama_index: Whether to use LlamaIndex implementation
        """
        super().__init__("character")
        self._use_llama_index = use_llama_index
        self._llama_splitter = None

        if use_llama_index:
            try:
                # Check if LlamaIndex is available
                import importlib.util

                spec = importlib.util.find_spec("llama_index.core.node_parser")
                self._llama_available = spec is not None
            except ImportError:
                logger.warning("LlamaIndex not available, falling back to domain implementation")
                self._llama_available = False
                self._use_llama_index = False
        else:
            self._llama_available = False

    def _init_llama_splitter(self, config: ChunkConfig) -> Any:
        """Initialize LlamaIndex splitter if needed."""
        if not self._use_llama_index or not self._llama_available:
            return None

        try:
            from llama_index.core.node_parser import TokenTextSplitter

            # Use overlap_tokens directly from config
            overlap_tokens = config.overlap_tokens

            return TokenTextSplitter(
                chunk_size=config.max_tokens,
                chunk_overlap=overlap_tokens,
                separator=" ",
            )
        except Exception as e:
            logger.error(f"Failed to initialize LlamaIndex splitter: {e}")
            return None

    def chunk(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk]:
        """
        Break content into fixed-size chunks.

        Args:
            content: The text content to chunk
            config: Configuration parameters
            progress_callback: Optional progress callback

        Returns:
            List of chunks
        """
        if not content:
            return []

        # Try LlamaIndex implementation if enabled
        if self._use_llama_index and self._llama_available:
            chunks = self._chunk_with_llama_index(content, config, progress_callback)
            if chunks is not None:
                return chunks

        # Fall back to domain implementation
        return self._chunk_with_domain(content, config, progress_callback)

    async def chunk_async(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk]:
        """
        Asynchronous chunking.

        Args:
            content: The text content to chunk
            config: Configuration parameters
            progress_callback: Optional progress callback

        Returns:
            List of chunks
        """
        if not content:
            return []

        # Run synchronous method in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.chunk,
            content,
            config,
            progress_callback,
        )

    def _chunk_with_llama_index(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk] | None:
        """
        Chunk using LlamaIndex TokenTextSplitter.

        Returns None if LlamaIndex is not available or fails.
        """
        try:
            from llama_index.core import Document

            # Initialize splitter
            splitter = self._init_llama_splitter(config)
            if not splitter:
                return None

            # Create a temporary document
            doc = Document(text=content)

            # Get nodes using token splitter
            nodes = splitter.get_nodes_from_documents([doc])

            if not nodes:
                return []

            chunks = []
            total_chars = len(content)

            # For LlamaIndex chunks, we need to calculate offsets based on the chunking pattern
            # Since LlamaIndex handles the overlap internally, we calculate offsets accordingly

            # Calculate approximate characters per token (LlamaIndex uses its own tokenizer)
            chars_per_token = 4
            config.max_tokens * chars_per_token
            overlap_chars = config.overlap_tokens * chars_per_token

            # Track the expected position based on chunk size and overlap
            current_position = 0

            # Convert LlamaIndex nodes to domain chunks
            for idx, node in enumerate(nodes):
                chunk_text = node.get_content()
                chunk_length = len(chunk_text)

                # Calculate offsets based on the chunking pattern
                if idx == 0:
                    # First chunk starts at the beginning
                    start_offset = 0
                    end_offset = min(chunk_length, total_chars)
                else:
                    # Subsequent chunks start with overlap from previous chunk
                    # The overlap means the new chunk starts before the previous ended
                    start_offset = current_position - overlap_chars
                    start_offset = max(0, start_offset)  # Ensure not negative
                    end_offset = min(start_offset + chunk_length, total_chars)

                # Update position for next chunk
                current_position = end_offset

                # Create chunk metadata
                token_count = self.count_tokens(chunk_text)

                metadata = ChunkMetadata(
                    chunk_id=f"{config.strategy_name}_{idx:04d}",
                    document_id="doc",
                    chunk_index=idx,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    token_count=token_count,
                    strategy_name=self.name,
                    semantic_density=0.5,
                    confidence_score=0.95,  # Higher confidence with LlamaIndex
                    created_at=datetime.now(tz=UTC),
                )

                # Create chunk entity
                effective_min_tokens = min(config.min_tokens, token_count, 1)

                chunk = Chunk(
                    content=chunk_text,
                    metadata=metadata,
                    min_tokens=effective_min_tokens,
                    max_tokens=config.max_tokens,
                )

                chunks.append(chunk)

                # Report progress
                if progress_callback:
                    progress = ((idx + 1) / len(nodes)) * 100
                    progress_callback(min(progress, 100.0))

            return chunks

        except Exception as e:
            logger.warning(f"LlamaIndex chunking failed, falling back to domain: {e}")
            return None

    def _chunk_with_domain(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk]:
        """
        Chunk using domain implementation (original character strategy).
        """
        chunks = []
        total_chars = len(content)

        # For very small text that's smaller than the min_tokens requirement,
        # return it as a single chunk
        estimated_tokens = max(1, total_chars // 4)  # Approximate token count, at least 1
        if estimated_tokens < config.min_tokens:
            metadata = ChunkMetadata(
                chunk_id=f"{config.strategy_name}_0000",
                document_id="doc",
                chunk_index=0,
                start_offset=0,
                end_offset=total_chars,
                token_count=estimated_tokens,
                strategy_name=self.name,
                semantic_density=0.5,
                confidence_score=0.9,
                created_at=datetime.now(tz=UTC),
            )

            # Create single chunk with relaxed min_tokens
            chunk = Chunk(
                content=content,
                metadata=metadata,
                min_tokens=min(config.min_tokens, estimated_tokens, 1),
                max_tokens=config.max_tokens,
            )

            if progress_callback:
                progress_callback(100.0)

            return [chunk]

        # Calculate chunk size in characters based on token limits
        chars_per_token = 4  # Domain approximation
        chunk_size_chars = config.max_tokens * chars_per_token
        overlap_chars = config.overlap_tokens * chars_per_token

        # Adjust for word boundaries if needed
        position = 0
        chunk_index = 0
        max_iterations = (total_chars // max(1, chunk_size_chars - overlap_chars)) + 100  # Safety limit

        while position < total_chars and chunk_index < max_iterations:
            # Calculate chunk boundaries
            if chunk_index == 0:
                # First chunk starts at beginning
                start = 0
                end = min(chunk_size_chars, total_chars)
            else:
                # Subsequent chunks include overlap
                start = position
                end = min(position + chunk_size_chars, total_chars)

            # Adjust boundaries to preserve words and sentences
            if end < total_chars:
                # Try to end at sentence boundary first
                sentence_boundary = self.find_sentence_boundary(content, end, prefer_before=True)
                if sentence_boundary > start and sentence_boundary <= end:
                    end = sentence_boundary
                else:
                    # Fall back to word boundary
                    end = self.find_word_boundary(content, end, prefer_before=True)

            # Adjust start to word boundary for non-first chunks
            if chunk_index > 0 and start > 0:
                # Find the next word boundary after start to avoid partial words
                adjusted_start = self.find_word_boundary(content, start, prefer_before=False)
                # Ensure adjusted start doesn't go beyond end
                if adjusted_start < end:
                    start = adjusted_start

            # Extract chunk text
            chunk_text = content[start:end]

            # Clean and validate chunk
            chunk_text = self.clean_chunk_text(chunk_text)
            if not chunk_text:
                # Ensure position advances to prevent infinite loop
                position = position + max(1, chunk_size_chars // 4) if end <= position else end
                continue

            # Create chunk metadata
            token_count = self.count_tokens(chunk_text)

            metadata = ChunkMetadata(
                chunk_id=f"{config.strategy_name}_{chunk_index:04d}",
                document_id="doc",
                chunk_index=chunk_index,
                start_offset=start,
                end_offset=end,
                token_count=token_count,
                strategy_name=self.name,
                semantic_density=0.5,  # Default for character-based chunking
                confidence_score=0.9,  # High confidence for simple strategy
                created_at=datetime.now(tz=UTC),
            )

            # Create chunk entity with adjusted min_tokens for small documents
            effective_min_tokens = min(config.min_tokens, token_count, 1)

            chunk = Chunk(
                content=chunk_text,
                metadata=metadata,
                min_tokens=effective_min_tokens,
                max_tokens=config.max_tokens,
            )

            chunks.append(chunk)
            chunk_index += 1

            # Update position for next chunk (with overlap)
            if end >= total_chars:
                break

            # Ensure position advances to prevent infinite loop
            new_position = end - overlap_chars
            position = position + max(1, chunk_size_chars // 2) if new_position <= position else new_position

            # Report progress
            if progress_callback:
                progress = (position / total_chars) * 100
                progress_callback(min(progress, 100.0))

        # Final progress report
        if progress_callback:
            progress_callback(100.0)

        return chunks

    def validate_content(self, content: str) -> tuple[bool, str | None]:
        """
        Validate content for character-based chunking.

        Args:
            content: Content to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not content:
            return False, "Content cannot be empty"

        if len(content) > 50_000_000:  # 50MB limit
            return False, f"Content too large: {len(content)} characters"

        return True, None

    def estimate_chunks(self, content_length: int, config: ChunkConfig) -> int:
        """
        Estimate the number of chunks.

        Args:
            content_length: Length of content in characters
            config: Chunking configuration

        Returns:
            Estimated chunk count
        """
        if content_length == 0:
            return 0

        # Convert character length to estimated tokens
        estimated_tokens = content_length // 4

        return int(config.estimate_chunks(estimated_tokens))
