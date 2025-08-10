#!/usr/bin/env python3
"""
Character/token-based chunking strategy.

This strategy breaks text into fixed-size chunks based on character or token count,
with optional overlap between consecutive chunks.
"""

from collections.abc import Callable
from datetime import UTC, datetime

from packages.shared.chunking.domain.entities.chunk import Chunk
from packages.shared.chunking.domain.services.chunking_strategies.base import (
    ChunkingStrategy,
)
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata


class CharacterChunkingStrategy(ChunkingStrategy):
    """
    Fixed-size chunking strategy based on character/token count.

    This strategy creates chunks of approximately equal size, with optional
    overlap to maintain context between chunks.
    """

    def __init__(self) -> None:
        """Initialize the character chunking strategy."""
        super().__init__("character")

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

        chunks = []
        total_chars = len(content)

        # For very small text that's smaller than the min_tokens requirement,
        # return it as a single chunk
        estimated_tokens = total_chars // 4  # Approximate token count
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
                # If adjusted_start >= end, keep original start to avoid empty chunks

            # Extract chunk text
            chunk_text = content[start:end]

            # Clean and validate chunk
            chunk_text = self.clean_chunk_text(chunk_text)
            if not chunk_text:
                # Ensure position advances to prevent infinite loop
                # If we got an empty chunk, force advancement
                # We're stuck, force advancement by at least 1 character
                position = position + max(1, chunk_size_chars // 4) if end <= position else end
                continue

            # Create chunk metadata
            token_count = self.count_tokens(chunk_text)

            metadata = ChunkMetadata(
                chunk_id=f"{config.strategy_name}_{chunk_index:04d}",
                document_id="doc",  # Will be set by operation
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
            # For very small documents or the last chunk, be lenient with min_tokens
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
            # Force advancement if we're stuck
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

        return config.estimate_chunks(estimated_tokens)
