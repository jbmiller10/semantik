#!/usr/bin/env python3
"""
Character/token-based chunking strategy.

This strategy breaks text into fixed-size chunks based on character or token count,
with optional overlap between consecutive chunks.
"""

from collections.abc import Callable
from datetime import datetime

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

        # Calculate chunk size in characters based on token limits
        chars_per_token = 4  # Domain approximation
        chunk_size_chars = config.max_tokens * chars_per_token
        overlap_chars = config.overlap_tokens * chars_per_token

        # Adjust for word boundaries if needed
        position = 0
        chunk_index = 0

        while position < total_chars:
            # Calculate chunk boundaries
            if chunk_index == 0:
                # First chunk starts at beginning
                start = 0
                end = min(chunk_size_chars, total_chars)
            else:
                # Subsequent chunks include overlap
                start = position
                end = min(position + chunk_size_chars, total_chars)

            # Adjust end to word boundary
            if end < total_chars:
                end = self.find_word_boundary(content, end, prefer_before=True)

            # Extract chunk text
            chunk_text = content[start:end]

            # Clean and validate chunk
            chunk_text = self.clean_chunk_text(chunk_text)
            if not chunk_text:
                position = end
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
                created_at=datetime.utcnow(),
            )

            # Create chunk entity
            chunk = Chunk(
                content=chunk_text,
                metadata=metadata,
                min_tokens=config.min_tokens,
                max_tokens=config.max_tokens,
            )

            chunks.append(chunk)
            chunk_index += 1

            # Update position for next chunk (with overlap)
            if end >= total_chars:
                break

            position = end - overlap_chars

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
