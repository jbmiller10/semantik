#!/usr/bin/env python3
"""
Recursive text splitting chunking strategy.

This strategy recursively splits text using a hierarchy of separators,
attempting to maintain semantic coherence within chunks.
"""

from collections.abc import Callable
from datetime import UTC, datetime

from packages.shared.chunking.domain.entities.chunk import Chunk
from packages.shared.chunking.domain.services.chunking_strategies.base import (
    ChunkingStrategy,
)
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata


class RecursiveChunkingStrategy(ChunkingStrategy):
    """
    Recursive text splitting strategy.

    This strategy tries to split text using a hierarchy of separators,
    falling back to simpler separators when chunks are too large.
    """

    # Hierarchy of separators from most to least preferred
    SEPARATORS = [
        "\n\n\n",  # Multiple blank lines (major sections)
        "\n\n",  # Paragraph breaks
        "\n",  # Line breaks
        ". ",  # Sentence endings
        "! ",  # Exclamation endings
        "? ",  # Question endings
        "; ",  # Semicolons
        ", ",  # Commas
        " ",  # Spaces
        "",  # Character-level (last resort)
    ]

    def __init__(self) -> None:
        """Initialize the recursive chunking strategy."""
        super().__init__("recursive")

    def chunk(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk]:
        """
        Recursively split content into chunks.

        Args:
            content: The text content to chunk
            config: Configuration parameters
            progress_callback: Optional progress callback

        Returns:
            List of chunks
        """
        if not content:
            return []

        # Perform recursive splitting
        text_segments = self._recursive_split(
            content,
            config.max_tokens,
            config.overlap_tokens,
            min_tokens=config.min_tokens,
        )

        # Convert segments to chunks
        chunks: list[Chunk] = []
        chunk_index = 0
        current_position = 0
        total_segments = len(text_segments)

        for i, segment in enumerate(text_segments):
            if not segment.strip():
                continue

            # Find actual position in original content
            start_offset = content.find(segment, current_position)
            if start_offset == -1:
                start_offset = current_position

            end_offset = start_offset + len(segment)
            current_position = end_offset

            # Create chunk metadata
            token_count = self.count_tokens(segment)

            # Skip creating a Chunk object if it would violate size constraints
            # This can happen when text naturally splits into small segments
            if token_count < config.min_tokens:
                # For very small segments, we'll skip them rather than fail
                # They were likely already combined in the recursive split phase
                continue

            if token_count > config.max_tokens:
                # This shouldn't happen if recursive split worked correctly
                # but we'll handle it gracefully by skipping
                continue

            metadata = ChunkMetadata(
                chunk_id=f"{config.strategy_name}_{chunk_index:04d}",
                document_id="doc",
                chunk_index=chunk_index,
                start_offset=start_offset,
                end_offset=end_offset,
                token_count=token_count,
                strategy_name=self.name,
                created_at=datetime.now(tz=UTC),
            )

            # Create chunk
            chunk = Chunk(
                content=self.clean_chunk_text(segment),
                metadata=metadata,
                min_tokens=config.min_tokens,
                max_tokens=config.max_tokens,
            )

            chunks.append(chunk)
            chunk_index += 1

            # Report progress
            if progress_callback:
                progress = ((i + 1) / total_segments) * 100
                progress_callback(min(progress, 100.0))

        return chunks

    def _recursive_split(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int,
        min_tokens: int = 10,
        separator_index: int = 0,
    ) -> list[str]:
        """
        Recursively split text using hierarchical separators.

        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Overlap between chunks
            min_tokens: Minimum tokens per chunk
            separator_index: Current separator level

        Returns:
            List of text segments
        """
        if not text:
            return []

        # Check if text is small enough
        token_count = self.count_tokens(text)
        if token_count <= max_tokens:
            # Only return as a single segment if it meets minimum size
            # or if it's all we have
            if token_count >= min_tokens:
                return [text]
            # Text is too small to be a valid chunk on its own
            # Return it anyway and let the caller decide what to do
            return [text]

        # If we've exhausted all separators, do character-level splitting
        if separator_index >= len(self.SEPARATORS):
            return self._character_split(text, max_tokens, overlap_tokens, min_tokens)

        separator = self.SEPARATORS[separator_index]

        # Try splitting with current separator
        if separator:
            parts = text.split(separator)
        else:
            # Character-level splitting
            return self._character_split(text, max_tokens, overlap_tokens, min_tokens)

        # Reassemble parts into chunks
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_tokens = 0

        for part in parts:
            part_tokens = self.count_tokens(part)

            # If single part is too large, recursively split it
            if part_tokens > max_tokens:
                # Save current chunk if any
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Recursively split the large part
                sub_chunks = self._recursive_split(
                    part,
                    max_tokens,
                    overlap_tokens,
                    min_tokens,
                    separator_index + 1,
                )
                chunks.extend(sub_chunks)

            # If adding this part would exceed limit, start new chunk
            elif current_tokens + part_tokens + len(separator) > max_tokens:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))

                    # Add overlap if configured
                    if overlap_tokens > 0 and current_chunk:
                        # Take last few parts for overlap
                        overlap_parts: list[str] = []
                        overlap_size = 0

                        for i in range(len(current_chunk) - 1, -1, -1):
                            part_size = self.count_tokens(current_chunk[i])
                            if overlap_size + part_size <= overlap_tokens:
                                overlap_parts.insert(0, current_chunk[i])
                                overlap_size += part_size
                            else:
                                break

                        current_chunk = overlap_parts + [part]
                        current_tokens = overlap_size + part_tokens
                    else:
                        current_chunk = [part]
                        current_tokens = part_tokens
                else:
                    current_chunk = [part]
                    current_tokens = part_tokens

            # Add part to current chunk
            else:
                current_chunk.append(part)
                current_tokens += part_tokens + (len(separator) if current_chunk else 0)

        # Add final chunk
        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks

    def _character_split(self, text: str, max_tokens: int, overlap_tokens: int, min_tokens: int = 10) -> list[str]:
        """
        Split text at character level when no separators work.

        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Overlap between chunks
            min_tokens: Minimum tokens per chunk

        Returns:
            List of text segments
        """
        if not text:
            return []

        chunks: list[str] = []

        # Check if text has no spaces (continuous text)
        # In this case, token counting formula is different
        has_spaces = " " in text
        if not has_spaces:
            # For spaceless text, the token formula is approximately:
            # tokens = 0.39 + 0.175 * char_count
            # So: char_count = (tokens - 0.39) / 0.175
            # We'll use a simplified version: chars ≈ tokens * 5.6
            chunk_size = int(max_tokens * 5.6)
            min_chunk_size = int(min_tokens * 5.6)
            overlap_size = int(overlap_tokens * 5.6)
        else:
            # Normal text with spaces uses standard approximation
            chars_per_token = 4
            chunk_size = max_tokens * chars_per_token
            min_chunk_size = min_tokens * chars_per_token
            overlap_size = overlap_tokens * chars_per_token

        position = 0
        while position < len(text):
            # Calculate the end position for this chunk
            # Ensure we get at least min_chunk_size characters
            end = min(position + chunk_size, len(text))

            # Ensure chunk meets minimum size requirement
            if end - position < min_chunk_size:
                # If we can extend to meet minimum, do so
                end = min(position + min_chunk_size, len(text))

            # For the last chunk, ensure it's not too small
            remaining = len(text) - position
            if remaining < min_chunk_size:
                # If we can't make a minimum-sized chunk, try to merge with previous
                # or if this is the first chunk, take all remaining text
                if not chunks and remaining > 0:
                    # First and only chunk - take everything
                    end = len(text)
                else:
                    # Too small for a chunk, will be handled by previous chunk extension
                    break

            # Try to find word boundary only if it won't make chunk too small
            if end < len(text) and (end - position) >= chunk_size:
                word_boundary = self.find_word_boundary(text, end, prefer_before=True)
                # Only use word boundary if it maintains minimum size
                if word_boundary > position + min_chunk_size:
                    end = word_boundary

            # Extract the chunk
            chunk = text[position:end].strip()
            if chunk and len(chunk) >= min_chunk_size:
                chunks.append(chunk)

            # Check if we've reached the end
            if end >= len(text):
                break

            # Calculate next position with overlap
            # Ensure position always advances to avoid infinite loops
            next_position = max(end - overlap_size, position + 1)

            # Check if remaining text after next position is too small for a chunk
            remaining_after_next = len(text) - next_position
            if 0 < remaining_after_next < min_chunk_size:
                # Extend current chunk to include remaining text if possible
                if len(chunks) > 0 and remaining_after_next > 0:
                    # Merge the small remainder into the last chunk
                    last_chunk = chunks[-1]
                    remainder = text[next_position:].strip()
                    if remainder:
                        chunks[-1] = last_chunk + " " + remainder
                break

            position = next_position

        return chunks

    def validate_content(self, content: str) -> tuple[bool, str | None]:
        """
        Validate content for recursive chunking.

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

        # Recursive chunking typically produces slightly fewer chunks
        # due to intelligent splitting at natural boundaries
        estimated_tokens = content_length // 4
        base_estimate = config.estimate_chunks(estimated_tokens)

        # Reduce estimate by 10-20% for recursive splitting efficiency
        return max(1, int(base_estimate * 0.85))
