#!/usr/bin/env python3
"""
Streaming character/token-based chunking strategy.

This strategy breaks text into fixed-size chunks based on character or token count,
with optional overlap between consecutive chunks. It's the simplest streaming strategy
requiring no additional buffer beyond the overlap zone.
"""

from datetime import UTC, datetime
from uuid import uuid4

from shared.chunking.domain.entities.chunk import Chunk
from shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata
from shared.chunking.infrastructure.streaming.window import StreamingWindow

from .base import StreamingChunkingStrategy


class StreamingCharacterStrategy(StreamingChunkingStrategy):
    """
    Streaming fixed-size chunking strategy based on character/token count.

    This strategy creates chunks of approximately equal size with optional
    overlap, maintaining output consistency with the non-streaming version
    while using minimal memory.
    """

    def __init__(self) -> None:
        """Initialize the streaming character chunking strategy."""
        super().__init__("character")
        self._overlap_buffer = ""
        self._chunk_index = 0
        self._byte_offset = 0
        self._char_offset = 0

    def find_word_boundary(self, text: str, target_position: int, prefer_before: bool = True) -> int:
        """
        Find the nearest word boundary to a target position.

        Args:
            text: The text to search in
            target_position: Target character position
            prefer_before: If True, prefer boundary before target

        Returns:
            Position of nearest word boundary
        """
        if not text or target_position < 0:
            return 0

        if target_position >= len(text):
            return len(text)

        if prefer_before:
            # Search backwards for a word boundary
            for i in range(target_position, -1, -1):
                if i < len(text) and (text[i].isspace() or text[i] in '.,;:!?"'):
                    # Return position after the boundary character
                    return min(i + 1, len(text))
            return 0
        # Search forwards for a word boundary
        for i in range(target_position, len(text)):
            if text[i].isspace() or text[i] in '.,;:!?"':
                return i
        return len(text)

    async def process_window(self, window: StreamingWindow, config: ChunkConfig, is_final: bool = False) -> list[Chunk]:
        """
        Process a window of text to produce fixed-size chunks.

        Args:
            window: StreamingWindow containing the data
            config: Chunk configuration parameters
            is_final: Whether this is the final window

        Returns:
            List of chunks produced from this window
        """
        chunks: list[Chunk] = []

        # Get text from window
        text = window.decode_safe()
        if not text and not is_final:
            return chunks

        # Prepend overlap buffer from previous window
        if self._overlap_buffer:
            text = self._overlap_buffer + text
            self._overlap_buffer = ""

        # Calculate chunk parameters
        chars_per_token = 4
        chunk_size_chars = config.max_tokens * chars_per_token
        overlap_chars = config.overlap_tokens * chars_per_token

        # Process text into chunks
        position = 0
        text_length = len(text)

        while position < text_length:
            # Calculate chunk boundaries
            start = position
            end = min(position + chunk_size_chars, text_length)

            # If not the final window and we're near the end, save for next window
            if not is_final and end >= text_length - overlap_chars:
                # Save remaining text for next window to ensure proper overlap
                self._overlap_buffer = text[start:]
                break

            # Adjust boundaries to preserve words and sentences
            if end < text_length:
                # Try sentence boundary first
                sentence_boundary = self.find_sentence_boundary(text, end, prefer_before=True)
                if sentence_boundary > start and sentence_boundary <= end:
                    end = sentence_boundary
                else:
                    # Fall back to word boundary
                    end = self.find_word_boundary(text, end, prefer_before=True)

            # Adjust start to word boundary for non-first chunks
            if self._chunk_index > 0 and start > 0:
                adjusted_start = self.find_word_boundary(text, start, prefer_before=False)
                if adjusted_start < end:
                    start = adjusted_start

            # Extract and clean chunk text
            chunk_text = text[start:end]
            chunk_text = self.clean_chunk_text(chunk_text)

            if not chunk_text:
                # Skip empty chunks but advance position
                position = end
                continue

            # Create chunk metadata
            token_count = self.count_tokens(chunk_text)

            # If token count exceeds max, trim the chunk
            if token_count > config.max_tokens:
                # Reduce chunk size to stay within limits
                # Use a more conservative estimate
                reduction_factor = config.max_tokens / token_count
                new_end = start + int((end - start) * reduction_factor * 0.95)  # 95% to be safe

                # Adjust to word boundary
                new_end = self.find_word_boundary(text, new_end, prefer_before=True)

                # Re-extract and clean chunk text
                chunk_text = text[start:new_end]
                chunk_text = self.clean_chunk_text(chunk_text)

                if not chunk_text:
                    position = end
                    continue

                # Recalculate token count
                token_count = self.count_tokens(chunk_text)
                end = new_end

            # For very small chunks or documents, be lenient with min_tokens
            effective_min_tokens = min(config.min_tokens, token_count, 1)

            metadata = ChunkMetadata(
                chunk_id=str(uuid4()),
                document_id="doc",  # Will be set by processor
                chunk_index=self._chunk_index,
                start_offset=self._char_offset + start,
                end_offset=self._char_offset + end,
                token_count=token_count,
                strategy_name=self.name,
                semantic_density=0.5,  # Default for character-based
                confidence_score=0.9,  # High confidence for simple strategy
                created_at=datetime.now(tz=UTC),
            )

            # Create chunk entity
            chunk = Chunk(
                content=chunk_text,
                metadata=metadata,
                min_tokens=effective_min_tokens,
                max_tokens=config.max_tokens,
            )

            chunks.append(chunk)
            self._chunk_index += 1

            # Update position with overlap
            if end >= text_length:
                break

            position = end - overlap_chars

            # Ensure we advance to prevent infinite loops
            if position <= start:
                position = end

        # Update offsets for next window
        if not is_final:
            # Account for text we've processed
            processed_length = len(text) - len(self._overlap_buffer)
            self._char_offset += processed_length

        return chunks

    async def finalize(self, config: ChunkConfig) -> list[Chunk]:
        """
        Process any remaining buffered text.

        Args:
            config: Chunk configuration parameters

        Returns:
            List of final chunks
        """
        chunks: list[Chunk] = []

        # Process any remaining overlap buffer as final text
        if self._overlap_buffer:
            text = self._overlap_buffer
            self._overlap_buffer = ""

            # Clean and validate
            chunk_text = self.clean_chunk_text(text)
            if chunk_text:
                token_count = self.count_tokens(chunk_text)
                effective_min_tokens = min(config.min_tokens, token_count, 1)

                metadata = ChunkMetadata(
                    chunk_id=str(uuid4()),
                    document_id="doc",
                    chunk_index=self._chunk_index,
                    start_offset=self._char_offset,
                    end_offset=self._char_offset + len(text),
                    token_count=token_count,
                    strategy_name=self.name,
                    semantic_density=0.5,
                    confidence_score=0.9,
                    created_at=datetime.now(tz=UTC),
                )

                chunk = Chunk(
                    content=chunk_text,
                    metadata=metadata,
                    min_tokens=effective_min_tokens,
                    max_tokens=config.max_tokens,
                )

                chunks.append(chunk)
                self._chunk_index += 1

        self._is_finalized = True
        return chunks

    def get_buffer_size(self) -> int:
        """
        Return the current buffer size in bytes.

        Returns:
            Size of overlap buffer in bytes
        """
        return len(self._overlap_buffer.encode("utf-8")) if self._overlap_buffer else 0

    def get_max_buffer_size(self) -> int:
        """
        Return the maximum allowed buffer size.

        Character strategy needs no buffer beyond overlap.

        Returns:
            2KB for overlap buffer
        """
        return 2 * 1024  # 2KB for overlap buffer

    def reset(self) -> None:
        """Reset the strategy state."""
        super().reset()
        self._overlap_buffer = ""
        self._chunk_index = 0
        self._byte_offset = 0
        self._char_offset = 0
