#!/usr/bin/env python3
"""
Streaming recursive text splitting strategy.

This strategy recursively splits text using a hierarchy of separators
(paragraphs, sentences, words) while maintaining bounded memory usage
by buffering only one paragraph at a time (max 10KB).
"""

from datetime import UTC, datetime
from uuid import uuid4

from shared.chunking.domain.entities.chunk import Chunk
from shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata
from shared.chunking.infrastructure.streaming.window import StreamingWindow

from .base import StreamingChunkingStrategy


class StreamingRecursiveStrategy(StreamingChunkingStrategy):
    """
    Streaming recursive text splitting strategy.

    Splits text hierarchically by paragraphs, then sentences, then words,
    maintaining output consistency while using minimal memory (max 10KB buffer).
    """

    MAX_BUFFER_SIZE = 10 * 1024  # 10KB max buffer

    def __init__(self) -> None:
        """Initialize the streaming recursive strategy."""
        super().__init__("recursive")
        self._paragraph_buffer: list[str] = []  # Buffer for current paragraph
        self._current_chunk: list[str] = []  # Current chunk being built
        self._current_chunk_size = 0  # Size in tokens
        self._chunk_index = 0
        self._char_offset = 0
        self._pending_text = ""  # Text from previous window

    async def process_window(self, window: StreamingWindow, config: ChunkConfig, is_final: bool = False) -> list[Chunk]:
        """
        Process a window using recursive splitting.

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

        # Combine with pending text from previous window
        if self._pending_text:
            text = self._pending_text + text
            self._pending_text = ""

        # Split into paragraphs (double newline)
        paragraphs = text.split("\n\n")

        # If not final, keep last paragraph for next window (might be incomplete)
        if not is_final and len(paragraphs) > 1:
            self._pending_text = paragraphs[-1]
            paragraphs = paragraphs[:-1]
        elif not is_final and not text.endswith("\n\n"):
            # Keep entire text if it's a single incomplete paragraph
            self._pending_text = text
            return chunks

        # Process each paragraph
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue

            # Check buffer size constraint
            paragraph_size = len(paragraph.encode("utf-8"))
            if paragraph_size > self.MAX_BUFFER_SIZE:
                # Paragraph too large, split by sentences
                chunks.extend(await self._process_large_paragraph(paragraph, config))
            else:
                # Process normal paragraph
                chunks.extend(await self._process_paragraph(paragraph, config))

        # If final, flush current chunk
        if is_final and self._current_chunk:
            chunks.extend(self._create_chunks_from_buffer(config))

        return chunks

    async def _process_paragraph(self, paragraph: str, config: ChunkConfig) -> list[Chunk]:
        """
        Process a single paragraph into chunks.

        Args:
            paragraph: Paragraph text to process
            config: Chunk configuration

        Returns:
            List of chunks created
        """
        chunks: list[Chunk] = []

        # Split paragraph into sentences
        sentences = self._split_sentences(paragraph)

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # Check if adding sentence would exceed max tokens
            if self._current_chunk_size + sentence_tokens > config.max_tokens and self._current_chunk:
                # Emit current chunk
                chunks.extend(self._create_chunks_from_buffer(config))

                # Start new chunk with overlap
                self._apply_overlap(config)

            # Add sentence to current chunk
            self._current_chunk.append(sentence)
            self._current_chunk_size += sentence_tokens

        # Check if we should emit chunk based on size
        if self._current_chunk_size >= config.max_tokens * 0.9:
            chunks.extend(self._create_chunks_from_buffer(config))
            self._apply_overlap(config)

        return chunks

    async def _process_large_paragraph(self, paragraph: str, config: ChunkConfig) -> list[Chunk]:
        """
        Process a paragraph that exceeds buffer limits.

        Split by sentences without buffering the entire paragraph.

        Args:
            paragraph: Large paragraph text
            config: Chunk configuration

        Returns:
            List of chunks created
        """
        chunks: list[Chunk] = []

        # Process sentence by sentence without buffering entire paragraph
        sentences = self._split_sentences(paragraph)

        for sentence in sentences:
            # If sentence itself is too large, split by words
            sentence_size = len(sentence.encode("utf-8"))
            if sentence_size > self.MAX_BUFFER_SIZE:
                # Split sentence by words
                words = sentence.split()
                word_buffer: list[str] = []
                word_buffer_size = 0

                for word in words:
                    word_size = len(word.encode("utf-8"))
                    if word_buffer_size + word_size > self.MAX_BUFFER_SIZE:
                        # Process word buffer
                        if word_buffer:
                            partial_sentence = " ".join(word_buffer)
                            chunks.extend(await self._process_paragraph(partial_sentence, config))
                        word_buffer = [word]
                        word_buffer_size = word_size
                    else:
                        word_buffer.append(word)
                        word_buffer_size += word_size + 1  # +1 for space

                # Process remaining words
                if word_buffer:
                    partial_sentence = " ".join(word_buffer)
                    chunks.extend(await self._process_paragraph(partial_sentence, config))
            else:
                # Process normal sentence
                chunks.extend(await self._process_paragraph(sentence, config))

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        sentences = []
        current_sentence = []

        for i, char in enumerate(text):
            current_sentence.append(char)

            # Check for sentence ending
            if char in ".!?" and i + 1 < len(text) and text[i + 1].isspace():
                sentences.append("".join(current_sentence).strip())
                current_sentence = []

        # Add remaining text as a sentence
        if current_sentence:
            remaining = "".join(current_sentence).strip()
            if remaining:
                sentences.append(remaining)

        return sentences

    def _split_to_token_limit(self, text: str, max_tokens: int) -> list[str]:
        """
        Split text into chunks that fit within max_tokens, using the streaming
        strategy's approximate token counting.
        """
        if not text or max_tokens <= 0:
            return []

        if self.count_tokens(text) <= max_tokens:
            return [text]

        parts: list[str] = []
        words = text.split()
        current_words: list[str] = []
        current_tokens = 0

        for word in words:
            word_tokens = max(1, self.count_tokens(word))

            # Handle extremely long "words" (URLs, base64, minified code) by splitting by character budget.
            if word_tokens > max_tokens:
                if current_words:
                    parts.append(" ".join(current_words))
                    current_words = []
                    current_tokens = 0

                max_chars = max(1, max_tokens * 4)
                for i in range(0, len(word), max_chars):
                    segment = word[i : i + max_chars]
                    if segment:
                        parts.append(segment)
                continue

            effective_tokens = word_tokens + (1 if current_words else 0)

            if current_tokens + effective_tokens > max_tokens and current_words:
                parts.append(" ".join(current_words))
                current_words = [word]
                current_tokens = word_tokens
            else:
                current_words.append(word)
                current_tokens += effective_tokens

        if current_words:
            parts.append(" ".join(current_words))

        return parts

    def _create_chunks_from_buffer(self, config: ChunkConfig) -> list[Chunk]:
        """
        Create one or more chunks from the current buffer.

        Args:
            config: Chunk configuration

        Returns:
            List of chunks produced from the buffer (may be empty)
        """
        if not self._current_chunk:
            return []

        # Join sentences/paragraphs
        chunk_text = " ".join(self._current_chunk)
        chunk_text = self.clean_chunk_text(chunk_text)

        if not chunk_text:
            return []

        parts = self._split_to_token_limit(chunk_text, config.max_tokens)
        produced: list[Chunk] = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            token_count = self.count_tokens(part)
            effective_min_tokens = max(1, min(config.min_tokens, token_count))

            metadata = ChunkMetadata(
                chunk_id=str(uuid4()),
                document_id="doc",
                chunk_index=self._chunk_index,
                start_offset=self._char_offset,
                end_offset=self._char_offset + len(part),
                token_count=token_count,
                strategy_name=self.name,
                semantic_density=0.6,  # Medium density for recursive
                confidence_score=0.85,
                created_at=datetime.now(tz=UTC),
            )

            produced.append(
                Chunk(
                    content=part,
                    metadata=metadata,
                    min_tokens=effective_min_tokens,
                    max_tokens=config.max_tokens,
                )
            )

            self._chunk_index += 1
            self._char_offset += len(part)

        # Clear buffer after emitting all parts
        self._current_chunk = []
        self._current_chunk_size = 0

        return produced

    def _apply_overlap(self, config: ChunkConfig) -> None:
        """
        Apply overlap by keeping last sentences in buffer.

        Args:
            config: Chunk configuration
        """
        if not self._current_chunk or config.overlap_tokens <= 0:
            self._current_chunk = []
            self._current_chunk_size = 0
            return

        # Keep last 20% of sentences for overlap
        overlap_count = max(1, len(self._current_chunk) // 5)
        overlap_sentences = self._current_chunk[-overlap_count:]

        self._current_chunk = overlap_sentences
        self._current_chunk_size = sum(self.count_tokens(s) for s in overlap_sentences)

    async def finalize(self, config: ChunkConfig) -> list[Chunk]:
        """
        Process any remaining buffered text.

        Args:
            config: Chunk configuration parameters

        Returns:
            List of final chunks
        """
        chunks: list[Chunk] = []

        # Process pending text
        if self._pending_text:
            paragraphs = self._pending_text.split("\n\n")
            for paragraph in paragraphs:
                if paragraph.strip():
                    chunks.extend(await self._process_paragraph(paragraph, config))
            self._pending_text = ""

        # Flush current chunk
        if self._current_chunk:
            chunks.extend(self._create_chunks_from_buffer(config))

        self._is_finalized = True
        return chunks

    def get_buffer_size(self) -> int:
        """
        Return the current buffer size in bytes.

        Returns:
            Size of buffers in bytes
        """
        size = 0

        # Pending text buffer
        if self._pending_text:
            size += len(self._pending_text.encode("utf-8"))

        # Current chunk buffer
        if self._current_chunk:
            chunk_text = " ".join(self._current_chunk)
            size += len(chunk_text.encode("utf-8"))

        return size

    def get_max_buffer_size(self) -> int:
        """
        Return the maximum allowed buffer size.

        Returns:
            10KB maximum buffer size
        """
        return self.MAX_BUFFER_SIZE

    def reset(self) -> None:
        """Reset the strategy state."""
        super().reset()
        self._paragraph_buffer = []
        self._current_chunk = []
        self._current_chunk_size = 0
        self._chunk_index = 0
        self._char_offset = 0
        self._pending_text = ""
