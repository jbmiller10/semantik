#!/usr/bin/env python3
"""
Streaming hybrid chunking strategy.

This strategy combines multiple chunking approaches, dynamically selecting
the best strategy based on content type detection. Uses up to 150KB buffer
to accommodate the combined approaches.
"""

import re
from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from packages.shared.chunking.domain.entities.chunk import Chunk
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata
from packages.shared.chunking.infrastructure.streaming.window import StreamingWindow

from .base import StreamingChunkingStrategy
from .character import StreamingCharacterStrategy
from .markdown import StreamingMarkdownStrategy
from .recursive import StreamingRecursiveStrategy
from .semantic import StreamingSemanticStrategy


class ContentType(Enum):
    """Types of content detected."""

    MARKDOWN = "markdown"
    CODE = "code"
    PROSE = "prose"
    STRUCTURED = "structured"
    MIXED = "mixed"


class StreamingHybridStrategy(StreamingChunkingStrategy):
    """
    Streaming hybrid chunking strategy.

    Dynamically selects and combines different chunking strategies
    based on content type, providing optimal chunking for mixed documents.
    """

    MAX_BUFFER_SIZE = 150 * 1024  # 150KB max buffer
    DETECTION_WINDOW = 1024  # Bytes to analyze for content type

    def __init__(self) -> None:
        """Initialize the streaming hybrid strategy."""
        super().__init__("hybrid")

        # Sub-strategies
        self._strategies = {
            ContentType.MARKDOWN: StreamingMarkdownStrategy(),
            ContentType.CODE: StreamingCharacterStrategy(),
            ContentType.PROSE: StreamingSemanticStrategy(),
            ContentType.STRUCTURED: StreamingRecursiveStrategy(),
        }

        # State
        self._current_strategy: StreamingChunkingStrategy | None = None
        self._content_buffer: list[str] = []
        self._buffer_size = 0
        self._detection_buffer = ""
        self._chunk_index = 0
        self._char_offset = 0
        self._strategy_scores: dict[ContentType, int] = {}
        self._pending_text = ""

    async def process_window(self, window: StreamingWindow, config: ChunkConfig, is_final: bool = False) -> list[Chunk]:
        """
        Process a window using hybrid approach.

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

        # Combine with pending text
        if self._pending_text:
            text = self._pending_text + text
            self._pending_text = ""

        # Add to detection buffer (limit size)
        if len(self._detection_buffer) < self.DETECTION_WINDOW:
            self._detection_buffer += text[: self.DETECTION_WINDOW - len(self._detection_buffer)]

        # Detect content type if not yet determined
        if not self._current_strategy:
            content_type = self._detect_content_type(self._detection_buffer)
            self._current_strategy = self._strategies[content_type]
            self._current_strategy.reset()

        # Split text into sections for processing
        sections = self._split_into_sections(text, is_final)

        for section_text, _ in sections:
            # Check if we should switch strategies
            if self._should_switch_strategy(section_text):
                # Process buffered content with current strategy
                if self._content_buffer:
                    buffer_text = "".join(self._content_buffer)
                    temp_window = self._create_temp_window(buffer_text)
                    strategy_chunks = await self._current_strategy.process_window(temp_window, config, is_final=False)
                    chunks.extend(self._enhance_chunks(strategy_chunks))
                    self._content_buffer = []
                    self._buffer_size = 0

                # Switch strategy
                new_type = self._detect_content_type(section_text)
                self._current_strategy = self._strategies[new_type]
                self._current_strategy.reset()

            # Check if adding this section would exceed buffer limit
            section_size = len(section_text.encode("utf-8"))

            # Get total size including sub-strategy buffers
            total_size = self.get_buffer_size()

            # If adding this section would exceed max buffer, process current buffer first
            # Use 70% threshold to leave room for sub-strategy buffers
            if (
                total_size + section_size > self.MAX_BUFFER_SIZE * 0.9
                or self._buffer_size + section_size > self.MAX_BUFFER_SIZE * 0.7
            ) and self._content_buffer:
                buffer_text = "".join(self._content_buffer)
                temp_window = self._create_temp_window(buffer_text)
                strategy_chunks = await self._current_strategy.process_window(temp_window, config, is_final=False)
                chunks.extend(self._enhance_chunks(strategy_chunks))
                self._content_buffer = []
                self._buffer_size = 0

            # Add to buffer
            self._content_buffer.append(section_text)
            self._buffer_size += section_size

            # Process if buffer is large enough or total size is getting high
            total_size = self.get_buffer_size()
            if self._buffer_size >= self.MAX_BUFFER_SIZE // 3 or total_size >= self.MAX_BUFFER_SIZE * 0.7:
                buffer_text = "".join(self._content_buffer)
                temp_window = self._create_temp_window(buffer_text)
                strategy_chunks = await self._current_strategy.process_window(temp_window, config, is_final=False)
                chunks.extend(self._enhance_chunks(strategy_chunks))

                # Keep overlap for context
                if config.overlap_tokens > 0 and self._content_buffer:
                    overlap_text = self._content_buffer[-1][-500:]  # Keep last 500 chars
                    self._content_buffer = [overlap_text]
                    self._buffer_size = len(overlap_text.encode("utf-8"))
                else:
                    self._content_buffer = []
                    self._buffer_size = 0

        # Process remaining buffer if final
        if is_final and self._content_buffer:
            buffer_text = "".join(self._content_buffer)
            temp_window = self._create_temp_window(buffer_text)
            strategy_chunks = await self._current_strategy.process_window(temp_window, config, is_final=True)
            chunks.extend(self._enhance_chunks(strategy_chunks))

            # Finalize current strategy
            final_chunks = await self._current_strategy.finalize(config)
            chunks.extend(self._enhance_chunks(final_chunks))

        return chunks

    def _detect_content_type(self, text: str) -> ContentType:
        """
        Detect the type of content.

        Args:
            text: Text to analyze

        Returns:
            Detected content type
        """
        if not text:
            return ContentType.PROSE

        # Calculate indicators
        indicators = {
            "markdown_headers": len(re.findall(r"^#{1,6}\s", text, re.MULTILINE)),
            "code_blocks": text.count("```") + text.count("~~~"),
            "tables": len(re.findall(r"\|.*\|.*\|", text)),
            "lists": len(re.findall(r"^[\s]*[-*+]\s", text, re.MULTILINE)),
            "numbered_lists": len(re.findall(r"^[\s]*\d+\.\s", text, re.MULTILINE)),
            "quotes": text.count('"') + text.count("'"),
            "sentences": len(re.findall(r"[.!?]\s+[A-Z]", text)),
            "code_indent": len(re.findall(r"^    ", text, re.MULTILINE)),
            "brackets": text.count("{") + text.count("[") + text.count("("),
        }

        # Score each content type
        scores = {
            ContentType.MARKDOWN: 0,
            ContentType.CODE: 0,
            ContentType.PROSE: 0,
            ContentType.STRUCTURED: 0,
        }

        # Markdown indicators
        if indicators["markdown_headers"] > 0:
            scores[ContentType.MARKDOWN] += 30
        if indicators["code_blocks"] > 0:
            scores[ContentType.MARKDOWN] += 20
        if indicators["tables"] > 0:
            scores[ContentType.MARKDOWN] += 15

        # Code indicators
        if indicators["code_indent"] > 5:
            scores[ContentType.CODE] += 25
        if indicators["brackets"] > len(text) / 50:
            scores[ContentType.CODE] += 20
        if indicators["code_blocks"] > 2:
            scores[ContentType.CODE] += 15

        # Prose indicators
        if indicators["sentences"] > 5:
            scores[ContentType.PROSE] += 30
        if indicators["quotes"] > 4:
            scores[ContentType.PROSE] += 10
        if indicators["sentences"] > indicators["lists"] * 2:
            scores[ContentType.PROSE] += 15

        # Structured indicators
        if indicators["lists"] + indicators["numbered_lists"] > 3:
            scores[ContentType.STRUCTURED] += 25
        if indicators["markdown_headers"] > 2:
            scores[ContentType.STRUCTURED] += 15

        # Return highest scoring type
        best_type = max(scores, key=lambda k: scores[k])

        # Store scores for potential strategy switching
        self._strategy_scores = scores

        return best_type

    def _should_switch_strategy(self, text: str) -> bool:
        """
        Determine if we should switch strategies.

        Args:
            text: Recent text to analyze

        Returns:
            True if strategy should switch
        """
        if len(text) < 100:
            return False

        # Detect content type of recent text
        new_type = self._detect_content_type(text)

        # Get current strategy type
        current_type = None
        for content_type, strategy in self._strategies.items():
            if strategy == self._current_strategy:
                current_type = content_type
                break

        # Switch if significantly different
        if current_type and new_type != current_type:
            # Check if the difference is significant
            current_score = self._strategy_scores.get(current_type, 0)
            new_score = self._strategy_scores.get(new_type, 0)

            if new_score > current_score * 1.5:
                return True

        return False

    def _split_into_sections(self, text: str, is_final: bool) -> list[tuple[str, ContentType]]:
        """
        Split text into logical sections.

        Args:
            text: Text to split
            is_final: Whether this is final window

        Returns:
            List of (section_text, section_type) tuples
        """
        sections = []

        # Simple splitting by double newlines
        parts = text.split("\n\n")

        # Keep last part if not final (might be incomplete)
        if not is_final and len(parts) > 1:
            self._pending_text = parts[-1]
            parts = parts[:-1]

        for part in parts:
            if part.strip():
                # Detect type for each section
                section_type = self._detect_content_type(part)
                sections.append((part, section_type))

        return sections

    def _create_temp_window(self, text: str) -> StreamingWindow:
        """
        Create a temporary window for sub-strategy processing.

        Args:
            text: Text for the window

        Returns:
            Temporary StreamingWindow
        """
        temp_window = StreamingWindow(max_size=len(text) * 2)
        temp_window.append(text.encode("utf-8"))
        return temp_window

    def _enhance_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Enhance chunks with hybrid metadata.

        Args:
            chunks: Chunks to enhance

        Returns:
            Enhanced chunks
        """
        enhanced = []

        for chunk in chunks:
            # Update chunk index
            new_metadata = ChunkMetadata(
                chunk_id=str(uuid4()),
                document_id=chunk.metadata.document_id,
                chunk_index=self._chunk_index,
                start_offset=self._char_offset,
                end_offset=self._char_offset + len(chunk.content),
                token_count=chunk.metadata.token_count,
                strategy_name="hybrid",
                semantic_density=chunk.metadata.semantic_density,
                confidence_score=chunk.metadata.confidence_score * 0.95,  # Slightly lower for hybrid
                created_at=datetime.now(tz=UTC),
                custom_attributes={
                    "sub_strategy": chunk.metadata.strategy_name,
                    "content_type": self._get_current_content_type(),
                    **chunk.metadata.custom_attributes,
                },
            )

            enhanced_chunk = Chunk(
                content=chunk.content,
                metadata=new_metadata,
                min_tokens=chunk.min_tokens,
                max_tokens=chunk.max_tokens,
            )

            enhanced.append(enhanced_chunk)
            self._chunk_index += 1
            self._char_offset += len(chunk.content)

        return enhanced

    def _get_current_content_type(self) -> str:
        """Get the current content type being processed."""
        for content_type, strategy in self._strategies.items():
            if strategy == self._current_strategy:
                return content_type.value
        return ContentType.MIXED.value

    async def finalize(self, config: ChunkConfig) -> list[Chunk]:
        """
        Process any remaining content.

        Args:
            config: Chunk configuration parameters

        Returns:
            List of final chunks
        """
        chunks: list[Chunk] = []

        # Process pending text
        if self._pending_text:
            self._content_buffer.append(self._pending_text)
            self._pending_text = ""

        # Process remaining buffer
        if self._content_buffer and self._current_strategy:
            buffer_text = "".join(self._content_buffer)
            temp_window = self._create_temp_window(buffer_text)
            strategy_chunks = await self._current_strategy.process_window(temp_window, config, is_final=True)
            chunks.extend(self._enhance_chunks(strategy_chunks))

            # Finalize sub-strategy
            final_chunks = await self._current_strategy.finalize(config)
            chunks.extend(self._enhance_chunks(final_chunks))

        self._is_finalized = True
        return chunks

    def get_buffer_size(self) -> int:
        """
        Return the current buffer size in bytes.

        Returns:
            Size of all buffers in bytes
        """
        size = self._buffer_size

        if self._pending_text:
            size += len(self._pending_text.encode("utf-8"))

        if self._detection_buffer:
            size += len(self._detection_buffer.encode("utf-8"))

        # Add sub-strategy buffer sizes
        if self._current_strategy:
            sub_size = self._current_strategy.get_buffer_size()
            # Cap sub-strategy contribution to prevent overflow
            size += min(sub_size, self.MAX_BUFFER_SIZE // 4)

        return size

    def get_max_buffer_size(self) -> int:
        """
        Return the maximum allowed buffer size.

        Returns:
            150KB maximum buffer size
        """
        return self.MAX_BUFFER_SIZE

    def reset(self) -> None:
        """Reset the strategy state."""
        super().reset()

        # Reset all sub-strategies
        for strategy in self._strategies.values():
            strategy.reset()

        self._current_strategy = None
        self._content_buffer = []
        self._buffer_size = 0
        self._detection_buffer = ""
        self._chunk_index = 0
        self._char_offset = 0
        self._strategy_scores = {}
        self._pending_text = ""
