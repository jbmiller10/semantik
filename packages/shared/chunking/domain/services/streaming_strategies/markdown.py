#!/usr/bin/env python3
"""
Streaming Markdown-aware chunking strategy.

This strategy processes Markdown documents while preserving structure,
tracking heading hierarchy, and maintaining code blocks and tables intact.
Uses up to 100KB buffer for current section.
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


class MarkdownBlockType(Enum):
    """Types of Markdown blocks."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    CODE_BLOCK = "code_block"
    LIST = "list"
    TABLE = "table"
    QUOTE = "quote"
    HORIZONTAL_RULE = "hr"


class StreamingMarkdownStrategy(StreamingChunkingStrategy):
    """
    Streaming Markdown-aware chunking strategy.

    Preserves document structure by tracking heading hierarchy,
    keeping code blocks and tables intact, and maintaining
    proper Markdown formatting.
    """

    MAX_BUFFER_SIZE = 100 * 1024  # 100KB max buffer

    def __init__(self) -> None:
        """Initialize the streaming Markdown strategy."""
        super().__init__("markdown")
        self._heading_stack: list[str] = []  # Current heading hierarchy
        self._current_section: list[str] = []  # Current section content
        self._section_size = 0  # Size in bytes
        self._in_code_block = False  # Track if we're in a code block
        self._code_fence = ""  # The fence used (``` or ~~~)
        self._pending_lines: list[str] = []  # Lines from previous window
        self._chunk_index = 0
        self._char_offset = 0
        self._table_buffer: list[str] = []  # Buffer for table rows
        self._in_table = False

    async def process_window(self, window: StreamingWindow, config: ChunkConfig, is_final: bool = False) -> list[Chunk]:
        """
        Process a window of Markdown text.

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

        # Split into lines
        lines = text.split("\n")

        # Handle pending lines from previous window
        if self._pending_lines:
            # Combine first line with last pending line
            if lines:
                self._pending_lines[-1] += lines[0]
                lines = self._pending_lines[:-1] + [self._pending_lines[-1]] + lines[1:]
            else:
                lines = self._pending_lines
            self._pending_lines = []

        # Process each line
        for i, line in enumerate(lines):
            # Check if this is the last line and not final window
            is_last_line = i == len(lines) - 1

            if is_last_line and not is_final:
                # Save incomplete line for next window
                self._pending_lines.append(line)
                continue

            # Process the line
            block_type = self._identify_block_type(line)

            # Handle based on block type
            if block_type == MarkdownBlockType.HEADING:
                # Process heading
                chunks.extend(await self._process_heading(line, config))

            elif block_type == MarkdownBlockType.CODE_BLOCK:
                # Toggle code block state
                self._handle_code_fence(line)
                self._current_section.append(line)
                self._section_size += len(line.encode("utf-8"))

            elif self._in_code_block:
                # Inside code block, accumulate lines
                self._current_section.append(line)
                self._section_size += len(line.encode("utf-8"))

            elif block_type == MarkdownBlockType.TABLE:
                # Handle table
                chunks.extend(await self._process_table_line(line, config))

            elif block_type == MarkdownBlockType.HORIZONTAL_RULE:
                # Horizontal rule marks section boundary
                chunk = await self._emit_current_section(config)
                if chunk:
                    chunks.append(chunk)
                self._current_section.append(line)

            else:
                # Regular content
                self._current_section.append(line)
                self._section_size += len(line.encode("utf-8"))

                # Check if section is getting too large
                if self._section_size > config.max_tokens * 4:  # Approx bytes
                    chunk = await self._emit_current_section(config)
                    if chunk:
                        chunks.append(chunk)

        # If final, emit remaining section
        if is_final:
            chunk = await self._emit_current_section(config)
            if chunk:
                chunks.append(chunk)

        return chunks

    def _identify_block_type(self, line: str) -> MarkdownBlockType:
        """
        Identify the type of Markdown block.

        Args:
            line: Line to analyze

        Returns:
            Type of Markdown block
        """
        stripped = line.strip()

        # Code fence
        if stripped.startswith(("```", "~~~")):
            return MarkdownBlockType.CODE_BLOCK

        # Heading
        if re.match(r"^#{1,6}\s", stripped):
            return MarkdownBlockType.HEADING

        # Horizontal rule
        if re.match(r"^(\*{3,}|-{3,}|_{3,})$", stripped):
            return MarkdownBlockType.HORIZONTAL_RULE

        # Table (simple detection)
        if "|" in line and len(line.split("|")) >= 3:
            # Check if it looks like a table row
            parts = [p.strip() for p in line.split("|")]
            if all(len(p) < 100 for p in parts):  # Reasonable cell size
                return MarkdownBlockType.TABLE

        # List
        if re.match(r"^(\s*[-*+]\s|\s*\d+\.\s)", line):
            return MarkdownBlockType.LIST

        # Quote
        if stripped.startswith(">"):
            return MarkdownBlockType.QUOTE

        return MarkdownBlockType.PARAGRAPH

    def _handle_code_fence(self, line: str) -> None:
        """
        Handle code fence detection and state.

        Args:
            line: Line containing potential code fence
        """
        stripped = line.strip()

        if stripped.startswith("```"):
            if not self._in_code_block:
                self._in_code_block = True
                self._code_fence = "```"
            elif self._code_fence == "```":
                self._in_code_block = False
                self._code_fence = ""

        elif stripped.startswith("~~~"):
            if not self._in_code_block:
                self._in_code_block = True
                self._code_fence = "~~~"
            elif self._code_fence == "~~~":
                self._in_code_block = False
                self._code_fence = ""

    async def _process_heading(self, line: str, config: ChunkConfig) -> list[Chunk]:
        """
        Process a heading line.

        Args:
            line: Heading line
            config: Chunk configuration

        Returns:
            List of chunks if section boundary detected
        """
        chunks: list[Chunk] = []

        # Extract heading level
        match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if not match:
            # Not a valid heading, treat as content
            self._current_section.append(line)
            self._section_size += len(line.encode("utf-8"))
            return chunks

        level = len(match.group(1))
        _ = match.group(2)  # heading_text - not used currently

        # Check if this heading starts a new section
        if self._heading_stack:
            current_level = len(self._heading_stack)

            if level <= current_level:
                # Same or higher level heading - emit current section
                chunk = await self._emit_current_section(config)
                if chunk:
                    chunks.append(chunk)

                # Update heading stack
                self._heading_stack = self._heading_stack[: level - 1]

        # Add new heading to stack
        self._heading_stack = self._heading_stack[: level - 1] + [line.strip()]

        # Add heading to current section
        self._current_section.append(line)
        self._section_size += len(line.encode("utf-8"))

        return chunks

    async def _process_table_line(self, line: str, config: ChunkConfig) -> list[Chunk]:  # noqa: ARG002
        """
        Process a table line.

        Args:
            line: Table line
            config: Chunk configuration

        Returns:
            List of chunks if table completed
        """
        chunks: list[Chunk] = []

        # Simple table detection
        if "|" in line:
            if not self._in_table:
                self._in_table = True
                self._table_buffer = []

            self._table_buffer.append(line)

            # Check if table separator (---|---|---)
            if re.match(r"^[\s\|:\-]+$", line):
                # This is likely the header separator
                pass

            # Check for table end (empty line or non-table content next)
            # This is simplified - in streaming we can't look ahead
        else:
            # Not a table line
            if self._in_table:
                # End of table - add to section
                self._current_section.extend(self._table_buffer)
                self._section_size += sum(len(line.encode("utf-8")) for line in self._table_buffer)
                self._table_buffer = []
                self._in_table = False

            # Process as normal line
            self._current_section.append(line)
            self._section_size += len(line.encode("utf-8"))

        return chunks

    async def _emit_current_section(self, config: ChunkConfig) -> Chunk | None:
        """
        Emit the current section as a chunk.

        Args:
            config: Chunk configuration

        Returns:
            Chunk if section has content
        """
        if not self._current_section:
            return None

        # Join lines
        content = "\n".join(self._current_section)

        # Clean but preserve Markdown formatting
        content = self._clean_markdown(content)

        if not content.strip():
            return None

        # Check token count and split if necessary
        token_count = self.count_tokens(content)

        # If content exceeds max_tokens, we need to split it
        if token_count > config.max_tokens:
            # Split content to fit within max_tokens
            # Estimate how much content we can keep
            ratio = config.max_tokens / token_count
            target_chars = int(len(content) * ratio * 0.9)  # Use 90% to be safe

            # Find a good split point (preferably at a line break)
            split_point = target_chars
            newline_pos = content.rfind("\n", 0, target_chars)
            if newline_pos > target_chars * 0.5:  # If we found a newline not too far back
                split_point = newline_pos

            # Take the first part
            content = content[:split_point].strip()

            # Recalculate token count for the truncated content
            token_count = self.count_tokens(content)

            # Keep the remaining lines for the next chunk
            remaining_content = content[split_point:].strip()
            if remaining_content:
                # Convert back to lines and keep them in pending
                self._pending_lines.extend(remaining_content.split("\n"))

        # Create metadata with heading context
        effective_min_tokens = min(config.min_tokens, token_count, 1)

        metadata = ChunkMetadata(
            chunk_id=str(uuid4()),
            document_id="doc",
            chunk_index=self._chunk_index,
            start_offset=self._char_offset,
            end_offset=self._char_offset + len(content),
            token_count=token_count,
            strategy_name=self.name,
            semantic_density=0.7,  # Good density for structured content
            confidence_score=0.85,
            created_at=datetime.now(tz=UTC),
            custom_attributes={
                "headings": self._heading_stack.copy(),
                "has_code": any("```" in line or "~~~" in line for line in self._current_section),
                "has_table": any("|" in line for line in self._current_section),
            },
        )

        # Create chunk
        chunk = Chunk(
            content=content,
            metadata=metadata,
            min_tokens=effective_min_tokens,
            max_tokens=config.max_tokens,
        )

        # Update state
        self._chunk_index += 1
        self._char_offset += len(content)

        # Clear section
        self._current_section = []
        self._section_size = 0

        return chunk

    def _clean_markdown(self, text: str) -> str:
        """
        Clean Markdown text while preserving formatting.

        Args:
            text: Raw Markdown text

        Returns:
            Cleaned text
        """
        lines = text.split("\n")
        cleaned = []

        for line in lines:
            # Remove excessive whitespace but preserve indentation
            if line.strip() or line.startswith("    "):  # Code block indent
                cleaned.append(line.rstrip())
            elif cleaned and cleaned[-1]:  # Preserve single blank lines
                cleaned.append("")

        # Remove trailing blank lines
        while cleaned and not cleaned[-1]:
            cleaned.pop()

        return "\n".join(cleaned)

    async def finalize(self, config: ChunkConfig) -> list[Chunk]:
        """
        Process any remaining content.

        Args:
            config: Chunk configuration parameters

        Returns:
            List of final chunks
        """
        chunks: list[Chunk] = []

        # Process pending lines
        if self._pending_lines:
            for line in self._pending_lines:
                self._current_section.append(line)
                self._section_size += len(line.encode("utf-8"))
            self._pending_lines = []

        # Process table buffer
        if self._table_buffer:
            self._current_section.extend(self._table_buffer)
            self._section_size += sum(len(line.encode("utf-8")) for line in self._table_buffer)
            self._table_buffer = []

        # Emit final section
        chunk = await self._emit_current_section(config)
        if chunk:
            chunks.append(chunk)

        self._is_finalized = True
        return chunks

    def get_buffer_size(self) -> int:
        """
        Return the current buffer size in bytes.

        Returns:
            Size of buffers in bytes
        """
        size = 0

        # Current section
        if self._current_section:
            content = "\n".join(self._current_section)
            size += len(content.encode("utf-8"))

        # Pending lines
        if self._pending_lines:
            pending = "\n".join(self._pending_lines)
            size += len(pending.encode("utf-8"))

        # Table buffer
        if self._table_buffer:
            table = "\n".join(self._table_buffer)
            size += len(table.encode("utf-8"))

        return size

    def get_max_buffer_size(self) -> int:
        """
        Return the maximum allowed buffer size.

        Returns:
            100KB maximum buffer size
        """
        return self.MAX_BUFFER_SIZE

    def reset(self) -> None:
        """Reset the strategy state."""
        super().reset()
        self._heading_stack = []
        self._current_section = []
        self._section_size = 0
        self._in_code_block = False
        self._code_fence = ""
        self._pending_lines = []
        self._chunk_index = 0
        self._char_offset = 0
        self._table_buffer = []
        self._in_table = False
