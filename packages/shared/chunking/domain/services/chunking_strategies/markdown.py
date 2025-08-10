#!/usr/bin/env python3
"""
Markdown/document structure chunking strategy.

This strategy chunks text based on document structure such as headers,
sections, and other structural elements.
"""

import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from packages.shared.chunking.domain.entities.chunk import Chunk
from packages.shared.chunking.domain.services.chunking_strategies.base import (
    ChunkingStrategy,
)
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata


class MarkdownChunkingStrategy(ChunkingStrategy):
    """
    Document structure-aware chunking strategy.

    This strategy respects document structure like headers, lists,
    code blocks, and paragraphs when creating chunks.
    """

    def __init__(self) -> None:
        """Initialize the markdown chunking strategy."""
        super().__init__("markdown")

    def chunk(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk]:
        """
        Create chunks based on document structure.

        Args:
            content: The text content to chunk
            config: Configuration parameters
            progress_callback: Optional progress callback

        Returns:
            List of chunks
        """
        if not content:
            return []

        # Parse document structure
        sections = self._parse_document_structure(content)

        # Group sections into chunks respecting size limits
        chunk_groups = self._group_sections(sections, config.max_tokens)

        # Convert groups to chunks
        chunks = []
        chunk_index = 0
        total_groups = len(chunk_groups)

        for i, group in enumerate(chunk_groups):
            # Combine sections in group
            group_text = self._combine_sections(group)

            # Clean text
            group_text = self.clean_chunk_text(group_text)
            if not group_text:
                continue

            # Calculate metadata
            start_offset = group[0]["start"]
            end_offset = group[-1]["end"]
            token_count = self.count_tokens(group_text)

            # Extract section title if available
            section_title = None
            for section in group:
                if section["type"] in ["h1", "h2", "h3"]:
                    section_title = section["content"].strip("#").strip()
                    break

            # Determine hierarchy level
            hierarchy_level = self._get_hierarchy_level(group)

            # Create metadata
            metadata = ChunkMetadata(
                chunk_id=f"{config.strategy_name}_{chunk_index:04d}",
                document_id="doc",
                chunk_index=chunk_index,
                start_offset=start_offset,
                end_offset=end_offset,
                token_count=token_count,
                strategy_name=self.name,
                section_title=section_title,
                hierarchy_level=hierarchy_level,
                created_at=datetime.now(tz=UTC),
            )

            # Create chunk
            chunk = Chunk(
                content=group_text,
                metadata=metadata,
                min_tokens=config.min_tokens,
                max_tokens=config.max_tokens,
            )

            chunks.append(chunk)
            chunk_index += 1

            # Report progress
            if progress_callback:
                progress = ((i + 1) / total_groups) * 100
                progress_callback(min(progress, 100.0))

        return chunks

    def _parse_document_structure(self, content: str) -> list[dict]:
        """
        Parse document structure into sections.

        Args:
            content: Document content

        Returns:
            List of section dictionaries
        """
        sections = []
        lines = content.split("\n")
        current_pos = 0

        i = 0
        while i < len(lines):
            line = lines[i]
            line_start = current_pos
            line_end = current_pos + len(line) + 1  # +1 for newline

            # Check for headers
            if line.startswith("#"):
                header_level = len(line) - len(line.lstrip("#"))
                if 1 <= header_level <= 6:
                    sections.append(
                        {
                            "type": f"h{header_level}",
                            "content": line,
                            "start": line_start,
                            "end": line_end,
                            "level": header_level,
                        }
                    )
                    current_pos = line_end
                    i += 1
                    continue

            # Check for code blocks
            if line.strip().startswith("```"):
                code_lines = [line]
                i += 1
                code_start = line_start

                # Find end of code block
                while i < len(lines):
                    current_pos += len(lines[i]) + 1
                    code_lines.append(lines[i])
                    if lines[i].strip().startswith("```"):
                        break
                    i += 1

                sections.append(
                    {
                        "type": "code",
                        "content": "\n".join(code_lines),
                        "start": code_start,
                        "end": current_pos,
                        "level": 0,
                    }
                )
                i += 1
                continue

            # Check for lists
            if re.match(r"^(\s*[-*+]|\s*\d+\.)\s+", line):
                list_lines = [line]
                list_start = line_start
                i += 1

                # Collect consecutive list items
                while i < len(lines):
                    if re.match(r"^(\s*[-*+]|\s*\d+\.)\s+", lines[i]) or (
                        lines[i].startswith("  ") and lines[i].strip()
                    ):
                        current_pos += len(lines[i]) + 1
                        list_lines.append(lines[i])
                        i += 1
                    else:
                        break

                sections.append(
                    {
                        "type": "list",
                        "content": "\n".join(list_lines),
                        "start": list_start,
                        "end": current_pos,
                        "level": 0,
                    }
                )
                continue

            # Check for blockquotes
            if line.startswith(">"):
                quote_lines = [line]
                quote_start = line_start
                i += 1

                # Collect consecutive quote lines
                while i < len(lines) and lines[i].startswith(">"):
                    current_pos += len(lines[i]) + 1
                    quote_lines.append(lines[i])
                    i += 1

                sections.append(
                    {
                        "type": "blockquote",
                        "content": "\n".join(quote_lines),
                        "start": quote_start,
                        "end": current_pos,
                        "level": 0,
                    }
                )
                continue

            # Check for horizontal rules
            if re.match(r"^(\*{3,}|-{3,}|_{3,})\s*$", line):
                sections.append(
                    {
                        "type": "hr",
                        "content": line,
                        "start": line_start,
                        "end": line_end,
                        "level": 0,
                    }
                )
                current_pos = line_end
                i += 1
                continue

            # Regular paragraph
            if line.strip():
                para_lines = [line]
                para_start = line_start
                i += 1

                # Collect consecutive non-empty lines
                while i < len(lines) and lines[i].strip() and not self._is_special_line(lines[i]):
                    current_pos += len(lines[i]) + 1
                    para_lines.append(lines[i])
                    i += 1

                sections.append(
                    {
                        "type": "paragraph",
                        "content": "\n".join(para_lines),
                        "start": para_start,
                        "end": current_pos,
                        "level": 0,
                    }
                )
            else:
                # Empty line
                current_pos = line_end
                i += 1

        return sections

    def _is_special_line(self, line: str) -> bool:
        """
        Check if a line is a special markdown element.

        Args:
            line: Line to check

        Returns:
            True if line is a special element
        """
        stripped = line.strip()

        # Headers
        if stripped.startswith("#"):
            return True

        # Lists
        if re.match(r"^[-*+]\s+", stripped) or re.match(r"^\d+\.\s+", stripped):
            return True

        # Code blocks
        if stripped.startswith("```"):
            return True

        # Blockquotes
        if stripped.startswith(">"):
            return True

        # Horizontal rules
        if re.match(r"^(\*{3,}|-{3,}|_{3,})\s*$", stripped):
            return True

        return False

    def _group_sections(self, sections: list[dict], max_tokens: int) -> list[list[dict]]:
        """
        Group sections into chunks respecting size limits.

        Args:
            sections: List of parsed sections
            max_tokens: Maximum tokens per chunk

        Returns:
            List of section groups
        """
        if not sections:
            return []

        groups = []
        current_group: list[dict[str, Any]] = []
        current_tokens = 0

        for section in sections:
            section_tokens = self.count_tokens(section["content"])

            # If single section exceeds limit, split it
            if section_tokens > max_tokens:
                # Save current group if any
                if current_group:
                    groups.append(current_group)
                    current_group = []
                    current_tokens = 0

                # Split large section
                split_sections = self._split_large_section(section, max_tokens)
                for split in split_sections:
                    groups.append([split])

            # If adding section would exceed limit, start new group
            elif current_tokens + section_tokens > max_tokens:
                if current_group:
                    groups.append(current_group)
                current_group = [section]
                current_tokens = section_tokens

            # Add section to current group
            else:
                current_group.append(section)
                current_tokens += section_tokens

        # Add final group
        if current_group:
            groups.append(current_group)

        return groups

    def _split_large_section(self, section: dict, max_tokens: int) -> list[dict]:
        """
        Split a large section into smaller chunks.

        Args:
            section: Section to split
            max_tokens: Maximum tokens per chunk

        Returns:
            List of split sections
        """
        content = section["content"]
        chars_per_token = 4
        chunk_size = max_tokens * chars_per_token

        splits = []
        position = 0

        while position < len(content):
            end = min(position + chunk_size, len(content))

            # Find sentence or word boundary
            if end < len(content):
                end = self.find_sentence_boundary(content, end, prefer_before=True)
                if end <= position:  # No sentence boundary found or would create empty chunk
                    end = self.find_word_boundary(content, position + chunk_size, prefer_before=True)
                    # Ensure we make progress even if no boundaries found
                    if end <= position:
                        end = min(position + chunk_size, len(content))

            # Ensure we always make progress to avoid infinite loops
            if end <= position:
                end = position + 1

            split_content = content[position:end]

            # Only add non-empty splits
            if split_content:
                splits.append(
                    {
                        "type": section["type"],
                        "content": split_content,
                        "start": section["start"] + position,
                        "end": section["start"] + end,
                        "level": section.get("level", 0),
                    }
                )

            position = end

        return splits

    def _combine_sections(self, sections: list[dict]) -> str:
        """
        Combine sections into a single text.

        Args:
            sections: List of sections to combine

        Returns:
            Combined text
        """
        if not sections:
            return ""

        # Preserve structure with appropriate spacing
        parts = []

        for i, section in enumerate(sections):
            content = section["content"]

            # Add spacing based on section type
            if i > 0:
                prev_type = sections[i - 1]["type"]
                curr_type = section["type"]

                # Add extra line break between different section types
                if prev_type != curr_type or curr_type in ["h1", "h2", "h3"]:
                    parts.append("")

            parts.append(content)

        return "\n".join(parts)

    def _get_hierarchy_level(self, sections: list[dict]) -> int:
        """
        Determine the hierarchy level of a chunk.

        Args:
            sections: List of sections in the chunk

        Returns:
            Hierarchy level (0-6)
        """
        # Find the highest level header in the chunk
        min_level = 7

        for section in sections:
            if section["type"].startswith("h"):
                level = int(section["type"][1])
                min_level = min(min_level, level)

        # If no headers, use 0
        return min_level if min_level < 7 else 0

    def validate_content(self, content: str) -> tuple[bool, str | None]:
        """
        Validate content for markdown chunking.

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

        # Markdown chunking typically produces well-structured chunks
        estimated_tokens = content_length // 4
        base_estimate = config.estimate_chunks(estimated_tokens)

        # Slight reduction due to structural grouping
        return max(1, int(base_estimate * 0.9))
