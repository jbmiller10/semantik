#!/usr/bin/env python3
"""
Unified markdown/document structure chunking strategy.

This module merges the domain-based and LlamaIndex-based markdown chunking 
implementations into a single unified strategy.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from packages.shared.chunking.domain.entities.chunk import Chunk
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata
from packages.shared.chunking.unified.base import UnifiedChunkingStrategy
from packages.shared.chunking.utils.safe_regex import SafeRegex

logger = logging.getLogger(__name__)


class MarkdownChunkingStrategy(UnifiedChunkingStrategy):
    """
    Unified markdown/document structure chunking strategy.

    This strategy respects document structure like headers, lists,
    code blocks, and paragraphs when creating chunks. Can optionally use
    LlamaIndex for enhanced markdown parsing.
    """

    def __init__(self, use_llama_index: bool = False) -> None:
        """
        Initialize the markdown chunking strategy.

        Args:
            use_llama_index: Whether to use LlamaIndex implementation
        """
        super().__init__("markdown")
        self._use_llama_index = use_llama_index
        self._llama_splitter = None
        self.safe_regex = SafeRegex(timeout=1.0)

        # Define safe patterns using RE2-compatible syntax
        self.patterns = {
            # Use atomic groups and possessive quantifiers where possible
            "heading": r"^#{1,6}\s+\S.*$",  # Non-greedy, bounded
            "code_block_start": r"^```[^`\n]*$",  # Simple code block detection
            "list_item": r"^[\*\-\+]\s+\S.*$",  # Bounded list item
            "numbered_list": r"^\d+\.\s+\S.*$",  # Numbered list item
            "blockquote": r"^>\s*\S.*$",  # Bounded blockquote
            "horizontal_rule": r"^(?:---|\*\*\*|___)$",  # Fixed alternatives
        }

        # Compile all patterns with safety checks
        self.compiled_patterns = {}
        for name, pattern in self.patterns.items():
            try:
                self.compiled_patterns[name] = self.safe_regex.compile_safe(pattern)
            except ValueError as e:
                logger.warning(f"Failed to compile pattern {name}: {e}")
                self.compiled_patterns[name] = None

        if use_llama_index:
            try:
                from llama_index.core.node_parser import MarkdownNodeParser

                self._llama_available = True
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
            from llama_index.core.node_parser import MarkdownNodeParser

            return MarkdownNodeParser()
        except Exception as e:
            logger.error(f"Failed to initialize LlamaIndex markdown splitter: {e}")
            return None

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

        # Try LlamaIndex implementation if enabled and content has markdown
        if self._use_llama_index and self._llama_available and self._has_markdown_structure(content):
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

    def _has_markdown_structure(self, content: str) -> bool:
        """Check if content has markdown structure."""
        # Look for markdown headers (# Header, ## Header, etc.)
        try:
            match = self.safe_regex.search_with_timeout(self.patterns["heading"], content, timeout=0.5)
            if match:
                return True
        except Exception as e:
            logger.debug(f"Failed to check for markdown headers: {e}")

        # Fallback: simple string check
        lines = content.split("\n")[:100]  # Check first 100 lines
        for line in lines:
            stripped = line.strip()
            if stripped and stripped[0] == "#" and len(stripped) > 1 and stripped[1] in "# \t":
                return True

        return False

    def _chunk_with_llama_index(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk] | None:
        """
        Chunk using LlamaIndex MarkdownNodeParser.

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

            # Get nodes using markdown parser
            nodes = splitter.get_nodes_from_documents([doc])

            if not nodes:
                return []

            chunks = []
            total_chars = len(content)

            # Convert LlamaIndex nodes to domain chunks
            for idx, node in enumerate(nodes):
                chunk_text = node.get_content()

                # Calculate offsets
                if idx == 0:
                    start_offset = 0
                else:
                    # Find the chunk text in the original content
                    prev_end = chunks[-1].metadata.end_offset
                    start_offset = content.find(chunk_text, prev_end - 100)  # Look near previous end
                    if start_offset == -1:
                        start_offset = prev_end

                end_offset = min(start_offset + len(chunk_text), total_chars)

                # Create chunk metadata
                token_count = self.count_tokens(chunk_text)

                # Extract structural info from node metadata if available
                is_heading = node.metadata.get("is_heading", False) if hasattr(node, "metadata") else False
                heading_level = node.metadata.get("heading_level", 0) if hasattr(node, "metadata") else 0

                metadata = ChunkMetadata(
                    chunk_id=f"{config.strategy_name}_{idx:04d}",
                    document_id="doc",
                    chunk_index=idx,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    token_count=token_count,
                    strategy_name=self.name,
                    semantic_density=0.7,  # Good for structured content
                    confidence_score=0.95,  # Higher confidence with LlamaIndex
                    is_heading=is_heading,
                    heading_level=heading_level,
                    created_at=datetime.now(tz=UTC),
                )

                # Create chunk entity
                effective_min_tokens = min(config.min_tokens, token_count, 1)
                # For markdown sections that exceed max_tokens, adjust the limit
                effective_max_tokens = max(config.max_tokens, token_count)

                chunk = Chunk(
                    content=chunk_text,
                    metadata=metadata,
                    min_tokens=effective_min_tokens,
                    max_tokens=effective_max_tokens,
                )

                chunks.append(chunk)

                # Report progress
                if progress_callback:
                    progress = ((idx + 1) / len(nodes)) * 100
                    progress_callback(min(progress, 100.0))

            return chunks

        except Exception as e:
            logger.warning(f"LlamaIndex markdown chunking failed, falling back to domain: {e}")
            return None

    def _chunk_with_domain(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk]:
        """
        Chunk using domain implementation (pattern-based structural chunking).
        """
        # Parse document structure
        sections = self._parse_structure(content)

        # Group sections into chunks based on size constraints
        chunk_groups = self._group_sections(sections, config.max_tokens, config.min_tokens)

        # Convert groups to chunks
        chunks = []
        chunk_index = 0
        total_groups = len(chunk_groups)

        for i, group in enumerate(chunk_groups):
            # Combine sections in group
            group_text = "\n\n".join([s["content"] for s in group["sections"]])

            # Clean text
            group_text = self.clean_chunk_text(group_text)
            if not group_text:
                continue

            # Calculate offsets
            start_offset = group["start_offset"]
            end_offset = group["end_offset"]
            token_count = self.count_tokens(group_text)

            # Create metadata
            metadata = ChunkMetadata(
                chunk_id=f"{config.strategy_name}_{chunk_index:04d}",
                document_id="doc",
                chunk_index=chunk_index,
                start_offset=start_offset,
                end_offset=end_offset,
                token_count=token_count,
                strategy_name=self.name,
                semantic_density=0.7,  # Good for structured content
                confidence_score=0.85,
                created_at=datetime.now(tz=UTC),
                custom_attributes={
                    "section_type": group.get("type", "mixed"),
                },
            )

            # Create chunk entity
            effective_min_tokens = min(config.min_tokens, token_count, 1)
            # For markdown sections that exceed max_tokens, adjust the limit
            effective_max_tokens = max(config.max_tokens, token_count)

            chunk = Chunk(
                content=group_text,
                metadata=metadata,
                min_tokens=effective_min_tokens,
                max_tokens=effective_max_tokens,
            )

            chunks.append(chunk)
            chunk_index += 1

            # Report progress
            if progress_callback:
                progress = ((i + 1) / total_groups) * 100
                progress_callback(min(progress, 100.0))

        return chunks

    def _parse_structure(self, content: str) -> list[dict[str, Any]]:
        """
        Parse document structure into sections.

        Args:
            content: Document content

        Returns:
            List of section dictionaries
        """
        sections = []
        lines = content.split("\n")

        current_section = {
            "type": "paragraph",
            "content": "",
            "start_offset": 0,
            "end_offset": 0,
            "level": 0,
        }

        offset = 0
        in_code_block = False

        for line in lines:
            line_with_newline = line + "\n"
            line_length = len(line_with_newline)

            # Check for code block
            if self._matches_pattern("code_block_start", line):
                in_code_block = not in_code_block
                if in_code_block:
                    # Start new code section
                    if current_section["content"]:
                        sections.append(current_section)
                    current_section = {
                        "type": "code",
                        "content": line_with_newline,
                        "start_offset": offset,
                        "end_offset": offset + line_length,
                        "level": 0,
                    }
                else:
                    # End code section
                    current_section["content"] += line_with_newline
                    current_section["end_offset"] = offset + line_length
                    sections.append(current_section)
                    current_section = {
                        "type": "paragraph",
                        "content": "",
                        "start_offset": offset + line_length,
                        "end_offset": offset + line_length,
                        "level": 0,
                    }
            elif in_code_block:
                # Inside code block
                current_section["content"] += line_with_newline
                current_section["end_offset"] = offset + line_length
            elif self._matches_pattern("heading", line):
                # Header - start new section
                if current_section["content"]:
                    sections.append(current_section)

                # Count heading level
                level = len(line) - len(line.lstrip("#"))

                current_section = {
                    "type": "heading",
                    "content": line_with_newline,
                    "start_offset": offset,
                    "end_offset": offset + line_length,
                    "level": level,
                }
                sections.append(current_section)

                # Start new paragraph section
                current_section = {
                    "type": "paragraph",
                    "content": "",
                    "start_offset": offset + line_length,
                    "end_offset": offset + line_length,
                    "level": 0,
                }
            elif self._matches_pattern("horizontal_rule", line):
                # Horizontal rule - section break
                if current_section["content"]:
                    sections.append(current_section)

                # Start new section
                current_section = {
                    "type": "paragraph",
                    "content": "",
                    "start_offset": offset + line_length,
                    "end_offset": offset + line_length,
                    "level": 0,
                }
            else:
                # Regular content
                current_section["content"] += line_with_newline
                current_section["end_offset"] = offset + line_length

            offset += line_length

        # Add final section
        if current_section["content"]:
            sections.append(current_section)

        return sections

    def _matches_pattern(self, pattern_name: str, text: str) -> bool:
        """
        Check if text matches a pattern.

        Args:
            pattern_name: Name of pattern to check
            text: Text to match against

        Returns:
            True if matches
        """
        pattern = self.compiled_patterns.get(pattern_name)
        if not pattern:
            return False

        try:
            return self.safe_regex.match_safe(self.patterns[pattern_name], text) is not None
        except Exception:
            return False

    def _group_sections(
        self,
        sections: list[dict[str, Any]],
        max_tokens: int,
        min_tokens: int,
    ) -> list[dict[str, Any]]:
        """
        Group sections into chunks respecting size constraints.

        Args:
            sections: List of section dictionaries
            max_tokens: Maximum tokens per chunk
            min_tokens: Minimum tokens per chunk

        Returns:
            List of grouped sections
        """
        if not sections:
            return []

        groups = []
        current_group = {
            "sections": [],
            "start_offset": sections[0]["start_offset"],
            "end_offset": sections[0]["end_offset"],
            "token_count": 0,
            "type": "mixed",
        }

        for section in sections:
            section_tokens = self.count_tokens(section["content"])

            # Check if adding this section would exceed max_tokens
            if current_group["token_count"] + section_tokens > max_tokens:
                # Save current group if it meets min_tokens
                if current_group["token_count"] >= min_tokens:
                    groups.append(current_group)

                    # Start new group
                    current_group = {
                        "sections": [section],
                        "start_offset": section["start_offset"],
                        "end_offset": section["end_offset"],
                        "token_count": section_tokens,
                        "type": section["type"],
                    }
                else:
                    # Current group is too small, keep adding
                    current_group["sections"].append(section)
                    current_group["end_offset"] = section["end_offset"]
                    current_group["token_count"] += section_tokens
            else:
                # Add section to current group
                current_group["sections"].append(section)
                current_group["end_offset"] = section["end_offset"]
                current_group["token_count"] += section_tokens

        # Add final group
        if current_group["sections"]:
            groups.append(current_group)

        return groups

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

        # Convert character length to estimated tokens
        estimated_tokens = content_length // 4

        # Markdown chunking creates chunks based on structure
        base_estimate = config.estimate_chunks(estimated_tokens)
        return max(1, base_estimate)
