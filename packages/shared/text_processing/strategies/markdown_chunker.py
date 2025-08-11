#!/usr/bin/env python3
"""
Markdown-aware chunking strategy using LlamaIndex.

This module implements markdown structure-aware text splitting using LlamaIndex's
MarkdownNodeParser.
"""

import asyncio
import logging
from typing import Any

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser

from packages.shared.chunking.utils.safe_regex import RegexTimeout, SafeRegex
from packages.shared.text_processing.base_chunker import BaseChunker, ChunkResult

logger = logging.getLogger(__name__)


class MarkdownChunker(BaseChunker):
    """Markdown-aware chunking using LlamaIndex MarkdownNodeParser."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize MarkdownChunker.

        Args:
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)

        # Initialize LlamaIndex markdown parser
        self.splitter = MarkdownNodeParser()

        # Initialize SafeRegex for pattern matching
        self.safe_regex = SafeRegex(timeout=1.0)

        logger.info("Initialized MarkdownChunker")

    def _has_markdown_headers(self, text: str) -> bool:
        """Check if text contains markdown headers."""
        # Look for markdown headers (# Header, ## Header, etc.) with safe regex
        try:
            # Use bounded pattern for safety
            pattern = r"^#{1,6}\s+\S.*$"
            match = self.safe_regex.search_with_timeout(pattern, text, timeout=0.5)
            return match is not None
        except (RegexTimeout, Exception) as e:
            logger.debug(f"Failed to check for markdown headers: {e}")
            # Fallback: simple string check
            lines = text.split("\n")
            for line in lines[:100]:  # Check first 100 lines only
                stripped = line.strip()
                if stripped and stripped[0] == "#" and len(stripped) > 1 and stripped[1] in "# \t":
                    return True
            return False

    def _is_markdown_file(self, metadata: dict[str, Any] | None) -> bool:
        """Check if the document is a markdown file based on metadata."""
        if not metadata:
            return False

        # Check file extension
        file_path = metadata.get("file_path", "")
        file_name = metadata.get("file_name", "")
        file_type = metadata.get("file_type", "")

        # Check for markdown extensions
        markdown_extensions = {".md", ".markdown", ".mdown", ".mkd", ".mdx"}
        for ext in markdown_extensions:
            if file_path.endswith(ext) or file_name.endswith(ext) or file_type == ext:
                return True

        return False

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Synchronous chunking using markdown-aware splitting.

        Args:
            text: The text to chunk
            doc_id: Unique identifier for the document
            metadata: Optional metadata to include with chunks

        Returns:
            List of ChunkResult objects
        """
        if not text.strip():
            return []

        # Check if this is actually markdown content
        is_markdown = self._is_markdown_file(metadata) or self._has_markdown_headers(text)

        if not is_markdown:
            logger.warning(
                f"Document {doc_id} doesn't appear to be markdown. Consider using recursive chunker instead."
            )

        # Create a temporary document
        doc = Document(text=text, metadata=metadata or {})

        try:
            # Get nodes using markdown parser
            nodes = self.splitter.get_nodes_from_documents([doc])
        except Exception as e:
            logger.error(f"Error parsing markdown: {e}. Falling back to basic splitting.")
            # If markdown parsing fails, create a single chunk
            return [
                self._create_chunk_result(
                    doc_id=doc_id,
                    chunk_index=0,
                    text=text,
                    start_offset=0,
                    end_offset=len(text),
                    metadata=metadata,
                )
            ]

        # Convert to ChunkResult
        results = []
        for idx, node in enumerate(nodes):
            # Extract section information from node metadata
            chunk_metadata = metadata.copy() if metadata else {}

            # Add markdown-specific metadata
            if hasattr(node, "metadata"):
                node_meta = node.metadata
                # Extract header level if available
                if "Header_1" in node_meta:
                    chunk_metadata["section"] = node_meta["Header_1"]
                if "Header_2" in node_meta:
                    chunk_metadata["subsection"] = node_meta["Header_2"]

            result = self._create_chunk_result(
                doc_id=doc_id,
                chunk_index=idx,
                text=node.get_content(),
                start_offset=(
                    node.start_char_idx if hasattr(node, "start_char_idx") and node.start_char_idx is not None else 0
                ),
                end_offset=(
                    node.end_char_idx
                    if hasattr(node, "end_char_idx") and node.end_char_idx is not None
                    else len(node.get_content())
                ),
                metadata=chunk_metadata,
            )
            results.append(result)

        logger.debug(f"Created {len(results)} chunks from {len(text)} characters of markdown content")
        return results

    async def chunk_text_async(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Asynchronous chunking using markdown-aware splitting.

        Args:
            text: The text to chunk
            doc_id: Unique identifier for the document
            metadata: Optional metadata to include with chunks

        Returns:
            List of ChunkResult objects
        """
        if not text.strip():
            return []

        # Run synchronous method in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.chunk_text,
            text,
            doc_id,
            metadata,
        )

    def validate_config(self, config: dict[str, Any]) -> bool:  # noqa: ARG002
        """Validate markdown chunker configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        # MarkdownNodeParser doesn't have many configurable parameters
        # Future versions might support max chunk size, etc.
        return True

    def estimate_chunks(self, text_length: int, config: dict[str, Any]) -> int:  # noqa: ARG002
        """Estimate number of chunks for capacity planning.

        Args:
            text_length: Length of text in characters
            config: Configuration parameters

        Returns:
            Estimated number of chunks
        """
        # Markdown chunking depends on structure, not just size
        # Estimate based on typical markdown document structure

        # More granular estimation:
        # - Headers typically create new chunks
        # - Code blocks often create separate chunks
        # - Lists and paragraphs may be grouped

        # Assume average chunk size of ~800 characters for markdown
        # (smaller than plain text due to structure preservation)
        estimated_chunks = max(1, text_length // 800)

        # Add buffer for additional structure-based splits
        estimated_chunks = int(estimated_chunks * 1.2)

        return max(1, estimated_chunks)
