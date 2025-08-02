#!/usr/bin/env python3
"""
Recursive sentence-based chunking strategy using LlamaIndex.

This module implements smart sentence-aware text splitting using LlamaIndex's
SentenceSplitter. It provides optimized parameters for code files.
"""

import asyncio
import logging
from typing import Any

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from packages.shared.text_processing.base_chunker import BaseChunker, ChunkResult

logger = logging.getLogger(__name__)


class RecursiveChunker(BaseChunker):
    """Recursive sentence-based chunking using LlamaIndex SentenceSplitter."""

    # Code file extensions that get optimized parameters
    CODE_EXTENSIONS = {
        ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp",
        ".cs", ".rb", ".go", ".rs", ".php", ".swift", ".kt", ".scala",
        ".r", ".m", ".mm", ".lua", ".dart", ".jsx", ".tsx", ".vue",
        ".sql", ".sh", ".bash", ".zsh", ".ps1", ".yaml", ".yml",
        ".json", ".xml", ".html", ".css", ".scss", ".sass", ".less",
    }

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        **kwargs: Any
    ) -> None:
        """Initialize RecursiveChunker.

        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)

        # Validate parameters
        if chunk_overlap >= chunk_size:
            logger.warning(
                f"chunk_overlap ({chunk_overlap}) >= chunk_size ({chunk_size}), "
                f"setting overlap to chunk_size/4"
            )
            chunk_overlap = chunk_size // 4

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize LlamaIndex splitter
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Optimized splitter for code files
        self.code_splitter = SentenceSplitter(
            chunk_size=400,  # Smaller chunks for code
            chunk_overlap=50,  # Less overlap for efficiency
        )

        logger.info(
            f"Initialized RecursiveChunker with chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )

    def _is_code_file(self, metadata: dict[str, Any] | None) -> bool:
        """Check if the document is a code file based on metadata."""
        if not metadata:
            return False

        # Check file extension
        file_path = metadata.get("file_path", "")
        file_name = metadata.get("file_name", "")
        file_type = metadata.get("file_type", "")

        # Check against code extensions
        for ext in self.CODE_EXTENSIONS:
            if file_path.endswith(ext) or file_name.endswith(ext) or file_type == ext:
                return True

        return False

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Synchronous chunking using sentence-based splitting.

        Args:
            text: The text to chunk
            doc_id: Unique identifier for the document
            metadata: Optional metadata to include with chunks

        Returns:
            List of ChunkResult objects
        """
        if not text.strip():
            return []

        # Choose appropriate splitter based on file type
        is_code = self._is_code_file(metadata)
        splitter = self.code_splitter if is_code else self.splitter

        # Create a temporary document
        doc = Document(text=text, metadata=metadata or {})

        # Get nodes using sentence splitter
        nodes = splitter.get_nodes_from_documents([doc])

        # Convert to ChunkResult
        results = []
        for idx, node in enumerate(nodes):
            # Add code file indicator to metadata
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["is_code_file"] = is_code

            result = self._create_chunk_result(
                doc_id=doc_id,
                chunk_index=idx,
                text=node.get_content(),
                start_offset=node.start_char_idx if hasattr(node, 'start_char_idx') and node.start_char_idx is not None else 0,
                end_offset=node.end_char_idx if hasattr(node, 'end_char_idx') and node.end_char_idx is not None else len(node.get_content()),
                metadata=chunk_metadata,
            )
            results.append(result)

        logger.debug(
            f"Created {len(results)} chunks from {len(text)} characters "
            f"(code_file={is_code})"
        )
        return results

    async def chunk_text_async(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Asynchronous chunking using sentence-based splitting.

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

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate recursive chunker configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            chunk_size = config.get("chunk_size", self.chunk_size)
            chunk_overlap = config.get("chunk_overlap", self.chunk_overlap)

            # Validate chunk size
            if not isinstance(chunk_size, int) or chunk_size <= 0:
                logger.error(f"Invalid chunk_size: {chunk_size}")
                return False

            # Validate chunk overlap
            if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
                logger.error(f"Invalid chunk_overlap: {chunk_overlap}")
                return False

            # Overlap should be less than chunk size
            if chunk_overlap >= chunk_size:
                logger.error(
                    f"chunk_overlap ({chunk_overlap}) must be less than "
                    f"chunk_size ({chunk_size})"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating config: {e}")
            return False

    def estimate_chunks(self, text_length: int, config: dict[str, Any]) -> int:
        """Estimate number of chunks for capacity planning.

        Args:
            text_length: Length of text in characters
            config: Configuration parameters

        Returns:
            Estimated number of chunks
        """
        chunk_size = config.get("chunk_size", self.chunk_size)
        chunk_overlap = config.get("chunk_overlap", self.chunk_overlap)

        # Check if it's a code file
        is_code = config.get("is_code_file", False)
        if is_code:
            chunk_size = 400
            chunk_overlap = 50

        # Rough estimate: assume ~4 characters per token
        estimated_tokens = text_length / 4

        if estimated_tokens <= chunk_size:
            return 1

        # Calculate with overlap
        effective_chunk_size = chunk_size - chunk_overlap
        return 1 + max(0, int((estimated_tokens - chunk_size) / effective_chunk_size))
