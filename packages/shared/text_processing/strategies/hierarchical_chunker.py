#!/usr/bin/env python3
"""
Hierarchical chunking strategy using LlamaIndex HierarchicalNodeParser.

This module implements multi-level chunking that creates parent-child relationships
between chunks at different granularities, enabling efficient context retrieval.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes

from packages.shared.text_processing.base_chunker import BaseChunker, ChunkResult

if TYPE_CHECKING:
    from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)


class HierarchicalChunker(BaseChunker):
    """Hierarchical chunking using LlamaIndex HierarchicalNodeParser for multi-level text organization."""

    def __init__(
        self,
        chunk_sizes: list[int] | None = None,
        chunk_overlap: int = 20,
        **kwargs: Any,
    ) -> None:
        """Initialize HierarchicalChunker.

        Args:
            chunk_sizes: List of chunk sizes from largest to smallest.
                         Defaults to [2048, 512, 128] for 3-level hierarchy.
            chunk_overlap: Number of overlapping tokens between chunks at same level
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)

        # Default chunk sizes for 3-level hierarchy
        if chunk_sizes is None:
            chunk_sizes = [2048, 512, 128]

        # Validate chunk sizes
        if not chunk_sizes:
            raise ValueError("chunk_sizes must contain at least one size")

        # Ensure chunk sizes are in descending order
        self.chunk_sizes = sorted(chunk_sizes, reverse=True)

        # Validate that each level is meaningfully smaller than the previous
        for i in range(1, len(self.chunk_sizes)):
            if self.chunk_sizes[i] >= self.chunk_sizes[i - 1]:
                raise ValueError(
                    f"Chunk sizes must be in descending order, but {self.chunk_sizes[i]} >= {self.chunk_sizes[i-1]}"
                )
            # Ensure at least 2x reduction between levels
            if self.chunk_sizes[i] > self.chunk_sizes[i - 1] / 2:
                logger.warning(
                    f"Chunk size {self.chunk_sizes[i]} is more than half of {self.chunk_sizes[i-1]}. "
                    "Consider larger differences between hierarchy levels for better performance."
                )

        self.chunk_overlap = chunk_overlap

        # Initialize the hierarchical parser
        self._parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=self.chunk_sizes,
            chunk_overlap=chunk_overlap,
        )

        logger.info(
            f"Initialized HierarchicalChunker with chunk_sizes={self.chunk_sizes}, " f"chunk_overlap={chunk_overlap}"
        )

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Synchronous hierarchical chunking.

        Args:
            text: The text to chunk
            doc_id: Unique identifier for the document
            metadata: Optional metadata to include with chunks

        Returns:
            List of ChunkResult objects with parent-child relationships
        """
        if not text.strip():
            return []

        try:
            # Track performance
            start_time = time.time()

            # Create document
            doc = Document(text=text, metadata=metadata or {})

            # Perform hierarchical chunking
            nodes = self._parser.get_nodes_from_documents([doc])

            # Get only the leaf nodes (smallest chunks) for the main results
            leaf_nodes = get_leaf_nodes(nodes)

            # Build a mapping of node IDs to nodes for relationship tracking
            node_map = {node.node_id: node for node in nodes}

            # Convert to ChunkResults, adding hierarchical metadata
            results = []
            chunk_index = 0

            # Process leaf nodes and enrich with hierarchical metadata
            for leaf_node in leaf_nodes:
                # Build hierarchy information
                hierarchy_info = self._build_hierarchy_info(leaf_node, node_map)

                # Calculate offsets (approximate, as HierarchicalNodeParser doesn't provide exact offsets)
                # We'll use a simple character-based estimation
                content = leaf_node.get_content()
                start_offset = text.find(content[:50])  # Find by first 50 chars to be more accurate
                if start_offset == -1:
                    # Fallback to approximate calculation
                    start_offset = chunk_index * (len(text) // len(leaf_nodes))

                end_offset = start_offset + len(content)

                # Prepare chunk metadata
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata.update(
                    {
                        "hierarchy_level": hierarchy_info["level"],
                        "parent_chunk_id": hierarchy_info["parent_id"],
                        "child_chunk_ids": hierarchy_info["child_ids"],
                        "chunk_sizes": self.chunk_sizes,
                        "node_id": leaf_node.node_id,
                        "is_leaf": True,
                    }
                )

                result = self._create_chunk_result(
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    text=content,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    metadata=chunk_metadata,
                )
                results.append(result)
                chunk_index += 1

            # Optionally, also include parent chunks for context retrieval
            # This allows retrieval systems to fetch broader context when needed
            parent_nodes = [node for node in nodes if node not in leaf_nodes]
            for parent_node in parent_nodes:
                hierarchy_info = self._build_hierarchy_info(parent_node, node_map)
                content = parent_node.get_content()

                # Find approximate offset
                start_offset = text.find(content[:50])
                if start_offset == -1:
                    start_offset = 0

                end_offset = min(start_offset + len(content), len(text))

                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata.update(
                    {
                        "hierarchy_level": hierarchy_info["level"],
                        "parent_chunk_id": hierarchy_info["parent_id"],
                        "child_chunk_ids": hierarchy_info["child_ids"],
                        "chunk_sizes": self.chunk_sizes,
                        "node_id": parent_node.node_id,
                        "is_leaf": False,
                    }
                )

                # Use a special chunk ID format for parent chunks
                parent_chunk_id = f"{doc_id}_parent_{chunk_index:04d}"
                result = ChunkResult(
                    chunk_id=parent_chunk_id,
                    text=content.strip(),
                    start_offset=start_offset,
                    end_offset=end_offset,
                    metadata=chunk_metadata,
                )
                results.append(result)
                chunk_index += 1

            # Log performance metrics
            elapsed_time = time.time() - start_time
            chunks_per_sec = len(results) / elapsed_time if elapsed_time > 0 else 0
            logger.debug(
                f"Hierarchical chunking completed: {len(results)} total chunks "
                f"({len(leaf_nodes)} leaf chunks) in {elapsed_time:.2f}s "
                f"({chunks_per_sec:.1f} chunks/sec)"
            )

            return results

        except Exception as e:
            logger.error(f"Error in hierarchical chunking: {e}")
            # Fallback to character-based chunking
            logger.warning("Falling back to character-based chunking due to hierarchical parsing error")
            from .character_chunker import CharacterChunker

            fallback_chunker = CharacterChunker(
                chunk_size=self.chunk_sizes[-1],  # Use smallest chunk size
                chunk_overlap=self.chunk_overlap,
            )
            return fallback_chunker.chunk_text(text, doc_id, metadata)

    async def chunk_text_async(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Asynchronous hierarchical chunking.

        Args:
            text: The text to chunk
            doc_id: Unique identifier for the document
            metadata: Optional metadata to include with chunks

        Returns:
            List of ChunkResult objects with parent-child relationships
        """
        if not text.strip():
            return []

        # Run synchronous chunking in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.chunk_text,
            text,
            doc_id,
            metadata,
        )

    def _build_hierarchy_info(
        self,
        node: BaseNode,
        node_map: dict[str, BaseNode],  # noqa: ARG002
    ) -> dict[str, Any]:
        """Build hierarchy information for a node.

        Args:
            node: The node to analyze
            node_map: Mapping of node IDs to nodes

        Returns:
            Dictionary with hierarchy information
        """
        hierarchy_info = {
            "level": 0,  # Will be determined by position in hierarchy
            "parent_id": None,
            "child_ids": [],
        }

        # Determine parent relationship
        if hasattr(node, "relationships") and node.relationships:
            # Check for parent relationship
            parent_rel = node.relationships.get("1")  # LlamaIndex uses "1" for parent
            if parent_rel and parent_rel.node_id:
                hierarchy_info["parent_id"] = parent_rel.node_id

            # Check for child relationships
            child_rel = node.relationships.get("2")  # LlamaIndex uses "2" for child
            if child_rel and child_rel.node_id:
                # Single child
                hierarchy_info["child_ids"] = [child_rel.node_id]
            elif hasattr(child_rel, "node_ids") and child_rel.node_ids:
                # Multiple children
                hierarchy_info["child_ids"] = list(child_rel.node_ids)

        # Determine hierarchy level based on chunk size used
        # This is an approximation based on content length
        content_length = len(node.get_content())
        for level, chunk_size in enumerate(self.chunk_sizes):
            # Approximate token count (4 chars per token)
            approx_tokens = content_length / 4
            if approx_tokens <= chunk_size * 1.5:  # Allow some flexibility
                hierarchy_info["level"] = level
                break

        return hierarchy_info

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate hierarchical chunker configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            chunk_sizes = config.get("chunk_sizes", self.chunk_sizes)
            chunk_overlap = config.get("chunk_overlap", self.chunk_overlap)

            # Validate chunk sizes
            if not isinstance(chunk_sizes, list) or not chunk_sizes:
                logger.error(f"Invalid chunk_sizes: {chunk_sizes}")
                return False

            # All sizes must be positive integers
            for size in chunk_sizes:
                if not isinstance(size, int) or size <= 0:
                    logger.error(f"Invalid chunk size: {size}")
                    return False

            # Sizes should be in descending order
            sorted_sizes = sorted(chunk_sizes, reverse=True)
            if chunk_sizes != sorted_sizes:
                logger.warning("chunk_sizes should be in descending order")

            # Validate chunk overlap
            if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
                logger.error(f"Invalid chunk_overlap: {chunk_overlap}")
                return False

            # Overlap should be less than smallest chunk size
            if chunk_overlap >= min(chunk_sizes):
                logger.error(
                    f"chunk_overlap ({chunk_overlap}) must be less than " f"smallest chunk size ({min(chunk_sizes)})"
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
            Estimated number of chunks (including all hierarchy levels)
        """
        chunk_sizes = config.get("chunk_sizes", self.chunk_sizes)
        chunk_overlap = config.get("chunk_overlap", self.chunk_overlap)

        # Estimate tokens (4 chars per token approximation)
        estimated_tokens = text_length / 4

        total_chunks = 0

        # Calculate chunks at each level
        for chunk_size in chunk_sizes:
            if estimated_tokens <= chunk_size:
                # Document fits in single chunk at this level
                total_chunks += 1
            else:
                # Calculate number of chunks with overlap
                effective_chunk_size = chunk_size - chunk_overlap
                level_chunks = 1 + max(0, int((estimated_tokens - chunk_size) / effective_chunk_size))
                total_chunks += level_chunks

        return total_chunks
