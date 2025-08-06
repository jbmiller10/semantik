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
    from collections.abc import AsyncIterator, Iterator

    from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

# Security constants
MAX_CHUNK_SIZE = 10000  # Maximum allowed chunk size to prevent memory exhaustion
MAX_HIERARCHY_DEPTH = 5  # Maximum hierarchy levels to prevent stack overflow
MAX_TEXT_LENGTH = 5_000_000  # 5MB text limit to prevent DOS
STREAMING_CHUNK_SIZE = 1_000_000  # 1MB chunks for streaming processing


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

        # Security validation: Prevent excessive hierarchy depth
        if len(chunk_sizes) > MAX_HIERARCHY_DEPTH:
            raise ValueError(f"Too many hierarchy levels: {len(chunk_sizes)} > {MAX_HIERARCHY_DEPTH}")

        # Security validation: Prevent excessive chunk sizes
        for size in chunk_sizes:
            if size > MAX_CHUNK_SIZE:
                raise ValueError(f"Chunk size {size} exceeds maximum allowed size of {MAX_CHUNK_SIZE}")
            if size <= 0:
                raise ValueError(f"Invalid chunk size: {size}. Must be positive.")

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
            f"Initialized HierarchicalChunker with chunk_sizes={self.chunk_sizes}, chunk_overlap={chunk_overlap}"
        )

    def _build_offset_map(
        self,
        text: str,
        nodes: list[BaseNode],
    ) -> dict[str, tuple[int, int]]:
        """Build efficient offset map for all nodes.

        This method avoids O(n²) complexity by using a single pass algorithm
        to match node content to text positions.

        Args:
            text: Original text
            nodes: List of nodes to map

        Returns:
            Dictionary mapping node_id to (start_offset, end_offset) tuples
        """
        offset_map: dict[str, tuple[int, int]] = {}

        # First, check if nodes have built-in offset information
        has_offsets = all(hasattr(node, "start_char_idx") and hasattr(node, "end_char_idx") for node in nodes)

        if has_offsets:
            # Use the provided offsets directly
            for node in nodes:
                start = getattr(node, "start_char_idx", 0)
                end = getattr(node, "end_char_idx", start + len(node.get_content()))
                offset_map[node.node_id] = (start, end)
            return offset_map

        # Otherwise, calculate offsets ourselves
        # Sort nodes by their content length (longest first) to match larger chunks first
        sorted_nodes = sorted(nodes, key=lambda n: len(n.get_content()), reverse=True)

        # Track assigned ranges instead of individual positions for memory efficiency
        assigned_ranges: list[tuple[int, int]] = []

        for node in sorted_nodes:
            content = node.get_content()

            # For matching, we'll use the exact content without stripping
            # to maintain accurate offsets
            if not content:
                continue

            # Try to find exact match first
            found = False
            search_start = 0

            while search_start < len(text):
                pos = text.find(content, search_start)
                if pos == -1:
                    # If exact match fails, try with stripped content
                    stripped_content = content.strip()
                    if stripped_content and stripped_content != content:
                        pos = text.find(stripped_content, search_start)
                        if pos != -1:
                            # Adjust for stripping
                            content = stripped_content

                if pos == -1:
                    break

                # Check if this position overlaps with already assigned ranges
                end_pos = pos + len(content)
                overlaps = any(not (end_pos <= start or pos >= end) for start, end in assigned_ranges)

                if not overlaps:
                    # Found a non-overlapping position
                    offset_map[node.node_id] = (pos, end_pos)
                    # Add this range to assigned ranges
                    assigned_ranges.append((pos, end_pos))
                    # Keep ranges sorted for potential optimizations
                    assigned_ranges.sort(key=lambda x: x[0])
                    found = True
                    break

                search_start = pos + 1

            if not found:
                # Fallback: estimate based on node relationships and hierarchy
                offset_map[node.node_id] = self._estimate_node_offset(node, nodes, text, offset_map)

        return offset_map

    def _estimate_node_offset(
        self,
        node: BaseNode,
        all_nodes: list[BaseNode],
        text: str,
        existing_offsets: dict[str, tuple[int, int]],
    ) -> tuple[int, int]:
        """Estimate node offset when exact match is not found.

        Args:
            node: Node to estimate offset for
            all_nodes: All nodes in the hierarchy
            text: Original text
            existing_offsets: Already calculated offsets

        Returns:
            Tuple of (start_offset, end_offset)
        """
        content_length = len(node.get_content())

        # Check if this is a parent node
        if hasattr(node, "relationships") and node.relationships:
            child_rel = node.relationships.get("2")  # LlamaIndex uses "2" for child relationship
            if child_rel:
                # For parent nodes, use the span of their children
                child_ids = []
                if hasattr(child_rel, "node_id") and child_rel.node_id:
                    child_ids = [child_rel.node_id]
                elif hasattr(child_rel, "node_ids") and child_rel.node_ids:
                    child_ids = list(child_rel.node_ids)

                if child_ids:
                    # Get min start and max end from children
                    child_offsets = [
                        existing_offsets.get(child_id, (0, 0)) for child_id in child_ids if child_id in existing_offsets
                    ]
                    if child_offsets:
                        start_offset = min(offset[0] for offset in child_offsets)
                        end_offset = max(offset[1] for offset in child_offsets)
                        return (start_offset, end_offset)

        # Fallback: proportional estimation
        # This is only used when all other methods fail
        node_idx = next((i for i, n in enumerate(all_nodes) if n.node_id == node.node_id), 0)
        proportion = node_idx / max(len(all_nodes), 1)
        start_offset = int(proportion * len(text))
        end_offset = min(start_offset + content_length, len(text))

        return (start_offset, end_offset)

    def chunk_text_stream(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
        include_parents: bool = False,
    ) -> Iterator[ChunkResult]:
        """Stream chunks as they are generated to minimize memory usage.

        This method is ideal for processing very large documents where
        loading all chunks into memory at once would be prohibitive.

        Args:
            text: The text to chunk
            doc_id: Unique identifier for the document
            metadata: Optional metadata to include with chunks
            include_parents: Whether to include parent chunks (default: False)

        Yields:
            ChunkResult objects as they are generated
        """
        if not text.strip():
            return

        # Security validation
        if len(text) > MAX_TEXT_LENGTH:
            logger.warning("Text exceeds maximum length limit")
            raise ValueError("Text too large to process")

        try:
            # Process text in smaller segments for very large documents
            if len(text) > STREAMING_CHUNK_SIZE:
                # For very large texts, process in segments
                segments = []
                for i in range(0, len(text), STREAMING_CHUNK_SIZE):
                    segment = text[i : i + STREAMING_CHUNK_SIZE]
                    segments.append(segment)

                # Process each segment
                all_nodes = []
                segment_offset = 0

                for segment in segments:
                    doc = Document(text=segment, metadata=metadata or {})
                    segment_nodes = self._parser.get_nodes_from_documents([doc])

                    # Adjust node offsets for this segment
                    for node in segment_nodes:
                        # Store original segment offset in metadata
                        if node.metadata is None:
                            node.metadata = {}
                        node.metadata["_segment_offset"] = segment_offset

                    all_nodes.extend(segment_nodes)
                    segment_offset += len(segment)

                nodes = all_nodes
            else:
                # Process normally for smaller texts
                doc = Document(text=text, metadata=metadata or {})
                nodes = self._parser.get_nodes_from_documents([doc])

            # Build offset map
            offset_map = self._build_offset_map(text, nodes)

            # Get leaf nodes
            leaf_nodes = get_leaf_nodes(nodes)
            node_map = {node.node_id: node for node in nodes}

            chunk_index = 0

            # Stream leaf nodes first
            for leaf_node in leaf_nodes:
                hierarchy_info = self._build_hierarchy_info(leaf_node, node_map)
                content = leaf_node.get_content()
                start_offset, end_offset = offset_map.get(leaf_node.node_id, (0, len(content)))

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
                yield result
                chunk_index += 1

            # Stream parent chunks if requested
            if include_parents:
                parent_nodes = [node for node in nodes if node not in leaf_nodes]
                for parent_node in parent_nodes:
                    hierarchy_info = self._build_hierarchy_info(parent_node, node_map)
                    content = parent_node.get_content()
                    start_offset, end_offset = offset_map.get(parent_node.node_id, (0, len(content)))

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

                    parent_chunk_id = f"{doc_id}_parent_{chunk_index:04d}"
                    result = ChunkResult(
                        chunk_id=parent_chunk_id,
                        text=content.strip(),
                        start_offset=start_offset,
                        end_offset=end_offset,
                        metadata=chunk_metadata,
                    )
                    yield result
                    chunk_index += 1

        except Exception as e:
            logger.error(f"Streaming hierarchical chunking failed for document {doc_id}")
            logger.debug(f"Internal error details: {e}")
            # Fall back to simple chunking
            logger.warning("Using fallback chunking strategy")
            from packages.shared.text_processing.chunking_factory import ChunkingFactory

            fallback_chunker = ChunkingFactory.create_chunker(
                {
                    "strategy": "character",
                    "params": {
                        "chunk_size": self.chunk_sizes[-1],
                        "chunk_overlap": self.chunk_overlap,
                    },
                }
            )
            yield from fallback_chunker.chunk_text(text, doc_id, metadata)

    async def chunk_text_stream_async(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
        include_parents: bool = False,
    ) -> AsyncIterator[ChunkResult]:
        """Asynchronous streaming chunking.

        Args:
            text: The text to chunk
            doc_id: Unique identifier for the document
            metadata: Optional metadata to include with chunks
            include_parents: Whether to include parent chunks

        Yields:
            ChunkResult objects as they are generated
        """
        if not text.strip():
            return

        # Run synchronous streaming in executor
        loop = asyncio.get_event_loop()

        # Create a generator function that yields chunks
        def chunk_generator() -> Iterator[ChunkResult]:
            return self.chunk_text_stream(text, doc_id, metadata, include_parents)

        # Yield chunks asynchronously
        for chunk in await loop.run_in_executor(None, list, chunk_generator()):
            yield chunk

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
        include_parents: bool = True,
    ) -> list[ChunkResult]:
        """Synchronous hierarchical chunking.

        Args:
            text: The text to chunk
            doc_id: Unique identifier for the document
            metadata: Optional metadata to include with chunks
            include_parents: Whether to include parent chunks (default: True for backward compatibility)

        Returns:
            List of ChunkResult objects with parent-child relationships
        """
        if not text.strip():
            return []

        # Security validation: Prevent processing of excessively large texts
        if len(text) > MAX_TEXT_LENGTH:
            logger.warning("Text exceeds maximum length limit")
            raise ValueError("Text too large to process")

        try:
            # Track performance
            start_time = time.time()

            # Create document
            doc = Document(text=text, metadata=metadata or {})

            # Perform hierarchical chunking
            nodes = self._parser.get_nodes_from_documents([doc])

            # Build efficient offset map for all nodes (O(n) complexity)
            offset_map = self._build_offset_map(text, nodes)

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

                # Get offsets from pre-calculated map
                content = leaf_node.get_content()
                start_offset, end_offset = offset_map.get(
                    leaf_node.node_id, (0, len(content))  # Fallback if not in map
                )

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
            if include_parents:
                parent_nodes = [node for node in nodes if node not in leaf_nodes]
                for parent_node in parent_nodes:
                    hierarchy_info = self._build_hierarchy_info(parent_node, node_map)
                    content = parent_node.get_content()

                    # Get offsets from pre-calculated map
                    start_offset, end_offset = offset_map.get(
                        parent_node.node_id, (0, len(content))  # Fallback if not in map
                    )

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
            # Security: Log detailed error internally but don't expose to external systems
            logger.error(f"Hierarchical chunking failed for document {doc_id}")
            logger.debug(f"Internal error details: {e}")  # Debug level for sensitive details

            # Generic error message for external consumption
            logger.warning("Using fallback chunking strategy")
            from packages.shared.text_processing.chunking_factory import ChunkingFactory

            fallback_chunker = ChunkingFactory.create_chunker(
                {
                    "strategy": "character",
                    "params": {
                        "chunk_size": self.chunk_sizes[-1],  # Use smallest chunk size
                        "chunk_overlap": self.chunk_overlap,
                    },
                }
            )
            fallback_results = fallback_chunker.chunk_text(text, doc_id, metadata)
            return fallback_results if fallback_results else []

    async def chunk_text_async(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
        include_parents: bool = True,
    ) -> list[ChunkResult]:
        """Asynchronous hierarchical chunking.

        Args:
            text: The text to chunk
            doc_id: Unique identifier for the document
            metadata: Optional metadata to include with chunks
            include_parents: Whether to include parent chunks (default: True for backward compatibility)

        Returns:
            List of ChunkResult objects with parent-child relationships
        """
        if not text.strip():
            return []

        # Security validation: Prevent processing of excessively large texts
        if len(text) > MAX_TEXT_LENGTH:
            logger.warning("Text exceeds maximum length limit")
            raise ValueError("Text too large to process")

        # Run synchronous chunking in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.chunk_text,
            text,
            doc_id,
            metadata,
            include_parents,
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
        hierarchy_info: dict[str, Any] = {
            "level": 0,  # Will be determined by position in hierarchy
            "parent_id": None,
            "child_ids": [],
        }

        # Determine parent relationship
        if hasattr(node, "relationships") and node.relationships:
            # Check for parent relationship
            parent_rel = node.relationships.get("1")  # LlamaIndex uses "1" for parent relationship
            if parent_rel:
                if hasattr(parent_rel, "node_id") and parent_rel.node_id:
                    hierarchy_info["parent_id"] = parent_rel.node_id

            # Check for child relationships
            child_rel = node.relationships.get("2")  # LlamaIndex uses "2" for child relationship
            if child_rel:
                if hasattr(child_rel, "node_id") and child_rel.node_id:
                    # Single child
                    hierarchy_info["child_ids"] = [child_rel.node_id]
                elif hasattr(child_rel, "node_ids"):
                    # Multiple children
                    node_ids = getattr(child_rel, "node_ids", None)
                    if node_ids:
                        hierarchy_info["child_ids"] = list(node_ids)

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

    def get_parent_chunks(
        self,
        text: str,
        doc_id: str,
        leaf_chunks: list[ChunkResult],
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Generate parent chunks on demand for given leaf chunks.

        This method allows lazy generation of parent chunks only when needed,
        helping to reduce memory usage for applications that don't always
        need the full hierarchy.

        Args:
            text: The original text
            doc_id: Document identifier
            leaf_chunks: List of leaf chunks to generate parents for
            metadata: Optional metadata

        Returns:
            List of parent ChunkResult objects
        """
        if not leaf_chunks:
            return []

        try:
            # Extract node IDs from leaf chunks
            leaf_node_ids = {chunk.metadata.get("node_id") for chunk in leaf_chunks if chunk.metadata.get("node_id")}

            if not leaf_node_ids:
                logger.warning("No valid node IDs found in leaf chunks")
                return []

            # Recreate document and get all nodes
            doc = Document(text=text, metadata=metadata or {})
            all_nodes = self._parser.get_nodes_from_documents([doc])

            # Build offset map
            offset_map = self._build_offset_map(text, all_nodes)
            node_map = {node.node_id: node for node in all_nodes}

            # Find parent nodes by checking relationships
            parent_nodes = []
            for node in all_nodes:
                if hasattr(node, "relationships") and node.relationships:
                    child_rel = node.relationships.get("2")  # LlamaIndex uses "2" for child relationship
                    if child_rel:
                        child_ids = []
                        if hasattr(child_rel, "node_id") and child_rel.node_id:
                            child_ids = [child_rel.node_id]
                        elif hasattr(child_rel, "node_ids") and child_rel.node_ids:
                            child_ids = list(child_rel.node_ids)

                        # Check if any of this node's children are in our leaf set
                        if any(child_id in leaf_node_ids for child_id in child_ids):
                            parent_nodes.append(node)

            # Generate parent chunks
            results = []

            for chunk_index, parent_node in enumerate(parent_nodes):
                hierarchy_info = self._build_hierarchy_info(parent_node, node_map)
                content = parent_node.get_content()

                start_offset, end_offset = offset_map.get(parent_node.node_id, (0, len(content)))

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

                parent_chunk_id = f"{doc_id}_parent_lazy_{chunk_index:04d}"
                result = ChunkResult(
                    chunk_id=parent_chunk_id,
                    text=content.strip(),
                    start_offset=start_offset,
                    end_offset=end_offset,
                    metadata=chunk_metadata,
                )
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Failed to generate parent chunks for document {doc_id}")
            logger.debug(f"Internal error details: {e}")
            return []

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
                logger.error("Invalid chunk_sizes configuration")
                return False

            # Security validation: Check hierarchy depth
            if len(chunk_sizes) > MAX_HIERARCHY_DEPTH:
                logger.error("Hierarchy depth exceeds maximum allowed")
                return False

            # All sizes must be positive integers within limits
            for size in chunk_sizes:
                if not isinstance(size, int) or size <= 0:
                    logger.error("Invalid chunk size detected")
                    return False
                if size > MAX_CHUNK_SIZE:
                    logger.error("Chunk size exceeds maximum allowed")
                    return False

            # Sizes should be in descending order
            sorted_sizes = sorted(chunk_sizes, reverse=True)
            if chunk_sizes != sorted_sizes:
                logger.warning("chunk_sizes should be in descending order")

            # Validate chunk overlap
            if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
                logger.error("Invalid chunk_overlap configuration")
                return False

            # Overlap should be less than smallest chunk size
            if chunk_overlap >= min(chunk_sizes):
                logger.error("chunk_overlap must be less than smallest chunk size")
                return False

            return True

        except Exception:
            # Security: Don't expose exception details
            logger.error("Configuration validation failed")
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
