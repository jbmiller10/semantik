#!/usr/bin/env python3
"""
Hierarchical chunking strategy with parent-child relationships.

This module implements hierarchical chunking using LlamaIndex's HierarchicalNodeParser,
which creates chunks at multiple levels with parent-child relationships.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from packages.shared.text_processing.base_chunker import BaseChunker, ChunkResult

# Conditional imports for CI compatibility
try:
    from llama_index.core import Document
    from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes, get_root_nodes
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    # Fallback for CI environments
    Document = None
    HierarchicalNodeParser = None
    get_leaf_nodes = None
    get_root_nodes = None
    LLAMA_INDEX_AVAILABLE = False

logger = logging.getLogger(__name__)


class HierarchicalChunker(BaseChunker):
    """Multi-level chunking for better context preservation and retrieval."""

    def __init__(
        self,
        chunk_sizes: Optional[List[int]] = None,
        chunk_overlap: int = 20,
        include_prev_next_rel: bool = True,
        include_metadata: bool = True,
    ):
        """Initialize hierarchical chunker with multiple chunk size levels.

        Args:
            chunk_sizes: List of chunk sizes from largest to smallest (default: [2048, 512, 128])
            chunk_overlap: Overlap between chunks at each level
            include_prev_next_rel: Include previous/next relationships
            include_metadata: Include metadata in nodes
        """
        # Set defaults
        self.chunk_sizes = chunk_sizes or [2048, 512, 128]
        self.chunk_overlap = chunk_overlap
        self.include_prev_next_rel = include_prev_next_rel
        self.include_metadata = include_metadata

        # Validate chunk sizes are in descending order
        if self.chunk_sizes != sorted(self.chunk_sizes, reverse=True):
            logger.warning("Chunk sizes should be in descending order, sorting automatically")
            self.chunk_sizes = sorted(self.chunk_sizes, reverse=True)

        if len(self.chunk_sizes) < 2:
            raise ValueError("Hierarchical chunking requires at least 2 chunk size levels")

        if len(self.chunk_sizes) > 5:
            logger.warning("More than 5 hierarchical levels may impact performance")

        # Initialize the LlamaIndex hierarchical parser
        try:
            if LLAMA_INDEX_AVAILABLE:
                self.splitter = HierarchicalNodeParser.from_defaults(
                    chunk_sizes=self.chunk_sizes,
                    chunk_overlap=self.chunk_overlap,
                    include_prev_next_rel=self.include_prev_next_rel,
                    include_metadata=self.include_metadata,
                )
            else:
                # Fallback for CI environments
                self.splitter = None
                logger.warning("HierarchicalNodeParser unavailable - using fallback mode")
        except Exception as e:
            logger.error(f"Failed to initialize HierarchicalNodeParser: {e}")
            raise ValueError(f"Invalid hierarchical chunker configuration: {e}")

        self.strategy_name = "hierarchical"

        logger.info(
            f"Initialized HierarchicalChunker with chunk_sizes={self.chunk_sizes}, "
            f"overlap={self.chunk_overlap}, levels={len(self.chunk_sizes)}"
        )

    async def chunk_text_async(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ChunkResult]:
        """Asynchronously create hierarchical chunks with parent-child relationships.

        Args:
            text: Text to chunk
            doc_id: Document identifier for chunk IDs
            metadata: Optional metadata to include in chunks

        Returns:
            List of ChunkResult objects with hierarchical relationships
        """
        if not text.strip():
            return []

        try:
            # Create document for LlamaIndex
            doc = Document(text=text, metadata=metadata or {})

            # Run hierarchical parsing in executor to avoid blocking
            loop = asyncio.get_event_loop()
            all_nodes = await loop.run_in_executor(
                None,
                self.splitter.get_nodes_from_documents,
                [doc],
            )

            if not all_nodes:
                logger.warning(f"Hierarchical parsing returned no nodes for document {doc_id}")
                return []

            # Process hierarchical nodes into ChunkResults
            results = self._process_hierarchical_nodes(all_nodes, doc_id, metadata)

            logger.info(
                f"Hierarchical chunking created {len(results)} chunks "
                f"({len([r for r in results if not r.metadata.get('is_parent_chunk', False)])} leaf nodes) "
                f"for document {doc_id} ({len(text)} characters)"
            )

            return results

        except Exception as e:
            logger.error(f"Hierarchical chunking failed for document {doc_id}: {e}")
            # Fallback to single-level chunking
            return await self._fallback_chunking(text, doc_id, metadata)

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ChunkResult]:
        """Synchronously create hierarchical chunks with parent-child relationships.

        Args:
            text: Text to chunk
            doc_id: Document identifier for chunk IDs
            metadata: Optional metadata to include in chunks

        Returns:
            List of ChunkResult objects with hierarchical relationships
        """
        if not text.strip():
            return []

        try:
            # Create document for LlamaIndex
            doc = Document(text=text, metadata=metadata or {})

            # Process with hierarchical parser
            all_nodes = self.splitter.get_nodes_from_documents([doc])

            if not all_nodes:
                logger.warning(f"Hierarchical parsing returned no nodes for document {doc_id}")
                return []

            # Process hierarchical nodes into ChunkResults
            results = self._process_hierarchical_nodes(all_nodes, doc_id, metadata)

            logger.info(
                f"Hierarchical chunking created {len(results)} chunks "
                f"({len([r for r in results if not r.metadata.get('is_parent_chunk', False)])} leaf nodes) "
                f"for document {doc_id} ({len(text)} characters)"
            )

            return results

        except Exception as e:
            logger.error(f"Hierarchical chunking failed for document {doc_id}: {e}")
            # Fallback to single-level chunking
            return self._fallback_chunking_sync(text, doc_id, metadata)

    def _process_hierarchical_nodes(
        self,
        all_nodes: List[Any],
        doc_id: str,
        metadata: Optional[Dict[str, Any]],
    ) -> List[ChunkResult]:
        """Process LlamaIndex hierarchical nodes into ChunkResult objects.

        Args:
            all_nodes: All nodes from hierarchical parsing
            doc_id: Document identifier
            metadata: Base metadata

        Returns:
            List of ChunkResult objects with proper hierarchy metadata
        """
        results = []
        base_metadata = metadata or {}

        # Separate leaf and parent nodes
        leaf_nodes = get_leaf_nodes(all_nodes)
        root_nodes = get_root_nodes(all_nodes)

        # Build node relationships mapping
        node_relationships = self._build_node_relationships(all_nodes)

        # Process leaf nodes (primary chunks for retrieval)
        leaf_chunk_idx = 0
        for node in leaf_nodes:
            # Get hierarchy information
            hierarchy_info = self._get_hierarchy_info(node, node_relationships)

            # Calculate character offsets
            start_offset = node.start_char_idx if node.start_char_idx is not None else 0
            end_offset = node.end_char_idx if node.end_char_idx is not None else start_offset + len(node.text)

            # Build comprehensive metadata
            chunk_metadata = {
                **base_metadata,
                "strategy": self.strategy_name,
                "chunk_type": "leaf",
                "chunk_level": hierarchy_info["level"],
                "chunk_size_target": self._get_target_chunk_size(len(node.text)),
                "parent_node_id": hierarchy_info.get("parent_id"),
                "child_node_ids": hierarchy_info.get("child_ids", []),
                "hierarchy_path": hierarchy_info.get("path", []),
                "chunk_index": leaf_chunk_idx,
                "total_leaf_chunks": len(leaf_nodes),
            }

            # Add LlamaIndex metadata if available
            if hasattr(node, "metadata") and node.metadata:
                chunk_metadata.update(node.metadata)

            results.append(
                ChunkResult(
                    chunk_id=f"{doc_id}_leaf_{leaf_chunk_idx:04d}",
                    text=node.text,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    metadata=chunk_metadata,
                )
            )
            leaf_chunk_idx += 1

        # Process parent nodes (context chunks for auto-merging)
        parent_chunk_idx = 0
        for node in all_nodes:
            if node in leaf_nodes:
                continue  # Skip leaf nodes, already processed

            # Get hierarchy information
            hierarchy_info = self._get_hierarchy_info(node, node_relationships)

            # Calculate character offsets
            start_offset = node.start_char_idx if node.start_char_idx is not None else 0
            end_offset = node.end_char_idx if node.end_char_idx is not None else start_offset + len(node.text)

            # Build parent chunk metadata
            chunk_metadata = {
                **base_metadata,
                "strategy": self.strategy_name,
                "chunk_type": "parent",
                "chunk_level": hierarchy_info["level"],
                "chunk_size_target": self._get_target_chunk_size(len(node.text)),
                "is_parent_chunk": True,
                "child_node_ids": hierarchy_info.get("child_ids", []),
                "parent_node_id": hierarchy_info.get("parent_id"),
                "hierarchy_path": hierarchy_info.get("path", []),
                "chunk_index": parent_chunk_idx,
            }

            # Add LlamaIndex metadata if available
            if hasattr(node, "metadata") and node.metadata:
                chunk_metadata.update(node.metadata)

            results.append(
                ChunkResult(
                    chunk_id=f"{doc_id}_parent_{parent_chunk_idx:04d}",
                    text=node.text,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    metadata=chunk_metadata,
                )
            )
            parent_chunk_idx += 1

        return results

    def _build_node_relationships(self, all_nodes: List[Any]) -> Dict[str, Dict[str, Any]]:
        """Build mapping of node relationships for hierarchy tracking.

        Args:
            all_nodes: All nodes from hierarchical parsing

        Returns:
            Dictionary mapping node IDs to relationship information
        """
        relationships = {}

        for node in all_nodes:
            node_id = getattr(node, "node_id", str(id(node)))
            relationships[node_id] = {
                "node": node,
                "parent_ids": [],
                "child_ids": [],
                "level": 0,
            }

        # Analyze relationships (simplified approach)
        # In practice, LlamaIndex provides relationship information
        for node in all_nodes:
            node_id = getattr(node, "node_id", str(id(node)))

            # Get relationships from node if available
            if hasattr(node, "relationships"):
                for rel_type, related_nodes in node.relationships.items():
                    if rel_type.value == "PARENT":
                        if isinstance(related_nodes, list):
                            relationships[node_id]["parent_ids"].extend([n.node_id for n in related_nodes])
                        else:
                            relationships[node_id]["parent_ids"].append(related_nodes.node_id)
                    elif rel_type.value == "CHILD":
                        if isinstance(related_nodes, list):
                            relationships[node_id]["child_ids"].extend([n.node_id for n in related_nodes])
                        else:
                            relationships[node_id]["child_ids"].append(related_nodes.node_id)

        # Calculate levels based on text length (larger = higher level)
        nodes_by_size = sorted(all_nodes, key=lambda n: len(n.text), reverse=True)
        for level, node in enumerate(nodes_by_size):
            node_id = getattr(node, "node_id", str(id(node)))
            relationships[node_id]["level"] = self._calculate_hierarchy_level(len(node.text))

        return relationships

    def _get_hierarchy_info(self, node: Any, relationships: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Get hierarchy information for a specific node.

        Args:
            node: LlamaIndex node
            relationships: Node relationships mapping

        Returns:
            Dictionary with hierarchy information
        """
        node_id = getattr(node, "node_id", str(id(node)))
        rel_info = relationships.get(node_id, {})

        return {
            "level": rel_info.get("level", 0),
            "parent_id": rel_info.get("parent_ids", [None])[0],
            "child_ids": rel_info.get("child_ids", []),
            "path": self._build_hierarchy_path(node_id, relationships),
        }

    def _build_hierarchy_path(self, node_id: str, relationships: Dict[str, Dict[str, Any]]) -> List[str]:
        """Build the full hierarchy path for a node.

        Args:
            node_id: Node identifier
            relationships: Node relationships mapping

        Returns:
            List of node IDs from root to current node
        """
        path = [node_id]
        current_id = node_id

        # Walk up the hierarchy
        max_depth = 10  # Prevent infinite loops
        depth = 0
        while depth < max_depth:
            rel_info = relationships.get(current_id, {})
            parent_ids = rel_info.get("parent_ids", [])

            if not parent_ids or parent_ids[0] is None:
                break

            parent_id = parent_ids[0]
            path.insert(0, parent_id)
            current_id = parent_id
            depth += 1

        return path

    def _calculate_hierarchy_level(self, text_length: int) -> int:
        """Calculate hierarchy level based on text length.

        Args:
            text_length: Length of text in characters

        Returns:
            Hierarchy level (0 = smallest chunks, higher = larger chunks)
        """
        # Map text length to chunk size level
        for level, chunk_size in enumerate(self.chunk_sizes):
            # Rough estimate: 4 characters per token
            if text_length <= chunk_size * 4:
                return len(self.chunk_sizes) - 1 - level

        # Default to highest level for very large chunks
        return 0

    def _get_target_chunk_size(self, actual_size: int) -> int:
        """Get the target chunk size that best matches the actual size.

        Args:
            actual_size: Actual chunk size in characters

        Returns:
            Target chunk size from configuration
        """
        # Find closest target size
        closest_size = self.chunk_sizes[-1]  # Start with smallest
        min_diff = abs(actual_size - closest_size * 4)  # Rough char-to-token conversion

        for size in self.chunk_sizes:
            diff = abs(actual_size - size * 4)
            if diff < min_diff:
                min_diff = diff
                closest_size = size

        return closest_size

    async def _fallback_chunking(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]],
    ) -> List[ChunkResult]:
        """Fallback to single-level chunking on hierarchical failure.

        Args:
            text: Text to chunk
            doc_id: Document identifier
            metadata: Metadata to preserve

        Returns:
            List of ChunkResult from single-level chunking
        """
        logger.info(f"Falling back to recursive chunking for document {doc_id}")

        try:
            # Import here to avoid circular import
            from packages.shared.text_processing.strategies.recursive_chunker import RecursiveChunker

            # Use the largest chunk size as fallback
            chunk_size = self.chunk_sizes[0] if self.chunk_sizes else 1024
            fallback = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=self.chunk_overlap)
            chunks = await fallback.chunk_text_async(text, doc_id, metadata)

            # Update metadata to indicate fallback was used
            for chunk in chunks:
                chunk.metadata.update(
                    {
                        "original_strategy": self.strategy_name,
                        "fallback_strategy": "recursive",
                        "fallback_reason": "hierarchical_chunking_failed",
                        "target_chunk_sizes": self.chunk_sizes,
                    }
                )

            return chunks

        except Exception as e:
            logger.error(f"Fallback chunking also failed for document {doc_id}: {e}")
            # Emergency fallback: single chunk
            return [
                ChunkResult(
                    chunk_id=f"{doc_id}_0000",
                    text=text[:10000],  # Limit to reasonable size
                    start_offset=0,
                    end_offset=min(len(text), 10000),
                    metadata={
                        **(metadata or {}),
                        "strategy": "emergency_fallback",
                        "original_strategy": self.strategy_name,
                        "fallback_reason": "all_chunking_failed",
                    },
                )
            ]

    def _fallback_chunking_sync(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]],
    ) -> List[ChunkResult]:
        """Synchronous fallback to single-level chunking."""
        logger.info(f"Falling back to recursive chunking for document {doc_id}")

        try:
            from packages.shared.text_processing.strategies.recursive_chunker import RecursiveChunker

            chunk_size = self.chunk_sizes[0] if self.chunk_sizes else 1024
            fallback = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=self.chunk_overlap)
            chunks = fallback.chunk_text(text, doc_id, metadata)

            for chunk in chunks:
                chunk.metadata.update(
                    {
                        "original_strategy": self.strategy_name,
                        "fallback_strategy": "recursive",
                        "fallback_reason": "hierarchical_chunking_failed",
                        "target_chunk_sizes": self.chunk_sizes,
                    }
                )

            return chunks

        except Exception as e:
            logger.error(f"Fallback chunking also failed for document {doc_id}: {e}")
            return [
                ChunkResult(
                    chunk_id=f"{doc_id}_0000",
                    text=text[:10000],
                    start_offset=0,
                    end_offset=min(len(text), 10000),
                    metadata={
                        **(metadata or {}),
                        "strategy": "emergency_fallback",
                        "original_strategy": self.strategy_name,
                        "fallback_reason": "all_chunking_failed",
                    },
                )
            ]

    def validate_config(self, params: Dict[str, Any]) -> bool:
        """Validate hierarchical chunking configuration parameters.

        Args:
            params: Configuration parameters to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate chunk_sizes
            chunk_sizes = params.get("chunk_sizes", self.chunk_sizes)
            if not isinstance(chunk_sizes, list):
                logger.error(f"chunk_sizes must be a list, got {type(chunk_sizes)}")
                return False

            if len(chunk_sizes) < 2:
                logger.error(f"chunk_sizes must have at least 2 levels, got {len(chunk_sizes)}")
                return False

            if not all(isinstance(size, int) and size > 0 for size in chunk_sizes):
                logger.error("All chunk_sizes must be positive integers")
                return False

            # Check if in descending order
            if chunk_sizes != sorted(chunk_sizes, reverse=True):
                logger.warning("chunk_sizes should be in descending order")

            # Validate chunk_overlap
            chunk_overlap = params.get("chunk_overlap", self.chunk_overlap)
            if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
                logger.error(f"Invalid chunk_overlap: {chunk_overlap}")
                return False

            # Validate overlap is not too large compared to smallest chunk
            if chunk_overlap >= min(chunk_sizes):
                logger.error(f"chunk_overlap ({chunk_overlap}) must be less than smallest chunk size ({min(chunk_sizes)})")
                return False

            return True

        except Exception as e:
            logger.error(f"Config validation error: {e}")
            return False

    def estimate_chunks(self, text_length: int, params: Dict[str, Any]) -> int:
        """Estimate total number of chunks (leaf + parent) for given text length.

        Args:
            text_length: Length of text in characters
            params: Chunking parameters

        Returns:
            Estimated total number of chunks
        """
        chunk_sizes = params.get("chunk_sizes", self.chunk_sizes)

        # Estimate chunks at each level
        total_estimated = 0
        remaining_text = text_length

        for chunk_size in chunk_sizes:
            # Convert tokens to approximate characters (4 chars per token)
            chunk_size_chars = chunk_size * 4
            level_chunks = max(1, remaining_text // chunk_size_chars)
            total_estimated += level_chunks

            # Each level processes the full text, so don't reduce remaining_text
            # This accounts for hierarchical overlap

        # Add some overhead for the hierarchical structure
        total_estimated = int(total_estimated * 1.2)

        logger.debug(
            f"Estimated {total_estimated} total chunks for {text_length} characters "
            f"across {len(chunk_sizes)} hierarchical levels"
        )

        return max(1, total_estimated)