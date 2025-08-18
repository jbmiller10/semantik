#!/usr/bin/env python3
"""
Unified hierarchical chunking strategy.

This module merges the domain-based and LlamaIndex-based hierarchical chunking
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

logger = logging.getLogger(__name__)


class HierarchicalChunkingStrategy(UnifiedChunkingStrategy):
    """
    Unified hierarchical chunking strategy.

    Creates multi-level chunks where larger parent chunks contain
    references to smaller child chunks, maintaining context at
    different levels of detail. Can optionally use LlamaIndex for
    enhanced hierarchical parsing.
    """

    def __init__(self, use_llama_index: bool = False) -> None:
        """
        Initialize the hierarchical chunking strategy.

        Args:
            use_llama_index: Whether to use LlamaIndex implementation
        """
        super().__init__("hierarchical")
        self._use_llama_index = use_llama_index
        self._llama_splitter = None

        if use_llama_index:
            try:
                # Check if LlamaIndex is available
                import importlib.util

                spec = importlib.util.find_spec("llama_index.core.node_parser")
                self._llama_available = spec is not None
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
            from llama_index.core.node_parser import HierarchicalNodeParser

            # Check if chunk_sizes are provided in config
            if hasattr(config, 'additional_params') and 'chunk_sizes' in config.additional_params:
                # Use provided chunk sizes (character-based)
                char_sizes = config.additional_params['chunk_sizes']
                # Convert character sizes to approximate token sizes (divide by 4)
                chunk_sizes = [size // 4 for size in char_sizes]
            else:
                # Calculate chunk sizes for hierarchy levels
                # Default: 3 levels with sizes [2048, 512, 128]
                levels = min(config.hierarchy_levels, 3)

                # Create chunk sizes from largest to smallest
                chunk_sizes = []
                base_size = config.max_tokens
                for i in range(levels):
                    chunk_sizes.append(base_size // (2**i))

            # Ensure overlap is smaller than the smallest chunk size
            smallest_chunk = min(chunk_sizes)
            safe_overlap = min(config.overlap_tokens, smallest_chunk // 2)

            return HierarchicalNodeParser.from_defaults(
                chunk_sizes=chunk_sizes,
                chunk_overlap=safe_overlap,
            )
        except Exception as e:
            logger.error(f"Failed to initialize LlamaIndex hierarchical splitter: {e}")
            return None

    def chunk(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk]:
        """
        Create hierarchical chunks at multiple levels.

        Args:
            content: The text content to chunk
            config: Configuration parameters
            progress_callback: Optional progress callback

        Returns:
            List of chunks at all hierarchy levels
        """
        if not content:
            return []

        # Try LlamaIndex implementation if enabled
        if self._use_llama_index and self._llama_available:
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

    def _chunk_with_llama_index(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk] | None:
        """
        Chunk using LlamaIndex HierarchicalNodeParser.

        Returns None if LlamaIndex is not available or fails.
        """
        try:
            from llama_index.core import Document
            from llama_index.core.node_parser import get_leaf_nodes
            from llama_index.core.schema import NodeRelationship

            # Initialize splitter
            splitter = self._init_llama_splitter(config)
            if not splitter:
                return None

            # Create a temporary document
            doc = Document(text=content)

            # Get nodes using hierarchical parser
            nodes = splitter.get_nodes_from_documents([doc])

            if not nodes:
                return []

            # Get leaf nodes (smallest chunks) for accurate count
            leaf_nodes = get_leaf_nodes(nodes)

            chunks: list[Chunk] = []
            total_chars = len(content)

            # Build a map to track node depths
            node_depths = {}

            # First pass: calculate depths for all nodes
            for node in nodes:
                if hasattr(node, "node_id"):
                    depth = self._calculate_node_depth(node, nodes)
                    node_depths[node.node_id] = depth

            # Find max depth to invert levels (0 should be top level)
            max_depth = max(node_depths.values()) if node_depths else 0

            # Process all nodes (including parent nodes)
            for idx, node in enumerate(nodes):
                chunk_text = node.get_content()

                # Determine hierarchy level (0 = top level, higher = deeper)
                # Invert the depth so that top-level nodes have level 0
                node_id = getattr(node, "node_id", None)
                level = max_depth - node_depths[node_id] if node_id and node_id in node_depths else 0

                # Calculate offsets
                if idx == 0:
                    start_offset = 0
                else:
                    # Find the chunk text in the original content
                    prev_end = chunks[-1].metadata.end_offset if chunks else 0
                    start_offset = content.find(chunk_text, max(0, prev_end - 100))
                    if start_offset == -1:
                        start_offset = prev_end

                end_offset = min(start_offset + len(chunk_text), total_chars)

                # Create chunk metadata
                token_count = self.count_tokens(chunk_text)

                # Get parent/child references
                parent_id = None
                child_ids = []

                if hasattr(node, "relationships"):
                    if NodeRelationship.PARENT in node.relationships:
                        parent_node = node.relationships[NodeRelationship.PARENT]
                        parent_id = parent_node.node_id if hasattr(parent_node, "node_id") else None

                    if NodeRelationship.CHILD in node.relationships:
                        children = node.relationships[NodeRelationship.CHILD]
                        if isinstance(children, list):
                            child_ids = [c.node_id for c in children if hasattr(c, "node_id")]

                # Determine if this is a leaf node
                is_leaf = node in leaf_nodes

                # Get chunk sizes - prefer original character sizes if available
                if hasattr(config, 'additional_params') and 'chunk_sizes' in config.additional_params:
                    chunk_sizes_list = config.additional_params['chunk_sizes']
                elif hasattr(splitter, 'chunk_sizes'):
                    # Convert token sizes back to character sizes (multiply by 4)
                    chunk_sizes_list = [size * 4 for size in splitter.chunk_sizes]
                else:
                    chunk_sizes_list = [config.max_tokens * 4]

                # Add hierarchy metadata to custom_attributes
                custom_attrs = {
                    "hierarchy_level": level,
                    "is_leaf": is_leaf,
                    "chunk_sizes": chunk_sizes_list,
                }
                if parent_id:
                    custom_attrs["parent_chunk_id"] = parent_id
                if child_ids:
                    custom_attrs["child_chunk_ids"] = child_ids

                metadata = ChunkMetadata(
                    chunk_id=f"{config.strategy_name}_{idx:04d}",
                    document_id="doc",
                    chunk_index=idx,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    token_count=token_count,
                    strategy_name=self.name,
                    hierarchy_level=level,
                    semantic_density=0.75,  # Good for hierarchical structure
                    confidence_score=0.95,  # Higher confidence with LlamaIndex
                    created_at=datetime.now(tz=UTC),
                    custom_attributes=custom_attrs,
                )

                # Create chunk entity
                effective_min_tokens = min(config.min_tokens, token_count, 1)

                chunk = Chunk(
                    content=chunk_text,
                    metadata=metadata,
                    min_tokens=effective_min_tokens,
                    max_tokens=config.max_tokens * (2**level),  # Larger max for parent chunks
                )

                chunks.append(chunk)

                # Report progress
                if progress_callback:
                    progress = ((idx + 1) / len(nodes)) * 100
                    progress_callback(min(progress, 100.0))

            return chunks

        except Exception as e:
            logger.warning(f"LlamaIndex hierarchical chunking failed, falling back to domain: {e}")
            return None

    def _chunk_with_domain(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk]:
        """
        Chunk using domain implementation (multi-level chunking).
        """
        all_chunks = []

        # Calculate chunk sizes for each level
        levels = min(config.hierarchy_levels, 3)  # Max 3 levels for practicality
        logger.debug(f"Hierarchical chunking with {levels} levels (from config.hierarchy_levels={config.hierarchy_levels})")
        level_configs = self._create_level_configs(config, levels)

        # Get all chunk sizes (in characters) - either from config or calculate them
        if hasattr(config, 'additional_params') and 'chunk_sizes' in config.additional_params:
            all_chunk_sizes = config.additional_params['chunk_sizes']
        else:
            # Calculate chunk sizes in characters from level configs
            all_chunk_sizes = [cfg['max_tokens'] * 4 for cfg in level_configs]

        # Process each level
        total_operations = levels
        parent_chunks: list[Chunk] = []
        level_chunks_map: dict[int, list[Chunk]] = {}
        global_chunk_index = 0  # Global index across all levels

        for level in range(levels):
            level_config = level_configs[level]

            # Create chunks for this level
            if level == 0:
                # Top level - chunk entire content
                level_chunks, global_chunk_index = self._create_level_chunks(
                    content,
                    level,
                    level_config,
                    config.strategy_name,
                    None,
                    global_chunk_index,
                    all_chunk_sizes,
                )
                parent_chunks = level_chunks
                level_chunks_map[level] = level_chunks
            else:
                # Child level - create child chunks and update parents
                level_chunks, updated_parents, global_chunk_index = self._create_child_level_chunks(
                    parent_chunks,
                    level,
                    level_config,
                    config.strategy_name,
                    global_chunk_index,
                    all_chunk_sizes,
                )
                # Update parent chunks with child references
                level_chunks_map[level - 1] = updated_parents
                level_chunks_map[level] = level_chunks
                # Use the child chunks as parents for next level
                parent_chunks = level_chunks

            # Report progress
            if progress_callback:
                progress = ((level + 1) / total_operations) * 100
                progress_callback(min(progress, 100.0))

        # Collect all chunks from all levels
        for level in range(levels):
            all_chunks.extend(level_chunks_map[level])

        return all_chunks

    def _create_level_configs(self, base_config: ChunkConfig, levels: int) -> list[dict[str, Any]]:
        """
        Create configuration for each hierarchy level.

        Args:
            base_config: Base configuration
            levels: Number of hierarchy levels

        Returns:
            List of level configurations
        """
        configs = []

        for level in range(levels):
            # Each level has progressively smaller chunks
            divisor = 2**level

            level_config = {
                "max_tokens": base_config.max_tokens // divisor,
                "min_tokens": base_config.min_tokens // divisor,
                "overlap_tokens": base_config.overlap_tokens // max(1, divisor),
                "level": level,
            }

            # Ensure reasonable minimums
            level_config["max_tokens"] = max(50, level_config["max_tokens"])
            level_config["min_tokens"] = max(10, level_config["min_tokens"])
            level_config["overlap_tokens"] = max(5, level_config["overlap_tokens"])

            configs.append(level_config)

        return configs

    def _create_level_chunks(
        self,
        content: str,
        level: int,
        level_config: dict[str, Any],
        strategy_name: str,
        _parent_chunks: list[Chunk] | None = None,
        global_chunk_index: int = 0,
        all_chunk_sizes: list[int] | None = None,
    ) -> tuple[list[Chunk], int]:
        """
        Create chunks for a specific hierarchy level (level 0 only).

        Args:
            content: Content to chunk
            level: Hierarchy level (0=top)
            level_config: Configuration for this level
            strategy_name: Strategy name for chunk IDs
            parent_chunks: Parent chunks from previous level (unused for level 0)

        Returns:
            List of chunks for this level
        """
        # This method now only handles level 0 (top level)
        chunks, new_global_index = self._create_base_chunks(
            content,
            level_config["max_tokens"],
            level_config["min_tokens"],
            level_config["overlap_tokens"],
            strategy_name,
            level,
            global_chunk_index=global_chunk_index,
            all_chunk_sizes=all_chunk_sizes,
        )
        return chunks, new_global_index

    def _create_child_level_chunks(
        self,
        parent_chunks: list[Chunk],
        level: int,
        level_config: dict[str, Any],
        strategy_name: str,
        global_chunk_index: int = 0,
        all_chunk_sizes: list[int] | None = None,
    ) -> tuple[list[Chunk], list[Chunk], int]:
        """
        Create child chunks from parent chunks and update parent references.

        Args:
            parent_chunks: Parent chunks to subdivide
            level: Hierarchy level for child chunks
            level_config: Configuration for this level
            strategy_name: Strategy name for chunk IDs

        Returns:
            Tuple of (child_chunks, updated_parent_chunks)
        """
        all_child_chunks = []
        updated_parents = []
        current_global_index = global_chunk_index

        for parent in parent_chunks:
            parent_content = parent.content
            child_chunks, current_global_index = self._create_base_chunks(
                parent_content,
                level_config["max_tokens"],
                level_config["min_tokens"],
                level_config["overlap_tokens"],
                strategy_name,
                level,
                parent_id=parent.metadata.chunk_id,
                parent_offset=parent.metadata.start_offset,
                global_chunk_index=current_global_index,
                all_chunk_sizes=all_chunk_sizes,
            )

            # Update parent with child references by creating a new chunk
            from dataclasses import replace

            updated_parent_metadata = replace(
                parent.metadata,
                custom_attributes={
                    **parent.metadata.custom_attributes,
                    "child_chunk_ids": [c.metadata.chunk_id for c in child_chunks],
                },
            )

            # Create updated parent chunk
            updated_parent = Chunk(
                content=parent.content,
                metadata=updated_parent_metadata,
                min_tokens=parent.min_tokens,
                max_tokens=parent.max_tokens,
            )

            updated_parents.append(updated_parent)
            all_child_chunks.extend(child_chunks)

        return all_child_chunks, updated_parents, current_global_index

    def _create_base_chunks(
        self,
        content: str,
        max_tokens: int,
        min_tokens: int,
        overlap_tokens: int,
        strategy_name: str,
        level: int,
        parent_id: str | None = None,
        parent_offset: int = 0,
        global_chunk_index: int = 0,
        all_chunk_sizes: list[int] | None = None,
    ) -> tuple[list[Chunk], int]:
        """
        Create basic chunks with size constraints.

        Args:
            content: Content to chunk
            max_tokens: Maximum tokens per chunk
            min_tokens: Minimum tokens per chunk
            overlap_tokens: Overlap between chunks
            strategy_name: Strategy name for chunk IDs
            level: Hierarchy level
            parent_id: Parent chunk ID if applicable
            parent_offset: Offset of parent in original document

        Returns:
            List of chunks
        """
        if not content:
            return [], global_chunk_index

        chunks: list[Chunk] = []
        chars_per_token = 4
        chunk_size_chars = max_tokens * chars_per_token
        overlap_chars = overlap_tokens * chars_per_token

        position = 0

        while position < len(content):
            # Calculate chunk boundaries
            start = position
            end = min(position + chunk_size_chars, len(content))

            # Adjust to word boundaries
            if end < len(content):
                word_boundary = self.find_word_boundary(content, end, prefer_before=True)
                # Ensure we don't go backwards
                if word_boundary > start:
                    end = word_boundary
                # If word boundary is at or before start, keep original end

            # Extract chunk
            chunk_text = content[start:end]
            chunk_text = self.clean_chunk_text(chunk_text)

            if not chunk_text:
                # Prevent infinite loop: if we're not making progress, break
                if end <= position:
                    logger.warning(f"Breaking potential infinite loop at position={position}, end={end}")
                    break
                position = end
                continue

            # Calculate token count
            token_count = self.count_tokens(chunk_text)

            # Skip if too small (unless it's the last chunk or the only chunk)
            if token_count < min_tokens and end < len(content) and len(chunks) > 0:
                position = end
                continue

            # Create metadata
            # Create metadata with hierarchy info in custom_attributes
            custom_attrs = {
                "hierarchy_level": level,
                "chunk_id": f"{strategy_name}_L{level}_{global_chunk_index:04d}",
                "is_leaf": level > 0,  # Non-zero levels are leaf chunks
                "chunk_sizes": all_chunk_sizes if all_chunk_sizes else [chunk_size_chars],  # Use full list if available
                "parent_chunk_id": parent_id,
                "child_chunk_ids": [],
            }

            metadata = ChunkMetadata(
                chunk_id=f"{strategy_name}_L{level}_{global_chunk_index:04d}",
                document_id="doc",
                chunk_index=global_chunk_index,
                start_offset=parent_offset + start,
                end_offset=parent_offset + end,
                token_count=token_count,
                strategy_name="hierarchical",
                hierarchy_level=level,
                semantic_density=0.75,
                confidence_score=0.85,
                created_at=datetime.now(tz=UTC),
                custom_attributes=custom_attrs,
            )

            # Create chunk
            chunk = Chunk(
                content=chunk_text,
                metadata=metadata,
                min_tokens=min(min_tokens, token_count, 1),
                max_tokens=max_tokens,
            )

            chunks.append(chunk)
            global_chunk_index += 1

            # Move position with overlap
            position = end - overlap_chars if overlap_chars > 0 else end

            # Ensure progress (prevent infinite loop)
            if position <= start:
                position = end
                # Double-check we're making progress
                if position <= start:
                    logger.warning(f"Breaking potential infinite loop: position={position}, start={start}, end={end}")
                    break

        return chunks, global_chunk_index

    def validate_content(self, content: str) -> tuple[bool, str | None]:
        """
        Validate content for hierarchical chunking.

        Args:
            content: Content to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not content:
            return False, "Content cannot be empty"

        if len(content) > 50_000_000:  # 50MB limit
            return False, f"Content too large: {len(content)} characters"

        # Hierarchical chunking needs reasonable amount of content
        if len(content) < 100:
            return False, "Content too short for hierarchical chunking"

        return True, None

    def _calculate_node_depth(self, node: Any, all_nodes: list[Any]) -> int:
        """Calculate the depth of a node in the hierarchy (0 = root)."""
        try:
            from llama_index.core.schema import NodeRelationship
        except ImportError:
            return 0

        if not hasattr(node, "relationships"):
            return 0

        # If node has a parent, calculate depth recursively
        if NodeRelationship.PARENT in node.relationships:
            parent_rel = node.relationships[NodeRelationship.PARENT]
            parent_id = getattr(parent_rel, "node_id", None)

            if parent_id:
                # Find parent node
                for pnode in all_nodes:
                    if hasattr(pnode, "node_id") and pnode.node_id == parent_id:
                        return self._calculate_node_depth(pnode, all_nodes) + 1

        # No parent means this is a root node
        return 0

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

        # Hierarchical creates multiple levels
        levels = min(config.hierarchy_levels, 3)
        total_chunks = 0

        for level in range(levels):
            divisor = 2**level
            level_tokens = config.max_tokens // divisor
            level_chunks = estimated_tokens // level_tokens
            total_chunks += max(1, level_chunks)

        return total_chunks
