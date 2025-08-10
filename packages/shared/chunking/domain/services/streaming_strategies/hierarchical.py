#!/usr/bin/env python3
"""
Streaming hierarchical chunking strategy.

This strategy builds a document tree structure incrementally and emits
chunks based on logical document hierarchy, maintaining a max 10KB buffer
for the tree state.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from packages.shared.chunking.domain.entities.chunk import Chunk
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata
from packages.shared.chunking.infrastructure.streaming.window import StreamingWindow

from .base import StreamingChunkingStrategy


@dataclass
class TreeNode:
    """Represents a node in the document hierarchy."""

    content: str
    level: int
    node_type: str  # heading, paragraph, list, etc.
    children: list["TreeNode"] = field(default_factory=list)
    parent: Optional["TreeNode"] = None
    size_bytes: int = 0
    token_count: int = 0

    def add_child(self, child: "TreeNode") -> None:
        """Add a child node."""
        child.parent = self
        self.children.append(child)
        self.size_bytes += child.size_bytes
        self.token_count += child.token_count

    def to_text(self, include_children: bool = True) -> str:
        """Convert node and optionally children to text."""
        parts = [self.content]

        if include_children:
            for child in self.children:
                child_text = child.to_text(include_children=True)
                if child_text:
                    parts.append(child_text)

        return "\n".join(parts)

    def get_depth(self) -> int:
        """Get the depth of the subtree."""
        if not self.children:
            return 1
        return 1 + max(child.get_depth() for child in self.children)

    def prune_to_size(self, max_size: int) -> list["TreeNode"]:
        """
        Prune the tree to stay within size limits.

        Returns nodes that were pruned and should be emitted.
        """
        pruned = []

        while self.size_bytes > max_size and self.children:
            # Remove oldest/largest children first
            child = self.children[0]
            self.children.remove(child)
            self.size_bytes -= child.size_bytes
            self.token_count -= child.token_count
            pruned.append(child)

        return pruned


class StreamingHierarchicalStrategy(StreamingChunkingStrategy):
    """
    Streaming hierarchical chunking strategy.

    Builds a document tree incrementally and emits chunks based on
    logical hierarchy, maintaining bounded memory usage.
    """

    MAX_BUFFER_SIZE = 10 * 1024  # 10KB max buffer
    MAX_TREE_DEPTH = 5  # Maximum tree depth

    def __init__(self):
        """Initialize the streaming hierarchical strategy."""
        super().__init__("hierarchical")
        self._root = TreeNode(content="", level=0, node_type="root")
        self._current_node = self._root
        self._pending_text = ""
        self._chunk_index = 0
        self._char_offset = 0

    async def process_window(self, window: StreamingWindow, config: ChunkConfig, is_final: bool = False) -> list[Chunk]:
        """
        Process a window building document hierarchy.

        Args:
            window: StreamingWindow containing the data
            config: Chunk configuration parameters
            is_final: Whether this is the final window

        Returns:
            List of chunks produced from this window
        """
        chunks = []

        # Get text from window
        text = window.decode_safe()
        if not text and not is_final:
            return chunks

        # Combine with pending text
        if self._pending_text:
            text = self._pending_text + text
            self._pending_text = ""

        # Parse text into hierarchical structure
        lines = text.split("\n")

        for i, line in enumerate(lines):
            # Check if last line and not final
            is_last_line = i == len(lines) - 1
            if is_last_line and not is_final:
                self._pending_text = line
                continue

            # Identify structure level and type
            level, node_type = self._identify_structure(line)

            if not line.strip():
                continue

            # Create node
            node = TreeNode(
                content=line,
                level=level,
                node_type=node_type,
                size_bytes=len(line.encode("utf-8")),
                token_count=self.count_tokens(line),
            )

            # Find appropriate parent based on level
            parent = self._find_parent_for_level(level)
            parent.add_child(node)

            # Check if we should emit chunks due to size
            if self._root.size_bytes > self.MAX_BUFFER_SIZE:
                # Prune tree and emit chunks
                pruned_chunks = await self._prune_and_emit(config)
                chunks.extend(pruned_chunks)

            # Check tree depth
            if self._root.get_depth() > self.MAX_TREE_DEPTH:
                # Emit deepest branches
                deep_chunks = await self._emit_deep_branches(config)
                chunks.extend(deep_chunks)

        # Emit based on logical boundaries
        if self._should_emit_subtree():
            subtree_chunks = await self._emit_logical_subtrees(config)
            chunks.extend(subtree_chunks)

        # If final, emit all remaining
        if is_final:
            final_chunks = await self._emit_all_remaining(config)
            chunks.extend(final_chunks)

        return chunks

    def _identify_structure(self, line: str) -> tuple[int, str]:
        """
        Identify the hierarchical level and type of a line.

        Args:
            line: Line to analyze

        Returns:
            Tuple of (level, node_type)
        """
        stripped = line.strip()

        # Headings (Markdown style)
        if stripped.startswith("#"):
            heading_level = 0
            for char in stripped:
                if char == "#":
                    heading_level += 1
                else:
                    break
            return (heading_level, "heading")

        # List items (check indentation)
        indent_level = 0
        for char in line:
            if char == " ":
                indent_level += 1
            elif char == "\t":
                indent_level += 4
            else:
                break

        # Detect list markers
        if stripped.startswith("-") or stripped.startswith("*") or stripped.startswith("+"):
            return (3 + indent_level // 2, "list_item")

        # Numbered lists
        import re

        if re.match(r"^\d+\.", stripped):
            return (3 + indent_level // 2, "numbered_list")

        # Code blocks (indented)
        if line.startswith("    "):
            return (4, "code")

        # Default paragraph
        return (5, "paragraph")

    def _find_parent_for_level(self, level: int) -> TreeNode:
        """
        Find the appropriate parent node for a given level.

        Args:
            level: Hierarchical level

        Returns:
            Parent node
        """
        # Start from current node and traverse up
        node = self._current_node

        while node.parent is not None:
            if node.level < level:
                self._current_node = node
                return node
            node = node.parent

        # Default to root
        self._current_node = self._root
        return self._root

    def _should_emit_subtree(self) -> bool:
        """
        Determine if we should emit subtrees based on structure.

        Returns:
            True if subtrees should be emitted
        """
        # Check if root has complete sections
        for child in self._root.children:
            if child.node_type == "heading" and child.children:
                # Check if section is complete (has content)
                has_content = any(c.node_type in ["paragraph", "list_item", "code"] for c in child.children)
                if has_content and child.token_count > 500:
                    return True

        return False

    async def _prune_and_emit(self, config: ChunkConfig) -> list[Chunk]:
        """
        Prune the tree to stay within memory limits and emit chunks.

        Args:
            config: Chunk configuration

        Returns:
            List of chunks from pruned nodes
        """
        chunks = []

        # Prune from root
        pruned_nodes = self._root.prune_to_size(self.MAX_BUFFER_SIZE // 2)

        for node in pruned_nodes:
            chunk = await self._create_chunk_from_node(node, config)
            if chunk:
                chunks.append(chunk)

        return chunks

    async def _emit_deep_branches(self, config: ChunkConfig) -> list[Chunk]:
        """
        Emit branches that are too deep.

        Args:
            config: Chunk configuration

        Returns:
            List of chunks from deep branches
        """
        chunks = []

        def find_deep_branches(node: TreeNode, depth: int = 0) -> list[TreeNode]:
            deep = []
            if depth >= self.MAX_TREE_DEPTH:
                deep.append(node)
            else:
                for child in node.children:
                    deep.extend(find_deep_branches(child, depth + 1))
            return deep

        deep_branches = find_deep_branches(self._root)

        for branch in deep_branches:
            chunk = await self._create_chunk_from_node(branch, config)
            if chunk:
                chunks.append(chunk)

            # Remove from tree
            if branch.parent:
                branch.parent.children.remove(branch)
                branch.parent.size_bytes -= branch.size_bytes
                branch.parent.token_count -= branch.token_count

        return chunks

    async def _emit_logical_subtrees(self, config: ChunkConfig) -> list[Chunk]:
        """
        Emit complete logical subtrees.

        Args:
            config: Chunk configuration

        Returns:
            List of chunks from subtrees
        """
        chunks = []

        # Find complete sections
        sections_to_emit = []

        for child in self._root.children[:]:  # Copy list for modification
            if child.node_type == "heading" and child.children:
                # Check if section is large enough
                if child.token_count >= config.min_tokens:
                    sections_to_emit.append(child)
                    self._root.children.remove(child)
                    self._root.size_bytes -= child.size_bytes
                    self._root.token_count -= child.token_count

        # Create chunks from sections
        for section in sections_to_emit:
            chunk = await self._create_chunk_from_node(section, config)
            if chunk:
                chunks.append(chunk)

        return chunks

    async def _emit_all_remaining(self, config: ChunkConfig) -> list[Chunk]:
        """
        Emit all remaining nodes as chunks.

        Args:
            config: Chunk configuration

        Returns:
            List of final chunks
        """
        chunks = []

        # Process pending text
        if self._pending_text:
            level, node_type = self._identify_structure(self._pending_text)
            node = TreeNode(
                content=self._pending_text,
                level=level,
                node_type=node_type,
                size_bytes=len(self._pending_text.encode("utf-8")),
                token_count=self.count_tokens(self._pending_text),
            )
            self._root.add_child(node)
            self._pending_text = ""

        # Emit all children of root
        for child in self._root.children:
            chunk = await self._create_chunk_from_node(child, config)
            if chunk:
                chunks.append(chunk)

        return chunks

    async def _create_chunk_from_node(self, node: TreeNode, config: ChunkConfig) -> Chunk | None:
        """
        Create a chunk from a tree node.

        Args:
            node: Tree node to convert
            config: Chunk configuration

        Returns:
            Chunk if node has content
        """
        if not node or not node.content:
            return None

        # Get text representation
        content = node.to_text(include_children=True)
        content = self.clean_chunk_text(content)

        if not content:
            return None

        # Calculate hierarchy path
        hierarchy_path = []
        current = node
        while current and current.parent:
            if current.node_type == "heading":
                hierarchy_path.insert(0, current.content.strip())
            current = current.parent

        # Create metadata
        token_count = node.token_count
        effective_min_tokens = min(config.min_tokens, token_count, 1)

        metadata = ChunkMetadata(
            chunk_id=str(uuid4()),
            document_id="doc",
            chunk_index=self._chunk_index,
            start_offset=self._char_offset,
            end_offset=self._char_offset + len(content),
            token_count=token_count,
            strategy_name=self.name,
            semantic_density=0.75,  # Good for hierarchical structure
            confidence_score=0.9,  # High confidence for structured approach
            created_at=datetime.now(tz=UTC),
            custom_attributes={
                "hierarchy_path": hierarchy_path,
                "node_type": node.node_type,
                "tree_depth": node.get_depth(),
                "child_count": len(node.children),
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

        return chunk

    async def finalize(self, config: ChunkConfig) -> list[Chunk]:
        """
        Process any remaining tree nodes.

        Args:
            config: Chunk configuration parameters

        Returns:
            List of final chunks
        """
        chunks = await self._emit_all_remaining(config)
        self._is_finalized = True
        return chunks

    def get_buffer_size(self) -> int:
        """
        Return the current buffer size in bytes.

        Returns:
            Size of tree structure in bytes
        """
        size = self._root.size_bytes

        if self._pending_text:
            size += len(self._pending_text.encode("utf-8"))

        return size

    def get_max_buffer_size(self) -> int:
        """
        Return the maximum allowed buffer size.

        Returns:
            10KB maximum buffer size
        """
        return self.MAX_BUFFER_SIZE

    def reset(self) -> None:
        """Reset the strategy state."""
        super().reset()
        self._root = TreeNode(content="", level=0, node_type="root")
        self._current_node = self._root
        self._pending_text = ""
        self._chunk_index = 0
        self._char_offset = 0
