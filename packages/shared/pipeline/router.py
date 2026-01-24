"""Pipeline routing for edge matching and DAG traversal.

This module provides the PipelineRouter class that routes files through
the DAG based on edge predicates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from shared.pipeline.predicates import matches_predicate
from shared.pipeline.validation import SOURCE_NODE

if TYPE_CHECKING:
    from shared.pipeline.types import FileReference, PipelineDAG, PipelineEdge, PipelineNode


class PipelineRouter:
    """Routes files through the DAG based on edge predicates.

    The router is responsible for determining which nodes a file should
    be processed by, based on the file's attributes and the DAG's edge
    predicates.

    Example:
        ```python
        router = PipelineRouter(dag)

        # Find entry node for a file
        entry_node = router.get_entry_node(file_ref)
        if entry_node:
            process(file_ref, entry_node)

            # Get next nodes after processing
            next_nodes = router.get_next_nodes(entry_node, file_ref)
            for node in next_nodes:
                process(file_ref, node)
        ```
    """

    def __init__(self, dag: PipelineDAG) -> None:
        """Initialize the router with a DAG.

        Args:
            dag: The pipeline DAG to route through
        """
        self.dag = dag
        self._node_index: dict[str, PipelineNode] = {node.id: node for node in dag.nodes}
        self._outgoing_edges: dict[str, list[PipelineEdge]] = {}
        self._build_edge_index()

    def _build_edge_index(self) -> None:
        """Build index of outgoing edges for each node."""
        for edge in self.dag.edges:
            if edge.from_node not in self._outgoing_edges:
                self._outgoing_edges[edge.from_node] = []
            self._outgoing_edges[edge.from_node].append(edge)

    def get_entry_node(self, file_ref: FileReference) -> PipelineNode | None:
        """Find the entry node for a file based on _source edges.

        Examines edges from _source and returns the first matching node.
        Edges with predicates are checked first (in order), with catch-all
        edges (when=None) checked last.

        Args:
            file_ref: The file reference to route

        Returns:
            The entry node to start processing, or None if no match
        """
        source_edges = self._outgoing_edges.get(SOURCE_NODE, [])

        if not source_edges:
            return None

        # Separate edges with predicates from catch-all edges
        predicate_edges: list[PipelineEdge] = []
        catchall_edges: list[PipelineEdge] = []

        for edge in source_edges:
            if edge.when is None or edge.when == {}:
                catchall_edges.append(edge)
            else:
                predicate_edges.append(edge)

        # Check predicate edges first (order matters - first match wins)
        for edge in predicate_edges:
            if matches_predicate(file_ref, edge.when):
                return self._node_index.get(edge.to_node)

        # Fall back to catch-all edges
        for edge in catchall_edges:
            return self._node_index.get(edge.to_node)

        return None

    def get_next_nodes(
        self,
        current: PipelineNode,
        file_ref: FileReference,
    ) -> list[PipelineNode]:
        """Find the next node to process after the current node completes.

        Examines outgoing edges from the current node and returns the first
        matching node. Uses first-match-wins semantics: predicate edges are
        evaluated in order, and the first matching edge determines the next
        node. Catch-all edges (when=None) are checked only if no predicate
        edges match.

        Args:
            current: The node that just completed processing
            file_ref: The file being processed

        Returns:
            List containing the first matching node, or empty list at terminal nodes.
        """
        outgoing = self._outgoing_edges.get(current.id, [])

        if not outgoing:
            return []

        # Separate edges with predicates from catch-all edges
        predicate_edges: list[PipelineEdge] = []
        catchall_edges: list[PipelineEdge] = []

        for edge in outgoing:
            if edge.when is None or edge.when == {}:
                catchall_edges.append(edge)
            else:
                predicate_edges.append(edge)

        # Check predicate edges first
        for edge in predicate_edges:
            if matches_predicate(file_ref, edge.when):
                node = self._node_index.get(edge.to_node)
                if node:
                    return [node]

        # Fall back to catch-all edges
        for edge in catchall_edges:
            node = self._node_index.get(edge.to_node)
            if node:
                return [node]

        return []

    def get_node(self, node_id: str) -> PipelineNode | None:
        """Get a node by ID.

        Args:
            node_id: The node ID to look up

        Returns:
            The node, or None if not found
        """
        return self._node_index.get(node_id)

    def get_all_paths(self, file_ref: FileReference) -> list[list[PipelineNode]]:
        """Get all possible processing paths for a file.

        This is useful for validation and debugging to see which nodes
        a file would be routed through.

        Args:
            file_ref: The file reference to route

        Returns:
            List of paths, where each path is a list of nodes from entry to embedder
        """
        paths: list[list[PipelineNode]] = []
        entry = self.get_entry_node(file_ref)

        if entry is None:
            return paths

        # Use DFS to find all paths
        self._find_paths(entry, file_ref, [entry], paths)
        return paths

    def _find_paths(
        self,
        current: PipelineNode,
        file_ref: FileReference,
        current_path: list[PipelineNode],
        all_paths: list[list[PipelineNode]],
    ) -> None:
        """Recursively find all paths from current node.

        Args:
            current: Current node
            file_ref: File being processed
            current_path: Path so far
            all_paths: Accumulator for all complete paths
        """
        next_nodes = self.get_next_nodes(current, file_ref)

        if not next_nodes:
            # Terminal node - save this path
            all_paths.append(list(current_path))
            return

        for next_node in next_nodes:
            # Avoid infinite loops
            if next_node not in current_path:
                current_path.append(next_node)
                self._find_paths(next_node, file_ref, current_path, all_paths)
                current_path.pop()


__all__ = [
    "PipelineRouter",
]
