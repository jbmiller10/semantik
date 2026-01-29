"""Pipeline routing for edge matching and DAG traversal.

This module provides the PipelineRouter class that routes files through
the DAG based on edge predicates. Supports both exclusive (first-match-wins)
and parallel (all-matching) edge semantics.

Edge Semantics
--------------
- **Exclusive edges** (``parallel=False``): First-match-wins. Only one exclusive
  edge fires per routing stage. Use for mutually exclusive routes like
  PDF-vs-text or language-specific parsing.

- **Parallel edges** (``parallel=True``): All matching parallel edges fire
  together, creating multiple execution paths. Use for fan-out scenarios
  like sending a document to both chunking and summarization pipelines.

Evaluation Order
----------------
At each routing stage, edges are evaluated in this order:

1. Parallel predicate edges - all matches fire
2. Exclusive predicate edges - first match wins
3. Parallel catch-all edges - all fire
4. Exclusive catch-all edges - first match wins (fallback)

This ordering ensures predicate edges take priority over catch-all edges,
and parallel edges can fire alongside exclusive edges.

Example
-------
>>> from shared.pipeline.router import PipelineRouter
>>> router = PipelineRouter(dag)
>>> entry_nodes = router.get_entry_nodes(file_ref)
>>> for node, path_name in entry_nodes:
...     print(f"File enters via {node.id} on path '{path_name}'")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from shared.pipeline.predicates import matches_predicate
from shared.pipeline.validation import SOURCE_NODE

if TYPE_CHECKING:
    from shared.pipeline.types import FileReference, PipelineDAG, PipelineEdge, PipelineNode


# Type alias for routing results: (node, path_name)
RoutedNode = tuple["PipelineNode", str]


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

        This is a backward-compatible method that returns only the first entry node.
        For parallel path support, use get_entry_nodes() instead.

        Examines edges from _source and returns the first matching node.
        Edges with predicates are checked first (in order), with catch-all
        edges (when=None) checked last.

        Args:
            file_ref: The file reference to route

        Returns:
            The entry node to start processing, or None if no match
        """
        entry_nodes = self.get_entry_nodes(file_ref)
        if entry_nodes:
            return entry_nodes[0][0]  # Return just the node, not the path_name
        return None

    def get_entry_nodes(self, file_ref: FileReference) -> list[RoutedNode]:
        """Find all entry nodes for a file based on _source edges.

        Handles both parallel and exclusive edge semantics:
        - Parallel edges (parallel=True): All matching parallel edges fire together
        - Exclusive edges (parallel=False): First-match-wins, at most one fires

        The evaluation order is:
        1. Check all parallel predicate edges (all matches fire)
        2. Check exclusive predicate edges (first match wins)
        3. Check parallel catch-all edges (all fire)
        4. Check exclusive catch-all edges (first match wins)

        Args:
            file_ref: The file reference to route

        Returns:
            List of (node, path_name) tuples for matched entry points
        """
        source_edges = self._outgoing_edges.get(SOURCE_NODE, [])

        if not source_edges:
            return []

        results: list[RoutedNode] = []

        # Categorize edges by parallel flag and predicate presence
        parallel_predicate: list[PipelineEdge] = []
        parallel_catchall: list[PipelineEdge] = []
        exclusive_predicate: list[PipelineEdge] = []
        exclusive_catchall: list[PipelineEdge] = []

        for edge in source_edges:
            is_catchall = edge.when is None or edge.when == {}
            if edge.parallel:
                if is_catchall:
                    parallel_catchall.append(edge)
                else:
                    parallel_predicate.append(edge)
            else:
                if is_catchall:
                    exclusive_catchall.append(edge)
                else:
                    exclusive_predicate.append(edge)

        # 1. Evaluate parallel predicate edges (all matches fire)
        for edge in parallel_predicate:
            if matches_predicate(file_ref, edge.when):
                node = self._node_index.get(edge.to_node)
                if node:
                    results.append((node, edge.get_path_name()))

        # 2. Evaluate exclusive predicate edges (first match wins)
        exclusive_matched = False
        for edge in exclusive_predicate:
            if matches_predicate(file_ref, edge.when):
                node = self._node_index.get(edge.to_node)
                if node:
                    results.append((node, edge.get_path_name()))
                    exclusive_matched = True
                    break

        # 3. Evaluate parallel catch-all edges (all fire)
        for edge in parallel_catchall:
            node = self._node_index.get(edge.to_node)
            if node:
                results.append((node, edge.get_path_name()))

        # 4. Fall back to exclusive catch-all if no exclusive match yet
        if not exclusive_matched:
            for edge in exclusive_catchall:
                node = self._node_index.get(edge.to_node)
                if node:
                    results.append((node, edge.get_path_name()))
                    break

        return results

    def get_next_nodes(
        self,
        current: PipelineNode,
        file_ref: FileReference,
    ) -> list[PipelineNode]:
        """Find the next node(s) to process after the current node completes.

        This is a backward-compatible method that returns only nodes without path names.
        For parallel path support with path names, use get_next_nodes_with_paths() instead.

        Uses first-match-wins semantics for exclusive edges: returns a single-element list
        with the first matching node, or empty list at terminal nodes.

        Args:
            current: The node that just completed processing
            file_ref: The file being processed (used for predicate matching)

        Returns:
            List of nodes. For exclusive routing, contains 0 or 1 nodes.
            For parallel routing, may contain multiple nodes.
        """
        routed = self.get_next_nodes_with_paths(current, file_ref)
        return [node for node, _ in routed]

    def get_next_nodes_with_paths(
        self,
        current: PipelineNode,
        file_ref: FileReference,
    ) -> list[RoutedNode]:
        """Find the next node(s) to process after the current node completes.

        Handles both parallel and exclusive edge semantics:
        - Parallel edges (parallel=True): All matching parallel edges fire together
        - Exclusive edges (parallel=False): First-match-wins, at most one fires

        The evaluation order is:
        1. Check all parallel predicate edges (all matches fire)
        2. Check exclusive predicate edges (first match wins)
        3. Check parallel catch-all edges (all fire)
        4. Check exclusive catch-all edges (first match wins)

        Args:
            current: The node that just completed processing
            file_ref: The file being processed (used for predicate matching)

        Returns:
            List of (node, path_name) tuples. For exclusive routing, contains 0 or 1.
            For parallel routing, may contain multiple entries.
        """
        outgoing = self._outgoing_edges.get(current.id, [])

        if not outgoing:
            return []

        results: list[RoutedNode] = []

        # Categorize edges by parallel flag and predicate presence
        parallel_predicate: list[PipelineEdge] = []
        parallel_catchall: list[PipelineEdge] = []
        exclusive_predicate: list[PipelineEdge] = []
        exclusive_catchall: list[PipelineEdge] = []

        for edge in outgoing:
            is_catchall = edge.when is None or edge.when == {}
            if edge.parallel:
                if is_catchall:
                    parallel_catchall.append(edge)
                else:
                    parallel_predicate.append(edge)
            else:
                if is_catchall:
                    exclusive_catchall.append(edge)
                else:
                    exclusive_predicate.append(edge)

        # 1. Evaluate parallel predicate edges (all matches fire)
        for edge in parallel_predicate:
            if matches_predicate(file_ref, edge.when):
                node = self._node_index.get(edge.to_node)
                if node:
                    results.append((node, edge.get_path_name()))

        # 2. Evaluate exclusive predicate edges (first match wins)
        exclusive_matched = False
        for edge in exclusive_predicate:
            if matches_predicate(file_ref, edge.when):
                node = self._node_index.get(edge.to_node)
                if node:
                    results.append((node, edge.get_path_name()))
                    exclusive_matched = True
                    break

        # 3. Evaluate parallel catch-all edges (all fire)
        for edge in parallel_catchall:
            node = self._node_index.get(edge.to_node)
            if node:
                results.append((node, edge.get_path_name()))

        # 4. Fall back to exclusive catch-all if no exclusive match yet
        if not exclusive_matched:
            for edge in exclusive_catchall:
                node = self._node_index.get(edge.to_node)
                if node:
                    results.append((node, edge.get_path_name()))
                    break

        return results

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
    "RoutedNode",
]
