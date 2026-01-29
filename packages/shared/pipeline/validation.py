"""DAG validation logic for pipeline definitions.

This module implements the validation rules for pipeline DAGs, ensuring
that the DAG structure is valid before it can be used for document processing.

Validation Rules:
    1. Exactly one EMBEDDER node
    2. All edge node refs exist (or are "_source")
    3. Every node is reachable from _source
    4. Every node has a path to the embedder
    5. No cycles
    6. At least one catch-all edge from _source (non-parallel)
    7. Node IDs are unique
    8. Plugin IDs are registered (if known_plugins provided)
    9. Parallel edges from same node must have unique path_names
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from shared.pipeline.types import DAGValidationError, NodeType

if TYPE_CHECKING:
    from shared.pipeline.types import PipelineDAG

# Special node ID for the entry point of the DAG
SOURCE_NODE = "_source"


def validate_dag(dag: PipelineDAG, known_plugins: set[str] | None = None) -> list[DAGValidationError]:
    """Validate a pipeline DAG and return any errors.

    Args:
        dag: The DAG to validate
        known_plugins: Optional set of registered plugin IDs

    Returns:
        List of validation errors (empty if valid)
    """
    errors: list[DAGValidationError] = []

    # Build lookup structures
    node_ids = {node.id for node in dag.nodes}

    # Rule 7: Node IDs must be unique
    seen_ids: set[str] = set()
    for node in dag.nodes:
        if node.id in seen_ids:
            errors.append(
                DAGValidationError(
                    rule="duplicate_node_id",
                    message=f"Duplicate node ID: {node.id}",
                    node_id=node.id,
                )
            )
        seen_ids.add(node.id)

    # Rule 1: Exactly one EMBEDDER node
    embedder_nodes = [node for node in dag.nodes if node.type == NodeType.EMBEDDER]
    if len(embedder_nodes) == 0:
        errors.append(
            DAGValidationError(
                rule="no_embedder",
                message="DAG must have exactly one EMBEDDER node",
            )
        )
    elif len(embedder_nodes) > 1:
        for node in embedder_nodes:
            errors.append(
                DAGValidationError(
                    rule="multiple_embedders",
                    message=f"DAG has multiple EMBEDDER nodes: {node.id}",
                    node_id=node.id,
                )
            )

    # Rule 2: All edge node refs exist (or are "_source")
    for i, edge in enumerate(dag.edges):
        if edge.from_node != SOURCE_NODE and edge.from_node not in node_ids:
            errors.append(
                DAGValidationError(
                    rule="invalid_from_node",
                    message=f"Edge references non-existent from_node: {edge.from_node}",
                    edge_index=i,
                )
            )
        if edge.to_node not in node_ids:
            errors.append(
                DAGValidationError(
                    rule="invalid_to_node",
                    message=f"Edge references non-existent to_node: {edge.to_node}",
                    edge_index=i,
                )
            )

    # Build adjacency list for graph traversal
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in dag.edges:
        adjacency[edge.from_node].append(edge.to_node)

    # Rule 3: Every node is reachable from _source
    reachable = _find_reachable_nodes(SOURCE_NODE, adjacency)
    for node in dag.nodes:
        if node.id not in reachable:
            errors.append(
                DAGValidationError(
                    rule="unreachable_node",
                    message=f"Node is not reachable from _source: {node.id}",
                    node_id=node.id,
                )
            )

    # Rule 4: Every node has a path to a valid terminal node (embedder or extractor)
    # The primary pipeline MUST reach an embedder, but parallel paths may terminate at extractors
    if embedder_nodes:
        embedder_id = embedder_nodes[0].id
        # Build reverse adjacency for backward traversal
        reverse_adjacency: dict[str, list[str]] = defaultdict(list)
        for edge in dag.edges:
            reverse_adjacency[edge.to_node].append(edge.from_node)

        # Find all nodes that can reach the embedder
        can_reach_embedder = _find_reachable_nodes(embedder_id, reverse_adjacency)

        # Find extractor nodes (valid sinks for parallel paths)
        extractor_nodes = [n for n in dag.nodes if n.type == NodeType.EXTRACTOR]

        # Find all nodes that can reach any extractor
        can_reach_extractor: set[str] = set()
        for extractor in extractor_nodes:
            can_reach_extractor.update(_find_reachable_nodes(extractor.id, reverse_adjacency))
            can_reach_extractor.add(extractor.id)  # Extractor itself is valid

        # A node is valid if it reaches embedder OR reaches an extractor
        valid_terminal_nodes = can_reach_embedder | can_reach_extractor

        for node in dag.nodes:
            if node.id != embedder_id and node.id not in valid_terminal_nodes:
                errors.append(
                    DAGValidationError(
                        rule="no_path_to_terminal",
                        message=f"Node has no path to embedder or extractor: {node.id}",
                        node_id=node.id,
                    )
                )

    # Rule 5: No cycles
    cycle_node = _find_cycle(dag.nodes, adjacency)
    if cycle_node:
        errors.append(
            DAGValidationError(
                rule="cycle_detected",
                message=f"Cycle detected involving node: {cycle_node}",
                node_id=cycle_node,
            )
        )

    # Rule 6: At least one non-parallel catch-all edge from _source
    # This ensures there's always a default path for files that don't match parallel predicates
    source_edges = [edge for edge in dag.edges if edge.from_node == SOURCE_NODE]
    has_exclusive_catchall = any((edge.when is None or edge.when == {}) and not edge.parallel for edge in source_edges)
    if not has_exclusive_catchall:
        errors.append(
            DAGValidationError(
                rule="no_source_catchall",
                message="No non-parallel catch-all edge from _source (at least one edge must have when=None and parallel=False)",
            )
        )

    # Rule 9: Parallel edges from same node must have unique path_names
    errors.extend(_validate_unique_path_names(dag))

    # Rule 8: Plugin IDs are registered (if known_plugins provided)
    if known_plugins is not None:
        for node in dag.nodes:
            if node.plugin_id not in known_plugins:
                errors.append(
                    DAGValidationError(
                        rule="unknown_plugin",
                        message=f"Unknown plugin: {node.plugin_id}",
                        node_id=node.id,
                    )
                )

    return errors


def _find_reachable_nodes(start: str, adjacency: dict[str, list[str]]) -> set[str]:
    """Find all nodes reachable from a starting node using BFS.

    Args:
        start: Starting node ID
        adjacency: Adjacency list (node -> list of neighbors)

    Returns:
        Set of reachable node IDs (excluding the start node)
    """
    visited: set[str] = set()
    queue = list(adjacency.get(start, []))

    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        queue.extend(adjacency.get(node, []))

    return visited


def _validate_unique_path_names(dag: PipelineDAG) -> list[DAGValidationError]:
    """Validate that parallel edges from the same node have unique path_names.

    When a document fans out through multiple parallel paths, each path must
    produce chunks with a unique path_id for search filtering. This requires
    that parallel edges from the same source node have distinct path_names.

    Args:
        dag: The DAG to validate

    Returns:
        List of validation errors for duplicate path_names
    """
    errors: list[DAGValidationError] = []

    # Group edges by source node
    edges_by_source: dict[str, list] = defaultdict(list)
    for edge in dag.edges:
        edges_by_source[edge.from_node].append(edge)

    # Check for duplicate path_names among parallel edges from each source
    for from_node, edges in edges_by_source.items():
        # Only check parallel edges
        parallel_edges = [e for e in edges if e.parallel]
        if len(parallel_edges) < 2:
            continue

        # Collect path_names (using get_path_name which defaults to to_node)
        seen_path_names: dict[str, int] = {}
        for i, edge in enumerate(dag.edges):
            if edge.from_node != from_node or not edge.parallel:
                continue

            path_name = edge.get_path_name()
            if path_name in seen_path_names:
                errors.append(
                    DAGValidationError(
                        rule="duplicate_path_name",
                        message=f"Duplicate path_name '{path_name}' for parallel edges from '{from_node}'",
                        edge_index=i,
                    )
                )
            else:
                seen_path_names[path_name] = i

    return errors


def _find_cycle(nodes: list, adjacency: dict[str, list[str]]) -> str | None:
    """Detect if there's a cycle in the DAG using DFS.

    Args:
        nodes: List of nodes in the DAG
        adjacency: Adjacency list (node -> list of neighbors)

    Returns:
        ID of a node involved in the cycle, or None if no cycle exists
    """
    # Track visit state: 0=unvisited, 1=visiting (in current path), 2=visited
    state: dict[str, int] = {node.id: 0 for node in nodes}

    def dfs(node_id: str) -> str | None:
        if state.get(node_id, 0) == 1:
            # Found a back edge - cycle detected
            return node_id
        if state.get(node_id, 0) == 2:
            # Already fully visited
            return None

        state[node_id] = 1  # Mark as visiting

        for neighbor in adjacency.get(node_id, []):
            result = dfs(neighbor)
            if result:
                return result

        state[node_id] = 2  # Mark as visited
        return None

    # Start DFS from _source
    result = dfs(SOURCE_NODE)
    if result:
        return result

    # Also check for isolated cycles (unreachable from _source)
    for node in nodes:
        if state.get(node.id, 0) == 0:
            result = dfs(node.id)
            if result:
                return result

    return None


__all__ = [
    "validate_dag",
    "SOURCE_NODE",
]
