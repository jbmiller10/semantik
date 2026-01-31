/**
 * Utility functions for DAG manipulation and analysis.
 */

import type { PipelineDAG, PipelineNode, PipelineEdge } from '@/types/pipeline';

/**
 * Find nodes that would have no incoming edges if a node is deleted.
 * Traverses downstream to find all transitively orphaned nodes.
 *
 * @param dag - The current pipeline DAG
 * @param deletedNodeId - The ID of the node being deleted
 * @param edges - Optional edge list to use instead of dag.edges (for simulating edge deletion)
 * @returns Array of nodes that would be orphaned
 */
export function findOrphanedNodes(
  dag: PipelineDAG,
  deletedNodeId: string,
  edges?: PipelineEdge[]
): PipelineNode[] {
  const edgesToCheck = edges || dag.edges;

  // Build adjacency map: which nodes have incoming edges from which sources
  const incomingEdgesMap = new Map<string, Set<string>>();

  // Initialize all nodes with empty sets
  for (const node of dag.nodes) {
    incomingEdgesMap.set(node.id, new Set());
  }

  // Add _source as a special node that's never orphaned
  // (it has no incoming edges by definition)

  // Populate the map with edges that don't involve the deleted node
  for (const edge of edgesToCheck) {
    if (edge.from_node === deletedNodeId) {
      // Skip edges originating from the deleted node
      continue;
    }
    const targetSet = incomingEdgesMap.get(edge.to_node);
    if (targetSet) {
      targetSet.add(edge.from_node);
    }
  }

  // Find directly orphaned nodes (nodes with no incoming edges after deletion)
  const orphaned: PipelineNode[] = [];
  const orphanedIds = new Set<string>();

  for (const node of dag.nodes) {
    // Skip the node being deleted
    if (node.id === deletedNodeId) continue;

    const incomingSet = incomingEdgesMap.get(node.id);
    if (!incomingSet || incomingSet.size === 0) {
      orphaned.push(node);
      orphanedIds.add(node.id);
    }
  }

  // Transitively find nodes that would be orphaned because their only
  // upstream nodes are also orphaned
  let foundNew = true;
  while (foundNew) {
    foundNew = false;
    for (const node of dag.nodes) {
      if (node.id === deletedNodeId) continue;
      if (orphanedIds.has(node.id)) continue;

      const incomingSet = incomingEdgesMap.get(node.id);
      if (!incomingSet) continue;

      // Check if all incoming edges are from orphaned nodes or deleted node
      const validSources = [...incomingSet].filter(
        (sourceId) => sourceId !== deletedNodeId && !orphanedIds.has(sourceId)
      );

      if (validSources.length === 0 && incomingSet.size > 0) {
        orphaned.push(node);
        orphanedIds.add(node.id);
        foundNew = true;
      }
    }
  }

  return orphaned;
}

/**
 * Find nodes that would be orphaned if an edge is deleted.
 *
 * @param dag - The current pipeline DAG
 * @param fromNode - The source node of the edge
 * @param toNode - The target node of the edge
 * @returns Array of nodes that would be orphaned, including the target and its downstream
 */
export function findOrphanedNodesAfterEdgeDeletion(
  dag: PipelineDAG,
  fromNode: string,
  toNode: string
): PipelineNode[] {
  // Create edge list without the deleted edge
  const remainingEdges = dag.edges.filter(
    (e) => !(e.from_node === fromNode && e.to_node === toNode)
  );

  // Check if the target node has any other incoming edges
  const otherIncomingEdges = remainingEdges.filter((e) => e.to_node === toNode);

  if (otherIncomingEdges.length > 0) {
    // Target node still has incoming edges, no orphans
    return [];
  }

  // Target node would be orphaned - find it and all downstream orphans
  const targetNode = dag.nodes.find((n) => n.id === toNode);
  if (!targetNode) return [];

  // Use findOrphanedNodes with the remaining edges to find all orphans
  // Pass toNode as the "deleted" node since we want to find what's orphaned without it
  const downstreamOrphans = findOrphanedNodes(dag, toNode, remainingEdges);

  return [targetNode, ...downstreamOrphans];
}
