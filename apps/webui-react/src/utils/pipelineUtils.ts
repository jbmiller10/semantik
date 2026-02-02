import type { PipelineDAG, PipelineEdge } from '@/types/pipeline';

/**
 * Ensures all parallel edges have a path_name set.
 * Auto-generates path_name for parallel edges that don't have one.
 * Returns a new DAG object (does not mutate the original).
 */
export function ensurePathNames(dag: PipelineDAG): PipelineDAG {
  const newEdges: PipelineEdge[] = dag.edges.map((edge, index) => {
    // Only auto-generate for parallel edges without a path_name
    if (edge.parallel && !edge.path_name) {
      return {
        ...edge,
        path_name: `path_${edge.from_node}_${index}`,
      };
    }
    return edge;
  });

  return {
    ...dag,
    edges: newEdges,
  };
}
