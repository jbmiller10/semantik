/**
 * Horizontal visualization of the routing path through the pipeline.
 * Shows nodes connected by arrows with the selected path highlighted.
 */

import { ChevronRight, FileInput, Box, Layers, Sparkles, Cpu } from 'lucide-react';
import type { PipelineDAG, PipelineNode as PipelineNodeType } from '@/types/pipeline';

interface PathVisualizationProps {
  /** The computed path (list of node IDs) */
  path: string[];
  /** The pipeline DAG for node lookup */
  dag: PipelineDAG;
}

/**
 * Get icon for a node type.
 */
function getNodeIcon(nodeId: string, type: string | undefined) {
  if (nodeId === '_source') {
    return <FileInput className="w-4 h-4" />;
  }
  switch (type) {
    case 'parser':
      return <Box className="w-4 h-4" />;
    case 'chunker':
      return <Layers className="w-4 h-4" />;
    case 'extractor':
      return <Sparkles className="w-4 h-4" />;
    case 'embedder':
      return <Cpu className="w-4 h-4" />;
    default:
      return <Box className="w-4 h-4" />;
  }
}

/**
 * Get display name for a node.
 */
function getNodeDisplayName(nodeId: string, nodes: PipelineNodeType[]): string {
  if (nodeId === '_source') {
    return 'Source';
  }
  const node = nodes.find((n) => n.id === nodeId);
  return node?.plugin_id || nodeId;
}

/**
 * Get node type.
 */
function getNodeType(nodeId: string, nodes: PipelineNodeType[]): string | undefined {
  if (nodeId === '_source') {
    return 'source';
  }
  const node = nodes.find((n) => n.id === nodeId);
  return node?.type;
}

export function PathVisualization({ path, dag }: PathVisualizationProps) {
  if (path.length === 0) {
    return (
      <div className="text-sm text-[var(--text-muted)] italic">No path computed</div>
    );
  }

  return (
    <div className="flex items-center gap-1 flex-wrap">
      {path.map((nodeId, index) => {
        const type = getNodeType(nodeId, dag.nodes);
        const displayName = getNodeDisplayName(nodeId, dag.nodes);
        const isLast = index === path.length - 1;

        return (
          <div key={nodeId} className="flex items-center gap-1">
            <div
              className={`
                flex items-center gap-1.5 px-2 py-1 rounded
                bg-green-500/10 border border-green-500/30
                text-green-400
              `}
              title={`${type || 'node'}: ${nodeId}`}
            >
              {getNodeIcon(nodeId, type)}
              <span className="text-sm font-medium">{displayName}</span>
            </div>
            {!isLast && (
              <ChevronRight className="w-4 h-4 text-[var(--text-muted)]" />
            )}
          </div>
        );
      })}
    </div>
  );
}

export default PathVisualization;
