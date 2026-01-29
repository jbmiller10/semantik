/**
 * Horizontal visualization of the routing path through the pipeline.
 * Shows nodes connected by arrows with the selected path highlighted.
 * Supports multiple paths for parallel fan-out visualization.
 */

import { ChevronRight, FileInput, Box, Layers, Sparkles, Cpu, GitFork } from 'lucide-react';
import type { PipelineDAG, PipelineNode as PipelineNodeType } from '@/types/pipeline';
import type { PathInfo } from '@/types/routePreview';

interface PathVisualizationProps {
  /** The computed primary path (list of node IDs) */
  path: string[];
  /** All execution paths for parallel fan-out (null for single-path DAGs) */
  paths?: PathInfo[] | null;
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

/**
 * Render a single path row.
 */
function SinglePathRow({
  pathNodes,
  dag,
  colorClass = 'bg-green-500/10 border-green-500/30 text-green-400',
}: {
  pathNodes: string[];
  dag: PipelineDAG;
  colorClass?: string;
}) {
  return (
    <div className="flex items-center gap-1 flex-wrap">
      {pathNodes.map((nodeId, index) => {
        const type = getNodeType(nodeId, dag.nodes);
        const displayName = getNodeDisplayName(nodeId, dag.nodes);
        const isLast = index === pathNodes.length - 1;

        return (
          <div key={nodeId} className="flex items-center gap-1">
            <div
              className={`flex items-center gap-1.5 px-2 py-1 rounded border ${colorClass}`}
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

export function PathVisualization({ path, paths, dag }: PathVisualizationProps) {
  if (path.length === 0) {
    return (
      <div className="text-sm text-[var(--text-muted)] italic">No path computed</div>
    );
  }

  // Multiple paths (parallel fan-out)
  if (paths && paths.length > 1) {
    return (
      <div className="space-y-3">
        {paths.map((p, idx) => {
          const isPrimary = idx === 0;
          const colorClass = isPrimary
            ? 'bg-green-500/10 border-green-500/30 text-green-400'
            : 'bg-blue-500/10 border-blue-500/30 text-blue-400';

          return (
            <div key={p.path_name} className="space-y-1">
              <div className="flex items-center gap-2">
                <GitFork className="w-4 h-4 text-blue-400" />
                <span className="text-xs text-[var(--text-secondary)] font-medium">
                  {p.path_name}
                </span>
                {isPrimary && (
                  <span className="text-xs text-green-400">(primary)</span>
                )}
              </div>
              <SinglePathRow pathNodes={p.nodes} dag={dag} colorClass={colorClass} />
            </div>
          );
        })}
      </div>
    );
  }

  // Single path (no parallel fan-out)
  return <SinglePathRow pathNodes={path} dag={dag} />;
}

export default PathVisualization;
