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

  // Multiple paths (parallel fan-out) - show tree structure
  if (paths && paths.length > 1) {
    // Find common prefix (nodes that appear in all paths at the same position)
    const findCommonPrefix = (): string[] => {
      const prefix: string[] = [];
      const minLength = Math.min(...paths.map(p => p.nodes.length));

      for (let i = 0; i < minLength; i++) {
        const nodeAtPosition = paths[0].nodes[i];
        if (paths.every(p => p.nodes[i] === nodeAtPosition)) {
          prefix.push(nodeAtPosition);
        } else {
          break;
        }
      }
      return prefix;
    };

    const commonPrefix = findCommonPrefix();
    const divergencePoint = commonPrefix.length;

    return (
      <div className="space-y-3">
        {/* Common prefix (shared path) */}
        {commonPrefix.length > 0 && (
          <div className="flex items-center gap-1 flex-wrap">
            {commonPrefix.map((nodeId, index) => {
              const type = getNodeType(nodeId, dag.nodes);
              const displayName = getNodeDisplayName(nodeId, dag.nodes);
              const isLast = index === commonPrefix.length - 1;

              return (
                <div key={nodeId} className="flex items-center gap-1">
                  <div
                    className="flex items-center gap-1.5 px-2 py-1 rounded border bg-gray-500/10 border-gray-500/30 text-gray-400"
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
            {/* Divergence indicator */}
            <div className="flex items-center gap-1">
              <ChevronRight className="w-4 h-4 text-[var(--text-muted)]" />
              <GitFork className="w-5 h-5 text-blue-400" />
            </div>
          </div>
        )}

        {/* Divergent paths */}
        <div className="path-divergence ml-4 pl-4 border-l-2 border-blue-500/30 space-y-2">
          {paths.map((p, idx) => {
            const isPrimary = idx === 0;
            const colorClass = isPrimary
              ? 'bg-green-500/10 border-green-500/30 text-green-400'
              : 'bg-blue-500/10 border-blue-500/30 text-blue-400';

            // Get nodes after divergence point
            const divergentNodes = p.nodes.slice(divergencePoint);

            return (
              <div key={p.path_name} className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className={`text-xs font-medium ${isPrimary ? 'text-green-400' : 'text-blue-400'}`}>
                    {p.path_name}
                  </span>
                  {isPrimary && (
                    <span className="text-xs bg-green-500/20 text-green-400 px-1.5 py-0.5 rounded">
                      primary
                    </span>
                  )}
                </div>
                {divergentNodes.length > 0 ? (
                  <SinglePathRow pathNodes={divergentNodes} dag={dag} colorClass={colorClass} />
                ) : (
                  <span className="text-xs text-[var(--text-muted)] italic">No additional nodes</span>
                )}
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  // Single path (no parallel fan-out)
  return <SinglePathRow pathNodes={path} dag={dag} />;
}

export default PathVisualization;
