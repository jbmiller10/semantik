/**
 * Main pipeline visualization component.
 * Renders a DAG of pipeline nodes and edges in SVG.
 */

import { useMemo, useCallback } from 'react';
import type { PipelineVisualizationProps } from '@/types/pipeline';
import { computeDAGLayout } from '@/utils/dagLayout';
import { PipelineNodeComponent } from './PipelineNode';
import { PipelineEdgeComponent } from './PipelineEdge';

export function PipelineVisualization({
  dag,
  selection = { type: 'none' },
  onSelectionChange,
  readOnly = false,
  className = '',
}: PipelineVisualizationProps) {
  // Compute layout
  const layout = useMemo(() => computeDAGLayout(dag), [dag]);

  // Handle node click
  const handleNodeClick = useCallback((nodeId: string) => {
    if (readOnly || !onSelectionChange) return;
    onSelectionChange({ type: 'node', nodeId });
  }, [readOnly, onSelectionChange]);

  // Handle edge click
  const handleEdgeClick = useCallback((fromNode: string, toNode: string) => {
    if (readOnly || !onSelectionChange) return;
    onSelectionChange({ type: 'edge', fromNode, toNode });
  }, [readOnly, onSelectionChange]);

  // Handle background click (clear selection)
  const handleBackgroundClick = useCallback((e: React.MouseEvent) => {
    // Only clear if clicking the SVG background itself
    if (e.target === e.currentTarget && onSelectionChange) {
      onSelectionChange({ type: 'none' });
    }
  }, [onSelectionChange]);

  // Check if node is selected
  const isNodeSelected = (nodeId: string) =>
    selection.type === 'node' && selection.nodeId === nodeId;

  // Check if edge is selected
  const isEdgeSelected = (fromNode: string, toNode: string) =>
    selection.type === 'edge' &&
    selection.fromNode === fromNode &&
    selection.toNode === toNode;

  // Empty state
  if (dag.nodes.length === 0) {
    return (
      <div className={`flex items-center justify-center h-full ${className}`}>
        <div className="text-center">
          <p className="text-[var(--text-muted)]">No pipeline configured</p>
          <p className="text-sm text-[var(--text-muted)] mt-1">
            Add nodes to build your pipeline
          </p>
        </div>
      </div>
    );
  }

  // Get source node position (always present when edges exist)
  const sourcePosition = layout.nodes.get('_source');

  return (
    <div className={`overflow-auto ${className}`}>
      <svg
        width={layout.width}
        height={layout.height}
        onClick={handleBackgroundClick}
        style={{ minWidth: '100%', minHeight: '100%' }}
      >
        {/* Arrow marker definition */}
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon
              points="0 0, 10 3.5, 0 7"
              fill="var(--text-muted)"
            />
          </marker>
          <marker
            id="arrowhead-selected"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon
              points="0 0, 10 3.5, 0 7"
              fill="var(--text-primary)"
            />
          </marker>
        </defs>

        {/* Render edges first (below nodes) */}
        <g className="edges">
          {dag.edges.map((edge, index) => {
            const fromPos = layout.nodes.get(edge.from_node);
            const toPos = layout.nodes.get(edge.to_node);

            if (!fromPos || !toPos) return null;

            const selected = isEdgeSelected(edge.from_node, edge.to_node);
            // Show catch-all (*) only for source edges
            const showCatchAll = edge.from_node === '_source' && edge.when === null;

            return (
              <PipelineEdgeComponent
                key={`${edge.from_node}-${edge.to_node}-${index}`}
                edge={edge}
                fromPosition={fromPos}
                toPosition={toPos}
                selected={selected}
                showCatchAll={showCatchAll}
                onClick={readOnly ? undefined : handleEdgeClick}
              />
            );
          })}
        </g>

        {/* Render source node */}
        {sourcePosition && (
          <PipelineNodeComponent
            node={{ id: '_source', type: 'parser', plugin_id: 'source', config: {} }}
            position={sourcePosition}
            selected={isNodeSelected('_source')}
            isSource={true}
            onClick={readOnly ? undefined : handleNodeClick}
          />
        )}

        {/* Render pipeline nodes */}
        <g className="nodes">
          {dag.nodes.map((node) => {
            const position = layout.nodes.get(node.id);
            if (!position) return null;

            return (
              <PipelineNodeComponent
                key={node.id}
                node={node}
                position={position}
                selected={isNodeSelected(node.id)}
                onClick={readOnly ? undefined : handleNodeClick}
              />
            );
          })}
        </g>
      </svg>
    </div>
  );
}

export default PipelineVisualization;
