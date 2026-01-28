/**
 * Main pipeline visualization component.
 * Renders a DAG of pipeline nodes and edges in SVG.
 */

import { useMemo, useCallback, useRef, useEffect } from 'react';
import type { PipelineVisualizationProps } from '@/types/pipeline';
import { computeDAGLayout, getNodeBottomCenter } from '@/utils/dagLayout';
import { useDragToConnect } from '@/hooks/useDragToConnect';
import { PipelineNodeComponent } from './PipelineNode';
import { PipelineEdgeComponent } from './PipelineEdge';
import { DragPreviewEdge } from './DragPreviewEdge';

/**
 * Convert screen coordinates to SVG coordinates.
 * Handles any transforms applied to the SVG.
 */
function screenToSVG(
  svg: SVGSVGElement,
  screenX: number,
  screenY: number
): { x: number; y: number } {
  const point = svg.createSVGPoint();
  point.x = screenX;
  point.y = screenY;
  const ctm = svg.getScreenCTM();
  if (!ctm) return { x: screenX, y: screenY };
  const transformed = point.matrixTransform(ctm.inverse());
  return { x: transformed.x, y: transformed.y };
}

export function PipelineVisualization({
  dag,
  selection = { type: 'none' },
  onSelectionChange,
  readOnly = false,
  className = '',
}: PipelineVisualizationProps) {
  // SVG ref for coordinate conversion
  const svgRef = useRef<SVGSVGElement>(null);

  // Compute layout
  const layout = useMemo(() => computeDAGLayout(dag), [dag]);

  // Drag-to-connect state and handlers
  const { dragState, startDrag, updateDrag, endDrag, cancelDrag, isValidDropTarget } =
    useDragToConnect({
      dag,
      layout,
    });

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

  // Handle drag start from a node's output port
  const handleStartDrag = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    (nodeId: string, screenPosition: { x: number; y: number }) => {
      if (readOnly || !svgRef.current) return;
      // Get the node's output port position in SVG coordinates
      const nodePosition = layout.nodes.get(nodeId);
      if (!nodePosition) return;
      const portPosition = getNodeBottomCenter(nodePosition);
      startDrag(nodeId, portPosition);
    },
    [readOnly, layout, startDrag]
  );

  // Handle mouse move during drag
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (!dragState.isDragging || !svgRef.current) return;
      const svgPosition = screenToSVG(svgRef.current, e.clientX, e.clientY);
      updateDrag(svgPosition);
    },
    [dragState.isDragging, updateDrag]
  );

  // Handle mouse up (end drag)
  const handleMouseUp = useCallback(() => {
    if (dragState.isDragging) {
      endDrag();
    }
  }, [dragState.isDragging, endDrag]);

  // Handle mouse leave (cancel drag)
  const handleMouseLeave = useCallback(() => {
    if (dragState.isDragging) {
      cancelDrag();
    }
  }, [dragState.isDragging, cancelDrag]);

  // Handle Escape key to cancel drag
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && dragState.isDragging) {
        cancelDrag();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [dragState.isDragging, cancelDrag]);

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
        ref={svgRef}
        width={layout.width}
        height={layout.height}
        onClick={handleBackgroundClick}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
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
            onStartDrag={readOnly ? undefined : handleStartDrag}
            showPorts={dragState.isDragging}
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
                onStartDrag={readOnly ? undefined : handleStartDrag}
                showPorts={dragState.isDragging}
                isValidDropTarget={isValidDropTarget(node.id)}
              />
            );
          })}
        </g>

        {/* Render preview edge during drag */}
        {dragState.isDragging && dragState.sourcePosition && dragState.cursorPosition && (
          <DragPreviewEdge
            from={dragState.sourcePosition}
            to={dragState.cursorPosition}
          />
        )}
      </svg>
    </div>
  );
}

export default PipelineVisualization;
