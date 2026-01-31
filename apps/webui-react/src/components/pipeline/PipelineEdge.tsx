/**
 * SVG component for rendering a pipeline edge (connection between nodes).
 */

import type { PipelineEdge, NodePosition } from '@/types/pipeline';
import { getNodeBottomCenter, getNodeTopCenter } from '@/utils/dagLayout';

// Offset in pixels between parallel edge lines
const PARALLEL_OFFSET = 3;

interface PipelineEdgeComponentProps {
  edge: PipelineEdge;
  fromPosition: NodePosition;
  toPosition: NodePosition;
  selected: boolean;
  showCatchAll?: boolean;
  onClick?: (fromNode: string, toNode: string) => void;
  /** Whether this is a newly created edge (for draw-in animation) */
  isNew?: boolean;
  /** Whether this edge is in the highlighted route preview path */
  isHighlighted?: boolean;
  /** Custom highlight color (defaults to green) */
  highlightColor?: string;
  /** Priority index (1-based) for edges from the same source node */
  priority?: number;
  /** Total number of edges from the same source node */
  totalFromSource?: number;
}

/**
 * Format full predicate for tooltip display.
 * Shows complete information about the condition.
 */
function formatTooltipPredicate(when: Record<string, unknown>): string {
  const entries = Object.entries(when);
  if (entries.length === 0) return 'Empty condition';

  return entries.map(([key, value]) => {
    const label = key.replace(/_/g, ' ').replace(/\./g, ' > ');
    const capitalizedLabel = label.charAt(0).toUpperCase() + label.slice(1);

    if (Array.isArray(value)) {
      return `${capitalizedLabel}: ${value.join(', ')}`;
    }
    return `${capitalizedLabel}: ${String(value)}`;
  }).join('\n');
}


export function PipelineEdgeComponent({
  edge,
  fromPosition,
  toPosition,
  selected,
  showCatchAll = false,
  onClick,
  isNew = false,
  isHighlighted = false,
  highlightColor,
  priority,
  totalFromSource,
}: PipelineEdgeComponentProps) {
  // Vertical layout: edges go from bottom of source to top of target
  const from = getNodeBottomCenter(fromPosition);
  const to = getNodeTopCenter(toPosition);

  // Curved path using cubic bezier (vertical flow)
  // Control points offset in Y direction for smooth vertical curves
  const midY = (from.y + to.y) / 2;
  const verticalPath = `M ${from.x} ${from.y} C ${from.x} ${midY}, ${to.x} ${midY}, ${to.x} ${to.y}`;

  const handleClick = () => {
    onClick?.(edge.from_node, edge.to_node);
  };

  const edgeId = `${edge.from_node}-${edge.to_node}`;
  const hasCondition = edge.when !== null;
  const isParallel = edge.parallel ?? false;

  // Create offset path for parallel edges (railroad-track effect)
  const createOffsetPath = (offset: number) => {
    const offsetFrom = { x: from.x + offset, y: from.y };
    const offsetTo = { x: to.x + offset, y: to.y };
    return `M ${offsetFrom.x} ${offsetFrom.y} C ${offsetFrom.x} ${midY}, ${offsetTo.x} ${midY}, ${offsetTo.x} ${offsetTo.y}`;
  };

  const leftPath = isParallel ? createOffsetPath(-PARALLEL_OFFSET) : verticalPath;
  const rightPath = isParallel ? createOffsetPath(PARALLEL_OFFSET) : null;

  // Indicator position: beside the edge midpoint (offset to the right to avoid overlap)
  const labelX = (from.x + to.x) / 2 + 40;
  const labelY = midY;

  // Common path styling
  const pathStroke = isHighlighted
    ? (highlightColor || 'rgb(34, 197, 94)')
    : selected
      ? 'var(--text-primary)'
      : 'var(--text-muted)';
  const pathStrokeWidth = isHighlighted || selected ? 2 : 1.5;
  const pathStrokeDasharray = !hasCondition && showCatchAll ? '4 2' : undefined;
  const pathMarkerEnd = isHighlighted ? 'url(#arrowhead-highlighted)' : 'url(#arrowhead)';
  const pathClassName = `${isNew ? 'pipeline-edge-new' : ''} ${isHighlighted ? 'pipeline-edge-highlighted' : ''}`;

  return (
    <g
      data-edge-id={edgeId}
      onClick={handleClick}
      style={{ cursor: onClick ? 'pointer' : 'default' }}
      className={isParallel ? 'pipeline-edge-parallel' : ''}
    >
      {/* Primary edge path */}
      <path
        d={leftPath}
        fill="none"
        stroke={pathStroke}
        strokeWidth={pathStrokeWidth}
        strokeDasharray={pathStrokeDasharray}
        markerEnd={pathMarkerEnd}
        className={pathClassName}
      />

      {/* Secondary path for parallel edges */}
      {rightPath && (
        <path
          d={rightPath}
          fill="none"
          stroke={pathStroke}
          strokeWidth={pathStrokeWidth}
          strokeDasharray={pathStrokeDasharray}
          markerEnd={pathMarkerEnd}
          className={pathClassName}
        />
      )}

      {/* Priority badge (shown when multiple edges from same source) */}
      {priority !== undefined && totalFromSource !== undefined && totalFromSource > 1 && (
        <g className="priority-badge">
          <circle
            cx={labelX - 45}
            cy={labelY}
            r={10}
            fill="var(--bg-secondary)"
            stroke="var(--border)"
            strokeWidth={1}
          />
          <text
            x={labelX - 45}
            y={labelY + 4}
            textAnchor="middle"
            fill="var(--text-secondary)"
            fontSize={10}
            fontWeight={500}
          >
            {priority}
          </text>
          <title>Evaluated #{priority} of {totalFromSource} edges from {edge.from_node === '_source' ? 'Source' : edge.from_node}</title>
        </g>
      )}

      {/* Indicator dot for conditional edges */}
      {hasCondition && (
        <g>
          <circle
            className="edge-indicator"
            cx={labelX}
            cy={labelY}
            r={6}
            fill="rgb(59, 130, 246)"
            opacity={0.9}
          >
            <title>{formatTooltipPredicate(edge.when!)}</title>
          </circle>
        </g>
      )}

      {/* Catch-all indicator (asterisk in small circle) */}
      {!hasCondition && showCatchAll && (
        <g>
          <circle
            className="edge-indicator edge-indicator-catchall"
            cx={labelX}
            cy={labelY}
            r={6}
            fill="var(--text-muted)"
            opacity={0.5}
          >
            <title>Catch-all route (matches everything)</title>
          </circle>
          <text
            x={labelX}
            y={labelY + 3}
            textAnchor="middle"
            fill="var(--bg-primary)"
            fontSize={8}
            fontWeight={700}
          >
            *
          </text>
        </g>
      )}
    </g>
  );
}

export default PipelineEdgeComponent;
