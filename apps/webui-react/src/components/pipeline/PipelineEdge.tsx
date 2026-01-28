/**
 * SVG component for rendering a pipeline edge (connection between nodes).
 */

import type { PipelineEdge, NodePosition } from '@/types/pipeline';
import { getNodeBottomCenter, getNodeTopCenter } from '@/utils/dagLayout';

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
}

/**
 * Format a predicate clause for display.
 * Extracts the most relevant info from the when clause.
 */
function formatPredicate(when: Record<string, unknown>): string {
  // Handle mime_type
  if ('mime_type' in when) {
    const mime = when.mime_type;
    if (Array.isArray(mime)) {
      return mime.map(m => extractMimeShortName(String(m))).join(', ');
    }
    return extractMimeShortName(String(mime));
  }

  // Handle extension
  if ('extension' in when) {
    const ext = when.extension;
    if (Array.isArray(ext)) {
      return ext.join(', ');
    }
    return String(ext);
  }

  // Generic: show first key
  const firstKey = Object.keys(when)[0];
  if (firstKey) {
    const value = when[firstKey];
    if (Array.isArray(value)) {
      return `${firstKey}: [...]`;
    }
    return `${firstKey}: ${value}`;
  }

  return '?';
}

/**
 * Extract short name from MIME type.
 * "application/pdf" -> "pdf"
 * "application/vnd.*" -> "office"
 */
function extractMimeShortName(mime: string): string {
  if (mime.includes('pdf')) return 'pdf';
  if (mime.includes('vnd.')) return 'office';
  if (mime.startsWith('text/')) return mime.replace('text/', '');
  if (mime.startsWith('image/')) return 'image';
  return mime.split('/').pop() || mime;
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
  const labelText = hasCondition
    ? formatPredicate(edge.when!)
    : (showCatchAll ? '*' : null);

  // Label position: beside the edge midpoint (offset to the right to avoid overlap)
  const labelX = (from.x + to.x) / 2 + 40;
  const labelY = midY;

  return (
    <g
      data-edge-id={edgeId}
      onClick={handleClick}
      style={{ cursor: onClick ? 'pointer' : 'default' }}
    >
      {/* Edge path */}
      <path
        d={verticalPath}
        fill="none"
        stroke={
          isHighlighted
            ? 'rgb(34, 197, 94)'
            : selected
              ? 'var(--text-primary)'
              : 'var(--text-muted)'
        }
        strokeWidth={isHighlighted || selected ? 2 : 1}
        markerEnd={isHighlighted ? 'url(#arrowhead-highlighted)' : 'url(#arrowhead)'}
        className={`${isNew ? 'pipeline-edge-new' : ''} ${isHighlighted ? 'pipeline-edge-highlighted' : ''}`}
      />

      {/* Predicate label */}
      {labelText && (
        <g>
          {/* Background for readability */}
          <rect
            x={labelX - 30}
            y={labelY - 10}
            width={60}
            height={16}
            rx={4}
            fill="var(--bg-primary)"
            opacity={0.9}
          />
          <text
            x={labelX}
            y={labelY}
            textAnchor="middle"
            fill={hasCondition ? 'var(--text-secondary)' : 'var(--text-muted)'}
            fontSize={11}
            fontFamily="monospace"
          >
            {labelText}
          </text>
        </g>
      )}
    </g>
  );
}

export default PipelineEdgeComponent;
