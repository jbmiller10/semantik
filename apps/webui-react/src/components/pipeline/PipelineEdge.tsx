/**
 * SVG component for rendering a pipeline edge (connection between nodes).
 */

import type { PipelineEdge, NodePosition } from '@/types/pipeline';
import { getNodeRightCenter, getNodeLeftCenter } from '@/utils/dagLayout';

interface PipelineEdgeComponentProps {
  edge: PipelineEdge;
  fromPosition: NodePosition;
  toPosition: NodePosition;
  selected: boolean;
  showCatchAll?: boolean;
  onClick?: (fromNode: string, toNode: string) => void;
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
}: PipelineEdgeComponentProps) {
  const from = getNodeRightCenter(fromPosition);
  const to = getNodeLeftCenter(toPosition);

  // Curved path using cubic bezier
  const controlOffset = 30;
  const simplePath = `M ${from.x} ${from.y} C ${from.x + controlOffset} ${from.y}, ${to.x - controlOffset} ${to.y}, ${to.x} ${to.y}`;

  const handleClick = () => {
    onClick?.(edge.from_node, edge.to_node);
  };

  const edgeId = `${edge.from_node}-${edge.to_node}`;
  const hasCondition = edge.when !== null;
  const labelText = hasCondition
    ? formatPredicate(edge.when!)
    : (showCatchAll ? '*' : null);

  // Label position: midpoint of the edge
  const labelX = (from.x + to.x) / 2;
  const labelY = (from.y + to.y) / 2 - 8;

  return (
    <g
      data-edge-id={edgeId}
      onClick={handleClick}
      style={{ cursor: onClick ? 'pointer' : 'default' }}
    >
      {/* Edge path */}
      <path
        d={simplePath}
        fill="none"
        stroke={selected ? 'var(--text-primary)' : 'var(--text-muted)'}
        strokeWidth={selected ? 2 : 1}
        markerEnd="url(#arrowhead)"
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
