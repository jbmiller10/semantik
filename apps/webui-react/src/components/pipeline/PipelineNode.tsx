/**
 * SVG component for rendering a single pipeline node.
 */

import type { PipelineNode, NodePosition, NodeType } from '@/types/pipeline';

// Color scheme for node types (neutral/white aesthetic per design language)
const NODE_COLORS: Record<NodeType, { bg: string; border: string }> = {
  parser: { bg: 'var(--bg-secondary)', border: 'var(--border)' },
  chunker: { bg: 'var(--bg-secondary)', border: 'var(--border)' },
  extractor: { bg: 'var(--bg-secondary)', border: 'var(--border)' },
  embedder: { bg: 'var(--bg-tertiary)', border: 'var(--border-strong, var(--border))' },
};

// Source node has distinct styling
const SOURCE_COLOR = { bg: 'var(--bg-tertiary)', border: 'var(--text-muted)' };

interface PipelineNodeComponentProps {
  node: PipelineNode;
  position: NodePosition;
  selected: boolean;
  isSource?: boolean;
  onClick?: (nodeId: string) => void;
  /** Callback when drag starts from output port */
  onStartDrag?: (nodeId: string, position: { x: number; y: number }) => void;
  /** Whether to show ports (during drag operations) */
  showPorts?: boolean;
  /** Whether this node is a valid drop target */
  isValidDropTarget?: boolean;
}

// Port styling constants
const PORT_RADIUS = 6;

export function PipelineNodeComponent({
  node,
  position,
  selected,
  isSource = false,
  onClick,
  onStartDrag,
  showPorts = false,
  isValidDropTarget = false,
}: PipelineNodeComponentProps) {
  const colors = isSource ? SOURCE_COLOR : NODE_COLORS[node.type];
  const borderRadius = 8;

  // Determine if this is an embedder (terminal node - no output port)
  const isEmbedder = node.type === 'embedder';

  // Show ports on hover or when dragging
  const shouldShowPorts = showPorts;

  const handleClick = () => {
    onClick?.(node.id);
  };

  return (
    <g
      data-node-id={node.id}
      onClick={handleClick}
      style={{ cursor: onClick ? 'pointer' : 'default' }}
    >
      {/* Node background */}
      <rect
        x={position.x}
        y={position.y}
        width={position.width}
        height={position.height}
        rx={borderRadius}
        ry={borderRadius}
        fill={colors.bg}
        stroke={selected ? 'var(--text-primary)' : colors.border}
        strokeWidth={selected ? 2 : 1}
      />

      {/* Type label (top) */}
      <text
        x={position.x + position.width / 2}
        y={position.y + 20}
        textAnchor="middle"
        fill="var(--text-muted)"
        fontSize={10}
        fontWeight={500}
        style={{ textTransform: 'uppercase', letterSpacing: '0.05em' }}
      >
        {isSource ? 'SOURCE' : node.type.toUpperCase()}
      </text>

      {/* Plugin name (center) */}
      <text
        x={position.x + position.width / 2}
        y={position.y + position.height / 2 + 5}
        textAnchor="middle"
        fill="var(--text-primary)"
        fontSize={14}
        fontWeight={600}
      >
        {isSource ? 'Source' : node.plugin_id}
      </text>

      {/* Config indicator (bottom) - only if has non-empty config */}
      {!isSource && Object.keys(node.config).length > 0 && (
        <text
          x={position.x + position.width / 2}
          y={position.y + position.height - 12}
          textAnchor="middle"
          fill="var(--text-muted)"
          fontSize={10}
        >
          {Object.keys(node.config).length} options
        </text>
      )}

      {/* Input port (top center) - not shown on source node */}
      {!isSource && (
        <circle
          cx={position.x + position.width / 2}
          cy={position.y}
          r={PORT_RADIUS}
          className="input-port"
          fill={isValidDropTarget ? 'var(--text-primary)' : 'var(--bg-tertiary)'}
          stroke={isValidDropTarget ? 'var(--text-primary)' : 'var(--border)'}
          strokeWidth={1}
          style={{
            opacity: shouldShowPorts || isValidDropTarget ? 1 : 0,
            transition: 'opacity 0.15s ease-in-out, fill 0.15s ease-in-out',
            pointerEvents: 'none',
          }}
        />
      )}

      {/* Output port (bottom center) - not shown on embedder (terminal) */}
      {!isEmbedder && (
        <circle
          cx={position.x + position.width / 2}
          cy={position.y + position.height}
          r={PORT_RADIUS}
          className="output-port"
          fill="var(--bg-tertiary)"
          stroke="var(--border)"
          strokeWidth={1}
          style={{
            opacity: shouldShowPorts ? 1 : 0,
            transition: 'opacity 0.15s ease-in-out',
            cursor: onStartDrag ? 'crosshair' : 'default',
          }}
          onMouseDown={(e) => {
            if (onStartDrag) {
              e.stopPropagation();
              onStartDrag(node.id, { x: e.clientX, y: e.clientY });
            }
          }}
        />
      )}
    </g>
  );
}

export default PipelineNodeComponent;
