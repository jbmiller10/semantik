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
}

export function PipelineNodeComponent({
  node,
  position,
  selected,
  isSource = false,
  onClick,
}: PipelineNodeComponentProps) {
  const colors = isSource ? SOURCE_COLOR : NODE_COLORS[node.type];
  const borderRadius = 8;

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
    </g>
  );
}

export default PipelineNodeComponent;
