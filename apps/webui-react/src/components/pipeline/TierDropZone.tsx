/**
 * Drop zone component for creating new nodes in a tier.
 * Displays a dashed rectangle that users can drop connections onto.
 *
 * Animations:
 * - Fades in when dragging starts (isActive becomes true)
 * - Pulses when cursor enters (isHovered becomes true)
 */

import type { NodeType } from '@/types/pipeline';
import { NODE_TYPE_LABELS } from '@/utils/pipelinePluginMapping';

export interface TierDropZoneProps {
  tier: NodeType;
  bounds: { x: number; y: number; width: number; height: number };
  isActive: boolean; // true when dragging and this tier is valid
  isHovered: boolean;
  onDrop?: () => void;
}

export function TierDropZone({
  tier,
  bounds,
  isActive,
  isHovered,
}: TierDropZoneProps) {
  if (!isActive) return null;

  const label = `+ Add ${NODE_TYPE_LABELS[tier].toLowerCase()}`;

  return (
    <g
      className={`tier-drop-zone tier-drop-zone-fade-in ${isHovered ? 'tier-drop-zone-hover' : ''}`}
      data-tier={tier}
    >
      <rect
        x={bounds.x}
        y={bounds.y}
        width={bounds.width}
        height={bounds.height}
        fill={isHovered ? 'var(--bg-tertiary)' : 'none'}
        fillOpacity={isHovered ? 0.5 : 0}
        stroke={isHovered ? 'var(--text-primary)' : 'var(--border)'}
        strokeDasharray={isHovered ? 'none' : '8 4'}
        strokeWidth={isHovered ? 2 : 1}
        rx={8}
        className="transition-all duration-200"
      />
      <text
        x={bounds.x + bounds.width / 2}
        y={bounds.y + bounds.height / 2}
        textAnchor="middle"
        dominantBaseline="middle"
        className={`text-sm select-none pointer-events-none transition-colors duration-150 ${
          isHovered ? 'fill-[var(--text-primary)]' : 'fill-[var(--text-muted)]'
        }`}
      >
        {label}
      </text>
    </g>
  );
}

export default TierDropZone;
