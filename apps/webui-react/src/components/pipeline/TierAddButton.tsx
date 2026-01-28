/**
 * "+" button component for adding nodes to a tier.
 * Provides a fallback interaction for touch devices and discoverability.
 */

import type { NodeType } from '@/types/pipeline';
import { NODE_TYPE_LABELS } from '@/utils/pipelinePluginMapping';

export interface TierAddButtonProps {
  tier: NodeType;
  position: { x: number; y: number };
  onClick: () => void;
  disabled?: boolean;
}

export function TierAddButton({
  tier,
  position,
  onClick,
  disabled = false,
}: TierAddButtonProps) {
  return (
    <g
      className={`tier-add-button ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
      transform={`translate(${position.x}, ${position.y})`}
      onClick={disabled ? undefined : onClick}
      data-tier={tier}
      role="button"
      aria-label={`Add ${NODE_TYPE_LABELS[tier].toLowerCase()}`}
    >
      {/* Background circle */}
      <circle
        r={14}
        fill="var(--bg-tertiary)"
        stroke="var(--border)"
        strokeWidth={1}
        className={`transition-all duration-150 ${
          !disabled ? 'hover:fill-[var(--bg-secondary)] hover:stroke-[var(--text-primary)]' : ''
        }`}
      />
      {/* Plus icon - rendered as SVG lines since we're in SVG context */}
      <line
        x1={-5}
        y1={0}
        x2={5}
        y2={0}
        stroke="var(--text-muted)"
        strokeWidth={2}
        strokeLinecap="round"
        className={`transition-all duration-150 ${
          !disabled ? 'group-hover:stroke-[var(--text-primary)]' : ''
        }`}
      />
      <line
        x1={0}
        y1={-5}
        x2={0}
        y2={5}
        stroke="var(--text-muted)"
        strokeWidth={2}
        strokeLinecap="round"
        className={`transition-all duration-150 ${
          !disabled ? 'group-hover:stroke-[var(--text-primary)]' : ''
        }`}
      />
    </g>
  );
}

export default TierAddButton;
