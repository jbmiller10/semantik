/**
 * "+" button component for adding nodes to a tier.
 * Provides a fallback interaction for touch devices and discoverability.
 *
 * On touch devices:
 * - Always visible (not hidden during drag since drag is disabled)
 * - Larger touch target (44x44px per Apple HIG)
 */

import type { NodeType } from '@/types/pipeline';
import { NODE_TYPE_LABELS } from '@/utils/pipelinePluginMapping';

// Button sizes: larger on touch for better tap targets (Apple HIG: min 44px)
const TOUCH_RADIUS = 22; // 44px diameter
const DESKTOP_RADIUS = 14; // 28px diameter

export interface TierAddButtonProps {
  tier: NodeType;
  position: { x: number; y: number };
  onClick: () => void;
  disabled?: boolean;
  /** True on touch devices - shows larger button */
  isTouch?: boolean;
}

export function TierAddButton({
  tier,
  position,
  onClick,
  disabled = false,
  isTouch = false,
}: TierAddButtonProps) {
  const radius = isTouch ? TOUCH_RADIUS : DESKTOP_RADIUS;
  const iconSize = isTouch ? 8 : 5; // Scale icon proportionally

  return (
    <g
      className={`tier-add-button ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
      transform={`translate(${position.x}, ${position.y})`}
      onClick={disabled ? undefined : onClick}
      data-tier={tier}
      data-touch={isTouch ? 'true' : undefined}
      role="button"
      aria-label={`Add ${NODE_TYPE_LABELS[tier].toLowerCase()}`}
    >
      {/* Background circle */}
      <circle
        r={radius}
        fill="var(--bg-tertiary)"
        stroke="var(--border)"
        strokeWidth={1}
        className={`transition-all duration-150 ${
          !disabled ? 'hover:fill-[var(--bg-secondary)] hover:stroke-[var(--text-primary)]' : ''
        }`}
      />
      {/* Plus icon - rendered as SVG lines since we're in SVG context */}
      <line
        x1={-iconSize}
        y1={0}
        x2={iconSize}
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
        y1={-iconSize}
        x2={0}
        y2={iconSize}
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
