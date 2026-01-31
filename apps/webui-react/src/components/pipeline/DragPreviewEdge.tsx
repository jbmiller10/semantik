/**
 * SVG component for rendering a preview edge during drag-to-connect.
 * Shows a dashed line from the source node's output port to the cursor position.
 */

interface DragPreviewEdgeProps {
  /** Starting position (source node's output port) */
  from: { x: number; y: number };
  /** Ending position (cursor position) */
  to: { x: number; y: number };
}

export function DragPreviewEdge({ from, to }: DragPreviewEdgeProps) {
  // Use same curve logic as PipelineEdge but dashed
  // Cubic bezier with control points for smooth vertical curve
  const midY = (from.y + to.y) / 2;
  const path = `M ${from.x} ${from.y} C ${from.x} ${midY}, ${to.x} ${midY}, ${to.x} ${to.y}`;

  return (
    <path
      d={path}
      fill="none"
      stroke="var(--text-muted)"
      strokeWidth={2}
      strokeDasharray="6 4"
      className="pointer-events-none preview-edge"
      style={{
        animation: 'dash-flow 0.5s linear infinite',
      }}
    />
  );
}

export default DragPreviewEdge;
