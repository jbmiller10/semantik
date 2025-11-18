import type { DataPoint } from 'embedding-atlas/react';
import { createRoot, type Root } from 'react-dom/client';
import type { ProjectionTooltipState } from '../hooks/useProjectionTooltip';

interface TooltipProps {
  tooltip: DataPoint | null;
  getTooltipIndex: (tooltip: DataPoint | null) => number | null;
  ids: Int32Array | undefined;
  tooltipState: ProjectionTooltipState;
}

// React component for the tooltip content
function TooltipContent({ tooltip, getTooltipIndex, ids, tooltipState }: TooltipProps) {
  if (!tooltip) return null;

  const idx = getTooltipIndex(tooltip);
  if (idx == null) return null;

  const selectedId = ids && idx >= 0 && idx < ids.length ? ids[idx] ?? null : null;
  const metadata =
    selectedId !== null && tooltipState.metadata?.selectedId === selectedId
      ? tooltipState.metadata
      : null;
  const status = metadata ? 'success' : tooltipState.status;

  if (status === 'idle' && !metadata) {
    return null;
  }

  const base =
    'pointer-events-none max-w-xs rounded-md border border-gray-200 bg-white/95 p-2 text-[12px] text-gray-700 shadow-md';

  if (status === 'loading' && !metadata) {
    return (
      <div role="tooltip" aria-live="polite" className={base}>
        <div className="text-gray-500">Loading...</div>
      </div>
    );
  }

  if (status === 'error' && !metadata) {
    return (
      <div role="tooltip" aria-live="polite" className={base}>
        <div className="text-gray-500">No metadata available</div>
      </div>
    );
  }

  if (!metadata) {
    return null;
  }

  const preview = metadata.contentPreview?.trim();
  const previewText =
    preview && preview.length > 0
      ? preview.slice(0, 200)
      : metadata.originalId
        ? 'No document metadata available for this point'
        : 'No metadata available';

  return (
    <div role="tooltip" aria-live="polite" className={base}>
      <div className="space-y-1">
        {metadata.originalId && !metadata.documentId && (
          <div className="text-gray-500">Point ID {metadata.originalId}</div>
        )}
        {metadata.documentLabel && (
          <div className="font-medium text-gray-800">{metadata.documentLabel}</div>
        )}
        {!metadata.documentLabel && metadata.documentId && (
          <div className="font-medium text-gray-800">Document {metadata.documentId}</div>
        )}
        {typeof metadata.chunkIndex === 'number' && (
          <div className="text-gray-500">Chunk #{metadata.chunkIndex}</div>
        )}
        <div className="text-gray-600">{previewText}</div>
      </div>
    </div>
  );
}

// Class-based component wrapper for Embedding Atlas
export class EmbeddingTooltipComponent {
  private root: Root | null = null;
  private props: TooltipProps;

  constructor(target: HTMLElement, props: TooltipProps) {
    this.props = props;

    // Create React root and render
    this.root = createRoot(target);
    this.render();
  }

  private render() {
    if (this.root) {
      this.root.render(
        <TooltipContent {...this.props} />
      );
    }
  }

  update(newProps: TooltipProps) {
    this.props = newProps;
    this.render();
  }

  destroy() {
    if (this.root) {
      this.root.unmount();
      this.root = null;
    }
  }
}

export class EmbeddingTooltipAdapter {
  private component: EmbeddingTooltipComponent | null;

  constructor(target: HTMLElement, props: TooltipProps) {
    this.component = new EmbeddingTooltipComponent(target, props);
  }

  update(props: TooltipProps) {
    this.component?.update(props);
  }

  destroy() {
    this.component?.destroy();
    this.component = null;
  }
}
