/**
 * Modal for selecting which upstream nodes to connect from.
 * Used when clicking "+" button and there are multiple potential upstream nodes.
 */

import { useState, useEffect, useRef } from 'react';
import { X, Check } from 'lucide-react';
import type { PipelineNode } from '@/types/pipeline';

export interface UpstreamNodePickerProps {
  upstreamNodes: PipelineNode[];
  position: { x: number; y: number };
  onSelect: (nodeIds: string[]) => void;
  onCancel: () => void;
}

export function UpstreamNodePicker({
  upstreamNodes,
  position,
  onSelect,
  onCancel,
}: UpstreamNodePickerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => {
    // Default all selected
    return new Set(upstreamNodes.map((n) => n.id));
  });

  // Handle click outside to cancel
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        onCancel();
      }
    };

    const timeoutId = setTimeout(() => {
      document.addEventListener('mousedown', handleClickOutside);
    }, 0);

    return () => {
      clearTimeout(timeoutId);
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [onCancel]);

  // Handle Escape key to cancel
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onCancel();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onCancel]);

  const toggleNode = (nodeId: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(nodeId)) {
        // Don't allow deselecting all
        if (next.size > 1) {
          next.delete(nodeId);
        }
      } else {
        next.add(nodeId);
      }
      return next;
    });
  };

  const handleConfirm = () => {
    onSelect(Array.from(selectedIds));
  };

  // Keep popover in viewport
  let left = position.x;
  let top = position.y + 20;

  if (typeof window !== 'undefined') {
    const popoverWidth = 260;
    const popoverHeight = 200;
    const padding = 16;

    if (left + popoverWidth + padding > window.innerWidth) {
      left = window.innerWidth - popoverWidth - padding;
    }
    if (left < padding) {
      left = padding;
    }
    if (top + popoverHeight + padding > window.innerHeight) {
      top = position.y - popoverHeight - 10;
    }
    if (top < padding) {
      top = padding;
    }
  }

  return (
    <div
      ref={containerRef}
      className="fixed z-50 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg shadow-lg min-w-[240px]"
      style={{ left, top }}
      role="dialog"
      aria-label="Select upstream nodes"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-[var(--border)]">
        <span className="text-sm font-medium text-[var(--text-secondary)]">
          Connect from
        </span>
        <button
          onClick={onCancel}
          className="p-1 rounded hover:bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
          aria-label="Cancel"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Node list */}
      <div className="p-2 max-h-[180px] overflow-y-auto">
        <div className="space-y-1">
          {upstreamNodes.map((node) => {
            const isSelected = selectedIds.has(node.id);
            return (
              <label
                key={node.id}
                className={`flex items-center gap-2 px-2 py-1.5 rounded cursor-pointer transition-colors ${
                  isSelected
                    ? 'bg-[var(--bg-tertiary)]'
                    : 'hover:bg-[var(--bg-tertiary)]'
                }`}
              >
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={() => toggleNode(node.id)}
                  className="sr-only"
                />
                <div
                  className={`w-4 h-4 rounded border flex items-center justify-center transition-colors ${
                    isSelected
                      ? 'bg-[var(--text-primary)] border-[var(--text-primary)]'
                      : 'border-[var(--border)]'
                  }`}
                >
                  {isSelected && (
                    <Check className="w-3 h-3 text-[var(--bg-primary)]" />
                  )}
                </div>
                <span className="text-sm text-[var(--text-primary)]">
                  {node.plugin_id}
                </span>
                <span className="text-xs text-[var(--text-muted)]">
                  ({node.type})
                </span>
              </label>
            );
          })}
        </div>
      </div>

      {/* Footer */}
      <div className="flex justify-end gap-2 px-3 py-2 border-t border-[var(--border)]">
        <button
          onClick={onCancel}
          className="px-3 py-1.5 text-sm rounded hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)] transition-colors"
        >
          Cancel
        </button>
        <button
          onClick={handleConfirm}
          disabled={selectedIds.size === 0}
          className="px-3 py-1.5 text-sm rounded bg-[var(--bg-tertiary)] hover:bg-[var(--border)] text-[var(--text-primary)] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Connect ({selectedIds.size})
        </button>
      </div>
    </div>
  );
}

export default UpstreamNodePicker;
