/**
 * Popover for selecting a plugin when creating a new node.
 * Shows a list of available plugins for the selected tier.
 */

import { useEffect, useRef, useCallback } from 'react';
import { X, Loader2 } from 'lucide-react';
import { useAvailablePlugins } from '@/hooks/useAvailablePlugins';
import { NODE_TYPE_LABELS } from '@/utils/pipelinePluginMapping';
import type { NodeType } from '@/types/pipeline';

export interface NodePickerPopoverProps {
  tier: NodeType;
  position: { x: number; y: number };
  onSelect: (pluginId: string) => void;
  onCancel: () => void;
}

export function NodePickerPopover({
  tier,
  position,
  onSelect,
  onCancel,
}: NodePickerPopoverProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const { plugins, isLoading, error } = useAvailablePlugins(tier);

  // Handle click outside to cancel
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        onCancel();
      }
    };

    // Delay adding listener to avoid immediate trigger from the drop event
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

  // Auto-select if only one plugin available
  useEffect(() => {
    if (!isLoading && plugins.length === 1) {
      onSelect(plugins[0].id);
    }
  }, [isLoading, plugins, onSelect]);

  // Keep popover in viewport
  const adjustedPosition = useCallback(() => {
    const popoverWidth = 220;
    const popoverHeight = 200;
    const padding = 16;

    let left = position.x;
    let top = position.y + 20; // Offset below cursor

    // Adjust horizontal position
    if (typeof window !== 'undefined') {
      if (left + popoverWidth + padding > window.innerWidth) {
        left = window.innerWidth - popoverWidth - padding;
      }
      if (left < padding) {
        left = padding;
      }

      // Adjust vertical position
      if (top + popoverHeight + padding > window.innerHeight) {
        top = position.y - popoverHeight - 10; // Show above cursor
      }
      if (top < padding) {
        top = padding;
      }
    }

    return { left, top };
  }, [position]);

  const { left, top } = adjustedPosition();

  return (
    <div
      ref={containerRef}
      className="fixed z-50 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg shadow-lg min-w-[200px] max-w-[280px]"
      style={{ left, top }}
      role="dialog"
      aria-label={`Select ${NODE_TYPE_LABELS[tier].toLowerCase()}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-[var(--border)]">
        <span className="text-sm font-medium text-[var(--text-secondary)]">
          Select {NODE_TYPE_LABELS[tier].toLowerCase()}
        </span>
        <button
          onClick={onCancel}
          className="p-1 rounded hover:bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
          aria-label="Cancel"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Content */}
      <div className="p-1 max-h-[240px] overflow-y-auto">
        {isLoading && (
          <div className="flex items-center justify-center py-6">
            <Loader2 className="w-5 h-5 animate-spin text-[var(--text-muted)]" />
          </div>
        )}

        {error && (
          <div className="px-3 py-4 text-sm text-red-500">
            Failed to load plugins
          </div>
        )}

        {!isLoading && !error && plugins.length === 0 && (
          <div className="px-3 py-4 text-sm text-[var(--text-muted)]">
            No plugins available
          </div>
        )}

        {!isLoading && !error && plugins.length > 0 && (
          <div className="space-y-0.5">
            {plugins.map((plugin) => (
              <button
                key={plugin.id}
                onClick={() => onSelect(plugin.id)}
                className="w-full text-left px-3 py-2 rounded hover:bg-[var(--bg-tertiary)] transition-colors group"
              >
                <div className="text-sm font-medium text-[var(--text-primary)] group-hover:text-[var(--text-primary)]">
                  {plugin.name}
                </div>
                {plugin.description && (
                  <div className="text-xs text-[var(--text-muted)] mt-0.5 line-clamp-2">
                    {plugin.description}
                  </div>
                )}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default NodePickerPopover;
