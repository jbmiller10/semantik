/**
 * Confirmation dialog for deleting nodes or edges.
 * Shows warning about orphaned nodes that will be cascade-deleted.
 */

import { useEffect, useCallback } from 'react';
import type { PipelineNode } from '@/types/pipeline';
import { AlertTriangle, X } from 'lucide-react';

interface DeleteConfirmationDialogProps {
  /** Type of element being deleted */
  type: 'node' | 'edge';
  /** Message describing what's being deleted */
  message: string;
  /** Nodes that will be orphaned by this deletion */
  orphanedNodes: PipelineNode[];
  /** Callback when deletion is confirmed */
  onConfirm: (deleteOrphans: boolean) => void;
  /** Callback when deletion is cancelled */
  onCancel: () => void;
}

export function DeleteConfirmationDialog({
  type,
  message,
  orphanedNodes,
  onConfirm,
  onCancel,
}: DeleteConfirmationDialogProps) {
  // Handle Escape key to cancel
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onCancel();
      }
    },
    [onCancel]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  // Handle click outside to cancel
  const handleBackdropClick = useCallback(
    (e: React.MouseEvent) => {
      if (e.target === e.currentTarget) {
        onCancel();
      }
    },
    [onCancel]
  );

  return (
    <div
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      onClick={handleBackdropClick}
      role="dialog"
      aria-modal="true"
      aria-labelledby="delete-dialog-title"
    >
      <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg p-6 max-w-md shadow-lg">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h3
            id="delete-dialog-title"
            className="text-lg font-semibold text-[var(--text-primary)] flex items-center gap-2"
          >
            <AlertTriangle className="w-5 h-5 text-amber-400" />
            Delete {type}?
          </h3>
          <button
            onClick={onCancel}
            className="p-1 rounded hover:bg-[var(--bg-tertiary)] text-[var(--text-muted)]"
            aria-label="Close dialog"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Message */}
        <p className="text-[var(--text-secondary)] mb-4">{message}</p>

        {/* Orphaned nodes warning */}
        {orphanedNodes.length > 0 && (
          <div className="bg-amber-500/10 border border-amber-500/30 rounded p-3 mb-4">
            <p className="text-amber-400 text-sm font-medium">
              This will also delete {orphanedNodes.length} orphaned node
              {orphanedNodes.length === 1 ? '' : 's'}:
            </p>
            <ul className="text-sm text-[var(--text-muted)] mt-2 space-y-1">
              {orphanedNodes.map((n) => (
                <li key={n.id} className="flex items-center gap-2">
                  <span className="text-[var(--text-secondary)]">•</span>
                  <span className="font-mono text-xs bg-[var(--bg-tertiary)] px-1.5 py-0.5 rounded">
                    {n.plugin_id}
                  </span>
                  <span className="text-[var(--text-muted)]">({n.type})</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Actions */}
        <div className="flex justify-end gap-2">
          <button
            onClick={onCancel}
            className="px-4 py-2 rounded bg-[var(--bg-tertiary)] text-[var(--text-primary)] hover:bg-[var(--border)] transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={() => onConfirm(true)}
            className="px-4 py-2 rounded bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
          >
            Delete
          </button>
        </div>
      </div>
    </div>
  );
}

export default DeleteConfirmationDialog;
