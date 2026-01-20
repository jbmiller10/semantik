import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import { RefreshCw, Search, HardDrive, AlertCircle, AlertTriangle, Loader2 } from 'lucide-react';
import {
  useModelManagerModels,
  useStartModelDownload,
  useModelDownloadProgress,
  useStartModelDelete,
  useModelDeleteProgress,
  modelManagerKeys,
} from '../../../hooks/useModelManager';
import type { ModelType, CuratedModelResponse, TaskProgressResponse } from '../../../types/model-manager';
import { MODEL_TYPE_LABELS, MODEL_TYPE_ORDER, groupModelsByType } from '../../../types/model-manager';
import ModelCard from './ModelCard';
import { useQueryClient } from '@tanstack/react-query';
import { useUIStore } from '../../../stores/uiStore';

type StatusFilter = 'all' | 'installed' | 'available';

function formatSize(mb: number): string {
  if (mb >= 1024) {
    return `${(mb / 1024).toFixed(1)} GB`;
  }
  return `${mb} MB`;
}

// =============================================================================
// Delete Confirmation Modal
// =============================================================================

interface DeleteConfirmationState {
  modelId: string;
  modelName: string;
  estimatedFreedSize: number | null;
  warnings: string[];
}

interface DeleteConfirmModalProps {
  confirmation: DeleteConfirmationState;
  onConfirm: () => void;
  onCancel: () => void;
  isDeleting: boolean;
}

function DeleteConfirmModal({
  confirmation,
  onConfirm,
  onCancel,
  isDeleting,
}: DeleteConfirmModalProps) {
  const modalRef = useRef<HTMLDivElement>(null);

  // Handle escape key to close modal
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !isDeleting) {
        onCancel();
      }
    };
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [onCancel, isDeleting]);

  // Focus trap for accessibility
  useEffect(() => {
    const modal = modalRef.current;
    if (!modal) return;

    const focusableElements = modal.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const firstElement = focusableElements[0] as HTMLElement;
    const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;

      if (e.shiftKey && document.activeElement === firstElement) {
        e.preventDefault();
        lastElement?.focus();
      } else if (!e.shiftKey && document.activeElement === lastElement) {
        e.preventDefault();
        firstElement?.focus();
      }
    };

    // Focus first focusable element on mount
    firstElement?.focus();

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <>
      <div
        className="fixed inset-0 bg-black/50 dark:bg-black/80 z-[60]"
        onClick={isDeleting ? undefined : onCancel}
      />
      <div
        ref={modalRef}
        className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 panel rounded-xl shadow-2xl z-[60] w-full max-w-md"
        role="dialog"
        aria-modal="true"
        aria-labelledby="delete-modal-title"
      >
        {/* Content */}
        <div className="p-6">
          <div className="flex items-center justify-center w-12 h-12 mx-auto bg-red-500/20 rounded-full">
            <AlertTriangle className="w-6 h-6 text-red-500" />
          </div>

          <h3
            id="delete-modal-title"
            className="mt-4 text-lg font-semibold text-[var(--text-primary)] text-center"
          >
            Delete Model
          </h3>

          <p className="mt-2 text-sm text-[var(--text-muted)] text-center">
            Are you sure you want to delete{' '}
            <span className="font-medium text-[var(--text-primary)]">
              "{confirmation.modelName}"
            </span>
            ?
          </p>

          {confirmation.estimatedFreedSize !== null && (
            <p className="mt-2 text-sm text-[var(--text-secondary)] text-center">
              This will free approximately{' '}
              <span className="font-medium">{formatSize(confirmation.estimatedFreedSize)}</span>{' '}
              of disk space.
            </p>
          )}

          {/* Warnings */}
          {confirmation.warnings.length > 0 && (
            <div className="mt-4 p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg">
              <div className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-amber-400 mt-0.5 flex-shrink-0" />
                <div className="text-sm text-amber-300">
                  <span className="font-medium">Warnings:</span>
                  <ul className="mt-1 list-disc pl-4 space-y-0.5">
                    {confirmation.warnings.map((warning, index) => (
                      <li key={index} className="text-amber-400">
                        {warning}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-[var(--bg-secondary)] border-t border-[var(--border)] flex justify-end gap-3 rounded-b-xl">
          <button
            onClick={onCancel}
            disabled={isDeleting}
            className="px-4 py-2 text-sm font-medium text-[var(--text-secondary)] bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg hover:bg-[var(--bg-primary)] focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white focus:ring-offset-1 focus:ring-offset-[var(--bg-primary)] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={isDeleting}
            className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-red-600 border border-transparent rounded-lg hover:bg-red-500 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-1 focus:ring-offset-[var(--bg-primary)] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isDeleting && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
            Delete Model
          </button>
        </div>
      </div>
    </>
  );
}

/**
 * Wrapper component for ModelCard that handles download and delete progress polling.
 * Each instance calls useModelDownloadProgress and useModelDeleteProgress for its own model.
 */
interface ModelCardWithDownloadProps {
  model: CuratedModelResponse;
  downloadTaskId: string | null;
  deleteTaskId: string | null;
  onDownload: (modelId: string) => void;
  onDelete?: (modelId: string) => void;
  onDownloadTerminal: (modelId: string, progress: TaskProgressResponse) => void;
  onDeleteTerminal: (modelId: string, progress: TaskProgressResponse) => void;
  onDismissDownloadError: (modelId: string) => void;
  onDismissDeleteError: (modelId: string) => void;
}

function ModelCardWithDownload({
  model,
  downloadTaskId,
  deleteTaskId,
  onDownload,
  onDelete,
  onDownloadTerminal,
  onDeleteTerminal,
  onDismissDownloadError,
  onDismissDeleteError,
}: ModelCardWithDownloadProps) {
  // Poll for download progress when we have a task ID
  const downloadProgress = useModelDownloadProgress(model.id, downloadTaskId, {
    onTerminal: (progress) => onDownloadTerminal(model.id, progress),
  });

  // Poll for delete progress when we have a task ID
  const deleteProgress = useModelDeleteProgress(model.id, deleteTaskId, {
    onTerminal: (progress) => onDeleteTerminal(model.id, progress),
  });

  return (
    <ModelCard
      model={model}
      onDownload={onDownload}
      onDelete={onDelete}
      downloadProgress={downloadProgress}
      deleteProgress={deleteProgress}
      onRetry={onDownload}
      onDismissError={onDismissDownloadError}
      onRetryDelete={onDelete}
      onDismissDeleteError={onDismissDeleteError}
    />
  );
}

export default function ModelsSettings() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  const [activeModelType, setActiveModelType] = useState<ModelType>('embedding');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');
  const [searchQuery, setSearchQuery] = useState('');

  // Lazy-load cache size to reduce unnecessary HF cache scans
  const [showCacheSize, setShowCacheSize] = useState(false);

  // Download management
  const {
    startDownload,
    getTaskId: getDownloadTaskId,
    clearTaskId: clearDownloadTaskId,
  } = useStartModelDownload();

  // Delete management
  const {
    startDelete,
    getTaskId: getDeleteTaskId,
    clearTaskId: clearDeleteTaskId,
    isPending: isDeletePending,
    lastConflict,
    clearConflict,
  } = useStartModelDelete();

  // Delete confirmation state
  const [deleteConfirmation, setDeleteConfirmation] = useState<DeleteConfirmationState | null>(
    null
  );

  const {
    data,
    isLoading,
    isError,
    error,
    refetch,
    isFetching,
  } = useModelManagerModels({ includeCacheSize: showCacheSize });

  // Group and filter models
  const filteredModels = useMemo(() => {
    if (!data?.models) return [];

    const grouped = groupModelsByType(data.models);
    let models = grouped[activeModelType] || [];

    // Apply status filter
    if (statusFilter === 'installed') {
      models = models.filter((m) => m.is_installed);
    } else if (statusFilter === 'available') {
      models = models.filter((m) => !m.is_installed);
    }

    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      models = models.filter(
        (m) =>
          m.name.toLowerCase().includes(query) ||
          m.description.toLowerCase().includes(query) ||
          m.id.toLowerCase().includes(query)
      );
    }

    // Sort: installed first, then by name
    return models.sort((a, b) => {
      if (a.is_installed !== b.is_installed) {
        return a.is_installed ? -1 : 1;
      }
      return a.name.localeCompare(b.name);
    });
  }, [data?.models, activeModelType, statusFilter, searchQuery]);

  // Count models by type and status
  const modelCounts = useMemo((): Record<ModelType, { total: number; installed: number }> => {
    const counts: Record<ModelType, { total: number; installed: number }> = {
      embedding: { total: 0, installed: 0 },
      llm: { total: 0, installed: 0 },
      reranker: { total: 0, installed: 0 },
      splade: { total: 0, installed: 0 },
    };
    if (!data?.models) return counts;
    const grouped = groupModelsByType(data.models);
    for (const type of MODEL_TYPE_ORDER) {
      const models = grouped[type] || [];
      counts[type] = {
        total: models.length,
        installed: models.filter((m) => m.is_installed).length,
      };
    }
    return counts;
  }, [data?.models]);

  // Handle download completion/failure
  const handleDownloadTerminal = useCallback(
    (modelId: string, progress: TaskProgressResponse) => {
      // Clear the task ID from our local state
      clearDownloadTaskId(modelId);

      // Handle based on status
      if (progress.status === 'completed') {
        addToast({
          type: 'success',
          message: 'Model download completed',
        });
        // Invalidate to refresh the model list
        queryClient.invalidateQueries({ queryKey: modelManagerKeys.lists() });
      } else if (progress.status === 'failed') {
        addToast({
          type: 'error',
          message: progress.error ?? 'Download failed',
        });
      } else if (progress.status === 'already_installed') {
        addToast({
          type: 'info',
          message: 'Model is already installed',
        });
        queryClient.invalidateQueries({ queryKey: modelManagerKeys.lists() });
      }
    },
    [clearDownloadTaskId, addToast, queryClient]
  );

  // Handle delete completion/failure
  const handleDeleteTerminal = useCallback(
    (modelId: string, progress: TaskProgressResponse) => {
      // Clear the task ID from our local state
      clearDeleteTaskId(modelId);

      // Handle based on status
      if (progress.status === 'completed' || progress.status === 'not_installed') {
        addToast({
          type: 'success',
          message: 'Model deleted successfully',
        });
        // Invalidate to refresh the model list
        queryClient.invalidateQueries({ queryKey: modelManagerKeys.lists() });
      } else if (progress.status === 'failed') {
        addToast({
          type: 'error',
          message: progress.error ?? 'Delete failed',
        });
      }
    },
    [clearDeleteTaskId, addToast, queryClient]
  );

  // Handle dismiss download error - clear task ID
  const handleDismissDownloadError = useCallback(
    (modelId: string) => {
      clearDownloadTaskId(modelId);
    },
    [clearDownloadTaskId]
  );

  // Handle dismiss delete error - clear task ID
  const handleDismissDeleteError = useCallback(
    (modelId: string) => {
      clearDeleteTaskId(modelId);
    },
    [clearDeleteTaskId]
  );

  // Get the effective download task ID for a model
  const getEffectiveDownloadTaskId = useCallback(
    (model: CuratedModelResponse): string | null => {
      // First check if we have a recently started download
      const startedTaskId = getDownloadTaskId(model.id);
      if (startedTaskId) return startedTaskId;

      // Fall back to the model's active download task (for resumability on page refresh)
      return model.active_download_task_id;
    },
    [getDownloadTaskId]
  );

  // Get the effective delete task ID for a model
  const getEffectiveDeleteTaskId = useCallback(
    (model: CuratedModelResponse): string | null => {
      // First check if we have a recently started delete
      const startedTaskId = getDeleteTaskId(model.id);
      if (startedTaskId) return startedTaskId;

      // Fall back to the model's active delete task (for resumability on page refresh)
      return model.active_delete_task_id;
    },
    [getDeleteTaskId]
  );

  // Handle delete button click - initiate delete (may trigger confirmation dialog)
  const handleDeleteClick = useCallback(
    (modelId: string) => {
      // Start delete without confirmation - API will return requires_confirmation if needed
      startDelete(modelId, false);
    },
    [startDelete]
  );

  // Handle confirm delete from modal
  const handleConfirmDelete = useCallback(() => {
    if (!deleteConfirmation) return;

    // Start delete with confirmation bypass
    startDelete(deleteConfirmation.modelId, true);

    // Close the dialog
    setDeleteConfirmation(null);
  }, [deleteConfirmation, startDelete]);

  // Handle cancel delete from modal
  const handleCancelDelete = useCallback(() => {
    setDeleteConfirmation(null);
    clearConflict();
  }, [clearConflict]);

  // Watch for requires_confirmation conflict and open the confirmation dialog
  useEffect(() => {
    if (lastConflict && lastConflict.conflict_type === 'requires_confirmation') {
      // Find the model to get its name and size
      const model = data?.models?.find((m) => m.id === lastConflict.model_id);
      if (model) {
        setDeleteConfirmation({
          modelId: lastConflict.model_id,
          modelName: model.name,
          estimatedFreedSize: model.size_on_disk_mb,
          warnings: lastConflict.warnings || [],
        });
      }
      // Clear the conflict so it doesn't trigger again
      clearConflict();
    }
  }, [lastConflict, data?.models, clearConflict]);

  // Sync active downloads/deletes from model data on mount (for resumability)
  useEffect(() => {
    // This effect intentionally runs on every data change to sync active tasks
    // No action needed here as getEffectiveTaskId functions already handle the model's active task IDs
  }, [data?.models]);

  if (isError) {
    return (
      <div className="p-6 text-center">
        <AlertCircle className="w-12 h-12 mx-auto text-red-400 mb-4" />
        <h3 className="text-lg font-medium text-[var(--text-primary)] mb-2">
          Failed to load models
        </h3>
        <p className="text-sm text-[var(--text-secondary)] mb-4">
          {error instanceof Error ? error.message : 'An unexpected error occurred'}
        </p>
        <button
          onClick={() => refetch()}
          className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded bg-gray-200 dark:bg-white text-gray-900 hover:bg-gray-300 dark:hover:bg-gray-100"
        >
          <RefreshCw className="w-4 h-4" />
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-medium text-[var(--text-primary)]">Model Manager</h3>
          <p className="mt-1 text-sm text-[var(--text-secondary)]">
            Manage curated AI models for embedding, local LLM inference, and reranking.
          </p>
        </div>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="inline-flex items-center gap-2 px-3 py-1.5 text-sm font-medium rounded border border-[var(--border)] bg-[var(--bg-secondary)] hover:bg-[var(--bg-tertiary)] text-[var(--text-primary)] disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${isFetching ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Cache Size Info */}
      {data?.cache_size ? (
        <div className="flex items-center gap-6 p-4 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg">
          <div className="flex items-center gap-2">
            <HardDrive className="w-5 h-5 text-[var(--text-muted)]" />
            <span className="text-sm text-[var(--text-secondary)]">Cache Usage</span>
          </div>
          <div className="flex items-center gap-4 text-sm">
            <span className="text-[var(--text-primary)]">
              <span className="font-medium">{formatSize(data.cache_size.total_cache_size_mb)}</span> total
            </span>
            <span className="text-[var(--text-muted)]">|</span>
            <span className="text-[var(--text-secondary)]">
              {formatSize(data.cache_size.managed_cache_size_mb)} managed
            </span>
            <span className="text-[var(--text-muted)]">|</span>
            <span className="text-[var(--text-secondary)]">
              {formatSize(data.cache_size.unmanaged_cache_size_mb)} unmanaged
              {data.cache_size.unmanaged_repo_count > 0 && (
                <span className="text-[var(--text-muted)]">
                  {' '}({data.cache_size.unmanaged_repo_count} repos)
                </span>
              )}
            </span>
          </div>
        </div>
      ) : (
        <div className="flex items-center gap-4 p-4 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg">
          <div className="flex items-center gap-2">
            <HardDrive className="w-5 h-5 text-[var(--text-muted)]" />
            <span className="text-sm text-[var(--text-secondary)]">Cache Usage</span>
          </div>
          <button
            onClick={() => setShowCacheSize(true)}
            disabled={isFetching}
            className="text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] underline underline-offset-2 disabled:opacity-50"
          >
            {isFetching && showCacheSize ? 'Loading...' : 'Show cache usage'}
          </button>
        </div>
      )}

      {/* Model Type Tabs */}
      <div className="border-b border-[var(--border)]">
        <nav className="-mb-px flex space-x-6" aria-label="Model type tabs">
          {MODEL_TYPE_ORDER.map((type) => {
            const count = modelCounts[type];
            const isActive = activeModelType === type;
            return (
              <button
                key={type}
                onClick={() => setActiveModelType(type)}
                className={`
                  whitespace-nowrap py-3 px-1 border-b-2 font-medium text-sm
                  ${
                    isActive
                      ? 'border-[var(--accent-primary)] text-[var(--accent-primary)]'
                      : 'border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--border-strong)]'
                  }
                `}
              >
                {MODEL_TYPE_LABELS[type]}
                {count && (
                  <span className="ml-2 text-xs text-[var(--text-muted)]">
                    {count.installed}/{count.total}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4">
        {/* Status Filter */}
        <div className="flex items-center gap-1 p-1 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg">
          {(['all', 'installed', 'available'] as StatusFilter[]).map((filter) => (
            <button
              key={filter}
              onClick={() => setStatusFilter(filter)}
              className={`px-3 py-1 text-xs font-medium rounded ${
                statusFilter === filter
                  ? 'bg-[var(--bg-tertiary)] text-[var(--text-primary)]'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
            >
              {filter.charAt(0).toUpperCase() + filter.slice(1)}
            </button>
          ))}
        </div>

        {/* Search */}
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--text-muted)]" />
          <input
            type="text"
            placeholder="Search models..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-1.5 text-sm bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:ring-1 focus:ring-[var(--accent-primary)]"
          />
        </div>
      </div>

      {/* Model List */}
      {isLoading ? (
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="h-24 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg animate-pulse"
            />
          ))}
        </div>
      ) : filteredModels.length === 0 ? (
        <div className="p-8 text-center border border-[var(--border)] rounded-lg bg-[var(--bg-secondary)]">
          <p className="text-sm text-[var(--text-secondary)]">
            {searchQuery
              ? 'No models match your search.'
              : statusFilter === 'installed'
                ? 'No installed models of this type.'
                : statusFilter === 'available'
                  ? 'No available models to download.'
                  : 'No models found.'}
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {filteredModels.map((model: CuratedModelResponse) => (
            <ModelCardWithDownload
              key={model.id}
              model={model}
              downloadTaskId={getEffectiveDownloadTaskId(model)}
              deleteTaskId={getEffectiveDeleteTaskId(model)}
              onDownload={startDownload}
              onDelete={handleDeleteClick}
              onDownloadTerminal={handleDownloadTerminal}
              onDeleteTerminal={handleDeleteTerminal}
              onDismissDownloadError={handleDismissDownloadError}
              onDismissDeleteError={handleDismissDeleteError}
            />
          ))}
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {deleteConfirmation && (
        <DeleteConfirmModal
          confirmation={deleteConfirmation}
          onConfirm={handleConfirmDelete}
          onCancel={handleCancelDelete}
          isDeleting={isDeletePending}
        />
      )}
    </div>
  );
}
