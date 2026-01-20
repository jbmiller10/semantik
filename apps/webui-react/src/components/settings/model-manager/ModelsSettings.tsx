import { useState, useMemo, useCallback, useEffect } from 'react';
import { RefreshCw, Search, HardDrive, AlertCircle } from 'lucide-react';
import {
  useModelManagerModels,
  useStartModelDownload,
  useModelDownloadProgress,
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

/**
 * Wrapper component for ModelCard that handles download progress polling.
 * Each instance calls useModelDownloadProgress for its own model.
 */
interface ModelCardWithDownloadProps {
  model: CuratedModelResponse;
  taskId: string | null;
  onDownload: (modelId: string) => void;
  onDelete?: (modelId: string) => void;
  onTerminal: (modelId: string, progress: TaskProgressResponse) => void;
  onDismissError: (modelId: string) => void;
}

function ModelCardWithDownload({
  model,
  taskId,
  onDownload,
  onDelete,
  onTerminal,
  onDismissError,
}: ModelCardWithDownloadProps) {
  // Poll for download progress when we have a task ID
  const downloadProgress = useModelDownloadProgress(model.id, taskId, {
    onTerminal: (progress) => onTerminal(model.id, progress),
  });

  return (
    <ModelCard
      model={model}
      onDownload={onDownload}
      onDelete={onDelete}
      downloadProgress={downloadProgress}
      onRetry={onDownload}
      onDismissError={onDismissError}
    />
  );
}

export default function ModelsSettings() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  const [activeModelType, setActiveModelType] = useState<ModelType>('embedding');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');
  const [searchQuery, setSearchQuery] = useState('');

  // Download management
  const { startDownload, getTaskId, clearTaskId } = useStartModelDownload();

  const {
    data,
    isLoading,
    isError,
    error,
    refetch,
    isFetching,
  } = useModelManagerModels({ includeCacheSize: true });

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
      clearTaskId(modelId);

      // Handle based on status
      if (progress.status === 'completed') {
        addToast({
          type: 'success',
          message: 'Model download completed',
        });
        // Invalidate to refresh the model list
        queryClient.invalidateQueries({ queryKey: modelManagerKeys.list() });
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
        queryClient.invalidateQueries({ queryKey: modelManagerKeys.list() });
      }
    },
    [clearTaskId, addToast, queryClient]
  );

  // Handle dismiss error - clear task ID
  const handleDismissError = useCallback(
    (modelId: string) => {
      clearTaskId(modelId);
    },
    [clearTaskId]
  );

  // Get the effective task ID for a model (from recently started download or from model data)
  const getEffectiveTaskId = useCallback(
    (model: CuratedModelResponse): string | null => {
      // First check if we have a recently started download
      const startedTaskId = getTaskId(model.id);
      if (startedTaskId) return startedTaskId;

      // Fall back to the model's active download task (for resumability on page refresh)
      return model.active_download_task_id;
    },
    [getTaskId]
  );

  // Sync active downloads from model data on mount (for resumability)
  useEffect(() => {
    // This effect intentionally runs on every data change to sync active downloads
    // No action needed here as getEffectiveTaskId already handles the model.active_download_task_id
  }, [data?.models]);

  // Phase 2C: Delete will be implemented later
  // const handleDelete = (modelId: string) => { ... }

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
      {data?.cache_size && (
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
              taskId={getEffectiveTaskId(model)}
              onDownload={startDownload}
              onDelete={undefined} // Phase 2C: Delete will be implemented later
              onTerminal={handleDownloadTerminal}
              onDismissError={handleDismissError}
            />
          ))}
        </div>
      )}
    </div>
  );
}
