import { useState } from 'react';
import {
  Download,
  Trash2,
  ChevronDown,
  ChevronUp,
  HardDrive,
  Cpu,
  Database,
  Check,
  AlertCircle,
  Loader2,
  RefreshCw,
  X,
} from 'lucide-react';
import type { CuratedModelResponse } from '../../../types/model-manager';
import { MODEL_TYPE_LABELS } from '../../../types/model-manager';
import type { FormattedDownloadProgress, FormattedDeleteProgress } from '../../../hooks/useModelManager';

interface ModelCardProps {
  model: CuratedModelResponse;
  onDownload?: (modelId: string) => void;
  onDelete?: (modelId: string) => void;
  /** Download progress from useModelDownloadProgress hook */
  downloadProgress?: FormattedDownloadProgress | null;
  /** Delete progress from useModelDeleteProgress hook */
  deleteProgress?: FormattedDeleteProgress | null;
  /** Callback to retry a failed download */
  onRetry?: (modelId: string) => void;
  /** Callback to dismiss error state */
  onDismissError?: (modelId: string) => void;
  /** Callback to retry a failed delete */
  onRetryDelete?: (modelId: string) => void;
  /** Callback to dismiss delete error state */
  onDismissDeleteError?: (modelId: string) => void;
}

function formatSize(mb: number | null): string {
  if (mb === null) return '--';
  if (mb >= 1024) {
    return `${(mb / 1024).toFixed(1)} GB`;
  }
  return `${mb} MB`;
}

interface DownloadProgressBarProps {
  progress: FormattedDownloadProgress;
  onRetry?: () => void;
  onDismiss?: () => void;
}

/**
 * Progress bar component for download operations.
 * Shows spinner + percentage + bytes during download.
 * Shows error message + Retry/Dismiss buttons on failure.
 */
function DownloadProgressBar({ progress, onRetry, onDismiss }: DownloadProgressBarProps) {
  const isDownloading = progress.status === 'pending' || progress.status === 'running';
  const isFailed = progress.status === 'failed';
  // Indeterminate state: downloading but total size not yet known
  const isIndeterminate = isDownloading && progress.bytesTotal === 0;

  if (isFailed) {
    return (
      <div className="mt-3 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-xs text-red-400 min-w-0">
            <AlertCircle className="w-3 h-3 flex-shrink-0" />
            <span className="truncate">{progress.error ?? 'Download failed'}</span>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            {onRetry && (
              <button
                onClick={onRetry}
                className="inline-flex items-center gap-1 px-2 py-1 text-xs font-medium rounded border border-red-500/50 text-red-400 hover:bg-red-500/10"
              >
                <RefreshCw className="w-3 h-3" />
                Retry
              </button>
            )}
            {onDismiss && (
              <button
                onClick={onDismiss}
                className="p-1 rounded hover:bg-red-500/10 text-red-400"
                title="Dismiss"
              >
                <X className="w-3 h-3" />
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (isDownloading) {
    return (
      <div className="mt-3 space-y-2">
        {/* Progress info */}
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-2 text-[var(--text-secondary)]">
            <Loader2 className="w-3 h-3 animate-spin" />
            <span>{isIndeterminate ? 'Initializing...' : 'Downloading...'}</span>
          </div>
          <div className="flex items-center gap-2 text-[var(--text-muted)]">
            {isIndeterminate ? (
              <span>Initializing...</span>
            ) : (
              <>
                <span>{progress.formattedBytes}</span>
                <span className="font-medium">{progress.percentage}%</span>
              </>
            )}
          </div>
        </div>
        {/* Progress bar */}
        <div className="h-1.5 bg-[var(--bg-tertiary)] rounded-full overflow-hidden">
          {isIndeterminate ? (
            <div className="h-full bg-blue-500 rounded-full animate-pulse w-full opacity-50" />
          ) : (
            <div
              className="h-full bg-blue-500 rounded-full transition-all duration-300 ease-out"
              style={{ width: `${progress.percentage}%` }}
            />
          )}
        </div>
      </div>
    );
  }

  // For other statuses (completed, already_installed, not_installed), don't show anything
  // The model list will refresh and show the updated state
  return null;
}

interface DeleteProgressBarProps {
  progress: FormattedDeleteProgress;
  onRetry?: () => void;
  onDismiss?: () => void;
}

/**
 * Progress indicator component for delete operations.
 * Shows spinner during delete, error message + Retry/Dismiss buttons on failure.
 * Simpler than download - no percentage or bytes needed.
 */
function DeleteProgressBar({ progress, onRetry, onDismiss }: DeleteProgressBarProps) {
  if (progress.isFailed) {
    return (
      <div className="mt-3 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-xs text-red-400 min-w-0">
            <AlertCircle className="w-3 h-3 flex-shrink-0" />
            <span className="truncate">{progress.error ?? 'Delete failed'}</span>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            {onRetry && (
              <button
                onClick={onRetry}
                className="inline-flex items-center gap-1 px-2 py-1 text-xs font-medium rounded border border-red-500/50 text-red-400 hover:bg-red-500/10"
              >
                <RefreshCw className="w-3 h-3" />
                Retry
              </button>
            )}
            {onDismiss && (
              <button
                onClick={onDismiss}
                className="p-1 rounded hover:bg-red-500/10 text-red-400"
                title="Dismiss"
              >
                <X className="w-3 h-3" />
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (progress.isDeleting) {
    return (
      <div className="mt-3 flex items-center gap-2 text-xs text-amber-400">
        <Loader2 className="w-3 h-3 animate-spin" />
        <span>Deleting model files...</span>
      </div>
    );
  }

  // For completed or other statuses, don't show anything
  return null;
}

export default function ModelCard({
  model,
  onDownload,
  onDelete,
  downloadProgress,
  deleteProgress,
  onRetry,
  onDismissError,
  onRetryDelete,
  onDismissDeleteError,
}: ModelCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  // Determine if there's an active download (from progress hook or model data)
  const isDownloading =
    downloadProgress?.status === 'pending' ||
    downloadProgress?.status === 'running' ||
    downloadProgress?.status === 'failed';

  // Determine if there's an active delete (from progress hook or model data)
  const isDeleting = deleteProgress?.isDeleting || deleteProgress?.isFailed;
  const hasActiveDeleteTask = !!model.active_delete_task_id || isDeleting;

  // Only show download button if not installed, not downloading, and handler provided
  const hasDownloadAction = !model.is_installed && !isDownloading && onDownload;
  const hasDeleteAction = model.is_installed && onDelete;

  // Get memory estimate for display (prefer int8 as common quantization)
  const memoryEstimate = model.memory_mb['int8'] ?? model.memory_mb['float16'] ?? null;

  return (
    <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg p-4">
      {/* Header Row */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          {/* Name and Status */}
          <div className="flex items-center gap-2 flex-wrap">
            <h4 className="text-sm font-medium text-[var(--text-primary)] truncate">
              {model.name}
            </h4>
            {/* Type Badge */}
            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-[var(--bg-tertiary)] text-[var(--text-secondary)]">
              {MODEL_TYPE_LABELS[model.model_type]}
            </span>
            {/* Installation Status */}
            {model.is_installed ? (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-green-500/10 text-green-400">
                <Check className="w-3 h-3" />
                Installed
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-[var(--bg-tertiary)] text-[var(--text-muted)]">
                Available
              </span>
            )}
          </div>

          {/* Description */}
          <p className="mt-1 text-xs text-[var(--text-secondary)] line-clamp-2">
            {model.description}
          </p>

          {/* Quick Stats */}
          <div className="mt-2 flex items-center gap-4 text-xs text-[var(--text-muted)]">
            {model.is_installed && model.size_on_disk_mb !== null && (
              <span className="flex items-center gap-1">
                <HardDrive className="w-3 h-3" />
                {formatSize(model.size_on_disk_mb)}
              </span>
            )}
            {memoryEstimate !== null && (
              <span className="flex items-center gap-1">
                <Cpu className="w-3 h-3" />
                ~{formatSize(memoryEstimate)} RAM
              </span>
            )}
            {model.model_type === 'embedding' && model.used_by_collections.length > 0 && (
              <span className="flex items-center gap-1">
                <Database className="w-3 h-3" />
                {model.used_by_collections.length} collection{model.used_by_collections.length !== 1 ? 's' : ''}
              </span>
            )}
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2 flex-shrink-0">
          {hasDownloadAction && (
            <button
              onClick={() => onDownload(model.id)}
              disabled={hasActiveDeleteTask}
              className="inline-flex items-center gap-1 px-3 py-1.5 text-xs font-medium rounded bg-gray-200 dark:bg-white text-gray-900 hover:bg-gray-300 dark:hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
              title={hasActiveDeleteTask ? 'Delete in progress' : 'Download model'}
            >
              <Download className="w-3 h-3" />
              Download
            </button>
          )}
          {hasDeleteAction && (
            <button
              onClick={() => onDelete(model.id)}
              disabled={hasActiveDeleteTask || isDownloading || model.used_by_collections.length > 0}
              className="inline-flex items-center gap-1 px-3 py-1.5 text-xs font-medium rounded border border-red-500/50 text-red-400 hover:bg-red-500/10 disabled:opacity-50 disabled:cursor-not-allowed"
              title={
                hasActiveDeleteTask
                  ? 'Delete in progress'
                  : isDownloading
                    ? 'Download in progress'
                    : model.used_by_collections.length > 0
                      ? 'Model is in use by collections'
                      : 'Delete model'
              }
            >
              <Trash2 className="w-3 h-3" />
              Delete
            </button>
          )}
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1.5 rounded hover:bg-[var(--bg-tertiary)] text-[var(--text-muted)]"
            title={isExpanded ? 'Hide details' : 'Show details'}
          >
            {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Download Progress */}
      {downloadProgress && (
        <DownloadProgressBar
          progress={downloadProgress}
          onRetry={onRetry ? () => onRetry(model.id) : undefined}
          onDismiss={onDismissError ? () => onDismissError(model.id) : undefined}
        />
      )}

      {/* Delete Progress (only show if no download progress) */}
      {deleteProgress && !downloadProgress && (
        <DeleteProgressBar
          progress={deleteProgress}
          onRetry={onRetryDelete ? () => onRetryDelete(model.id) : undefined}
          onDismiss={onDismissDeleteError ? () => onDismissDeleteError(model.id) : undefined}
        />
      )}

      {/* Delete Task Warning (fallback for active task without progress tracking) */}
      {hasActiveDeleteTask && !downloadProgress && !deleteProgress && (
        <div className="mt-3 flex items-center gap-2 text-xs text-amber-400">
          <Loader2 className="w-3 h-3 animate-spin" />
          Deletion in progress...
        </div>
      )}

      {/* Expanded Details */}
      {isExpanded && (
        <div className="mt-4 pt-4 border-t border-[var(--border)] space-y-3">
          {/* Model ID */}
          <div>
            <dt className="text-xs font-medium text-[var(--text-muted)]">Model ID</dt>
            <dd className="mt-0.5 text-xs text-[var(--text-secondary)] font-mono break-all">
              {model.id}
            </dd>
          </div>

          {/* Memory Estimates */}
          {Object.keys(model.memory_mb).length > 0 && (
            <div>
              <dt className="text-xs font-medium text-[var(--text-muted)]">Memory by Quantization</dt>
              <dd className="mt-1 flex flex-wrap gap-2">
                {Object.entries(model.memory_mb).map(([quant, mb]) => (
                  <span
                    key={quant}
                    className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-[var(--bg-tertiary)] text-[var(--text-secondary)]"
                  >
                    {quant}: {formatSize(mb)}
                  </span>
                ))}
              </dd>
            </div>
          )}

          {/* Embedding-specific details */}
          {model.embedding_details && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {model.embedding_details.dimension !== null && (
                <div>
                  <dt className="text-xs font-medium text-[var(--text-muted)]">Dimension</dt>
                  <dd className="mt-0.5 text-xs text-[var(--text-secondary)]">
                    {model.embedding_details.dimension}
                  </dd>
                </div>
              )}
              {model.embedding_details.max_sequence_length !== null && (
                <div>
                  <dt className="text-xs font-medium text-[var(--text-muted)]">Max Sequence</dt>
                  <dd className="mt-0.5 text-xs text-[var(--text-secondary)]">
                    {model.embedding_details.max_sequence_length.toLocaleString()}
                  </dd>
                </div>
              )}
              {model.embedding_details.pooling_method && (
                <div>
                  <dt className="text-xs font-medium text-[var(--text-muted)]">Pooling</dt>
                  <dd className="mt-0.5 text-xs text-[var(--text-secondary)]">
                    {model.embedding_details.pooling_method}
                  </dd>
                </div>
              )}
              {model.embedding_details.is_asymmetric && (
                <div>
                  <dt className="text-xs font-medium text-[var(--text-muted)]">Asymmetric</dt>
                  <dd className="mt-0.5 text-xs text-[var(--text-secondary)]">Yes</dd>
                </div>
              )}
            </div>
          )}

          {/* LLM-specific details */}
          {model.llm_details && model.llm_details.context_window !== null && (
            <div>
              <dt className="text-xs font-medium text-[var(--text-muted)]">Context Window</dt>
              <dd className="mt-0.5 text-xs text-[var(--text-secondary)]">
                {model.llm_details.context_window.toLocaleString()} tokens
              </dd>
            </div>
          )}

          {/* Collections using this model */}
          {model.used_by_collections.length > 0 && (
            <div>
              <dt className="text-xs font-medium text-[var(--text-muted)]">Used by Collections</dt>
              <dd className="mt-1 flex flex-wrap gap-1">
                {model.used_by_collections.map((name) => (
                  <span
                    key={name}
                    className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-blue-500/10 text-blue-400"
                  >
                    {name}
                  </span>
                ))}
              </dd>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
