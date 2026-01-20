/**
 * React Query hooks for Model Manager feature.
 * Note: This is separate from useModels.ts which handles legacy embedding models.
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useCallback, useRef, useState } from 'react';
import { modelManagerApi, type ListModelsParams } from '../services/api/v2/model-manager';
import { handleApiError } from '../services/api/v2/collections';
import { useUIStore } from '../stores/uiStore';
import type {
  ModelListResponse,
  ModelType,
  TaskProgressResponse,
  TaskStatus,
} from '../types/model-manager';
import { isTerminalStatus } from '../types/model-manager';
import { AxiosError } from 'axios';

// Query key factory for consistent key generation
export const modelManagerKeys = {
  all: ['model-manager'] as const,
  list: (params?: ListModelsParams) => [...modelManagerKeys.all, 'list', params ?? {}] as const,
  usage: (modelId: string) => [...modelManagerKeys.all, 'usage', modelId] as const,
  task: (taskId: string) => [...modelManagerKeys.all, 'task', taskId] as const,
};

export interface UseModelManagerModelsOptions {
  modelType?: ModelType;
  installedOnly?: boolean;
  includeCacheSize?: boolean;
  enabled?: boolean;
}

/**
 * Hook to fetch curated models from the model manager.
 *
 * @example
 * // Fetch all models with cache size info
 * const { data, isLoading, refetch } = useModelManagerModels({ includeCacheSize: true });
 *
 * // Fetch only installed embedding models
 * const { data } = useModelManagerModels({ modelType: 'embedding', installedOnly: true });
 */
export function useModelManagerModels(options?: UseModelManagerModelsOptions) {
  const params: ListModelsParams = {
    model_type: options?.modelType,
    installed_only: options?.installedOnly,
    include_cache_size: options?.includeCacheSize,
  };

  return useQuery<ModelListResponse>({
    queryKey: modelManagerKeys.list(params),
    queryFn: () => modelManagerApi.listModels(params),
    staleTime: 2 * 60 * 1000, // 2 minutes - models change more frequently during downloads
    gcTime: 5 * 60 * 1000, // Keep in cache for 5 minutes
    enabled: options?.enabled ?? true,
  });
}

// =============================================================================
// Download Progress State
// =============================================================================

/**
 * Download progress information for a single model.
 */
export interface DownloadProgress {
  taskId: string;
  modelId: string;
  status: TaskStatus;
  bytesDownloaded: number;
  bytesTotal: number;
  error: string | null;
  updatedAt: number;
}

/**
 * Formatted download progress with computed values.
 */
export interface FormattedDownloadProgress extends DownloadProgress {
  percentage: number;
  formattedBytes: string;
}

/**
 * Format bytes for display.
 */
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

// =============================================================================
// Task Progress Hook
// =============================================================================

export interface UseTaskProgressOptions {
  /** Callback when task reaches terminal status */
  onTerminal?: (progress: TaskProgressResponse) => void;
  /** Polling interval in ms (default: 2000) */
  pollingInterval?: number;
  /** Max consecutive 404s before stopping (default: 3) */
  grace404Count?: number;
}

/**
 * Hook to poll task progress by task ID.
 * Automatically stops polling when task reaches terminal status.
 * Handles 404 grace period for task creation latency.
 */
export function useTaskProgress(
  taskId: string | null,
  options?: UseTaskProgressOptions
) {
  const { onTerminal, pollingInterval = 2000, grace404Count = 3 } = options ?? {};

  // Track 404 count for grace period
  const consecutive404s = useRef(0);

  return useQuery({
    queryKey: modelManagerKeys.task(taskId ?? ''),
    queryFn: async () => {
      if (!taskId) return null;

      try {
        const response = await modelManagerApi.getTaskProgress(taskId);
        // Reset 404 counter on success
        consecutive404s.current = 0;
        return response;
      } catch (error) {
        // Handle 404 with grace period
        if (error instanceof AxiosError && error.response?.status === 404) {
          consecutive404s.current++;
          if (consecutive404s.current < grace404Count) {
            // Return null to continue polling
            return null;
          }
        }
        throw error;
      }
    },
    enabled: !!taskId,
    staleTime: 1000,
    refetchInterval: (query) => {
      // Stop polling if no task ID
      if (!taskId) return false;

      // Stop polling on error (after grace period)
      if (query.state.error) return false;

      // Stop polling on terminal status
      const status = query.state.data?.status;
      if (isTerminalStatus(status)) {
        // Call onTerminal callback if provided
        if (onTerminal && query.state.data) {
          onTerminal(query.state.data);
        }
        return false;
      }

      return pollingInterval;
    },
  });
}

// =============================================================================
// Download Mutation Hook
// =============================================================================

export interface StartModelDownloadResult {
  /** Start a download for a model */
  startDownload: (modelId: string) => void;
  /** Get task ID for a model (from recently started downloads) */
  getTaskId: (modelId: string) => string | undefined;
  /** Clear task ID for a model (after completion/dismissal) */
  clearTaskId: (modelId: string) => void;
  /** Check if mutation is pending */
  isPending: boolean;
}

/**
 * Hook to start model downloads.
 * Tracks task IDs for recently started downloads.
 * Progress polling should be done separately via useModelDownloadProgress.
 */
export function useStartModelDownload(): StartModelDownloadResult {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  // Track recently started downloads: modelId -> taskId
  // This provides immediate access to task IDs before the model list refreshes
  const [startedDownloads, setStartedDownloads] = useState<Map<string, string>>(new Map());

  // Start download mutation
  const downloadMutation = useMutation({
    mutationFn: async (modelId: string) => {
      return modelManagerApi.startDownload(modelId);
    },
    onSuccess: (response, modelId) => {
      if (response.status === 'already_installed') {
        // Handle immediate already_installed response
        addToast({
          type: 'info',
          message: 'Model is already installed',
        });
        // Invalidate to refresh status
        queryClient.invalidateQueries({ queryKey: modelManagerKeys.list() });
        return;
      }

      if (response.task_id) {
        // Track the task ID for immediate progress polling
        setStartedDownloads((prev) => {
          const next = new Map(prev);
          next.set(modelId, response.task_id!);
          return next;
        });

        // Invalidate model list to pick up active_download_task_id
        queryClient.invalidateQueries({ queryKey: modelManagerKeys.list() });

        // Show warnings if any
        if (response.warnings.length > 0) {
          addToast({
            type: 'warning',
            message: response.warnings.join(', '),
          });
        }
      }
    },
    onError: (error) => {
      // Handle 409 conflict
      if (error instanceof AxiosError && error.response?.status === 409) {
        const detail = error.response.data?.detail || 'Operation conflict';
        const conflictType = error.response.data?.conflict_type;

        if (conflictType === 'cross_op_exclusion') {
          addToast({
            type: 'warning',
            message: 'Cannot download: another operation is in progress',
          });
        } else {
          addToast({
            type: 'error',
            message: detail,
          });
        }
        return;
      }

      const errorMessage = handleApiError(error);
      addToast({
        type: 'error',
        message: `Download failed: ${errorMessage}`,
      });
    },
  });

  // Start download
  const startDownload = useCallback(
    (modelId: string) => {
      // Clear any existing entry for this model (for retry)
      setStartedDownloads((prev) => {
        if (prev.has(modelId)) {
          const next = new Map(prev);
          next.delete(modelId);
          return next;
        }
        return prev;
      });

      downloadMutation.mutate(modelId);
    },
    [downloadMutation]
  );

  // Get task ID for a model
  const getTaskId = useCallback(
    (modelId: string): string | undefined => {
      return startedDownloads.get(modelId);
    },
    [startedDownloads]
  );

  // Clear task ID for a model
  const clearTaskId = useCallback((modelId: string) => {
    setStartedDownloads((prev) => {
      const next = new Map(prev);
      next.delete(modelId);
      return next;
    });
  }, []);

  return {
    startDownload,
    getTaskId,
    clearTaskId,
    isPending: downloadMutation.isPending,
  };
}

// =============================================================================
// Model Download Progress Hook
// =============================================================================

export interface UseModelDownloadProgressOptions {
  /** Callback when download reaches terminal status */
  onTerminal?: (progress: TaskProgressResponse) => void;
}

/**
 * Hook to track download progress for a specific model.
 * Combines useTaskProgress with download-specific logic.
 */
export function useModelDownloadProgress(
  modelId: string | null,
  taskId: string | null,
  options?: UseModelDownloadProgressOptions
): FormattedDownloadProgress | null {
  const { onTerminal } = options ?? {};

  const { data } = useTaskProgress(taskId, {
    onTerminal,
  });

  if (!data || !modelId) return null;

  const percentage =
    data.bytes_total > 0
      ? Math.round((data.bytes_downloaded / data.bytes_total) * 100)
      : 0;

  const formattedBytes =
    data.bytes_total > 0
      ? `${formatBytes(data.bytes_downloaded)} / ${formatBytes(data.bytes_total)}`
      : 'Starting...';

  return {
    taskId: data.task_id,
    modelId: data.model_id,
    status: data.status,
    bytesDownloaded: data.bytes_downloaded,
    bytesTotal: data.bytes_total,
    error: data.error,
    updatedAt: data.updated_at,
    percentage,
    formattedBytes,
  };
}

// Phase 2C hooks - placeholders for future implementation
// export function useModelUsage(modelId: string) { ... }
// export function useDeleteModel() { ... }
