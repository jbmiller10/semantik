import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { AxiosError } from 'axios';

import {
  modelManagerKeys,
  useModelManagerModels,
  useTaskProgress,
  useStartModelDownload,
  useModelDownloadProgress,
  useModelUsage,
  useStartModelDelete,
  useModelDeleteProgress,
} from '../useModelManager';
import { modelManagerApi } from '../../services/api/v2/model-manager';
import { handleApiError } from '../../services/api/v2/collections';
import type {
  ModelListResponse,
  ModelUsageResponse,
  ModelManagerConflictResponse,
  TaskProgressResponse,
  TaskResponse,
} from '../../types/model-manager';

const mockAddToast = vi.fn();

vi.mock('../../services/api/v2/model-manager', () => ({
  modelManagerApi: {
    listModels: vi.fn(),
    getTaskProgress: vi.fn(),
    getModelUsage: vi.fn(),
    startDownload: vi.fn(),
    deleteModel: vi.fn(),
  },
}));

vi.mock('../../services/api/v2/collections', () => ({
  handleApiError: vi.fn(),
}));

vi.mock('../../stores/uiStore', () => ({
  useUIStore: () => ({
    addToast: mockAddToast,
  }),
}));

const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        staleTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  });

const createWrapper = (queryClient: QueryClient) => {
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

const createAxiosError = (status: number, data: unknown) => {
  const axiosError = new AxiosError('Request failed');
  axiosError.response = {
    status,
    data,
    statusText: 'Error',
    headers: {},
    config: {} as never,
  };
  return axiosError;
};

const createTaskProgress = (
  overrides?: Partial<TaskProgressResponse>
): TaskProgressResponse => ({
  task_id: 'task-1',
  model_id: 'model-1',
  operation: 'download',
  status: 'running',
  bytes_downloaded: 0,
  bytes_total: 0,
  error: null,
  updated_at: Date.now(),
  ...overrides,
});

describe('useModelManager hooks', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockAddToast.mockClear();
  });

  describe('modelManagerKeys', () => {
    it('should generate correct query keys', () => {
      expect(modelManagerKeys.all).toEqual(['model-manager']);
      expect(modelManagerKeys.lists()).toEqual(['model-manager', 'list']);
      expect(modelManagerKeys.list()).toEqual(['model-manager', 'list', {}]);
      expect(modelManagerKeys.list({ model_type: 'embedding' })).toEqual([
        'model-manager',
        'list',
        { model_type: 'embedding' },
      ]);
      expect(modelManagerKeys.usage('m-1')).toEqual(['model-manager', 'usage', 'm-1']);
      expect(modelManagerKeys.task('t-1')).toEqual(['model-manager', 'task', 't-1']);
    });
  });

	  describe('useModelManagerModels', () => {
	    it('should fetch curated models with params', async () => {
	      const response: ModelListResponse = { models: [], cache_size: null, hf_cache_scan_error: null };
	      vi.mocked(modelManagerApi.listModels).mockResolvedValue(response);

      const queryClient = createTestQueryClient();
      const { result } = renderHook(
        () =>
          useModelManagerModels({
            modelType: 'embedding',
            installedOnly: true,
            includeCacheSize: true,
          }),
        { wrapper: createWrapper(queryClient) }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(modelManagerApi.listModels).toHaveBeenCalledWith({
        model_type: 'embedding',
        installed_only: true,
        include_cache_size: true,
      });
    });

	    it('should not fetch when disabled', async () => {
	      vi.mocked(modelManagerApi.listModels).mockResolvedValue({
	        models: [],
	        cache_size: null,
	        hf_cache_scan_error: null,
	      });

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useModelManagerModels({ enabled: false }), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(result.current.fetchStatus).toBe('idle');
      expect(modelManagerApi.listModels).not.toHaveBeenCalled();
    });
  });

  describe('useTaskProgress', () => {
    it('should tolerate 404s during task creation up to grace count', async () => {
      vi.mocked(modelManagerApi.getTaskProgress)
        .mockRejectedValueOnce(createAxiosError(404, { detail: 'Not found' }))
        .mockRejectedValueOnce(createAxiosError(404, { detail: 'Not found' }))
        .mockResolvedValueOnce(createTaskProgress({ status: 'running' }));

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useTaskProgress('task-1', { grace404Count: 3 }), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });
      expect(result.current.data).toBeNull();

      await act(async () => {
        await result.current.refetch();
      });
      expect(result.current.data).toBeNull();

      await act(async () => {
        await result.current.refetch();
      });

      await waitFor(() => {
        expect(result.current.data?.status).toBe('running');
      });

      expect(modelManagerApi.getTaskProgress).toHaveBeenCalledTimes(3);
    });

    it('should surface a 404 when grace count is 1', async () => {
      vi.mocked(modelManagerApi.getTaskProgress).mockRejectedValue(
        createAxiosError(404, { detail: 'Not found' })
      );

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useTaskProgress('task-1', { grace404Count: 1 }), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });
      expect(result.current.error).toBeTruthy();
    });

    it('should call onTerminal exactly once per task id', async () => {
      const onTerminal = vi.fn();
      vi.mocked(modelManagerApi.getTaskProgress).mockResolvedValue(
        createTaskProgress({ task_id: 'task-1', status: 'completed' })
      );

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useTaskProgress('task-1', { onTerminal }), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.data?.status).toBe('completed');
      });

      await waitFor(() => {
        expect(onTerminal).toHaveBeenCalledTimes(1);
      });

      await act(async () => {
        await result.current.refetch();
      });

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(onTerminal).toHaveBeenCalledTimes(1);
    });

    it('should reset onTerminal tracking when taskId changes', async () => {
      const onTerminal = vi.fn();

      vi.mocked(modelManagerApi.getTaskProgress).mockImplementation(async (taskId: string) => {
        return createTaskProgress({ task_id: taskId, status: 'completed' });
      });

      const queryClient = createTestQueryClient();
      const { rerender } = renderHook(({ taskId }) => useTaskProgress(taskId, { onTerminal }), {
        wrapper: createWrapper(queryClient),
        initialProps: { taskId: 'task-1' },
      });

      await waitFor(() => {
        expect(onTerminal).toHaveBeenCalledWith(expect.objectContaining({ task_id: 'task-1' }));
      });

      rerender({ taskId: 'task-2' });

      await waitFor(() => {
        expect(onTerminal).toHaveBeenCalledWith(expect.objectContaining({ task_id: 'task-2' }));
      });
      expect(onTerminal).toHaveBeenCalledTimes(2);
    });
  });

  describe('useStartModelDownload', () => {
    it('should track task IDs and show warning toasts', async () => {
      const queryClient = createTestQueryClient();
      const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

      const response: TaskResponse = {
        task_id: 'download-task-1',
        model_id: 'model-1',
        operation: 'download',
        status: 'running',
        warnings: ['Low disk space'],
      };
      vi.mocked(modelManagerApi.startDownload).mockResolvedValue(response);

      const { result } = renderHook(() => useStartModelDownload(), {
        wrapper: createWrapper(queryClient),
      });

      act(() => {
        result.current.startDownload('model-1');
      });

      await waitFor(() => {
        expect(result.current.getTaskId('model-1')).toBe('download-task-1');
      });

      expect(invalidateSpy).toHaveBeenCalled();
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'warning',
        message: 'Low disk space',
      });

      act(() => {
        result.current.clearTaskId('model-1');
      });

      expect(result.current.getTaskId('model-1')).toBeUndefined();
    });

    it('should handle already_installed response', async () => {
      const queryClient = createTestQueryClient();
      const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

      vi.mocked(modelManagerApi.startDownload).mockResolvedValue({
        task_id: null,
        model_id: 'model-1',
        operation: 'download',
        status: 'already_installed',
        warnings: [],
      });

      const { result } = renderHook(() => useStartModelDownload(), {
        wrapper: createWrapper(queryClient),
      });

      act(() => {
        result.current.startDownload('model-1');
      });

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          type: 'info',
          message: 'Model is already installed',
        });
      });
      expect(invalidateSpy).toHaveBeenCalled();
    });

    it('should handle 409 cross operation conflicts', async () => {
      const queryClient = createTestQueryClient();

      const conflict: ModelManagerConflictResponse = {
        conflict_type: 'cross_op_exclusion',
        detail: 'Operation blocked',
        model_id: 'model-1',
        active_operation: 'delete',
        active_task_id: 'task-2',
        blocked_by_collections: [],
        requires_confirmation: false,
        warnings: [],
      };

      vi.mocked(modelManagerApi.startDownload).mockRejectedValue(
        createAxiosError(409, { detail: conflict })
      );

      const { result } = renderHook(() => useStartModelDownload(), {
        wrapper: createWrapper(queryClient),
      });

      act(() => {
        result.current.startDownload('model-1');
      });

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          type: 'warning',
          message: 'Cannot download: another operation is in progress',
        });
      });
    });

    it('should clear existing task id on retry, even if the retry fails', async () => {
      const queryClient = createTestQueryClient();
      const response: TaskResponse = {
        task_id: 'download-task-1',
        model_id: 'model-1',
        operation: 'download',
        status: 'running',
        warnings: [],
      };

      vi.mocked(modelManagerApi.startDownload)
        .mockResolvedValueOnce(response)
        .mockRejectedValueOnce(new Error('boom'));
      vi.mocked(handleApiError).mockReturnValue('boom');

      const { result } = renderHook(() => useStartModelDownload(), {
        wrapper: createWrapper(queryClient),
      });

      act(() => {
        result.current.startDownload('model-1');
      });

      await waitFor(() => {
        expect(result.current.getTaskId('model-1')).toBe('download-task-1');
      });

      act(() => {
        result.current.startDownload('model-1');
      });

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          type: 'error',
          message: 'Download failed: boom',
        });
      });

      expect(result.current.getTaskId('model-1')).toBeUndefined();
    });
  });

  describe('useModelDownloadProgress', () => {
    it('should format byte counts and percentage', async () => {
      vi.mocked(modelManagerApi.getTaskProgress).mockResolvedValue(
        createTaskProgress({
          task_id: 'download-task-1',
          model_id: 'model-1',
          bytes_downloaded: 50,
          bytes_total: 100,
          status: 'running',
        })
      );

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useModelDownloadProgress('model-1', 'download-task-1'), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current).not.toBeNull();
      });

      expect(result.current?.percentage).toBe(50);
      expect(result.current?.formattedBytes).toBe('50.0 B / 100.0 B');
    });

    it('should return null when modelId is missing', async () => {
      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useModelDownloadProgress(null, null), {
        wrapper: createWrapper(queryClient),
      });
      expect(result.current).toBeNull();
    });
  });

  describe('useModelUsage', () => {
    it('should not fetch when modelId is null', async () => {
      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useModelUsage(null), {
        wrapper: createWrapper(queryClient),
      });

      expect(result.current.data).toBeNull();
      expect(modelManagerApi.getModelUsage).not.toHaveBeenCalled();
    });

    it('should fetch usage when enabled and modelId provided', async () => {
	      const usage: ModelUsageResponse = {
	        model_id: 'model-1',
	        is_installed: true,
	        size_on_disk_mb: 123,
	        estimated_freed_size_mb: 123,
	        blocked_by_collections: [],
	        user_preferences_count: 0,
	        llm_config_count: 0,
	        is_default_embedding_model: false,
	        loaded_in_vecpipe: false,
	        loaded_vecpipe_model_types: [],
	        hf_cache_scan_error: null,
	        vecpipe_query_error: null,
	        warnings: [],
	        can_delete: true,
	        requires_confirmation: false,
	      };
      vi.mocked(modelManagerApi.getModelUsage).mockResolvedValue(usage);

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useModelUsage('model-1'), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.data).toEqual(usage);
      });
      expect(modelManagerApi.getModelUsage).toHaveBeenCalledWith('model-1');
    });
  });

  describe('useStartModelDelete', () => {
    it('should surface requires_confirmation conflicts without a toast', async () => {
      const queryClient = createTestQueryClient();

      const conflict: ModelManagerConflictResponse = {
        conflict_type: 'requires_confirmation',
        detail: 'Model has user preferences',
        model_id: 'model-1',
        active_operation: null,
        active_task_id: null,
        blocked_by_collections: [],
        requires_confirmation: true,
        warnings: ['Will remove preferences'],
      };
      vi.mocked(modelManagerApi.deleteModel).mockRejectedValue(
        createAxiosError(409, { detail: conflict })
      );

      const { result } = renderHook(() => useStartModelDelete(), {
        wrapper: createWrapper(queryClient),
      });

      act(() => {
        result.current.startDelete('model-1', false);
      });

      await waitFor(() => {
        expect(result.current.lastConflict?.conflict_type).toBe('requires_confirmation');
      });
      expect(result.current.lastConflict?.model_id).toBe('model-1');
      expect(mockAddToast).not.toHaveBeenCalled();
    });

    it('should show in_use_block conflict message', async () => {
      const queryClient = createTestQueryClient();

      const conflict: ModelManagerConflictResponse = {
        conflict_type: 'in_use_block',
        detail: 'In use',
        model_id: 'model-1',
        active_operation: null,
        active_task_id: null,
        blocked_by_collections: ['c1', 'c2'],
        requires_confirmation: false,
        warnings: [],
      };
      vi.mocked(modelManagerApi.deleteModel).mockRejectedValue(createAxiosError(409, { detail: conflict }));

      const { result } = renderHook(() => useStartModelDelete(), {
        wrapper: createWrapper(queryClient),
      });

      act(() => {
        result.current.startDelete('model-1', false);
      });

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          type: 'error',
          message: 'Cannot delete: model is used by 2 collection(s)',
        });
      });
    });
  });

  describe('useModelDeleteProgress', () => {
    it('should map running status to isDeleting', async () => {
      vi.mocked(modelManagerApi.getTaskProgress).mockResolvedValue(
        createTaskProgress({
          operation: 'delete',
          status: 'running',
          task_id: 'delete-task-1',
          model_id: 'model-1',
        })
      );

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useModelDeleteProgress('model-1', 'delete-task-1'), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current).not.toBeNull();
      });

      expect(result.current?.isDeleting).toBe(true);
      expect(result.current?.isFailed).toBe(false);
    });

    it('should map failed status to isFailed', async () => {
      vi.mocked(modelManagerApi.getTaskProgress).mockResolvedValue(
        createTaskProgress({
          operation: 'delete',
          status: 'failed',
          task_id: 'delete-task-1',
          model_id: 'model-1',
          error: 'boom',
        })
      );

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useModelDeleteProgress('model-1', 'delete-task-1'), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current?.isFailed).toBe(true);
      });

      expect(result.current?.error).toBe('boom');
    });
  });
});
