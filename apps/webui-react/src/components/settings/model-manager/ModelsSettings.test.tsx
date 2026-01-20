import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { useEffect } from 'react';

import ModelsSettings from './ModelsSettings';
import {
  useModelManagerModels,
  useStartModelDownload,
  useModelDownloadProgress,
  useStartModelDelete,
  useModelDeleteProgress,
} from '../../../hooks/useModelManager';
import type {
  CuratedModelResponse,
  ModelListResponse,
  ModelManagerConflictResponse,
  TaskProgressResponse,
} from '../../../types/model-manager';

const mockAddToast = vi.fn();

vi.mock('../../../stores/uiStore', () => ({
  useUIStore: () => ({
    addToast: mockAddToast,
  }),
}));

vi.mock('../../../hooks/useModelManager', () => ({
  modelManagerKeys: {
    lists: () => ['model-manager', 'list'],
  },
  useModelManagerModels: vi.fn(),
  useStartModelDownload: vi.fn(),
  useModelDownloadProgress: vi.fn(),
  useStartModelDelete: vi.fn(),
  useModelDeleteProgress: vi.fn(),
}));

vi.mock('./ModelCard', () => ({
  default: ({
    model,
    onDownload,
    onDelete,
    onDismissError,
    onDismissDeleteError,
  }: {
    model: CuratedModelResponse;
    onDownload?: (modelId: string) => void;
    onDelete?: (modelId: string) => void;
    onDismissError?: (modelId: string) => void;
    onDismissDeleteError?: (modelId: string) => void;
  }) => (
    <div data-testid={`model-card-${model.id}`}>
      <div>{model.name}</div>
      {!model.is_installed && onDownload && (
        <button onClick={() => onDownload(model.id)}>Download</button>
      )}
      {model.is_installed && onDelete && <button onClick={() => onDelete(model.id)}>Delete</button>}
      {onDismissError && <button onClick={() => onDismissError(model.id)}>DismissDownloadError</button>}
      {onDismissDeleteError && (
        <button onClick={() => onDismissDeleteError(model.id)}>DismissDeleteError</button>
      )}
    </div>
  ),
}));

const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: 0, refetchInterval: false },
      mutations: { retry: false },
    },
  });

const createWrapper = (queryClient: QueryClient) => {
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>{children}</MemoryRouter>
    </QueryClientProvider>
  );
};

const createModel = (overrides: Partial<CuratedModelResponse>): CuratedModelResponse => ({
  id: 'model-1',
  name: 'Model',
  description: 'Description',
  model_type: 'embedding',
  memory_mb: { int8: 123 },
  is_installed: false,
  size_on_disk_mb: null,
  used_by_collections: [],
  active_download_task_id: null,
  active_delete_task_id: null,
  embedding_details: null,
  llm_details: null,
  ...overrides,
});

describe('ModelsSettings', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockAddToast.mockClear();
  });

  it('shows an error state and retries via refetch()', async () => {
    const refetch = vi.fn();
    vi.mocked(useModelManagerModels).mockReturnValue({
      data: null,
      isLoading: false,
      isError: true,
      error: new Error('boom'),
      refetch,
      isFetching: false,
    } as unknown as ReturnType<typeof useModelManagerModels>);

    vi.mocked(useStartModelDownload).mockReturnValue({
      startDownload: vi.fn(),
      getTaskId: vi.fn(() => undefined),
      clearTaskId: vi.fn(),
      isPending: false,
    } as unknown as ReturnType<typeof useStartModelDownload>);

    vi.mocked(useStartModelDelete).mockReturnValue({
      startDelete: vi.fn(),
      getTaskId: vi.fn(() => undefined),
      clearTaskId: vi.fn(),
      isPending: false,
      lastConflict: null,
      clearConflict: vi.fn(),
    } as unknown as ReturnType<typeof useStartModelDelete>);

    const queryClient = createTestQueryClient();
    render(<ModelsSettings />, { wrapper: createWrapper(queryClient) });

    expect(screen.getByText('Failed to load models')).toBeInTheDocument();
    expect(screen.getByText('boom')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(refetch).toHaveBeenCalled();
  });

  it('renders models, filters/tabs, and can lazy-load cache size info', async () => {
    const startDownload = vi.fn();
    const clearDownloadTaskId = vi.fn();
    const getDownloadTaskId = vi.fn((modelId: string) =>
      modelId === 'embed-available' ? 'started-download-task' : undefined
    );

    const startDelete = vi.fn();
    const clearDeleteTaskId = vi.fn();
    const getDeleteTaskId = vi.fn((modelId: string) =>
      modelId === 'embed-installed' ? 'started-delete-task' : undefined
    );

    vi.mocked(useStartModelDownload).mockReturnValue({
      startDownload,
      getTaskId: getDownloadTaskId,
      clearTaskId: clearDownloadTaskId,
      isPending: false,
    } as unknown as ReturnType<typeof useStartModelDownload>);

    vi.mocked(useStartModelDelete).mockReturnValue({
      startDelete,
      getTaskId: getDeleteTaskId,
      clearTaskId: clearDeleteTaskId,
      isPending: false,
      lastConflict: null,
      clearConflict: vi.fn(),
    } as unknown as ReturnType<typeof useStartModelDelete>);

    const models: CuratedModelResponse[] = [
      createModel({
        id: 'embed-installed',
        name: 'Alpha',
        is_installed: true,
        model_type: 'embedding',
        active_delete_task_id: 'active-delete-task',
      }),
      createModel({
        id: 'embed-available',
        name: 'Beta',
        is_installed: false,
        model_type: 'embedding',
        active_download_task_id: 'active-download-task',
      }),
      createModel({
        id: 'llm-1',
        name: 'Local LLM 1',
        is_installed: false,
        model_type: 'llm',
      }),
    ];

    vi.mocked(useModelManagerModels).mockImplementation((options) => {
      const includeCacheSize = options?.includeCacheSize ?? false;
	      const response: ModelListResponse = {
	        models,
	        cache_size: includeCacheSize
	          ? {
	              total_cache_size_mb: 1024,
	              managed_cache_size_mb: 512,
	              unmanaged_cache_size_mb: 512,
	              unmanaged_repo_count: 2,
	            }
	          : null,
	        hf_cache_scan_error: null,
	      };

      return {
        data: response,
        isLoading: false,
        isError: false,
        error: null,
        refetch: vi.fn(),
        isFetching: false,
      } as unknown as ReturnType<typeof useModelManagerModels>;
    });

    const completedDownloadProgress: TaskProgressResponse = {
      task_id: 'started-download-task',
      model_id: 'embed-available',
      operation: 'download',
      status: 'completed',
      bytes_downloaded: 100,
      bytes_total: 100,
      error: null,
      updated_at: Date.now(),
    };

    const completedDeleteProgress: TaskProgressResponse = {
      task_id: 'started-delete-task',
      model_id: 'embed-installed',
      operation: 'delete',
      status: 'completed',
      bytes_downloaded: 0,
      bytes_total: 0,
      error: null,
      updated_at: Date.now(),
    };

    const downloadTerminalSent = new Set<string>();
    vi.mocked(useModelDownloadProgress).mockImplementation((modelId, taskId, options) => {
      useEffect(() => {
        if (!taskId) return;
        if (modelId !== 'embed-available') return;
        if (downloadTerminalSent.has(taskId)) return;
        downloadTerminalSent.add(taskId);
        options?.onTerminal?.(completedDownloadProgress);
      }, [modelId, taskId, options]);
      return null;
    });

    const deleteTerminalSent = new Set<string>();
    vi.mocked(useModelDeleteProgress).mockImplementation((modelId, taskId, options) => {
      useEffect(() => {
        if (!taskId) return;
        if (modelId !== 'embed-installed') return;
        if (deleteTerminalSent.has(taskId)) return;
        deleteTerminalSent.add(taskId);
        options?.onTerminal?.(completedDeleteProgress);
      }, [modelId, taskId, options]);
      return null;
    });

    const queryClient = createTestQueryClient();
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    render(<ModelsSettings />, { wrapper: createWrapper(queryClient) });

    // Default tab is Embedding
    expect(screen.getByText('Alpha')).toBeInTheDocument();
    expect(screen.getByText('Beta')).toBeInTheDocument();
    expect(screen.queryByText('Local LLM 1')).not.toBeInTheDocument();

    // Tabs filter models by type
    await userEvent.click(screen.getByRole('button', { name: /Local LLM/ }));
    expect(screen.getByText('Local LLM 1')).toBeInTheDocument();

    // Status filter filters to installed models
    await userEvent.click(screen.getByRole('button', { name: /Embedding/ }));
    await userEvent.click(screen.getByRole('button', { name: 'Installed' }));
    expect(screen.getByText('Alpha')).toBeInTheDocument();
    expect(screen.queryByText('Beta')).not.toBeInTheDocument();

    // Search filters by name/id/description
    await userEvent.click(screen.getByRole('button', { name: 'All' }));
    await userEvent.type(screen.getByPlaceholderText('Search models...'), 'beta');
    expect(screen.getByText('Beta')).toBeInTheDocument();
    expect(screen.queryByText('Alpha')).not.toBeInTheDocument();

    // Actions delegate to hook callbacks
    await userEvent.click(screen.getByRole('button', { name: 'Download' }));
    expect(startDownload).toHaveBeenCalledWith('embed-available');

    await userEvent.clear(screen.getByPlaceholderText('Search models...'));
    await userEvent.click(screen.getByRole('button', { name: 'Installed' }));
    await userEvent.click(screen.getByRole('button', { name: 'Delete' }));
    expect(startDelete).toHaveBeenCalledWith('embed-installed', false);

    // Effective task ids prefer recently started tasks over model.active_* task ids
    await waitFor(() => {
      expect(vi.mocked(useModelDownloadProgress).mock.calls).toEqual(
        expect.arrayContaining([
          expect.arrayContaining(['embed-available', 'started-download-task']),
        ])
      );
    });
    expect(vi.mocked(useModelDeleteProgress).mock.calls).toEqual(
      expect.arrayContaining([
        expect.arrayContaining(['embed-installed', 'started-delete-task']),
      ])
    );

    // Lazy-load cache size info
    expect(screen.getByRole('button', { name: 'Show cache usage' })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Show cache usage' }));
    expect(await screen.findByText(/\bmanaged\b/)).toBeInTheDocument();

    // Terminal status handlers invalidate queries and clear task ids
    await waitFor(() => {
      expect(clearDownloadTaskId).toHaveBeenCalledWith('embed-available');
      expect(clearDeleteTaskId).toHaveBeenCalledWith('embed-installed');
      expect(invalidateSpy).toHaveBeenCalled();
      expect(mockAddToast).toHaveBeenCalled();
    });
  });

  it('shows a delete confirmation modal when API requires confirmation', async () => {
    const clearConflict = vi.fn();
    const startDelete = vi.fn();

    const models: CuratedModelResponse[] = [
      createModel({
        id: 'model-requires-confirm',
        name: 'Danger Model',
        is_installed: true,
        size_on_disk_mb: 2048,
      }),
    ];

    vi.mocked(useModelManagerModels).mockReturnValue({
      data: { models, cache_size: null },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
      isFetching: false,
    } as unknown as ReturnType<typeof useModelManagerModels>);

    vi.mocked(useStartModelDownload).mockReturnValue({
      startDownload: vi.fn(),
      getTaskId: vi.fn(() => undefined),
      clearTaskId: vi.fn(),
      isPending: false,
    } as unknown as ReturnType<typeof useStartModelDownload>);

    const conflict: ModelManagerConflictResponse = {
      conflict_type: 'requires_confirmation',
      detail: 'Will remove preferences',
      model_id: 'model-requires-confirm',
      active_operation: null,
      active_task_id: null,
      blocked_by_collections: [],
      requires_confirmation: true,
      warnings: ['Will remove stored preferences'],
    };

    vi.mocked(useStartModelDelete).mockReturnValue({
      startDelete,
      getTaskId: vi.fn(() => undefined),
      clearTaskId: vi.fn(),
      isPending: false,
      lastConflict: { ...conflict },
      clearConflict,
    } as unknown as ReturnType<typeof useStartModelDelete>);

    vi.mocked(useModelDownloadProgress).mockReturnValue(null);
    vi.mocked(useModelDeleteProgress).mockReturnValue(null);

    const queryClient = createTestQueryClient();
    const { rerender } = render(<ModelsSettings />, { wrapper: createWrapper(queryClient) });

    // Effect opens modal and clears the conflict
    const cancelButton = await screen.findByRole('button', { name: 'Cancel' });
    expect(clearConflict).toHaveBeenCalled();
    expect(cancelButton).toHaveFocus();
    expect(screen.getByText('Will remove stored preferences')).toBeInTheDocument();
    expect(screen.getByText('2.0 GB')).toBeInTheDocument();

    // Escape closes the modal when not deleting
    await userEvent.keyboard('{Escape}');
    await waitFor(() => {
      expect(screen.queryByText('Delete Model')).not.toBeInTheDocument();
    });

    // Re-open and confirm
    vi.mocked(useStartModelDelete).mockReturnValue({
      startDelete,
      getTaskId: vi.fn(() => undefined),
      clearTaskId: vi.fn(),
      isPending: false,
      lastConflict: { ...conflict, warnings: [...conflict.warnings] },
      clearConflict,
    } as unknown as ReturnType<typeof useStartModelDelete>);
    rerender(<ModelsSettings />);

    await screen.findByRole('button', { name: 'Delete Model' });
    await userEvent.click(screen.getByRole('button', { name: 'Delete Model' }));

    expect(startDelete).toHaveBeenCalledWith('model-requires-confirm', true);
  });
});
