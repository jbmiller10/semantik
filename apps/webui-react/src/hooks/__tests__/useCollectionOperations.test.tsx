import { renderHook, waitFor, act } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import {
  useCollectionOperations,
  useAddSource,
  useRemoveSource,
  useReindexCollection,
  useUpdateOperationInCache,
  operationKeys,
} from '../useCollectionOperations';
import { collectionKeys } from '../useCollections';
import { collectionsV2Api } from '../../services/api/v2/collections';
import { useUIStore } from '../../stores/uiStore';
import type { Operation, Collection, AddSourceRequest } from '../../types/collection';

// Mock the API module
vi.mock('../../services/api/v2/collections', () => ({
  collectionsV2Api: {
    listOperations: vi.fn(),
    addSource: vi.fn(),
    removeSource: vi.fn(),
    reindex: vi.fn(),
  },
  handleApiError: vi.fn((error) => error.message || 'API Error'),
}));

// Mock the UI store
vi.mock('../../stores/uiStore', () => ({
  useUIStore: vi.fn(() => ({
    addToast: vi.fn(),
  })),
}));

// Test helpers
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

// Mock data
const mockOperations: Operation[] = [
  {
    id: 'op-1',
    collection_id: 'col-1',
    type: 'index',
    status: 'processing',
    config: { source_path: '/data/docs' },
    created_at: '2024-01-01T00:00:00Z',
    progress: 50,
  },
  {
    id: 'op-2',
    collection_id: 'col-1',
    type: 'append',
    status: 'completed',
    config: { source_path: '/data/images' },
    created_at: '2024-01-01T01:00:00Z',
    completed_at: '2024-01-01T02:00:00Z',
  },
  {
    id: 'op-3',
    collection_id: 'col-1',
    type: 'reindex',
    status: 'pending',
    config: {},
    created_at: '2024-01-01T03:00:00Z',
  },
];

const mockCollection: Collection = {
  id: 'col-1',
  name: 'Test Collection',
  description: 'Test description',
  owner_id: 1,
  vector_store_name: 'test_collection',
  embedding_model: 'test-model',
  quantization: 'float16',
  chunk_size: 1000,
  chunk_overlap: 200,
  is_public: false,
  status: 'ready',
  metadata: {},
  document_count: 10,
  vector_count: 100,
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
};

describe('useCollectionOperations', () => {
  let queryClient: QueryClient;
  let mockAddToast: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.clearAllMocks();
    queryClient = createTestQueryClient();
    mockAddToast = vi.fn();
    vi.mocked(useUIStore).mockReturnValue({ addToast: mockAddToast });
  });

  describe('useCollectionOperations hook', () => {
    it('should fetch operations for a collection', async () => {
      vi.mocked(collectionsV2Api.listOperations).mockResolvedValue({
        data: mockOperations,
      } as any);

      const { result } = renderHook(() => useCollectionOperations('col-1'), {
        wrapper: createWrapper(queryClient),
      });

      expect(result.current.isLoading).toBe(true);

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockOperations);
      expect(collectionsV2Api.listOperations).toHaveBeenCalledWith('col-1', undefined);
    });

    it('should accept limit option', async () => {
      vi.mocked(collectionsV2Api.listOperations).mockResolvedValue({
        data: mockOperations.slice(0, 2),
      } as any);

      const { result } = renderHook(() => useCollectionOperations('col-1', { limit: 2 }), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(collectionsV2Api.listOperations).toHaveBeenCalledWith('col-1', { limit: 2 });
    });

    it('should auto-refetch when there are active operations', async () => {
      vi.mocked(collectionsV2Api.listOperations).mockResolvedValue({
        data: mockOperations,
      } as any);

      const { result } = renderHook(() => useCollectionOperations('col-1'), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      // The refetchInterval function should return 5000ms for active operations
      const refetchInterval = queryClient.getQueryDefaults(operationKeys.list('col-1')).queries?.refetchInterval;
      if (typeof refetchInterval === 'function') {
        const query = queryClient.getQueryCache().find(operationKeys.list('col-1'));
        const interval = refetchInterval({ state: { data: mockOperations } } as any, query);
        expect(interval).toBe(5000); // 5 seconds for active operations
      }
    });

    it('should not auto-refetch when all operations are completed', async () => {
      const completedOperations = mockOperations.map(op => ({
        ...op,
        status: 'completed' as const,
      }));

      vi.mocked(collectionsV2Api.listOperations).mockResolvedValue({
        data: completedOperations,
      } as any);

      const { result } = renderHook(() => useCollectionOperations('col-1'), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      const refetchInterval = queryClient.getQueryDefaults(operationKeys.list('col-1')).queries?.refetchInterval;
      if (typeof refetchInterval === 'function') {
        const query = queryClient.getQueryCache().find(operationKeys.list('col-1'));
        const interval = refetchInterval({ state: { data: completedOperations } } as any, query);
        expect(interval).toBe(false);
      }
    });
  });

  describe('useAddSource hook', () => {
    it('should add source with optimistic updates', async () => {
      const sourcePath = '/new/data/path';
      const config: AddSourceRequest['config'] = { recursive: true };
      
      const newOperation: Operation = {
        id: 'real-op-id',
        collection_id: 'col-1',
        type: 'append',
        status: 'pending',
        config: { source_path: sourcePath, ...config },
        created_at: '2024-01-03T00:00:00Z',
      };

      vi.mocked(collectionsV2Api.addSource).mockResolvedValue({
        data: newOperation,
      } as any);

      // Pre-populate caches
      queryClient.setQueryData(operationKeys.list('col-1'), mockOperations);
      queryClient.setQueryData(collectionKeys.detail('col-1'), mockCollection);

      const { result } = renderHook(() => useAddSource(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        await result.current.mutateAsync({
          collectionId: 'col-1',
          sourcePath,
          config,
        });
      });

      // Check operations cache was updated
      const operationsCache = queryClient.getQueryData<Operation[]>(operationKeys.list('col-1'));
      expect(operationsCache).toHaveLength(4); // 3 existing + 1 new
      expect(operationsCache?.[0]).toEqual(newOperation); // New operation should be first

      // Check collection status was updated
      const collectionCache = queryClient.getQueryData<Collection>(collectionKeys.detail('col-1'));
      expect(collectionCache?.status).toBe('processing');
      expect(collectionCache?.isProcessing).toBe(true);

      // Check success toast
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'success',
        message: 'Data source added, indexing started',
      });
    });

    it('should rollback on error', async () => {
      const error = new Error('Failed to add source');
      vi.mocked(collectionsV2Api.addSource).mockRejectedValue(error);

      // Pre-populate caches
      queryClient.setQueryData(operationKeys.list('col-1'), mockOperations);
      queryClient.setQueryData(collectionKeys.detail('col-1'), mockCollection);

      const { result } = renderHook(() => useAddSource(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        try {
          await result.current.mutateAsync({
            collectionId: 'col-1',
            sourcePath: '/test',
          });
        } catch (e) {
          // Expected to throw
        }
      });

      // Check caches were rolled back
      const operationsCache = queryClient.getQueryData<Operation[]>(operationKeys.list('col-1'));
      expect(operationsCache).toEqual(mockOperations);

      const collectionCache = queryClient.getQueryData<Collection>(collectionKeys.detail('col-1'));
      expect(collectionCache).toEqual(mockCollection);

      // Check error toast
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'error',
        message: 'Failed to add source',
      });
    });
  });

  describe('useRemoveSource hook', () => {
    it('should remove source with optimistic updates', async () => {
      const sourcePath = '/data/docs';
      
      const removeOperation: Operation = {
        id: 'remove-op-id',
        collection_id: 'col-1',
        type: 'remove_source',
        status: 'pending',
        config: { source_path: sourcePath },
        created_at: '2024-01-03T00:00:00Z',
      };

      vi.mocked(collectionsV2Api.removeSource).mockResolvedValue({
        data: removeOperation,
      } as any);

      // Pre-populate caches
      queryClient.setQueryData(operationKeys.list('col-1'), mockOperations);
      queryClient.setQueryData(collectionKeys.detail('col-1'), mockCollection);

      const { result } = renderHook(() => useRemoveSource(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        await result.current.mutateAsync({
          collectionId: 'col-1',
          sourcePath,
        });
      });

      // Check operations cache was updated
      const operationsCache = queryClient.getQueryData<Operation[]>(operationKeys.list('col-1'));
      expect(operationsCache?.[0]).toEqual(removeOperation);

      // Check collection status was updated
      const collectionCache = queryClient.getQueryData<Collection>(collectionKeys.detail('col-1'));
      expect(collectionCache?.status).toBe('processing');

      // Check success toast
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'success',
        message: 'Source removed successfully',
      });
    });
  });

  describe('useReindexCollection hook', () => {
    it('should trigger reindex with optimistic updates', async () => {
      const reindexOperation: Operation = {
        id: 'reindex-op-id',
        collection_id: 'col-1',
        type: 'reindex',
        status: 'pending',
        config: {},
        created_at: '2024-01-03T00:00:00Z',
      };

      vi.mocked(collectionsV2Api.reindex).mockResolvedValue({
        data: reindexOperation,
      } as any);

      // Pre-populate caches
      queryClient.setQueryData(operationKeys.list('col-1'), mockOperations);
      queryClient.setQueryData(collectionKeys.detail('col-1'), mockCollection);

      const { result } = renderHook(() => useReindexCollection(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        await result.current.mutateAsync({
          collectionId: 'col-1',
        });
      });

      // Check operations cache was updated
      const operationsCache = queryClient.getQueryData<Operation[]>(operationKeys.list('col-1'));
      expect(operationsCache?.[0]).toEqual(reindexOperation);

      // Check collection status was updated
      const collectionCache = queryClient.getQueryData<Collection>(collectionKeys.detail('col-1'));
      expect(collectionCache?.status).toBe('processing');

      // Check success toast with collection name
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'success',
        message: 'Re-indexing started for "Test Collection". You can track progress on the collection page.',
      });
    });

    it('should accept configuration options', async () => {
      const config = { force: true, chunk_size: 2000 };
      
      vi.mocked(collectionsV2Api.reindex).mockResolvedValue({
        data: mockOperations[0],
      } as any);

      const { result } = renderHook(() => useReindexCollection(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        await result.current.mutateAsync({
          collectionId: 'col-1',
          config,
        });
      });

      expect(collectionsV2Api.reindex).toHaveBeenCalledWith('col-1', config);
    });
  });

  describe('useUpdateOperationInCache utility', () => {
    it('should update operation progress in cache', () => {
      // Pre-populate operations cache
      queryClient.setQueryData(operationKeys.list('col-1'), mockOperations);

      const { result } = renderHook(() => useUpdateOperationInCache(), {
        wrapper: createWrapper(queryClient),
      });

      act(() => {
        result.current('op-1', {
          progress: 75,
          status: 'processing',
        });
      });

      const operationsCache = queryClient.getQueryData<Operation[]>(operationKeys.list('col-1'));
      const updatedOp = operationsCache?.find(op => op.id === 'op-1');
      expect(updatedOp?.progress).toBe(75);
    });

    it('should update collection status when operation completes', () => {
      // Pre-populate caches
      queryClient.setQueryData(operationKeys.list('col-1'), mockOperations);
      queryClient.setQueryData(collectionKeys.detail('col-1'), {
        ...mockCollection,
        activeOperation: mockOperations[0],
        isProcessing: true,
        status: 'processing',
      });
      queryClient.setQueryData(collectionKeys.lists(), [{
        ...mockCollection,
        activeOperation: mockOperations[0],
        isProcessing: true,
        status: 'processing',
      }]);

      const { result } = renderHook(() => useUpdateOperationInCache(), {
        wrapper: createWrapper(queryClient),
      });

      act(() => {
        result.current('op-1', {
          status: 'completed',
          completed_at: '2024-01-01T04:00:00Z',
        });
      });

      // Check operation was updated
      const operationsCache = queryClient.getQueryData<Operation[]>(operationKeys.list('col-1'));
      const updatedOp = operationsCache?.find(op => op.id === 'op-1');
      expect(updatedOp?.status).toBe('completed');

      // Check collection status was updated in detail view
      const collectionDetail = queryClient.getQueryData<Collection>(collectionKeys.detail('col-1'));
      expect(collectionDetail?.activeOperation).toBeUndefined();
      expect(collectionDetail?.isProcessing).toBe(false);
      expect(collectionDetail?.status).toBe('ready');

      // Check collection status was updated in list view
      const collectionsList = queryClient.getQueryData<Collection[]>(collectionKeys.lists());
      const collectionInList = collectionsList?.find(c => c.id === 'col-1');
      expect(collectionInList?.activeOperation).toBeUndefined();
      expect(collectionInList?.isProcessing).toBe(false);
      expect(collectionInList?.status).toBe('ready');
    });

    it('should handle operation failure', () => {
      // Pre-populate caches
      queryClient.setQueryData(operationKeys.list('col-1'), mockOperations);
      queryClient.setQueryData(collectionKeys.detail('col-1'), {
        ...mockCollection,
        activeOperation: mockOperations[0],
        status: 'processing',
      });

      const { result } = renderHook(() => useUpdateOperationInCache(), {
        wrapper: createWrapper(queryClient),
      });

      act(() => {
        result.current('op-1', {
          status: 'failed',
          error: 'Out of memory',
        });
      });

      // Check collection status was set to error
      const collectionDetail = queryClient.getQueryData<Collection>(collectionKeys.detail('col-1'));
      expect(collectionDetail?.status).toBe('error');
      expect(collectionDetail?.activeOperation).toBeUndefined();
    });

    it('should invalidate queries on successful completion', async () => {
      const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

      // Pre-populate caches
      queryClient.setQueryData(operationKeys.list('col-1'), mockOperations);
      queryClient.setQueryData(collectionKeys.detail('col-1'), {
        ...mockCollection,
        activeOperation: mockOperations[0],
      });

      const { result } = renderHook(() => useUpdateOperationInCache(), {
        wrapper: createWrapper(queryClient),
      });

      act(() => {
        result.current('op-1', {
          status: 'completed',
        });
      });

      // Check that queries were invalidated to refresh data
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: collectionKeys.detail('col-1') });
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: collectionKeys.lists() });
    });
  });

  describe('cache invalidation patterns', () => {
    it('should invalidate all relevant queries after mutation', async () => {
      vi.mocked(collectionsV2Api.addSource).mockResolvedValue({
        data: mockOperations[0],
      } as any);

      const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

      const { result } = renderHook(() => useAddSource(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        await result.current.mutateAsync({
          collectionId: 'col-1',
          sourcePath: '/test',
        });
      });

      await waitFor(() => {
        expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: operationKeys.list('col-1') });
        expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: collectionKeys.detail('col-1') });
        expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: collectionKeys.lists() });
      });
    });
  });

  describe('error handling', () => {
    it('should handle API errors gracefully', async () => {
      const error = new Error('Network error');
      vi.mocked(collectionsV2Api.listOperations).mockRejectedValue(error);

      const { result } = renderHook(() => useCollectionOperations('col-1'), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error).toBe(error);
    });
  });
});