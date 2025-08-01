import { renderHook, waitFor, act } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import {
  useCollections,
  useCollection,
  useCreateCollection,
  useUpdateCollection,
  useDeleteCollection,
  useUpdateCollectionInCache,
  collectionKeys,
} from '../useCollections';
import { collectionsV2Api } from '../../services/api/v2/collections';
import { useUIStore } from '../../stores/uiStore';
import type { Collection, CreateCollectionRequest, UpdateCollectionRequest } from '../../types/collection';
import type { MockAxiosResponse } from '../../tests/types/test-types';

// Mock the API module
vi.mock('../../services/api/v2/collections', () => ({
  collectionsV2Api: {
    list: vi.fn(),
    get: vi.fn(),
    create: vi.fn(),
    update: vi.fn(),
    delete: vi.fn(),
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
const mockCollections: Collection[] = [
  {
    id: '1',
    name: 'Test Collection 1',
    description: 'Test description',
    owner_id: 1,
    vector_store_name: 'test_1',
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
  },
  {
    id: '2',
    name: 'Test Collection 2',
    description: 'Another test',
    owner_id: 1,
    vector_store_name: 'test_2',
    embedding_model: 'test-model',
    quantization: 'float16',
    chunk_size: 1000,
    chunk_overlap: 200,
    is_public: true,
    status: 'processing',
    metadata: {},
    document_count: 5,
    vector_count: 50,
    created_at: '2024-01-02T00:00:00Z',
    updated_at: '2024-01-02T00:00:00Z',
    activeOperation: {
      id: 'op-1',
      collection_id: '2',
      type: 'index',
      status: 'processing',
      config: {},
      created_at: '2024-01-02T00:00:00Z',
    },
  },
];

describe('useCollections', () => {
  let queryClient: QueryClient;
  let mockAddToast: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.clearAllMocks();
    queryClient = createTestQueryClient();
    mockAddToast = vi.fn();
    vi.mocked(useUIStore).mockReturnValue({ addToast: mockAddToast });
  });

  describe('useCollections hook', () => {
    it('should fetch and return collections list', async () => {
      vi.mocked(collectionsV2Api.list).mockResolvedValue({
        data: { collections: mockCollections },
      } as MockAxiosResponse<{ collections: Collection[] }>);

      const { result } = renderHook(() => useCollections(), {
        wrapper: createWrapper(queryClient),
      });

      expect(result.current.isLoading).toBe(true);
      expect(result.current.data).toBeUndefined();

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.data).toEqual(mockCollections);
      expect(collectionsV2Api.list).toHaveBeenCalledTimes(1);
    });

    it('should handle fetch error gracefully', async () => {
      const error = new Error('Network error');
      vi.mocked(collectionsV2Api.list).mockRejectedValue(error);

      const { result } = renderHook(() => useCollections(), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error).toBe(error);
      expect(result.current.data).toBeUndefined();
    });

    it('should auto-refetch when there are active operations', async () => {
      vi.mocked(collectionsV2Api.list).mockResolvedValue({
        data: { collections: mockCollections },
      } as MockAxiosResponse<{ collections: Collection[] }>);

      const { result } = renderHook(() => useCollections(), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      // The refetchInterval function should return 30000ms for active operations
      const refetchInterval = queryClient.getQueryDefaults(collectionKeys.lists()).queries?.refetchInterval;
      if (typeof refetchInterval === 'function') {
        const query = queryClient.getQueryCache().find(collectionKeys.lists());
        const interval = refetchInterval({ state: { data: mockCollections } } as Parameters<typeof refetchInterval>[0], query);
        expect(interval).toBe(30000); // 30 seconds for active operations
      }
    });

    it('should not auto-refetch when no active operations', async () => {
      const collectionsWithoutActive = mockCollections.map(c => ({
        ...c,
        status: 'ready' as const,
        activeOperation: undefined,
      }));

      vi.mocked(collectionsV2Api.list).mockResolvedValue({
        data: { collections: collectionsWithoutActive },
      } as MockAxiosResponse<{ collections: Collection[] }>);

      const { result } = renderHook(() => useCollections(), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      // The refetchInterval function should return false for no active operations
      const refetchInterval = queryClient.getQueryDefaults(collectionKeys.lists()).queries?.refetchInterval;
      if (typeof refetchInterval === 'function') {
        const query = queryClient.getQueryCache().find(collectionKeys.lists());
        const interval = refetchInterval({ state: { data: collectionsWithoutActive } } as Parameters<typeof refetchInterval>[0], query);
        expect(interval).toBe(false);
      }
    });
  });

  describe('useCollection hook', () => {
    it('should fetch a single collection by id', async () => {
      const collection = mockCollections[0];
      vi.mocked(collectionsV2Api.get).mockResolvedValue({
        data: collection,
      } as MockAxiosResponse<Collection>);

      const { result } = renderHook(() => useCollection('1'), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(collection);
      expect(collectionsV2Api.get).toHaveBeenCalledWith('1');
    });

    it('should not fetch if id is empty', () => {
      const { result } = renderHook(() => useCollection(''), {
        wrapper: createWrapper(queryClient),
      });

      // When enabled is false, query should be in fetching: 'idle' state
      expect(result.current.isLoading).toBe(false);
      expect(result.current.fetchStatus).toBe('idle');
      expect(collectionsV2Api.get).not.toHaveBeenCalled();
    });
  });

  describe('useCreateCollection hook', () => {
    it('should create collection with optimistic update', async () => {
      const createData: CreateCollectionRequest = {
        name: 'New Collection',
        description: 'New collection description',
        embedding_model: 'test-model',
        quantization: 'float16',
        chunk_size: 1000,
        chunk_overlap: 200,
      };

      const createdCollection: Collection = {
        id: 'real-id',
        ...createData,
        owner_id: 1,
        vector_store_name: 'new_collection',
        is_public: false,
        status: 'pending',
        metadata: {},
        document_count: 0,
        vector_count: 0,
        created_at: '2024-01-03T00:00:00Z',
        updated_at: '2024-01-03T00:00:00Z',
      };

      vi.mocked(collectionsV2Api.create).mockResolvedValue({
        data: createdCollection,
      } as MockAxiosResponse<Collection>);

      // Pre-populate cache with existing collections
      queryClient.setQueryData(collectionKeys.lists(), mockCollections);

      const { result } = renderHook(() => useCreateCollection(), {
        wrapper: createWrapper(queryClient),
      });

      let mutationResult: Collection | undefined;

      await act(async () => {
        mutationResult = await result.current.mutateAsync(createData);
      });

      // Check optimistic update happened
      const cachedData = queryClient.getQueryData<Collection[]>(collectionKeys.lists());
      expect(cachedData).toHaveLength(3); // 2 existing + 1 new
      expect(cachedData?.find(c => c.id === 'real-id')).toEqual(createdCollection);

      // Check success toast
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'success',
        message: `Collection "New Collection" created successfully`,
      });

      expect(mutationResult).toEqual(createdCollection);
    });

    it('should rollback on error', async () => {
      const createData: CreateCollectionRequest = {
        name: 'New Collection',
        description: 'New collection description',
      };

      const error = new Error('Creation failed');
      vi.mocked(collectionsV2Api.create).mockRejectedValue(error);

      // Pre-populate cache
      queryClient.setQueryData(collectionKeys.lists(), mockCollections);

      const { result } = renderHook(() => useCreateCollection(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        try {
          await result.current.mutateAsync(createData);
        } catch {
          // Expected to throw
        }
      });

      // Check cache was rolled back
      const cachedData = queryClient.getQueryData<Collection[]>(collectionKeys.lists());
      expect(cachedData).toEqual(mockCollections);

      // Check error toast
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'error',
        message: 'Creation failed',
      });
    });
  });

  describe('useUpdateCollection hook', () => {
    it('should update collection with optimistic update', async () => {
      const updates: UpdateCollectionRequest = {
        name: 'Updated Name',
        description: 'Updated description',
      };

      const updatedCollection = {
        ...mockCollections[0],
        ...updates,
        updated_at: '2024-01-03T00:00:00Z',
      };

      vi.mocked(collectionsV2Api.update).mockResolvedValue({} as MockAxiosResponse<Collection>);
      vi.mocked(collectionsV2Api.get).mockResolvedValue({
        data: updatedCollection,
      } as MockAxiosResponse<Collection>);

      // Pre-populate caches
      queryClient.setQueryData(collectionKeys.lists(), mockCollections);
      queryClient.setQueryData(collectionKeys.detail('1'), mockCollections[0]);

      const { result } = renderHook(() => useUpdateCollection(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        await result.current.mutateAsync({ id: '1', updates });
      });

      // Check both caches were updated
      const detailCache = queryClient.getQueryData<Collection>(collectionKeys.detail('1'));
      expect(detailCache?.name).toBe('Updated Name');

      const listCache = queryClient.getQueryData<Collection[]>(collectionKeys.lists());
      const updatedInList = listCache?.find(c => c.id === '1');
      expect(updatedInList?.name).toBe('Updated Name');

      // Check success toast
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'success',
        message: `Collection "Updated Name" updated successfully`,
      });
    });

    it('should rollback both caches on error', async () => {
      const updates: UpdateCollectionRequest = {
        name: 'Updated Name',
      };

      const error = new Error('Update failed');
      vi.mocked(collectionsV2Api.update).mockRejectedValue(error);

      // Pre-populate caches
      queryClient.setQueryData(collectionKeys.lists(), mockCollections);
      queryClient.setQueryData(collectionKeys.detail('1'), mockCollections[0]);

      const { result } = renderHook(() => useUpdateCollection(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        try {
          await result.current.mutateAsync({ id: '1', updates });
        } catch {
          // Expected to throw
        }
      });

      // Check caches were rolled back
      const detailCache = queryClient.getQueryData<Collection>(collectionKeys.detail('1'));
      expect(detailCache).toEqual(mockCollections[0]);

      const listCache = queryClient.getQueryData<Collection[]>(collectionKeys.lists());
      expect(listCache).toEqual(mockCollections);

      // Check error toast
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'error',
        message: 'Update failed',
      });
    });
  });

  describe('useDeleteCollection hook', () => {
    it('should delete collection with cascade cleanup', async () => {
      vi.mocked(collectionsV2Api.delete).mockResolvedValue({} as MockAxiosResponse<void>);

      // Pre-populate various caches
      queryClient.setQueryData(collectionKeys.lists(), mockCollections);
      queryClient.setQueryData(collectionKeys.detail('1'), mockCollections[0]);
      queryClient.setQueryData(['collection-operations', '1'], []);
      queryClient.setQueryData(['collection-documents', '1'], []);

      const { result } = renderHook(() => useDeleteCollection(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        await result.current.mutateAsync('1');
      });

      // Check collection was removed from list
      const listCache = queryClient.getQueryData<Collection[]>(collectionKeys.lists());
      expect(listCache).toHaveLength(1);
      expect(listCache?.find(c => c.id === '1')).toBeUndefined();

      // Check related queries were removed
      expect(queryClient.getQueryData(collectionKeys.detail('1'))).toBeUndefined();
      expect(queryClient.getQueryData(['collection-operations', '1'])).toBeUndefined();
      expect(queryClient.getQueryData(['collection-documents', '1'])).toBeUndefined();

      // Check success toast
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'success',
        message: 'Collection deleted successfully',
      });
    });

    it('should rollback on delete error', async () => {
      const error = new Error('Delete failed');
      vi.mocked(collectionsV2Api.delete).mockRejectedValue(error);

      // Pre-populate cache
      queryClient.setQueryData(collectionKeys.lists(), mockCollections);

      const { result } = renderHook(() => useDeleteCollection(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        try {
          await result.current.mutateAsync('1');
        } catch {
          // Expected to throw
        }
      });

      // Check cache was rolled back
      const listCache = queryClient.getQueryData<Collection[]>(collectionKeys.lists());
      expect(listCache).toEqual(mockCollections);

      // Check error toast
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'error',
        message: 'Delete failed',
      });
    });
  });

  describe('useUpdateCollectionInCache utility', () => {
    it('should update collection in both detail and list caches', () => {
      // Pre-populate caches
      queryClient.setQueryData(collectionKeys.lists(), mockCollections);
      queryClient.setQueryData(collectionKeys.detail('1'), mockCollections[0]);

      const { result } = renderHook(() => useUpdateCollectionInCache(), {
        wrapper: createWrapper(queryClient),
      });

      const updates = {
        status: 'processing' as const,
        document_count: 20,
      };

      act(() => {
        result.current('1', updates);
      });

      // Check both caches were updated
      const detailCache = queryClient.getQueryData<Collection>(collectionKeys.detail('1'));
      expect(detailCache?.status).toBe('processing');
      expect(detailCache?.document_count).toBe(20);

      const listCache = queryClient.getQueryData<Collection[]>(collectionKeys.lists());
      const updatedInList = listCache?.find(c => c.id === '1');
      expect(updatedInList?.status).toBe('processing');
      expect(updatedInList?.document_count).toBe(20);
    });

    it('should handle missing data gracefully', () => {
      const { result } = renderHook(() => useUpdateCollectionInCache(), {
        wrapper: createWrapper(queryClient),
      });

      // Should not throw when caches are empty
      expect(() => {
        result.current('non-existent', { status: 'ready' });
      }).not.toThrow();
    });
  });

  describe('cache invalidation patterns', () => {
    it('should invalidate queries after successful mutations', async () => {
      vi.mocked(collectionsV2Api.create).mockResolvedValue({
        data: { ...mockCollections[0], id: 'new-id' },
      } as MockAxiosResponse<Collection>);

      const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

      const { result } = renderHook(() => useCreateCollection(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        await result.current.mutateAsync({ name: 'Test' });
      });

      await waitFor(() => {
        expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: collectionKeys.lists() });
      });
    });
  });

  describe('loading states and transitions', () => {
    it('should properly transition through loading states', async () => {
      let resolvePromise: (value: MockAxiosResponse<{ collections: Collection[] }>) => void;
      const promise = new Promise((resolve) => {
        resolvePromise = resolve;
      });

      vi.mocked(collectionsV2Api.list).mockReturnValue(promise as Promise<MockAxiosResponse<{ collections: Collection[] }>>);

      const { result } = renderHook(() => useCollections(), {
        wrapper: createWrapper(queryClient),
      });

      // Initial loading state
      expect(result.current.isLoading).toBe(true);
      expect(result.current.isPending).toBe(true);
      expect(result.current.data).toBeUndefined();

      // Resolve the promise
      act(() => {
        resolvePromise!({ data: { collections: mockCollections } });
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      // Final success state
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
      expect(result.current.data).toEqual(mockCollections);
    });
  });
});