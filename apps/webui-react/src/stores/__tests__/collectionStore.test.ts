import { renderHook, act } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useCollectionStore } from '../collectionStore';
import { collectionsV2Api } from '../../services/api/v2/collections';
import type { Collection, Operation } from '../../types/collection';

// Mock the API module
vi.mock('../../services/api/v2/collections', () => ({
  collectionsV2Api: {
    list: vi.fn(),
    get: vi.fn(),
    create: vi.fn(),
    update: vi.fn(),
    delete: vi.fn(),
    addSource: vi.fn(),
    removeSource: vi.fn(),
    reindex: vi.fn(),
    listOperations: vi.fn(),
  },
  handleApiError: vi.fn((error) => error.message || 'API Error'),
}));

describe('collectionStore', () => {
  beforeEach(() => {
    // Clear all mocks before each test
    vi.clearAllMocks();
    
    // Reset store state
    const { result } = renderHook(() => useCollectionStore());
    act(() => {
      result.current.clearStore();
    });
  });

  describe('fetchCollections', () => {
    it('should fetch and store collections', async () => {
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
        },
      ];

      vi.mocked(collectionsV2Api.list).mockResolvedValue({
        data: {
          items: mockCollections,
          total: 2,
          page: 1,
          limit: 10,
        },
      } as any);

      const { result } = renderHook(() => useCollectionStore());

      await act(async () => {
        await result.current.fetchCollections();
      });

      expect(result.current.isLoading).toBe(false);
      expect(result.current.error).toBe(null);
      expect(result.current.getCollectionsArray()).toHaveLength(2);
      expect(result.current.getCollectionById('1')).toEqual(mockCollections[0]);
      expect(result.current.getCollectionById('2')).toEqual(mockCollections[1]);
    });

    it('should handle fetch errors', async () => {
      const error = new Error('Network error');
      vi.mocked(collectionsV2Api.list).mockRejectedValue(error);

      const { result } = renderHook(() => useCollectionStore());

      try {
        await act(async () => {
          await result.current.fetchCollections();
        });
      } catch (e) {
        // Expected to throw
      }

      expect(result.current.isLoading).toBe(false);
      expect(result.current.error).toBe('Network error');
    });
  });

  describe('createCollection', () => {
    it('should create a collection with optimistic updates', async () => {
      const createData = {
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
      } as any);

      const { result } = renderHook(() => useCollectionStore());

      let returnedCollection: Collection;
      await act(async () => {
        returnedCollection = await result.current.createCollection(createData);
      });

      // Check that the real collection replaced the temporary one
      expect(result.current.getCollectionsArray()).toHaveLength(1);
      expect(result.current.getCollectionById('real-id')).toEqual(createdCollection);
      expect(returnedCollection!).toEqual(createdCollection);
    });
  });

  describe('updateCollection', () => {
    it('should update a collection optimistically', async () => {
      const existingCollection: Collection = {
        id: '1',
        name: 'Original Name',
        description: 'Original description',
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
      };

      const updates = {
        name: 'Updated Name',
        description: 'Updated description',
      };

      const updatedCollection = {
        ...existingCollection,
        ...updates,
        updated_at: '2024-01-03T00:00:00Z',
      };

      vi.mocked(collectionsV2Api.update).mockResolvedValue({} as any);
      vi.mocked(collectionsV2Api.get).mockResolvedValue({
        data: updatedCollection,
      } as any);

      const { result } = renderHook(() => useCollectionStore());

      // Set initial collection
      act(() => {
        result.current.collections.set('1', existingCollection);
      });

      await act(async () => {
        await result.current.updateCollection('1', updates);
      });

      expect(result.current.getCollectionById('1')).toEqual(updatedCollection);
    });
  });

  describe('operations', () => {
    it('should add source with optimistic updates', async () => {
      const collectionId = '1';
      const sourcePath = '/path/to/source';
      
      const realOperation: Operation = {
        id: 'real-op-id',
        collection_id: collectionId,
        type: 'append',
        status: 'pending',
        config: { source_path: sourcePath },
        created_at: '2024-01-03T00:00:00Z',
      };

      vi.mocked(collectionsV2Api.addSource).mockResolvedValue({
        data: realOperation,
      } as any);

      const { result } = renderHook(() => useCollectionStore());

      let returnedOperation: Operation;
      await act(async () => {
        returnedOperation = await result.current.addSource(collectionId, sourcePath);
      });

      // Check that operation was added
      const operations = result.current.getCollectionOperations(collectionId);
      expect(operations).toHaveLength(1);
      expect(operations[0]).toEqual(realOperation);
      expect(result.current.activeOperations.has('real-op-id')).toBe(true);
      expect(returnedOperation!).toEqual(realOperation);
    });

    it('should update operation progress', () => {
      const { result } = renderHook(() => useCollectionStore());

      const operation: Operation = {
        id: 'op-1',
        collection_id: 'col-1',
        type: 'index',
        status: 'processing',
        config: {},
        created_at: '2024-01-01T00:00:00Z',
        progress: 50,
      };

      // Add initial operation
      act(() => {
        result.current.optimisticAddOperation('col-1', operation);
      });

      // Update progress
      act(() => {
        result.current.updateOperationProgress('op-1', {
          progress: 75,
          status: 'processing',
        });
      });

      const operations = result.current.getCollectionOperations('col-1');
      expect(operations[0].progress).toBe(75);

      // Complete operation
      act(() => {
        result.current.updateOperationProgress('op-1', {
          status: 'completed',
          completed_at: '2024-01-01T01:00:00Z',
        });
      });

      expect(result.current.activeOperations.has('op-1')).toBe(false);
    });
  });

  describe('selectors', () => {
    it('should get active operations across all collections', () => {
      const { result } = renderHook(() => useCollectionStore());

      const operations: Operation[] = [
        {
          id: 'op-1',
          collection_id: 'col-1',
          type: 'index',
          status: 'processing',
          config: {},
          created_at: '2024-01-01T00:00:00Z',
        },
        {
          id: 'op-2',
          collection_id: 'col-1',
          type: 'append',
          status: 'completed',
          config: {},
          created_at: '2024-01-01T00:00:00Z',
        },
        {
          id: 'op-3',
          collection_id: 'col-2',
          type: 'reindex',
          status: 'pending',
          config: {},
          created_at: '2024-01-01T00:00:00Z',
        },
      ];

      act(() => {
        result.current.operations.set('col-1', [operations[0], operations[1]]);
        result.current.operations.set('col-2', [operations[2]]);
        result.current.activeOperations.add('op-1');
        result.current.activeOperations.add('op-3');
      });

      const activeOps = result.current.getActiveOperations();
      expect(activeOps).toHaveLength(2);
      expect(activeOps).toContainEqual(operations[0]);
      expect(activeOps).toContainEqual(operations[2]);
    });
  });

  describe('utility actions', () => {
    it('should clear the store', () => {
      const { result } = renderHook(() => useCollectionStore());

      // Add some data
      act(() => {
        result.current.collections.set('1', {} as Collection);
        result.current.operations.set('1', []);
        result.current.activeOperations.add('op-1');
        result.current.selectedCollectionId = '1';
        result.current.error = 'Some error';
      });

      // Clear store
      act(() => {
        result.current.clearStore();
      });

      expect(result.current.collections.size).toBe(0);
      expect(result.current.operations.size).toBe(0);
      expect(result.current.activeOperations.size).toBe(0);
      expect(result.current.selectedCollectionId).toBe(null);
      expect(result.current.error).toBe(null);
    });
  });
});