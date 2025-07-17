import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { 
  Collection, 
  Operation, 
  CreateCollectionRequest,
  UpdateCollectionRequest,
  AddSourceRequest,
  ReindexRequest,
  CollectionStatus
} from '../types/collection';
import { collectionsV2Api, handleApiError } from '../services/api/v2/collections';

interface CollectionStore {
  // State
  collections: Map<string, Collection>;
  operations: Map<string, Operation[]>; // collection_id -> operations
  activeOperations: Set<string>; // operation IDs currently in progress
  selectedCollectionId: string | null;
  isLoading: boolean;
  error: string | null;

  // Actions
  fetchCollections: () => Promise<void>;
  fetchCollectionById: (id: string) => Promise<Collection>;
  createCollection: (data: CreateCollectionRequest) => Promise<Collection>;
  updateCollection: (id: string, updates: UpdateCollectionRequest) => Promise<void>;
  deleteCollection: (id: string) => Promise<void>;
  
  // Operation actions
  addSource: (collectionId: string, sourcePath: string, config?: AddSourceRequest['config']) => Promise<Operation>;
  removeSource: (collectionId: string, sourcePath: string) => Promise<Operation>;
  reindexCollection: (collectionId: string, config?: ReindexRequest) => Promise<Operation>;
  fetchOperations: (collectionId: string) => Promise<void>;
  
  // Optimistic updates
  optimisticUpdateCollection: (id: string, updates: Partial<Collection>) => void;
  optimisticAddOperation: (collectionId: string, operation: Operation) => void;
  
  // WebSocket updates
  updateOperationProgress: (operationId: string, updates: Partial<Operation>) => void;
  updateCollectionStatus: (collectionId: string, status: CollectionStatus, message?: string) => void;
  
  // Selectors
  getCollectionById: (id: string) => Collection | undefined;
  getCollectionOperations: (collectionId: string) => Operation[];
  getActiveOperations: () => Operation[];
  getCollectionsArray: () => Collection[];
  
  // Utility actions
  setSelectedCollection: (id: string | null) => void;
  setError: (error: string | null) => void;
  clearStore: () => void;
}

export const useCollectionStore = create<CollectionStore>()(
  devtools(
    (set, get) => ({
      // Initial state
      collections: new Map(),
      operations: new Map(),
      activeOperations: new Set(),
      selectedCollectionId: null,
      isLoading: false,
      error: null,

      // Fetch all collections
      fetchCollections: async () => {
        set({ isLoading: true, error: null });
        try {
          const response = await collectionsV2Api.list();
          const collections = new Map(response.data.items.map(c => [c.id, c]));
          set({ collections, isLoading: false });
        } catch (error) {
          const errorMessage = handleApiError(error);
          set({ error: errorMessage, isLoading: false });
          throw error;
        }
      },

      // Fetch single collection
      fetchCollectionById: async (id: string) => {
        try {
          const response = await collectionsV2Api.get(id);
          const collection = response.data;
          set(state => ({
            collections: new Map(state.collections).set(id, collection)
          }));
          return collection;
        } catch (error) {
          const errorMessage = handleApiError(error);
          set({ error: errorMessage });
          throw error;
        }
      },

      // Create new collection
      createCollection: async (data: CreateCollectionRequest) => {
        try {
          // Optimistic update
          const tempId = `temp-${Date.now()}`;
          const tempCollection: Collection = {
            id: tempId,
            name: data.name,
            description: data.description,
            owner_id: 0, // Will be set by backend
            vector_store_name: '',
            embedding_model: data.embedding_model || 'Qwen/Qwen3-Embedding-0.6B',
            chunk_size: data.chunk_size || 1000,
            chunk_overlap: data.chunk_overlap || 200,
            is_public: data.is_public || false,
            status: 'pending',
            metadata: data.metadata,
            document_count: 0,
            vector_count: 0,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            isProcessing: true,
          };
          
          set(state => ({
            collections: new Map(state.collections).set(tempId, tempCollection)
          }));

          const response = await collectionsV2Api.create(data);
          const collection = response.data;
          
          // Remove temp and add real collection
          set(state => {
            const newCollections = new Map(state.collections);
            newCollections.delete(tempId);
            newCollections.set(collection.id, collection);
            return { collections: newCollections };
          });
          
          return collection;
        } catch (error) {
          const errorMessage = handleApiError(error);
          set({ error: errorMessage });
          throw error;
        }
      },

      // Update collection
      updateCollection: async (id: string, updates: UpdateCollectionRequest) => {
        // Optimistic update
        get().optimisticUpdateCollection(id, updates);
        
        try {
          await collectionsV2Api.update(id, updates);
          const response = await collectionsV2Api.get(id);
          const updated = response.data;
          set(state => ({
            collections: new Map(state.collections).set(id, updated)
          }));
        } catch (error) {
          // Revert optimistic update
          await get().fetchCollectionById(id);
          const errorMessage = handleApiError(error);
          set({ error: errorMessage });
          throw error;
        }
      },

      // Delete collection
      deleteCollection: async (id: string) => {
        // Optimistic update
        set(state => {
          const newCollections = new Map(state.collections);
          newCollections.delete(id);
          const newOperations = new Map(state.operations);
          newOperations.delete(id);
          return { 
            collections: newCollections, 
            operations: newOperations,
            selectedCollectionId: state.selectedCollectionId === id ? null : state.selectedCollectionId
          };
        });
        
        try {
          await collectionsV2Api.delete(id);
        } catch (error) {
          // Revert by re-fetching
          await get().fetchCollections();
          const errorMessage = handleApiError(error);
          set({ error: errorMessage });
          throw error;
        }
      },

      // Add source to collection
      addSource: async (collectionId: string, sourcePath: string, config?: AddSourceRequest['config']) => {
        try {
          // Optimistic update with temporary operation
          const tempOperation: Operation = {
            id: `temp-op-${Date.now()}`,
            collection_id: collectionId,
            type: 'append',
            status: 'pending',
            config: { source_path: sourcePath, ...config },
            created_at: new Date().toISOString(),
          };
          
          get().optimisticAddOperation(collectionId, tempOperation);
          get().optimisticUpdateCollection(collectionId, { 
            status: 'processing',
            isProcessing: true,
            activeOperation: tempOperation
          });
          
          const response = await collectionsV2Api.addSource(collectionId, { source_path: sourcePath, config });
          const operation = response.data;
          
          // Replace temp operation with real one
          set(state => {
            const ops = state.operations.get(collectionId) || [];
            const newOps = ops.map(op => op.id === tempOperation.id ? operation : op);
            const newOperations = new Map(state.operations).set(collectionId, newOps);
            
            // Update active operations
            const newActiveOps = new Set(state.activeOperations);
            newActiveOps.delete(tempOperation.id);
            newActiveOps.add(operation.id);
            
            return { operations: newOperations, activeOperations: newActiveOps };
          });
          
          return operation;
        } catch (error) {
          const errorMessage = handleApiError(error);
          set({ error: errorMessage });
          throw error;
        }
      },

      // Remove source from collection
      removeSource: async (collectionId: string, sourcePath: string) => {
        try {
          // Optimistic update with temporary operation
          const tempOperation: Operation = {
            id: `temp-op-${Date.now()}`,
            collection_id: collectionId,
            type: 'remove_source',
            status: 'pending',
            config: { source_path: sourcePath },
            created_at: new Date().toISOString(),
          };
          
          get().optimisticAddOperation(collectionId, tempOperation);
          get().optimisticUpdateCollection(collectionId, { 
            status: 'processing',
            isProcessing: true,
            activeOperation: tempOperation
          });
          
          const response = await collectionsV2Api.removeSource(collectionId, { source_path: sourcePath });
          const operation = response.data;
          
          // Replace temp operation with real one
          set(state => {
            const ops = state.operations.get(collectionId) || [];
            const newOps = ops.map(op => op.id === tempOperation.id ? operation : op);
            const newOperations = new Map(state.operations).set(collectionId, newOps);
            
            // Update active operations
            const newActiveOps = new Set(state.activeOperations);
            newActiveOps.delete(tempOperation.id);
            newActiveOps.add(operation.id);
            
            return { operations: newOperations, activeOperations: newActiveOps };
          });
          
          return operation;
        } catch (error) {
          const errorMessage = handleApiError(error);
          set({ error: errorMessage });
          throw error;
        }
      },

      // Reindex collection
      reindexCollection: async (collectionId: string, config?: ReindexRequest) => {
        try {
          // Optimistic update with temporary operation
          const tempOperation: Operation = {
            id: `temp-op-${Date.now()}`,
            collection_id: collectionId,
            type: 'reindex',
            status: 'pending',
            config: config || {},
            created_at: new Date().toISOString(),
          };
          
          get().optimisticAddOperation(collectionId, tempOperation);
          get().optimisticUpdateCollection(collectionId, { 
            status: 'processing',
            isProcessing: true,
            activeOperation: tempOperation
          });
          
          const response = await collectionsV2Api.reindex(collectionId, config);
          const operation = response.data;
          
          // Replace temp operation with real one
          set(state => {
            const ops = state.operations.get(collectionId) || [];
            const newOps = ops.map(op => op.id === tempOperation.id ? operation : op);
            const newOperations = new Map(state.operations).set(collectionId, newOps);
            
            // Update active operations
            const newActiveOps = new Set(state.activeOperations);
            newActiveOps.delete(tempOperation.id);
            newActiveOps.add(operation.id);
            
            return { operations: newOperations, activeOperations: newActiveOps };
          });
          
          return operation;
        } catch (error) {
          const errorMessage = handleApiError(error);
          set({ error: errorMessage });
          throw error;
        }
      },

      // Fetch operations for a collection
      fetchOperations: async (collectionId: string) => {
        try {
          const response = await collectionsV2Api.listOperations(collectionId);
          const operations = response.data.items;
          
          set(state => {
            const newOperations = new Map(state.operations).set(collectionId, operations);
            
            // Update active operations
            const newActiveOps = new Set(state.activeOperations);
            operations.forEach(op => {
              if (op.status === 'pending' || op.status === 'processing') {
                newActiveOps.add(op.id);
              } else {
                newActiveOps.delete(op.id);
              }
            });
            
            return { operations: newOperations, activeOperations: newActiveOps };
          });
        } catch (error) {
          const errorMessage = handleApiError(error);
          set({ error: errorMessage });
          throw error;
        }
      },

      // Optimistic updates
      optimisticUpdateCollection: (id: string, updates: Partial<Collection>) => {
        set(state => {
          const collection = state.collections.get(id);
          if (!collection) return state;
          
          const updatedCollection = { ...collection, ...updates };
          return {
            collections: new Map(state.collections).set(id, updatedCollection)
          };
        });
      },

      optimisticAddOperation: (collectionId: string, operation: Operation) => {
        set(state => {
          const ops = state.operations.get(collectionId) || [];
          const newOps = [operation, ...ops];
          
          const newActiveOps = new Set(state.activeOperations);
          if (operation.status === 'pending' || operation.status === 'processing') {
            newActiveOps.add(operation.id);
          }
          
          return {
            operations: new Map(state.operations).set(collectionId, newOps),
            activeOperations: newActiveOps
          };
        });
      },

      // WebSocket updates
      updateOperationProgress: (operationId: string, updates: Partial<Operation>) => {
        set(state => {
          const newOperations = new Map(state.operations);
          const newActiveOps = new Set(state.activeOperations);
          
          // Find and update the operation
          for (const [collectionId, ops] of newOperations) {
            const opIndex = ops.findIndex(op => op.id === operationId);
            if (opIndex !== -1) {
              const updatedOp = { ...ops[opIndex], ...updates };
              ops[opIndex] = updatedOp;
              
              // Update active operations set
              if (updatedOp.status === 'completed' || updatedOp.status === 'failed' || updatedOp.status === 'cancelled') {
                newActiveOps.delete(operationId);
                
                // Update collection status if this was the active operation
                const collection = state.collections.get(collectionId);
                if (collection?.activeOperation?.id === operationId) {
                  const updatedCollection = { 
                    ...collection, 
                    activeOperation: undefined,
                    isProcessing: false,
                    status: (updatedOp.status === 'completed' ? 'ready' : 'error') as CollectionStatus
                  };
                  state.collections.set(collectionId, updatedCollection);
                }
              }
              break;
            }
          }
          
          return { operations: newOperations, activeOperations: newActiveOps };
        });
      },

      updateCollectionStatus: (collectionId: string, status: CollectionStatus, message?: string) => {
        set(state => ({
          collections: new Map(state.collections).set(
            collectionId,
            {
              ...state.collections.get(collectionId)!,
              status,
              status_message: message
            }
          )
        }));
      },

      // Selectors
      getCollectionById: (id: string) => get().collections.get(id),
      
      getCollectionOperations: (collectionId: string) => {
        return get().operations.get(collectionId) || [];
      },
      
      getActiveOperations: () => {
        const state = get();
        const activeOps: Operation[] = [];
        
        for (const ops of state.operations.values()) {
          activeOps.push(...ops.filter(op => state.activeOperations.has(op.id)));
        }
        
        return activeOps;
      },
      
      getCollectionsArray: () => Array.from(get().collections.values()),

      // Utility actions
      setSelectedCollection: (id: string | null) => set({ selectedCollectionId: id }),
      
      setError: (error: string | null) => set({ error }),
      
      clearStore: () => set({
        collections: new Map(),
        operations: new Map(),
        activeOperations: new Set(),
        selectedCollectionId: null,
        isLoading: false,
        error: null,
      }),
    }),
    {
      name: 'collection-store',
    }
  )
);