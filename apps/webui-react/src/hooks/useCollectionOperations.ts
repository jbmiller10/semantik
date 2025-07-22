import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { collectionsV2Api, handleApiError } from '../services/api/v2/collections';
import { useUIStore } from '../stores/uiStore';
import { collectionKeys } from './useCollections';
import type { 
  Operation, 
  AddSourceRequest, 
  ReindexRequest,
  Collection 
} from '../types/collection';

// Query key factory for operations
export const operationKeys = {
  all: ['operations'] as const,
  lists: () => [...operationKeys.all, 'list'] as const,
  list: (collectionId: string) => [...operationKeys.lists(), collectionId] as const,
};

// Hook to fetch operations for a collection
export function useCollectionOperations(collectionId: string, options?: { limit?: number }) {
  return useQuery({
    queryKey: operationKeys.list(collectionId),
    queryFn: async () => {
      const response = await collectionsV2Api.listOperations(collectionId, options);
      return response.data;
    },
    enabled: !!collectionId,
    // Refetch more frequently if there are active operations
    refetchInterval: (query) => {
      const hasActiveOps = query.state.data?.some(
        (op: Operation) => op.status === 'pending' || op.status === 'processing'
      );
      return hasActiveOps ? 5000 : false; // 5 seconds for active operations
    },
  });
}

// Hook to add a source to a collection
export function useAddSource() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async ({ 
      collectionId, 
      sourcePath, 
      config 
    }: { 
      collectionId: string; 
      sourcePath: string; 
      config?: AddSourceRequest['config'];
    }) => {
      const response = await collectionsV2Api.addSource(collectionId, { 
        source_path: sourcePath, 
        config 
      });
      return { operation: response.data, collectionId };
    },
    onMutate: async ({ collectionId, sourcePath, config }) => {
      // Cancel any outgoing refetches
      await queryClient.cancelQueries({ queryKey: operationKeys.list(collectionId) });
      await queryClient.cancelQueries({ queryKey: collectionKeys.detail(collectionId) });

      // Snapshot the previous values
      const previousOperations = queryClient.getQueryData<Operation[]>(
        operationKeys.list(collectionId)
      );
      const previousCollection = queryClient.getQueryData<Collection>(
        collectionKeys.detail(collectionId)
      );

      // Create temporary operation
      const tempOperation: Operation = {
        id: `temp-op-${Date.now()}`,
        collection_id: collectionId,
        type: 'append',
        status: 'pending',
        config: { source_path: sourcePath, ...config },
        created_at: new Date().toISOString(),
      };

      // Optimistically update operations
      queryClient.setQueryData<Operation[]>(
        operationKeys.list(collectionId),
        old => [tempOperation, ...(old || [])]
      );

      // Optimistically update collection status
      if (previousCollection) {
        queryClient.setQueryData<Collection>(
          collectionKeys.detail(collectionId),
          { 
            ...previousCollection, 
            status: 'processing',
            isProcessing: true,
            activeOperation: tempOperation
          }
        );
      }

      return { previousOperations, previousCollection, tempOpId: tempOperation.id };
    },
    onError: (error, variables, context) => {
      // Rollback on error
      if (context?.previousOperations !== undefined) {
        queryClient.setQueryData(
          operationKeys.list(variables.collectionId),
          context.previousOperations
        );
      }
      if (context?.previousCollection) {
        queryClient.setQueryData(
          collectionKeys.detail(variables.collectionId),
          context.previousCollection
        );
      }
      
      const errorMessage = handleApiError(error);
      addToast({ type: 'error', message: errorMessage });
    },
    onSuccess: ({ operation, collectionId }, _variables, context) => {
      // Replace temp operation with real one
      queryClient.setQueryData<Operation[]>(
        operationKeys.list(collectionId),
        old => {
          if (!old) return [operation];
          return old.map(op => op.id === context?.tempOpId ? operation : op);
        }
      );
      
      addToast({
        type: 'success',
        message: 'Data source added, indexing started',
      });
    },
    onSettled: (data) => {
      // Always refetch after error or success
      if (data?.collectionId) {
        queryClient.invalidateQueries({ queryKey: operationKeys.list(data.collectionId) });
        queryClient.invalidateQueries({ queryKey: collectionKeys.detail(data.collectionId) });
        queryClient.invalidateQueries({ queryKey: collectionKeys.lists() });
      }
    },
  });
}

// Hook to remove a source from a collection
export function useRemoveSource() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async ({ 
      collectionId, 
      sourcePath 
    }: { 
      collectionId: string; 
      sourcePath: string; 
    }) => {
      const response = await collectionsV2Api.removeSource(collectionId, { 
        source_path: sourcePath 
      });
      return { operation: response.data, collectionId };
    },
    onMutate: async ({ collectionId, sourcePath }) => {
      // Similar optimistic update pattern as addSource
      await queryClient.cancelQueries({ queryKey: operationKeys.list(collectionId) });
      await queryClient.cancelQueries({ queryKey: collectionKeys.detail(collectionId) });

      const previousOperations = queryClient.getQueryData<Operation[]>(
        operationKeys.list(collectionId)
      );
      const previousCollection = queryClient.getQueryData<Collection>(
        collectionKeys.detail(collectionId)
      );

      const tempOperation: Operation = {
        id: `temp-op-${Date.now()}`,
        collection_id: collectionId,
        type: 'remove_source',
        status: 'pending',
        config: { source_path: sourcePath },
        created_at: new Date().toISOString(),
      };

      queryClient.setQueryData<Operation[]>(
        operationKeys.list(collectionId),
        old => [tempOperation, ...(old || [])]
      );

      if (previousCollection) {
        queryClient.setQueryData<Collection>(
          collectionKeys.detail(collectionId),
          { 
            ...previousCollection, 
            status: 'processing',
            isProcessing: true,
            activeOperation: tempOperation
          }
        );
      }

      return { previousOperations, previousCollection, tempOpId: tempOperation.id };
    },
    onError: (error, variables, context) => {
      if (context?.previousOperations !== undefined) {
        queryClient.setQueryData(
          operationKeys.list(variables.collectionId),
          context.previousOperations
        );
      }
      if (context?.previousCollection) {
        queryClient.setQueryData(
          collectionKeys.detail(variables.collectionId),
          context.previousCollection
        );
      }
      
      const errorMessage = handleApiError(error);
      addToast({ type: 'error', message: errorMessage });
    },
    onSuccess: ({ operation, collectionId }, _variables, context) => {
      queryClient.setQueryData<Operation[]>(
        operationKeys.list(collectionId),
        old => {
          if (!old) return [operation];
          return old.map(op => op.id === context?.tempOpId ? operation : op);
        }
      );
      
      addToast({
        type: 'success',
        message: 'Source removed successfully',
      });
    },
    onSettled: (data) => {
      if (data?.collectionId) {
        queryClient.invalidateQueries({ queryKey: operationKeys.list(data.collectionId) });
        queryClient.invalidateQueries({ queryKey: collectionKeys.detail(data.collectionId) });
        queryClient.invalidateQueries({ queryKey: collectionKeys.lists() });
      }
    },
  });
}

// Hook to reindex a collection
export function useReindexCollection() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async ({ 
      collectionId, 
      config 
    }: { 
      collectionId: string; 
      config?: ReindexRequest;
    }) => {
      const response = await collectionsV2Api.reindex(collectionId, config);
      return { operation: response.data, collectionId };
    },
    onMutate: async ({ collectionId, config }) => {
      await queryClient.cancelQueries({ queryKey: operationKeys.list(collectionId) });
      await queryClient.cancelQueries({ queryKey: collectionKeys.detail(collectionId) });

      const previousOperations = queryClient.getQueryData<Operation[]>(
        operationKeys.list(collectionId)
      );
      const previousCollection = queryClient.getQueryData<Collection>(
        collectionKeys.detail(collectionId)
      );

      const tempOperation: Operation = {
        id: `temp-op-${Date.now()}`,
        collection_id: collectionId,
        type: 'reindex',
        status: 'pending',
        config: config || {},
        created_at: new Date().toISOString(),
      };

      queryClient.setQueryData<Operation[]>(
        operationKeys.list(collectionId),
        old => [tempOperation, ...(old || [])]
      );

      if (previousCollection) {
        queryClient.setQueryData<Collection>(
          collectionKeys.detail(collectionId),
          { 
            ...previousCollection, 
            status: 'processing',
            isProcessing: true,
            activeOperation: tempOperation
          }
        );
      }

      return { previousOperations, previousCollection, tempOpId: tempOperation.id };
    },
    onError: (error, variables, context) => {
      if (context?.previousOperations !== undefined) {
        queryClient.setQueryData(
          operationKeys.list(variables.collectionId),
          context.previousOperations
        );
      }
      if (context?.previousCollection) {
        queryClient.setQueryData(
          collectionKeys.detail(variables.collectionId),
          context.previousCollection
        );
      }
      
      const errorMessage = handleApiError(error);
      addToast({ type: 'error', message: errorMessage });
    },
    onSuccess: ({ operation, collectionId }, _variables, context) => {
      queryClient.setQueryData<Operation[]>(
        operationKeys.list(collectionId),
        old => {
          if (!old) return [operation];
          return old.map(op => op.id === context?.tempOpId ? operation : op);
        }
      );
      
      const collectionName = queryClient.getQueryData<Collection>(
        collectionKeys.detail(collectionId)
      )?.name || 'collection';
      
      addToast({
        type: 'success',
        message: `Re-indexing started for "${collectionName}". You can track progress on the collection page.`,
      });
    },
    onSettled: (data) => {
      if (data?.collectionId) {
        queryClient.invalidateQueries({ queryKey: operationKeys.list(data.collectionId) });
        queryClient.invalidateQueries({ queryKey: collectionKeys.detail(data.collectionId) });
        queryClient.invalidateQueries({ queryKey: collectionKeys.lists() });
      }
    },
  });
}

// Utility function to update operation in cache (useful for WebSocket updates)
export function useUpdateOperationInCache() {
  const queryClient = useQueryClient();
  
  return (operationId: string, updates: Partial<Operation>) => {
    // Find which collection this operation belongs to
    const cache = queryClient.getQueryCache();
    const queries = cache.findAll({ queryKey: operationKeys.lists() });
    
    queries.forEach(query => {
      const collectionId = query.queryKey[2] as string;
      if (collectionId) {
        queryClient.setQueryData<Operation[]>(
          operationKeys.list(collectionId),
          old => {
            if (!old) return old;
            const hasOperation = old.some(op => op.id === operationId);
            if (!hasOperation) return old;
            
            return old.map(op => 
              op.id === operationId ? { ...op, ...updates } : op
            );
          }
        );
        
        // If operation is completed/failed, update collection status
        if (updates.status === 'completed' || updates.status === 'failed' || updates.status === 'cancelled') {
          queryClient.setQueryData<Collection>(
            collectionKeys.detail(collectionId),
            old => {
              if (!old || old.activeOperation?.id !== operationId) return old;
              
              return {
                ...old,
                activeOperation: undefined,
                isProcessing: false,
                status: updates.status === 'completed' ? 'ready' : 'error'
              };
            }
          );
        }
      }
    });
  };
}