import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { collectionsV2Api, handleApiError } from '../services/api/v2/collections';
import { useUIStore } from '../stores/uiStore';
import type { 
  Collection, 
  CreateCollectionRequest, 
  UpdateCollectionRequest 
} from '../types/collection';

// Query key factory for consistent key generation
export const collectionKeys = {
  all: ['collections'] as const,
  lists: () => [...collectionKeys.all, 'list'] as const,
  list: (filters?: unknown) => [...collectionKeys.lists(), filters] as const,
  details: () => [...collectionKeys.all, 'detail'] as const,
  detail: (id: string) => [...collectionKeys.details(), id] as const,
};

// Hook to fetch all collections
export function useCollections() {
  const isTestEnv = import.meta.env.MODE === 'test'

  return useQuery({
    queryKey: collectionKeys.lists(),
    queryFn: async () => {
      const response = await collectionsV2Api.list();
      if (!response.data || !response.data.collections) {
        throw new Error('Invalid response structure');
      }
      return response.data.collections;
    },
    // Automatically refetch every 30 seconds if there are active operations
    refetchInterval: isTestEnv
      ? false
      : (query) => {
          const hasActiveOperations = query.state.data?.some(
            (c: Collection) => c.status === 'processing' || c.activeOperation
          );
          return hasActiveOperations ? 30000 : false;
        },
    staleTime: 5000, // Consider data stale after 5 seconds
  });
}

// Hook to fetch a single collection
export function useCollection(id: string) {
  return useQuery({
    queryKey: collectionKeys.detail(id),
    queryFn: async () => {
      const response = await collectionsV2Api.get(id);
      return response.data;
    },
    enabled: !!id,
  });
}

// Hook to create a new collection
export function useCreateCollection() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async (data: CreateCollectionRequest) => {
      const response = await collectionsV2Api.create(data);
      return response.data;
    },
    onMutate: async (newCollection) => {
      // Cancel any outgoing refetches
      await queryClient.cancelQueries({ queryKey: collectionKeys.lists() });

      // Snapshot the previous value
      const previousCollections = queryClient.getQueryData<Collection[]>(
        collectionKeys.lists()
      );

      // Optimistically update to the new value
      const tempCollection: Collection = {
        id: `temp-${Date.now()}`,
        name: newCollection.name,
        description: newCollection.description,
        owner_id: 0, // Will be set by backend
        vector_store_name: '',
        embedding_model: newCollection.embedding_model || 'Qwen/Qwen3-Embedding-0.6B',
        quantization: newCollection.quantization || 'float16',
        chunk_size: newCollection.chunk_size || 1000,
        chunk_overlap: newCollection.chunk_overlap || 200,
        is_public: newCollection.is_public || false,
        status: 'pending',
        metadata: newCollection.metadata,
        document_count: 0,
        vector_count: 0,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        isProcessing: true,
      };

      queryClient.setQueryData<Collection[]>(
        collectionKeys.lists(),
        old => [...(old || []), tempCollection]
      );

      // Return a context object with the previous value
      return { previousCollections, tempId: tempCollection.id };
    },
    onError: (error, _newCollection, context) => {
      // If the mutation fails, rollback to the previous value
      queryClient.setQueryData(
        collectionKeys.lists(),
        context?.previousCollections
      );
      
      const errorMessage = handleApiError(error);
      addToast({ type: 'error', message: errorMessage });
    },
    onSuccess: (data, _variables, context) => {
      // Remove the temporary collection and add the real one
      queryClient.setQueryData<Collection[]>(
        collectionKeys.lists(),
        old => {
          if (!old) return [data];
          return old.map(c => c.id === context?.tempId ? data : c);
        }
      );
      
      addToast({
        type: 'success',
        message: `Collection "${data.name}" created successfully`,
      });
    },
    onSettled: () => {
      // Always refetch after error or success
      queryClient.invalidateQueries({ queryKey: collectionKeys.lists() });
    },
  });
}

// Hook to update a collection
export function useUpdateCollection() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async ({ 
      id, 
      updates 
    }: { 
      id: string; 
      updates: UpdateCollectionRequest 
    }) => {
      await collectionsV2Api.update(id, updates);
      // Fetch the updated collection
      const response = await collectionsV2Api.get(id);
      return response.data;
    },
    onMutate: async ({ id, updates }) => {
      // Cancel any outgoing refetches
      await queryClient.cancelQueries({ queryKey: collectionKeys.detail(id) });
      await queryClient.cancelQueries({ queryKey: collectionKeys.lists() });

      // Snapshot the previous values
      const previousCollection = queryClient.getQueryData<Collection>(
        collectionKeys.detail(id)
      );
      const previousCollections = queryClient.getQueryData<Collection[]>(
        collectionKeys.lists()
      );

      // Optimistically update the collection
      const updatedCollection = previousCollection 
        ? { ...previousCollection, ...updates }
        : null;

      if (updatedCollection) {
        queryClient.setQueryData(collectionKeys.detail(id), updatedCollection);
        queryClient.setQueryData<Collection[]>(
          collectionKeys.lists(),
          old => old?.map(c => c.id === id ? updatedCollection : c) || []
        );
      }

      return { previousCollection, previousCollections };
    },
    onError: (error, { id }, context) => {
      // Rollback on error
      if (context?.previousCollection) {
        queryClient.setQueryData(
          collectionKeys.detail(id), 
          context.previousCollection
        );
      }
      if (context?.previousCollections) {
        queryClient.setQueryData(
          collectionKeys.lists(), 
          context.previousCollections
        );
      }
      
      const errorMessage = handleApiError(error);
      addToast({ type: 'error', message: errorMessage });
    },
    onSuccess: (data) => {
      addToast({
        type: 'success',
        message: `Collection "${data.name}" updated successfully`,
      });
    },
    onSettled: (_data, _error, { id }) => {
      // Always refetch after error or success
      queryClient.invalidateQueries({ queryKey: collectionKeys.detail(id) });
      queryClient.invalidateQueries({ queryKey: collectionKeys.lists() });
    },
  });
}

// Hook to delete a collection
export function useDeleteCollection() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async (id: string) => {
      await collectionsV2Api.delete(id);
      return id;
    },
    onMutate: async (id) => {
      // Cancel any outgoing refetches
      await queryClient.cancelQueries({ queryKey: collectionKeys.lists() });

      // Snapshot the previous value
      const previousCollections = queryClient.getQueryData<Collection[]>(
        collectionKeys.lists()
      );

      // Optimistically remove the collection
      queryClient.setQueryData<Collection[]>(
        collectionKeys.lists(),
        old => old?.filter(c => c.id !== id) || []
      );

      // Remove from cache
      queryClient.removeQueries({ queryKey: collectionKeys.detail(id) });

      return { previousCollections };
    },
    onError: (error, _id, context) => {
      // Rollback on error
      if (context?.previousCollections) {
        queryClient.setQueryData(
          collectionKeys.lists(), 
          context.previousCollections
        );
      }
      
      const errorMessage = handleApiError(error);
      addToast({ type: 'error', message: errorMessage });
    },
    onSuccess: (id) => {
      // Remove any related queries
      queryClient.removeQueries({ queryKey: ['collection-operations', id] });
      queryClient.removeQueries({ queryKey: ['collection-documents', id] });
      
      addToast({
        type: 'success',
        message: 'Collection deleted successfully',
      });
    },
    onSettled: () => {
      // Always refetch after error or success
      queryClient.invalidateQueries({ queryKey: collectionKeys.lists() });
    },
  });
}

// Utility function to update collection in cache (useful for WebSocket updates)
export function useUpdateCollectionInCache() {
  const queryClient = useQueryClient();
  
  return (id: string, updates: Partial<Collection>) => {
    // Update in the detail query
    queryClient.setQueryData<Collection>(
      collectionKeys.detail(id),
      old => old ? { ...old, ...updates } : old
    );
    
    // Update in the list query
    queryClient.setQueryData<Collection[]>(
      collectionKeys.lists(),
      old => old?.map(c => c.id === id ? { ...c, ...updates } : c) || []
    );
  };
}
