/**
 * Sparse Index React Query Hooks
 *
 * Custom hooks for sparse indexing (BM25/SPLADE) management.
 * Uses React Query for server state with optimistic updates.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useEffect } from 'react';
import { sparseIndexApi, sparseIndexKeys } from '../services/api/v2/sparse-index';
import { handleApiError } from '../services/api/v2/collections';
import { collectionKeys } from './useCollections';
import { useUIStore } from '../stores/uiStore';
import type {
  SparseIndexStatus,
  EnableSparseIndexRequest,
  SparseReindexStatus,
} from '../types/sparse-index';

function normalizeSparseReindexStatus(status: string | null | undefined): SparseReindexStatus {
  if (!status) return 'pending';

  const lowered = status.toLowerCase();
  if (
    lowered === 'pending' ||
    lowered === 'processing' ||
    lowered === 'completed' ||
    lowered === 'failed'
  ) {
    return lowered as SparseReindexStatus;
  }

  switch (status.toUpperCase()) {
    case 'QUEUED':
    case 'RECEIVED':
    case 'PENDING':
      return 'pending';
    case 'STARTED':
    case 'PROGRESS':
    case 'RETRY':
      return 'processing';
    case 'SUCCESS':
      return 'completed';
    case 'FAILURE':
    case 'REVOKED':
      return 'failed';
    default:
      // Best-effort fallback: keep polling rather than stopping early.
      return 'processing';
  }
}

/**
 * Hook to fetch sparse index status for a collection
 */
export function useSparseIndexStatus(collectionUuid: string | undefined) {
  return useQuery({
    queryKey: sparseIndexKeys.status(collectionUuid || ''),
    queryFn: async () => {
      if (!collectionUuid) {
        return null;
      }
      const response = await sparseIndexApi.getStatus(collectionUuid);
      return response.data;
    },
    enabled: !!collectionUuid,
    staleTime: 30000, // 30 seconds
  });
}

/**
 * Hook to enable sparse indexing on a collection
 */
export function useEnableSparseIndex() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async ({
      collectionUuid,
      data,
    }: {
      collectionUuid: string;
      data: EnableSparseIndexRequest;
    }) => {
      const response = await sparseIndexApi.enable(collectionUuid, data);
      return response.data;
    },
    onMutate: async ({ collectionUuid }) => {
      // Cancel any outgoing refetches
      await queryClient.cancelQueries({
        queryKey: sparseIndexKeys.status(collectionUuid),
      });

      // Snapshot previous value for rollback
      const previousStatus = queryClient.getQueryData<SparseIndexStatus>(
        sparseIndexKeys.status(collectionUuid)
      );

      return { previousStatus, collectionUuid };
    },
    onSuccess: (data, { collectionUuid }) => {
      // Update cache with new status
      queryClient.setQueryData(sparseIndexKeys.status(collectionUuid), data);

      // Invalidate collection details to refresh sparse-related UI
      queryClient.invalidateQueries({
        queryKey: collectionKeys.detail(collectionUuid),
      });

      addToast({
        type: 'success',
        message: 'Sparse indexing enabled successfully',
      });
    },
    onError: (error: Error, { collectionUuid }, context) => {
      // Rollback on error
      if (context?.previousStatus) {
        queryClient.setQueryData(
          sparseIndexKeys.status(collectionUuid),
          context.previousStatus
        );
      }

      addToast({
        type: 'error',
        message: `Failed to enable sparse indexing: ${handleApiError(error)}`,
      });
    },
  });
}

/**
 * Hook to disable sparse indexing on a collection
 */
export function useDisableSparseIndex() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async (collectionUuid: string) => {
      await sparseIndexApi.disable(collectionUuid);
    },
    onMutate: async (collectionUuid) => {
      await queryClient.cancelQueries({
        queryKey: sparseIndexKeys.status(collectionUuid),
      });

      const previousStatus = queryClient.getQueryData<SparseIndexStatus>(
        sparseIndexKeys.status(collectionUuid)
      );

      // Optimistically update to disabled state
      queryClient.setQueryData<SparseIndexStatus>(
        sparseIndexKeys.status(collectionUuid),
        { enabled: false }
      );

      return { previousStatus, collectionUuid };
    },
    onSuccess: (_, collectionUuid) => {
      queryClient.invalidateQueries({
        queryKey: collectionKeys.detail(collectionUuid),
      });

      addToast({
        type: 'success',
        message: 'Sparse indexing disabled',
      });
    },
    onError: (error: Error, collectionUuid, context) => {
      if (context?.previousStatus) {
        queryClient.setQueryData(
          sparseIndexKeys.status(collectionUuid),
          context.previousStatus
        );
      }

      addToast({
        type: 'error',
        message: `Failed to disable sparse indexing: ${handleApiError(error)}`,
      });
    },
  });
}

/**
 * Hook to trigger a sparse reindex job
 */
export function useTriggerSparseReindex() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async (collectionUuid: string) => {
      const response = await sparseIndexApi.triggerReindex(collectionUuid);
      return {
        collectionUuid,
        ...response.data,
        status: normalizeSparseReindexStatus(response.data.status),
      };
    },
    onSuccess: ({ collectionUuid, job_id }) => {
      addToast({
        type: 'info',
        message: 'Sparse reindex job started',
      });

      // Start polling for progress
      queryClient.invalidateQueries({
        queryKey: sparseIndexKeys.reindexProgress(collectionUuid, job_id),
      });
    },
    onError: (error: Error) => {
      addToast({
        type: 'error',
        message: `Failed to start reindex: ${handleApiError(error)}`,
      });
    },
  });
}

/**
 * Hook to poll sparse reindex progress
 * Automatically stops polling when job completes or fails
 */
export function useSparseReindexProgress(
  collectionUuid: string | undefined,
  jobId: string | undefined
) {
  const queryClient = useQueryClient();

  const query = useQuery({
    queryKey: sparseIndexKeys.reindexProgress(collectionUuid || '', jobId || ''),
    queryFn: async () => {
      if (!collectionUuid || !jobId) {
        return null;
      }
      const response = await sparseIndexApi.getReindexProgress(
        collectionUuid,
        jobId
      );
      return {
        ...response.data,
        status: normalizeSparseReindexStatus(response.data.status),
      };
    },
    enabled: !!collectionUuid && !!jobId,
    // Poll every 2 seconds while job is in progress
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === 'pending' || status === 'processing') {
        return 2000;
      }
      return false; // Stop polling when done
    },
  });

  // Handle completion - invalidate sparse status to refresh document count
  useEffect(() => {
    if (!collectionUuid) return;
    if (query.data?.status !== 'completed') return;

    queryClient.invalidateQueries({
      queryKey: sparseIndexKeys.status(collectionUuid),
    });
  }, [collectionUuid, query.data?.status, queryClient]);

  return query;
}

/**
 * Hook that combines status and active reindex tracking
 * Useful for components that need both pieces of information
 */
export function useSparseIndexWithReindex(collectionUuid: string | undefined) {
  const statusQuery = useSparseIndexStatus(collectionUuid);
  const enableMutation = useEnableSparseIndex();
  const disableMutation = useDisableSparseIndex();
  const reindexMutation = useTriggerSparseReindex();

  return {
    // Status
    status: statusQuery.data,
    isLoading: statusQuery.isLoading,
    isError: statusQuery.isError,
    error: statusQuery.error,
    refetch: statusQuery.refetch,

    // Mutations
    enable: enableMutation.mutate,
    enableAsync: enableMutation.mutateAsync,
    isEnabling: enableMutation.isPending,

    disable: disableMutation.mutate,
    disableAsync: disableMutation.mutateAsync,
    isDisabling: disableMutation.isPending,

    triggerReindex: reindexMutation.mutate,
    triggerReindexAsync: reindexMutation.mutateAsync,
    isReindexing: reindexMutation.isPending,
    reindexJobId: reindexMutation.data?.job_id,
  };
}
