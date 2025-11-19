import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { projectionsV2Api } from '../services/api/v2/projections';
import { handleApiError } from '../services/api/v2/collections';
import { useUIStore } from '../stores/uiStore';
import type {
  ProjectionMetadata,
  ProjectionStatus,
  StartProjectionRequest,
} from '../types/projection';

const DEFAULT_REFETCH_INTERVAL = 5000;

export const projectionKeys = {
  root: ['projections'] as const,
  lists: (collectionId: string) => [...projectionKeys.root, collectionId, 'list'] as const,
  detail: (collectionId: string, projectionId: string) =>
    [...projectionKeys.root, collectionId, 'detail', projectionId] as const,
};

export function useCollectionProjections(collectionId: string | null) {
  const safeId = collectionId ?? '__none__';

  return useQuery<ProjectionMetadata[]>({
    queryKey: projectionKeys.lists(safeId),
    queryFn: async () => {
      if (!collectionId) return [];
      const response = await projectionsV2Api.list(collectionId);
      return response.data?.projections ?? [];
    },
    enabled: !!collectionId,
    initialData: [],
    refetchInterval: (query) => {
      const projections = query.state.data as ProjectionMetadata[] | undefined;
      const hasRunning = projections?.some((projection) =>
        ['pending', 'running'].includes(projection.status as ProjectionStatus)
      );
      return hasRunning ? DEFAULT_REFETCH_INTERVAL : false;
    },
  });
}

export function useProjectionMetadata(collectionId: string | null, projectionId: string | null) {
  const safeCollectionId = collectionId ?? '__none__';
  const safeProjectionId = projectionId ?? '__none__';

  return useQuery<ProjectionMetadata | null>({
    queryKey: projectionKeys.detail(safeCollectionId, safeProjectionId),
    queryFn: async () => {
      if (!collectionId || !projectionId) return null;
      const response = await projectionsV2Api.getMetadata(collectionId, projectionId);
      return response.data;
    },
    enabled: !!collectionId && !!projectionId,
  });
}

export function useStartProjection(collectionId: string | null) {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async (payload: StartProjectionRequest) => {
      if (!collectionId) throw new Error('Collection ID is required');
      const response = await projectionsV2Api.start(collectionId, payload);
      return response.data;
    },
    onMutate: async () => {
      if (!collectionId) return undefined;
      await queryClient.cancelQueries({ queryKey: projectionKeys.lists(collectionId) });
      const previous = queryClient.getQueryData<ProjectionMetadata[]>(
        projectionKeys.lists(collectionId)
      );
      return { previous };
    },
    onError: (error, _variables, context) => {
      const message = handleApiError(error);
      addToast({ type: 'error', message });
      const collectionIdSafe = collectionId;
      if (collectionIdSafe && context?.previous) {
        queryClient.setQueryData(projectionKeys.lists(collectionIdSafe), context.previous);
      }
    },
    onSuccess: (data) => {
      if (!collectionId) return;
      const isReuse = data?.idempotent_reuse === true;
      addToast({
        type: 'success',
        message: isReuse ? 'Reused latest projection' : 'Projection started successfully',
      });
      queryClient.setQueryData<ProjectionMetadata[]>(
        projectionKeys.lists(collectionId),
        (prev = []) => {
          if (!data) return prev;
          const filtered = prev.filter((projection) => projection.id !== data.id);
          return [data, ...filtered];
        }
      );
    },
    onSettled: () => {
      if (!collectionId) return;
      queryClient.invalidateQueries({ queryKey: projectionKeys.lists(collectionId) });
    },
  });
}

export function useDeleteProjection(collectionId: string | null) {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async (projectionId: string) => {
      if (!collectionId) throw new Error('Collection ID is required');
      await projectionsV2Api.delete(collectionId, projectionId);
      return projectionId;
    },
    onMutate: async (projectionId) => {
      if (!collectionId) return undefined;
      await queryClient.cancelQueries({ queryKey: projectionKeys.lists(collectionId) });
      const previous = queryClient.getQueryData<ProjectionMetadata[]>(
        projectionKeys.lists(collectionId)
      );
      queryClient.setQueryData<ProjectionMetadata[]>(
        projectionKeys.lists(collectionId),
        (prev = []) => prev.filter((projection) => projection.id !== projectionId)
      );
      return { previous };
    },
    onError: (error, _projectionId, context) => {
      const message = handleApiError(error);
      addToast({ type: 'error', message });
      if (!collectionId) return;
      if (context?.previous) {
        queryClient.setQueryData(
          projectionKeys.lists(collectionId),
          context.previous
        );
      }
    },
    onSuccess: () => {
      addToast({ type: 'success', message: 'Projection deleted successfully' });
    },
    onSettled: () => {
      if (!collectionId) return;
      queryClient.invalidateQueries({ queryKey: projectionKeys.lists(collectionId) });
    },
  });
}

export function useUpdateProjectionInCache() {
  const queryClient = useQueryClient();

  return (collectionId: string, updated: ProjectionMetadata) => {
    queryClient.setQueryData<ProjectionMetadata[]>(
      projectionKeys.lists(collectionId),
      (prev = []) => prev.map((projection) =>
        projection.id === updated.id ? { ...projection, ...updated } : projection
      )
    );
    queryClient.setQueryData(
      projectionKeys.detail(collectionId, updated.id),
      updated
    );
  };
}
