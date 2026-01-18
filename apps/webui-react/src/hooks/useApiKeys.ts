import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { AxiosError } from 'axios';
import { apiKeysApi } from '../services/api/v2/api-keys';
import { useUIStore } from '../stores/uiStore';
import { ApiErrorHandler } from '../utils/api-error-handler';
import type {
  ApiKeyResponse,
  ApiKeyCreate,
  ApiKeyCreateResponse,
} from '../types/api-key';

/**
 * Query key factory for API key queries
 */
export const apiKeyKeys = {
  all: ['api-keys'] as const,
  lists: () => [...apiKeyKeys.all, 'list'] as const,
  list: () => [...apiKeyKeys.lists()] as const,
  details: () => [...apiKeyKeys.all, 'detail'] as const,
  detail: (id: string) => [...apiKeyKeys.details(), id] as const,
};

/**
 * Hook to fetch all API keys for the current user
 */
export function useApiKeys() {
  return useQuery({
    queryKey: apiKeyKeys.list(),
    queryFn: async (): Promise<ApiKeyResponse[]> => {
      try {
        const response = await apiKeysApi.list();
        return response.data.api_keys;
      } catch (error) {
        console.error('Failed to fetch API keys:', error);
        throw error;
      }
    },
    staleTime: 30 * 1000, // Cache for 30 seconds
    gcTime: 5 * 60 * 1000, // Keep in cache for 5 minutes
  });
}

/**
 * Hook to create a new API key
 * Returns the full response including the api_key field (shown once)
 */
export function useCreateApiKey() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation<ApiKeyCreateResponse, Error, ApiKeyCreate>({
    mutationFn: async (data: ApiKeyCreate) => {
      const response = await apiKeysApi.create(data);
      return response.data;
    },
    onSuccess: () => {
      // Invalidate all list queries
      queryClient.invalidateQueries({ queryKey: apiKeyKeys.lists() });
      // Note: Don't show toast here - the caller will handle showing the key
    },
    onError: (error) => {
      // Check for specific error codes
      if (error instanceof AxiosError) {
        const status = error.response?.status;
        const detail = error.response?.data?.detail;

        if (status === 400 && typeof detail === 'string' && detail.includes('Maximum')) {
          addToast({ type: 'error', message: 'Maximum API keys limit reached' });
          return;
        }
        // Note: 409 (duplicate name) falls through to generic handler
        // Form components may additionally show field-level validation
      }

      const errorMessage = ApiErrorHandler.getMessage(error);
      addToast({ type: 'error', message: errorMessage });
    },
  });
}

/**
 * Hook to revoke or reactivate an API key
 * Uses optimistic updates with rollback on error
 */
export function useRevokeApiKey() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation<
    ApiKeyResponse,
    Error,
    { keyId: string; isActive: boolean; keyName: string },
    { previousKeys: ApiKeyResponse[] | undefined }
  >({
    mutationFn: async ({ keyId, isActive }) => {
      const response = await apiKeysApi.update(keyId, { is_active: isActive });
      return response.data;
    },
    onMutate: async ({ keyId, isActive }) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey: apiKeyKeys.lists() });

      // Snapshot previous value
      const previousKeys = queryClient.getQueryData<ApiKeyResponse[]>(
        apiKeyKeys.list()
      );

      // Optimistically update
      if (previousKeys) {
        queryClient.setQueryData<ApiKeyResponse[]>(
          apiKeyKeys.list(),
          previousKeys.map((k) =>
            k.id === keyId ? { ...k, is_active: isActive } : k
          )
        );
      }

      return { previousKeys };
    },
    onError: (error, _, context) => {
      // Rollback on error
      if (context?.previousKeys) {
        queryClient.setQueryData(apiKeyKeys.list(), context.previousKeys);
      }
      const errorMessage = ApiErrorHandler.getMessage(error);
      addToast({ type: 'error', message: errorMessage });
    },
    onSuccess: (_data, { keyName, isActive }) => {
      addToast({
        type: 'success',
        message: `API key "${keyName}" ${isActive ? 'reactivated' : 'revoked'}`,
      });
    },
    onSettled: () => {
      // Always refetch to ensure consistency
      queryClient.invalidateQueries({ queryKey: apiKeyKeys.lists() });
    },
  });
}
