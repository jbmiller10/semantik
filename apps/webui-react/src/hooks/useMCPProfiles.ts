import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { mcpProfilesApi } from '../services/api/v2/mcp-profiles';
import { useUIStore } from '../stores/uiStore';
import { handleApiError } from '../services/api/v2/collections';
import type {
  MCPProfile,
  MCPProfileCreate,
  MCPProfileUpdate,
  MCPClientConfig,
} from '../types/mcp-profile';

/**
 * Query key factory for MCP profile queries
 */
export const mcpProfileKeys = {
  all: ['mcp-profiles'] as const,
  lists: () => [...mcpProfileKeys.all, 'list'] as const,
  list: (enabled?: boolean) => [...mcpProfileKeys.lists(), { enabled }] as const,
  details: () => [...mcpProfileKeys.all, 'detail'] as const,
  detail: (id: string) => [...mcpProfileKeys.details(), id] as const,
  configs: () => [...mcpProfileKeys.all, 'config'] as const,
  config: (id: string) => [...mcpProfileKeys.configs(), id] as const,
};

/**
 * Hook to fetch all MCP profiles for the current user
 * @param enabled Optional filter by enabled state
 */
export function useMCPProfiles(enabled?: boolean) {
  return useQuery({
    queryKey: mcpProfileKeys.list(enabled),
    queryFn: async (): Promise<MCPProfile[]> => {
      const response = await mcpProfilesApi.list(enabled);
      return response.data.profiles;
    },
    staleTime: 30 * 1000, // Cache for 30 seconds
    gcTime: 5 * 60 * 1000, // Keep in cache for 5 minutes
  });
}

/**
 * Hook to fetch a specific MCP profile
 * @param profileId The profile UUID
 */
export function useMCPProfile(profileId: string) {
  return useQuery({
    queryKey: mcpProfileKeys.detail(profileId),
    queryFn: async (): Promise<MCPProfile> => {
      const response = await mcpProfilesApi.get(profileId);
      return response.data;
    },
    enabled: !!profileId,
    staleTime: 30 * 1000,
  });
}

/**
 * Hook to fetch the MCP client configuration for a profile
 * @param profileId The profile UUID
 */
export function useMCPProfileConfig(profileId: string) {
  return useQuery({
    queryKey: mcpProfileKeys.config(profileId),
    queryFn: async (): Promise<MCPClientConfig> => {
      const response = await mcpProfilesApi.getConfig(profileId);
      return response.data;
    },
    enabled: !!profileId,
    staleTime: 5 * 60 * 1000, // Config doesn't change often
  });
}

/**
 * Hook to create a new MCP profile
 */
export function useCreateMCPProfile() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation<MCPProfile, Error, MCPProfileCreate>({
    mutationFn: async (data: MCPProfileCreate) => {
      const response = await mcpProfilesApi.create(data);
      return response.data;
    },
    onSuccess: (data) => {
      // Invalidate all list queries
      queryClient.invalidateQueries({ queryKey: mcpProfileKeys.lists() });
      addToast({
        type: 'success',
        message: `Profile "${data.name}" created successfully`,
      });
    },
    onError: (error) => {
      const errorMessage = handleApiError(error);
      addToast({ type: 'error', message: errorMessage });
    },
  });
}

/**
 * Hook to update an existing MCP profile
 */
export function useUpdateMCPProfile() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation<
    MCPProfile,
    Error,
    { profileId: string; data: MCPProfileUpdate }
  >({
    mutationFn: async ({ profileId, data }) => {
      const response = await mcpProfilesApi.update(profileId, data);
      return response.data;
    },
    onSuccess: (data, { profileId }) => {
      // Update the cache with the returned profile
      queryClient.setQueryData(mcpProfileKeys.detail(profileId), data);
      // Invalidate all list queries
      queryClient.invalidateQueries({ queryKey: mcpProfileKeys.lists() });
      addToast({
        type: 'success',
        message: `Profile "${data.name}" updated successfully`,
      });
    },
    onError: (error) => {
      const errorMessage = handleApiError(error);
      addToast({ type: 'error', message: errorMessage });
    },
  });
}

/**
 * Hook to delete an MCP profile
 */
export function useDeleteMCPProfile() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation<void, Error, { profileId: string; profileName: string }>({
    mutationFn: async ({ profileId }) => {
      await mcpProfilesApi.delete(profileId);
    },
    onSuccess: (_, { profileId, profileName }) => {
      // Remove from cache
      queryClient.removeQueries({ queryKey: mcpProfileKeys.detail(profileId) });
      queryClient.removeQueries({ queryKey: mcpProfileKeys.config(profileId) });
      // Invalidate all list queries
      queryClient.invalidateQueries({ queryKey: mcpProfileKeys.lists() });
      addToast({
        type: 'success',
        message: `Profile "${profileName}" deleted successfully`,
      });
    },
    onError: (error) => {
      const errorMessage = handleApiError(error);
      addToast({ type: 'error', message: errorMessage });
    },
  });
}

/**
 * Hook to quickly toggle a profile's enabled state
 */
export function useToggleMCPProfileEnabled() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation<
    MCPProfile,
    Error,
    { profileId: string; enabled: boolean; profileName: string }
  >({
    mutationFn: async ({ profileId, enabled }) => {
      const response = await mcpProfilesApi.update(profileId, { enabled });
      return response.data;
    },
    onMutate: async ({ profileId, enabled }) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey: mcpProfileKeys.lists() });

      // Snapshot previous value - use list(undefined) to match the actual query key
      const previousProfiles = queryClient.getQueryData<MCPProfile[]>(
        mcpProfileKeys.list(undefined)
      );

      // Optimistically update
      if (previousProfiles) {
        queryClient.setQueryData<MCPProfile[]>(
          mcpProfileKeys.list(undefined),
          previousProfiles.map((p) =>
            p.id === profileId ? { ...p, enabled } : p
          )
        );
      }

      return { previousProfiles };
    },
    onError: (error, _, context) => {
      // Rollback on error
      if (context?.previousProfiles) {
        queryClient.setQueryData(
          mcpProfileKeys.list(undefined),
          context.previousProfiles
        );
      }
      const errorMessage = handleApiError(error);
      addToast({ type: 'error', message: errorMessage });
    },
    onSuccess: (data, { profileName, enabled }) => {
      addToast({
        type: 'success',
        message: `Profile "${profileName}" ${enabled ? 'enabled' : 'disabled'}`,
      });
    },
    onSettled: () => {
      // Always refetch to ensure consistency
      queryClient.invalidateQueries({ queryKey: mcpProfileKeys.lists() });
    },
  });
}
