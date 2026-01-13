/**
 * React Query hooks for User Preferences management.
 * Provides hooks for fetching, updating, and resetting user preferences.
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { preferencesApi } from '../services/api/v2/preferences';
import { useUIStore } from '../stores/uiStore';
import { ApiErrorHandler } from '../utils/api-error-handler';
import type { UserPreferencesResponse, UserPreferencesUpdate } from '../types/preferences';

/**
 * Query key factory for preferences queries.
 * Enables hierarchical cache invalidation.
 */
export const preferencesKeys = {
  all: ['preferences'] as const,
  settings: () => [...preferencesKeys.all, 'settings'] as const,
};

/**
 * Hook to fetch current user's preferences.
 * Creates defaults if not configured yet.
 */
export function usePreferences() {
  return useQuery({
    queryKey: preferencesKeys.settings(),
    queryFn: async () => {
      const response = await preferencesApi.get();
      return response.data;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchOnWindowFocus: true, // Multi-tab sync
  });
}

/**
 * Hook to update user preferences.
 * Supports partial updates - only send fields you want to change.
 * Shows toast notifications on success/error.
 */
export function useUpdatePreferences() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation<UserPreferencesResponse, Error, UserPreferencesUpdate>({
    mutationFn: async (data: UserPreferencesUpdate) => {
      const response = await preferencesApi.update(data);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: preferencesKeys.settings() });
      addToast({
        type: 'success',
        message: 'Preferences saved successfully',
      });
    },
    onError: (error) => {
      const apiError = ApiErrorHandler.handle(error);
      addToast({
        type: 'error',
        message: apiError.message || 'Failed to save preferences',
      });
    },
  });
}

/**
 * Hook to reset search preferences to defaults.
 * Shows toast notifications on success/error.
 */
export function useResetSearchPreferences() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation<UserPreferencesResponse, Error>({
    mutationFn: async () => {
      const response = await preferencesApi.resetSearch();
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: preferencesKeys.settings() });
      addToast({
        type: 'success',
        message: 'Search preferences reset to defaults',
      });
    },
    onError: (error) => {
      const apiError = ApiErrorHandler.handle(error);
      addToast({
        type: 'error',
        message: apiError.message || 'Failed to reset search preferences',
      });
    },
  });
}

/**
 * Hook to reset collection defaults to system defaults.
 * Shows toast notifications on success/error.
 */
export function useResetCollectionDefaults() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation<UserPreferencesResponse, Error>({
    mutationFn: async () => {
      const response = await preferencesApi.resetCollectionDefaults();
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: preferencesKeys.settings() });
      addToast({
        type: 'success',
        message: 'Collection defaults reset to system defaults',
      });
    },
    onError: (error) => {
      const apiError = ApiErrorHandler.handle(error);
      addToast({
        type: 'error',
        message: apiError.message || 'Failed to reset collection defaults',
      });
    },
  });
}

/**
 * Hook to reset interface preferences to defaults.
 * Shows toast notifications on success/error.
 */
export function useResetInterfacePreferences() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation<UserPreferencesResponse, Error>({
    mutationFn: async () => {
      const response = await preferencesApi.resetInterface();
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: preferencesKeys.settings() });
      addToast({
        type: 'success',
        message: 'Interface preferences reset to defaults',
      });
    },
    onError: (error) => {
      const apiError = ApiErrorHandler.handle(error);
      addToast({
        type: 'error',
        message: apiError.message || 'Failed to reset interface preferences',
      });
    },
  });
}
