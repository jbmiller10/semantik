/**
 * React Query hooks for System Settings management.
 * Provides hooks for fetching and updating admin-only system settings.
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { systemSettingsApi } from '../services/api/v2/system-settings';
import { useUIStore } from '../stores/uiStore';
import { ApiErrorHandler } from '../utils/api-error-handler';
import type {
  SystemSettingsUpdateRequest,
  SystemSettingsUpdateResponse,
} from '../types/system-settings';

/**
 * Query key factory for system settings queries.
 * Enables hierarchical cache invalidation.
 */
export const systemSettingsKeys = {
  all: ['system-settings'] as const,
  settings: () => [...systemSettingsKeys.all, 'settings'] as const,
  effective: () => [...systemSettingsKeys.all, 'effective'] as const,
  defaults: () => [...systemSettingsKeys.all, 'defaults'] as const,
};

/**
 * Hook to fetch effective system settings (resolved DB value or env fallback).
 * Admin-only. Returns the actual values used by the system.
 */
export function useEffectiveSettings() {
  return useQuery({
    queryKey: systemSettingsKeys.effective(),
    queryFn: async () => {
      const response = await systemSettingsApi.getEffective();
      return response.data;
    },
    staleTime: 60 * 1000, // 1 minute - matches backend cache TTL
    refetchOnWindowFocus: true,
  });
}

/**
 * Hook to fetch default values for all settings.
 * Useful for "Reset to Defaults" functionality.
 */
export function useSystemSettingsDefaults() {
  return useQuery({
    queryKey: systemSettingsKeys.defaults(),
    queryFn: async () => {
      const response = await systemSettingsApi.getDefaults();
      return response.data;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes - defaults don't change often
  });
}

/**
 * Hook to update system settings.
 * Admin-only. Shows toast notifications on success/error.
 */
export function useUpdateSystemSettings() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation<
    SystemSettingsUpdateResponse,
    Error,
    SystemSettingsUpdateRequest
  >({
    mutationFn: async (data: SystemSettingsUpdateRequest) => {
      const response = await systemSettingsApi.updateSettings(data);
      return response.data;
    },
    onSuccess: (data) => {
      // Invalidate all system settings queries to refresh data
      queryClient.invalidateQueries({ queryKey: systemSettingsKeys.all });

      const count = data.updated.length;
      addToast({
        type: 'success',
        message: `${count} setting${count !== 1 ? 's' : ''} saved successfully`,
      });
    },
    onError: (error) => {
      const apiError = ApiErrorHandler.handle(error);
      addToast({
        type: 'error',
        message: apiError.message || 'Failed to save settings',
      });
    },
  });
}

/**
 * Hook to reset specific settings to their defaults.
 * Convenience wrapper that fetches defaults and updates settings.
 */
export function useResetSettingsToDefaults() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation<
    SystemSettingsUpdateResponse,
    Error,
    string[] // keys to reset
  >({
    mutationFn: async (keys: string[]) => {
      // First, get the defaults
      const defaultsResponse = await systemSettingsApi.getDefaults();
      const defaults = defaultsResponse.data.defaults;

      // Build update request with default values for specified keys
      const settings: Record<string, unknown> = {};
      for (const key of keys) {
        if (key in defaults) {
          settings[key] = defaults[key];
        }
      }

      // Update settings with defaults
      const response = await systemSettingsApi.updateSettings({ settings });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: systemSettingsKeys.all });
      addToast({
        type: 'success',
        message: 'Settings reset to defaults',
      });
    },
    onError: (error) => {
      const apiError = ApiErrorHandler.handle(error);
      addToast({
        type: 'error',
        message: apiError.message || 'Failed to reset settings',
      });
    },
  });
}
