/**
 * React Query hooks for System Information.
 * Provides hooks for fetching system info, health status, and GPU status.
 */
import { useQuery } from '@tanstack/react-query';
import { systemApi } from '../services/api/v2/system';

/**
 * Query key factory for system queries.
 * Enables hierarchical cache invalidation.
 */
export const systemKeys = {
  all: ['system'] as const,
  info: () => [...systemKeys.all, 'info'] as const,
  health: () => [...systemKeys.all, 'health'] as const,
  status: () => [...systemKeys.all, 'status'] as const,
};

/**
 * Hook to fetch system information.
 * Returns version, environment, limits, and rate limits.
 */
export function useSystemInfo() {
  return useQuery({
    queryKey: systemKeys.info(),
    queryFn: async () => {
      const response = await systemApi.getInfo();
      return response.data;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes - system info rarely changes
    gcTime: 30 * 60 * 1000, // Keep in cache for 30 minutes
  });
}

/**
 * Hook to fetch health status for all backend services.
 * Auto-refreshes every 30 seconds for real-time monitoring.
 */
export function useSystemHealth() {
  return useQuery({
    queryKey: systemKeys.health(),
    queryFn: async () => {
      const response = await systemApi.getHealth();
      return response.data;
    },
    staleTime: 15 * 1000, // 15 seconds
    refetchInterval: 30 * 1000, // Auto-refresh every 30 seconds
    refetchOnWindowFocus: true,
  });
}

/**
 * Hook to fetch system status including GPU availability.
 */
export function useSystemStatus() {
  return useQuery({
    queryKey: systemKeys.status(),
    queryFn: async () => {
      return await systemApi.getStatus();
    },
    staleTime: 60 * 1000, // 1 minute - GPU status doesn't change often
    gcTime: 10 * 60 * 1000, // Keep in cache for 10 minutes
  });
}
