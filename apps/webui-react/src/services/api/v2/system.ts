import apiClient from './client';
import type { SystemInfo, SystemHealth, SystemStatus } from '../../../types/system';

// Re-export SystemStatus for backwards compatibility
export type { SystemStatus };

/**
 * System API service for checking system capabilities, health, and info.
 */
export const systemApi = {
  /**
   * Check system status including GPU and reranking availability.
   */
  async getStatus(): Promise<SystemStatus> {
    try {
      const response = await apiClient.get<SystemStatus>('/api/v2/system/status');
      return response.data;
    } catch {
      // If the endpoint doesn't exist, return defaults
      console.warn('System status endpoint not available, returning defaults');
      return {
        gpu_available: false,
        reranking_available: true, // Assume available unless we know otherwise
        available_reranking_models: [],
        cuda_device_count: 0,
        cuda_device_name: null,
      };
    }
  },

  /**
   * Get system information including version, limits, and rate limits.
   * This endpoint is public (no auth required).
   */
  getInfo: () => apiClient.get<SystemInfo>('/api/v2/system/info'),

  /**
   * Get health status for all backend services.
   * Always returns 200 with per-service status (partial results).
   */
  getHealth: () => apiClient.get<SystemHealth>('/api/v2/system/health'),
};