import apiClient from './client';

export interface SystemStatus {
  gpu_available: boolean;
  reranking_available: boolean;
  available_reranking_models?: string[];
}

/**
 * System API service for checking system capabilities
 */
export const systemApi = {
  /**
   * Check system status including GPU and reranking availability
   */
  async getStatus(): Promise<SystemStatus> {
    try {
      const response = await apiClient.get<SystemStatus>('/api/v2/system/status');
      return response.data;
    } catch (error) {
      // If the endpoint doesn't exist, try to infer from error responses
      console.warn('System status endpoint not available, returning defaults');
      return {
        gpu_available: false,
        reranking_available: true, // Assume available unless we know otherwise
      };
    }
  },
};