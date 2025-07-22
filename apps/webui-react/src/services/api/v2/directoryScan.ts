import { apiClient } from './client';
import type { DirectoryScanRequest, DirectoryScanResponse } from './types';

/**
 * Directory scan API endpoints for v2 API
 */
export const directoryScanV2Api = {
  /**
   * Scan a directory to preview its contents
   * This doesn't create a collection, just returns a list of supported files
   */
  preview: async (request: DirectoryScanRequest): Promise<DirectoryScanResponse> => {
    const response = await apiClient.post<DirectoryScanResponse>(
      '/api/v2/directory-scan/preview',
      request
    );
    return response.data;
  },

  /**
   * Get the WebSocket URL for directory scan progress
   * @param scanId - The scan ID to track
   * @returns WebSocket URL with authentication token
   */
  getWebSocketUrl: (scanId: string): string => {
    const token = localStorage.getItem('access_token');
    const baseUrl = window.location.origin.replace(/^http/, 'ws');
    return `${baseUrl}/ws/directory-scan/${scanId}?token=${encodeURIComponent(token || '')}`;
  },
};

// Helper function to generate a scan ID
export const generateScanId = (): string => {
  return crypto.randomUUID();
};