import apiClient from './client';
import type { DirectoryScanRequest, DirectoryScanResponse } from './types';
import { buildWebSocketUrl, getAuthToken } from '../baseUrl';

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
   * @returns WebSocket URL (token passed via subprotocol, not URL)
   */
  getWebSocketUrl: (scanId: string): string | null => {
    return buildWebSocketUrl(`/ws/directory-scan/${scanId}`);
  },

  /**
   * Get authentication token for WebSocket subprotocol
   * @returns The auth token to use as subprotocol
   */
  getAuthToken: (): string | null => {
    return getAuthToken();
  },
};

// Helper function to generate a scan ID
export const generateScanId = (): string => {
  const globalCrypto = typeof globalThis !== 'undefined' ? globalThis.crypto : undefined;

  if (globalCrypto?.randomUUID) {
    return globalCrypto.randomUUID();
  }

  if (globalCrypto?.getRandomValues) {
    const bytes = globalCrypto.getRandomValues(new Uint8Array(16));
    // RFC 4122 variant 1 UUID layout
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }

  // Last-resort fallback for very old browsers: timestamp + random suffix
  return `scan-${Date.now().toString(16)}-${Math.random().toString(16).slice(2, 10)}`;
};
