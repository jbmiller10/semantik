import apiClient from './client';
import type { Operation } from '../../../types/collection';

export const operationsV2Api = {
  /**
   * Get a specific operation by ID
   */
  get: async (operationId: string): Promise<Operation> => {
    const response = await apiClient.get<Operation>(`/api/v2/operations/${operationId}`);
    return response.data;
  },

  /**
   * List operations with optional filters
   */
  list: async (params?: {
    collection_id?: string;
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ operations: Operation[]; total: number }> => {
    const queryParams = new URLSearchParams();
    if (params?.collection_id) queryParams.append('collection_id', params.collection_id);
    if (params?.status) queryParams.append('status', params.status);
    if (params?.limit) queryParams.append('limit', params.limit.toString());
    if (params?.offset) queryParams.append('offset', params.offset.toString());
    
    const query = queryParams.toString();
    const response = await apiClient.get<{ operations: Operation[]; total: number }>(
      `/api/v2/operations${query ? `?${query}` : ''}`
    );
    return response.data;
  },

  /**
   * Cancel a pending or processing operation
   */
  cancel: async (operationId: string): Promise<Operation> => {
    const response = await apiClient.delete<Operation>(`/api/v2/operations/${operationId}`);
    return response.data;
  },

  /**
   * Get the WebSocket URL for operation progress
   * @param operationId - The operation ID to track
   * @returns WebSocket URL with authentication token
   */
  getWebSocketUrl: (operationId: string): string => {
    // Get auth state from Zustand store persisted in localStorage
    const authStorage = localStorage.getItem('auth-storage');
    let token = '';
    
    if (authStorage) {
      try {
        const authState = JSON.parse(authStorage);
        token = authState.state?.token || '';
      } catch (error) {
        console.error('Failed to parse auth storage:', error);
      }
    }
    
    const baseUrl = window.location.origin.replace(/^http/, 'ws');
    return `${baseUrl}/ws/operations/${operationId}?token=${encodeURIComponent(token)}`;
  },
};