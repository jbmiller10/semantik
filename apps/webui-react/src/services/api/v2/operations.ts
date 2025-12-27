import apiClient from './client';
import type { Operation, OperationListResponse } from '../../../types/collection';
import { buildWebSocketUrl, getAuthToken } from '../baseUrl';

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
    status?: string;
    operation_type?: string;
    page?: number;
    per_page?: number;
    limit?: number;
    offset?: number;
  }): Promise<OperationListResponse> => {
    const queryParams = new URLSearchParams();
    if (params?.status) queryParams.append('status', params.status);
    if (params?.operation_type) queryParams.append('operation_type', params.operation_type);

    const perPage = params?.per_page ?? params?.limit;
    const page =
      params?.page ??
      (params?.offset !== undefined && params?.limit ? Math.floor(params.offset / params.limit) + 1 : undefined);

    if (page) queryParams.append('page', page.toString());
    if (perPage) queryParams.append('per_page', perPage.toString());
    
    const query = queryParams.toString();
    const response = await apiClient.get<OperationListResponse>(
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
  getWebSocketUrl: (operationId: string, token?: string | null): string | null => {
    return buildWebSocketUrl(`/ws/operations/${operationId}`, getAuthToken(token));
  },

  /**
   * Global operations feed (broadcast for all operations)
   */
  getGlobalWebSocketUrl: (token?: string | null): string | null => {
    return buildWebSocketUrl('/ws/operations', getAuthToken(token));
  },
};
