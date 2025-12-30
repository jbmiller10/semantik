import apiClient from './client';
import type { Operation, OperationListResponse } from '../../../types/collection';
import { buildWebSocketUrl, buildWebSocketProtocols, getAuthToken } from '../baseUrl';

/** WebSocket connection info with URL and authentication protocols */
export interface WebSocketConnectionInfo {
  url: string;
  protocols: string[];
}

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
   * Get WebSocket connection info for operation progress
   * @param operationId - The operation ID to track
   * @param token - Optional auth token override (defaults to stored token)
   * @returns WebSocket URL and authentication protocols
   */
  getWebSocketConnectionInfo: (operationId: string, token?: string | null): WebSocketConnectionInfo | null => {
    const url = buildWebSocketUrl(`/ws/operations/${operationId}`);
    if (!url) return null;
    return {
      url,
      protocols: buildWebSocketProtocols(getAuthToken(token)),
    };
  },

  /**
   * Get WebSocket connection info for global operations feed
   * @param token - Optional auth token override (defaults to stored token)
   * @returns WebSocket URL and authentication protocols
   */
  getGlobalWebSocketConnectionInfo: (token?: string | null): WebSocketConnectionInfo | null => {
    const url = buildWebSocketUrl('/ws/operations');
    if (!url) return null;
    return {
      url,
      protocols: buildWebSocketProtocols(getAuthToken(token)),
    };
  },

  // Deprecated: Use getWebSocketConnectionInfo instead
  getWebSocketUrl: (operationId: string): string | null => {
    return buildWebSocketUrl(`/ws/operations/${operationId}`);
  },

  // Deprecated: Use getGlobalWebSocketConnectionInfo instead
  getGlobalWebSocketUrl: (): string | null => {
    return buildWebSocketUrl('/ws/operations');
  },
};
