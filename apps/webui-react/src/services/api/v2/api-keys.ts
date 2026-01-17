import apiClient from './client';
import type {
  ApiKeyCreate,
  ApiKeyResponse,
  ApiKeyCreateResponse,
  ApiKeyListResponse,
  ApiKeyUpdate,
} from '../../../types/api-key';

/**
 * V2 API Keys API client
 * Provides endpoints for managing API keys
 */
export const apiKeysApi = {
  /**
   * List all API keys for the current user
   */
  list: () => apiClient.get<ApiKeyListResponse>('/api/v2/api-keys'),

  /**
   * Get a specific API key
   * @param keyId The API key UUID
   */
  get: (keyId: string) => apiClient.get<ApiKeyResponse>(`/api/v2/api-keys/${keyId}`),

  /**
   * Create a new API key
   * @param data The API key creation data
   */
  create: (data: ApiKeyCreate) =>
    apiClient.post<ApiKeyCreateResponse>('/api/v2/api-keys', data),

  /**
   * Update an existing API key (revoke/reactivate)
   * @param keyId The API key UUID
   * @param data The update data
   */
  update: (keyId: string, data: ApiKeyUpdate) =>
    apiClient.patch<ApiKeyResponse>(`/api/v2/api-keys/${keyId}`, data),
};
