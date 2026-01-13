/**
 * User Preferences API client.
 * Implements endpoints for search preferences and collection defaults.
 */
import apiClient from './client';
import type { UserPreferencesResponse, UserPreferencesUpdate } from '../../../types/preferences';

export const preferencesApi = {
  /**
   * Get current user's preferences.
   * Creates defaults if not configured yet.
   */
  get: () => apiClient.get<UserPreferencesResponse>('/api/v2/preferences'),

  /**
   * Update user preferences.
   * Supports partial updates - only send fields you want to change.
   * @param data Partial preferences to update
   */
  update: (data: UserPreferencesUpdate) =>
    apiClient.put<UserPreferencesResponse>('/api/v2/preferences', data),

  /**
   * Reset search preferences to default values.
   */
  resetSearch: () =>
    apiClient.post<UserPreferencesResponse>('/api/v2/preferences/reset/search'),

  /**
   * Reset collection defaults to system defaults.
   */
  resetCollectionDefaults: () =>
    apiClient.post<UserPreferencesResponse>('/api/v2/preferences/reset/collection-defaults'),
};
