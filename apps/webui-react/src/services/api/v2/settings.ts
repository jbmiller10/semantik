import apiClient from './client';

/**
 * Settings API client
 * Note: Settings endpoints are not versioned and remain at /api/settings
 */
export const settingsApi = {
  getStats: () => apiClient.get('/api/settings/stats'),
  
  resetDatabase: () => apiClient.post('/api/settings/reset-database'),
};