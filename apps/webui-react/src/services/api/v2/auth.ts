import apiClient from './client';

/**
 * Authentication API client
 * Note: Auth endpoints are not versioned and remain at /api/auth
 */
export const authApi = {
  login: (credentials: { username: string; password: string }) =>
    apiClient.post('/api/auth/login', credentials),
    
  register: (credentials: { username: string; email: string; password: string; full_name?: string }) =>
    apiClient.post('/api/auth/register', credentials),
    
  me: () => apiClient.get('/api/auth/me'),
  
  logout: () => apiClient.post('/api/auth/logout'),
};