import axios, { AxiosError } from 'axios';
import type { InternalAxiosRequestConfig } from 'axios';
import { useAuthStore } from '../../../stores/authStore';
import { navigateTo } from '../../navigation';
import { getApiBaseUrl } from '../baseUrl';

/**
 * Axios instance configured for v2 API endpoints
 */
const apiClient = axios.create({
  baseURL: getApiBaseUrl(),
});

// Token refresh state
let isRefreshing = false;
let failedQueue: Array<{
  resolve: (token: string) => void;
  reject: (error: Error) => void;
}> = [];

/**
 * Process queued requests after token refresh
 */
const processQueue = (error: Error | null, token: string | null = null) => {
  failedQueue.forEach((prom) => {
    if (error) {
      prom.reject(error);
    } else if (token) {
      prom.resolve(token);
    }
  });
  failedQueue = [];
};

// Request interceptor to add auth token
apiClient.interceptors.request.use(
  (config) => {
    // Get token from store state at request time
    const state = useAuthStore.getState();
    const token = state.token;

    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle auth errors with token refresh
apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & {
      _retry?: boolean;
    };

    // Only attempt refresh on 401 errors that haven't been retried
    if (error.response?.status === 401 && originalRequest && !originalRequest._retry) {
      // If already refreshing, queue this request
      if (isRefreshing) {
        return new Promise<string>((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        })
          .then((token) => {
            originalRequest.headers.Authorization = `Bearer ${token}`;
            return apiClient(originalRequest);
          })
          .catch((err) => {
            return Promise.reject(err);
          });
      }

      originalRequest._retry = true;
      isRefreshing = true;

      try {
        const refreshToken = useAuthStore.getState().refreshToken;

        // No refresh token available - logout immediately
        if (!refreshToken) {
          throw new Error('No refresh token available');
        }

        // Attempt to refresh the token
        const response = await fetch(`${getApiBaseUrl()}/api/auth/refresh`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: refreshToken }),
        });

        if (!response.ok) {
          throw new Error('Token refresh failed');
        }

        const data = await response.json();
        const newAccessToken = data.access_token;
        const newRefreshToken = data.refresh_token || refreshToken;

        // Update tokens in store
        useAuthStore.getState().setTokens(newAccessToken, newRefreshToken);

        // Process queued requests with new token
        processQueue(null, newAccessToken);

        // Retry the original request with new token
        originalRequest.headers.Authorization = `Bearer ${newAccessToken}`;
        return apiClient(originalRequest);
      } catch (refreshError) {
        // Token refresh failed - logout and clear queue
        processQueue(refreshError as Error, null);
        await useAuthStore.getState().logout();
        navigateTo('/login');
        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
      }
    }

    return Promise.reject(error);
  }
);

export default apiClient;
