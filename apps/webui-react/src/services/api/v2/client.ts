import axios from 'axios';
import { useAuthStore } from '../../../stores/authStore';
import { navigateTo } from '../../navigation';

/**
 * Axios instance configured for v2 API endpoints
 */
const apiClient = axios.create({
  baseURL: '',
  headers: {
    'Content-Type': 'application/json',
  },
});

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

// Response interceptor to handle auth errors
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      // Clear auth state
      await useAuthStore.getState().logout();
      
      navigateTo('/login');
    }
    return Promise.reject(error);
  }
);

export default apiClient;
