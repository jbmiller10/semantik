import axios from 'axios';
import { useAuthStore } from '../stores/authStore';

const api = axios.create({
  baseURL: '',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
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
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      useAuthStore.getState().logout();
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default api;

// Search API endpoints
export const searchApi = {
  search: (params: {
    query: string;
    collection: string;
    top_k: number;
    score_threshold: number;
    search_type: 'vector' | 'hybrid';
    rerank_model?: string;
    rerank_quantization?: string;
    use_reranker?: boolean;
    hybrid_alpha?: number;
    hybrid_mode?: 'rerank' | 'filter';
    keyword_mode?: 'any' | 'all';
  }) => api.post('/api/search', params),
};

// Document API endpoints
export const documentsApi = {
  getDocument: (docId: string) => api.get(`/api/documents/${docId}`),
  getChunk: (docId: string, chunkIndex: number) =>
    api.get(`/api/documents/${docId}/chunks/${chunkIndex}`),
};

// Auth API endpoints
export const authApi = {
  login: (credentials: { username: string; password: string }) =>
    api.post('/api/auth/login', credentials),
  register: (credentials: { username: string; email: string; password: string; full_name?: string }) =>
    api.post('/api/auth/register', credentials),
  me: () => api.get('/api/auth/me'),
  logout: () => api.post('/api/auth/logout'),
};


// Models API endpoints
export const modelsApi = {
  list: () => api.get('/api/models'),
};

// Settings API endpoints
export const settingsApi = {
  getStats: () => api.get('/api/settings/stats'),
  resetDatabase: () => api.post('/api/settings/reset-database'),
};