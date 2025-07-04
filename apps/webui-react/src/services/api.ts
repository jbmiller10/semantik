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
    const token = useAuthStore.getState().token;
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

// Job API endpoints
export const jobsApi = {
  list: () => api.get('/api/jobs'),
  create: (data: {
    directory: string;
    collection_name: string;
    model_name?: string;
    chunk_size?: number;
    chunk_overlap?: number;
    batch_size?: number;
    vector_dim?: number;
    quantization?: string;
    instruction?: string;
  }) =>
    api.post('/api/jobs', {
      name: data.collection_name,
      description: '',
      directory_path: data.directory,
      model_name: data.model_name,
      chunk_size: data.chunk_size,
      chunk_overlap: data.chunk_overlap,
      batch_size: data.batch_size,
      vector_dim: data.vector_dim,
      quantization: data.quantization,
      instruction: data.instruction,
    }),
  delete: (jobId: string) => api.delete(`/api/jobs/${jobId}`),
  getMetrics: () => api.get('/api/metrics'),
  cancel: (jobId: string) => api.post(`/api/jobs/${jobId}/cancel`),
  getCollectionsStatus: () => api.get('/api/jobs/collections-status'),
};

// Search API endpoints
export const searchApi = {
  search: (params: {
    query: string;
    collection: string;
    top_k: number;
    score_threshold: number;
    search_type: 'vector' | 'hybrid';
    rerank_model?: string;
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

// Collections API endpoints
export const collectionsApi = {
  list: () => api.get('/api/collections'),
  getDetails: (name: string) => api.get(`/api/collections/${encodeURIComponent(name)}`),
  rename: (name: string, newName: string) => 
    api.put(`/api/collections/${encodeURIComponent(name)}`, { new_name: newName }),
  delete: (name: string) => api.delete(`/api/collections/${encodeURIComponent(name)}`),
  getFiles: (name: string, page: number = 1, limit: number = 50) => 
    api.get(`/api/collections/${encodeURIComponent(name)}/files?page=${page}&limit=${limit}`),
  addData: (collectionName: string, directoryPath: string, description?: string) =>
    api.post('/api/jobs/add-to-collection', {
      collection_name: collectionName,
      directory_path: directoryPath,
      description: description || `Adding documents to ${collectionName}`,
    }),
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