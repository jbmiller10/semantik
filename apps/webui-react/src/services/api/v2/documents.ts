import apiClient from './client';
import type { DocumentResponse } from './types';
import { useAuthStore } from '../../../stores/authStore';

/**
 * V2 Documents API client
 * Implements document-related endpoints for the collection-centric API
 */
export const documentsV2Api = {
  /**
   * Get document content for viewing
   * Returns the raw document file for display in the DocumentViewer
   */
  getContent: (collectionUuid: string, documentUuid: string) => {
    // For file downloads/viewing, we need to construct the URL directly
    // as axios would try to parse the response as JSON by default
    const baseURL = apiClient.defaults.baseURL || '';
    // Get token from auth store
    const state = useAuthStore.getState();
    const token = state.token;
    
    // Construct the full URL for the document content endpoint
    const url = `${baseURL}/api/v2/collections/${collectionUuid}/documents/${documentUuid}/content`;
    
    // Return URL and headers for direct fetch or iframe use
    return {
      url,
      headers: token ? { Authorization: `Bearer ${token}` } : {}
    };
  },

  /**
   * Get document metadata (if we add this endpoint in the future)
   */
  get: (collectionUuid: string, documentUuid: string) =>
    apiClient.get<DocumentResponse>(`/api/v2/collections/${collectionUuid}/documents/${documentUuid}`),

  /**
   * Get chunk by ID
   * Returns chunk content and position data for highlighting
   */
  getChunk: async (collectionUuid: string, chunkId: string) => {
    const response = await apiClient.get<{
      id: string;
      content: string;
      start_offset: number | null;
      end_offset: number | null;
      chunk_index: number;
      document_id: string;
    }>(`/api/v2/collections/${collectionUuid}/chunks/${chunkId}`);
    return response.data;
  },
};