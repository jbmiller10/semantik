import apiClient from './client';
import type { Document } from './types';

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
    const token = localStorage.getItem('access_token');
    
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
    apiClient.get<Document>(`/api/v2/collections/${collectionUuid}/documents/${documentUuid}`),
};