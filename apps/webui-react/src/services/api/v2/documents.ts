import apiClient from './client';
import type { DocumentResponse, FailedDocumentCountResponse, RetryDocumentsResponse } from './types';
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
   * Get document metadata
   */
  get: (collectionUuid: string, documentUuid: string) =>
    apiClient.get<DocumentResponse>(`/api/v2/collections/${collectionUuid}/documents/${documentUuid}`),

  /**
   * Retry a single failed document
   * Resets the document status to PENDING so it can be reprocessed
   */
  retry: (collectionUuid: string, documentUuid: string) =>
    apiClient.post<DocumentResponse>(`/api/v2/collections/${collectionUuid}/documents/${documentUuid}/retry`),

  /**
   * Bulk retry all failed documents in a collection
   * Only resets documents with transient or unknown errors that haven't exceeded max retries
   */
  retryFailed: (collectionUuid: string) =>
    apiClient.post<RetryDocumentsResponse>(`/api/v2/collections/${collectionUuid}/documents/retry-failed`),

  /**
   * Get count of failed documents by error category
   */
  getFailedCount: (collectionUuid: string, retryableOnly: boolean = false) =>
    apiClient.get<FailedDocumentCountResponse>(
      `/api/v2/collections/${collectionUuid}/documents/failed/count`,
      { params: { retryable_only: retryableOnly } }
    ),
};