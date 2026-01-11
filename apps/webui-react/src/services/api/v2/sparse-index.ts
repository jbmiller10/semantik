/**
 * Sparse Index API Client
 *
 * API client for sparse indexing (BM25/SPLADE) management endpoints.
 * Follows the collection-centric URL pattern: /api/v2/collections/{uuid}/sparse-index
 */

import apiClient from './client';
import type {
  SparseIndexStatus,
  EnableSparseIndexRequest,
  SparseReindexJobResponse,
  SparseReindexProgress,
} from './types';

/**
 * Sparse Index API client
 * Manages BM25/SPLADE sparse indexing for collections
 */
export const sparseIndexApi = {
  /**
   * Get sparse index status for a collection
   */
  getStatus: (collectionUuid: string) =>
    apiClient.get<SparseIndexStatus>(
      `/api/v2/collections/${collectionUuid}/sparse-index`
    ),

  /**
   * Enable sparse indexing on a collection
   * Creates the sparse collection in Qdrant and optionally reindexes existing documents
   */
  enable: (collectionUuid: string, data: EnableSparseIndexRequest) =>
    apiClient.post<SparseIndexStatus>(
      `/api/v2/collections/${collectionUuid}/sparse-index`,
      data
    ),

  /**
   * Disable sparse indexing on a collection
   * Deletes the sparse collection from Qdrant
   */
  disable: (collectionUuid: string) =>
    apiClient.delete<void>(
      `/api/v2/collections/${collectionUuid}/sparse-index`
    ),

  /**
   * Trigger a sparse reindex job for the collection
   * Returns a job ID for progress tracking
   */
  triggerReindex: (collectionUuid: string) =>
    apiClient.post<SparseReindexJobResponse>(
      `/api/v2/collections/${collectionUuid}/sparse-index/reindex`,
      {}
    ),

  /**
   * Get reindex job progress
   * Poll this endpoint to track reindex progress
   */
  getReindexProgress: (collectionUuid: string, jobId: string) =>
    apiClient.get<SparseReindexProgress>(
      `/api/v2/collections/${collectionUuid}/sparse-index/reindex/${jobId}`
    ),
};

/**
 * React Query keys for sparse index queries
 * Use these for consistent cache key management
 */
export const sparseIndexKeys = {
  all: ['sparse-index'] as const,

  /** Key for sparse index status query */
  status: (collectionUuid: string) =>
    [...sparseIndexKeys.all, 'status', collectionUuid] as const,

  /** Key for reindex job progress query */
  reindexProgress: (collectionUuid: string, jobId: string) =>
    [...sparseIndexKeys.all, 'reindex', collectionUuid, jobId] as const,
};
