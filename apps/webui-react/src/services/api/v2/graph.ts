/**
 * Graph API client for entity browsing and graph traversal.
 *
 * Endpoints:
 * - GET  /api/graph/collections/{id}/stats - Get graph statistics
 * - POST /api/graph/collections/{id}/entities/search - Search entities
 * - GET  /api/graph/collections/{id}/entities/{entityId} - Get single entity
 * - POST /api/graph/collections/{id}/traverse - Traverse graph
 * - GET  /api/graph/collections/{id}/entity-types - Get entity type counts
 * - GET  /api/graph/collections/{id}/relationship-types - Get relationship type counts
 */

import apiClient from './client';
import type {
  EntityResponse,
  EntitySearchRequest,
  EntitySearchResponse,
  GraphTraversalRequest,
  GraphResponse,
  GraphStatsResponse,
} from '../../../types/graph';

/**
 * Graph API client functions
 */
export const graphApi = {
  /**
   * Get graph statistics for a collection.
   * Returns entity and relationship counts, broken down by type.
   */
  getStats: (collectionId: string) =>
    apiClient.get<GraphStatsResponse>(`/api/graph/collections/${collectionId}/stats`),

  /**
   * Search entities in a collection.
   * Supports name prefix search and filtering by entity type.
   */
  searchEntities: (collectionId: string, request: EntitySearchRequest) =>
    apiClient.post<EntitySearchResponse>(
      `/api/graph/collections/${collectionId}/entities/search`,
      request
    ),

  /**
   * Get a single entity by ID.
   */
  getEntity: (collectionId: string, entityId: number) =>
    apiClient.get<EntityResponse>(
      `/api/graph/collections/${collectionId}/entities/${entityId}`
    ),

  /**
   * Traverse the graph from a starting entity.
   * Returns nodes and edges suitable for visualization.
   */
  traverseGraph: (collectionId: string, request: GraphTraversalRequest) =>
    apiClient.post<GraphResponse>(
      `/api/graph/collections/${collectionId}/traverse`,
      request
    ),

  /**
   * Get entity types and counts for a collection.
   * Useful for building filter UI.
   */
  getEntityTypes: (collectionId: string) =>
    apiClient.get<Record<string, number>>(
      `/api/graph/collections/${collectionId}/entity-types`
    ),

  /**
   * Get relationship types and counts for a collection.
   * Useful for building filter UI.
   */
  getRelationshipTypes: (collectionId: string) =>
    apiClient.get<Record<string, number>>(
      `/api/graph/collections/${collectionId}/relationship-types`
    ),
};

// Also export individual functions for convenience
export const {
  getStats,
  searchEntities,
  getEntity,
  traverseGraph,
  getEntityTypes,
  getRelationshipTypes,
} = graphApi;
