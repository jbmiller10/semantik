/**
 * React Query hooks for Graph API.
 *
 * Provides data fetching hooks for:
 * - Graph statistics
 * - Entity search
 * - Single entity lookup
 * - Graph traversal
 * - Entity/relationship type filters
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { graphApi } from '../services/api/v2/graph';
import type {
  EntitySearchRequest,
  EntitySearchResponse,
  EntityResponse,
  GraphResponse,
  GraphStatsResponse,
  GraphTraversalRequest,
} from '../types/graph';

// ============================================================================
// Query Key Factory
// ============================================================================

/**
 * Query key factory for consistent cache key generation.
 * All graph-related queries should use these keys for proper cache invalidation.
 */
export const graphKeys = {
  /** Root key for all graph queries */
  all: ['graph'] as const,

  /** Keys for graph statistics */
  stats: (collectionId: string) => [...graphKeys.all, 'stats', collectionId] as const,

  /** Keys for entity searches */
  entitySearches: (collectionId: string) =>
    [...graphKeys.all, 'entity-search', collectionId] as const,
  entitySearch: (collectionId: string, request: EntitySearchRequest) =>
    [...graphKeys.entitySearches(collectionId), request] as const,

  /** Keys for single entity lookups */
  entities: (collectionId: string) =>
    [...graphKeys.all, 'entities', collectionId] as const,
  entity: (collectionId: string, entityId: number) =>
    [...graphKeys.entities(collectionId), entityId] as const,

  /** Keys for graph traversal */
  traversals: (collectionId: string) =>
    [...graphKeys.all, 'traversal', collectionId] as const,
  traversal: (collectionId: string, entityId: number, maxHops: number) =>
    [...graphKeys.traversals(collectionId), entityId, maxHops] as const,

  /** Keys for entity types */
  entityTypes: (collectionId: string) =>
    [...graphKeys.all, 'entity-types', collectionId] as const,

  /** Keys for relationship types */
  relationshipTypes: (collectionId: string) =>
    [...graphKeys.all, 'relationship-types', collectionId] as const,
};

// ============================================================================
// Statistics Hook
// ============================================================================

/**
 * Fetch graph statistics for a collection.
 *
 * @param collectionId - Collection UUID
 * @param enabled - Whether to enable the query (default: true when collectionId exists)
 * @returns Query result with graph statistics
 *
 * @example
 * ```tsx
 * const { data: stats, isLoading } = useGraphStats(collectionId);
 * if (stats?.graph_enabled) {
 *   console.log(`${stats.total_entities} entities`);
 * }
 * ```
 */
export function useGraphStats(collectionId: string | null, enabled = true) {
  return useQuery<GraphStatsResponse>({
    queryKey: graphKeys.stats(collectionId ?? ''),
    queryFn: async () => {
      if (!collectionId) {
        throw new Error('Collection ID is required');
      }
      const response = await graphApi.getStats(collectionId);
      return response.data;
    },
    enabled: enabled && !!collectionId,
    staleTime: 30000, // 30 seconds
  });
}

// ============================================================================
// Entity Search Hook
// ============================================================================

/**
 * Search entities in a collection.
 *
 * @param collectionId - Collection UUID
 * @param request - Search parameters (query, entity_types, limit, offset)
 * @param enabled - Whether to enable the query
 * @returns Query result with search results
 *
 * @example
 * ```tsx
 * const { data, isLoading } = useEntitySearch(collectionId, {
 *   query: 'John',
 *   entity_types: ['PERSON'],
 *   limit: 20,
 * });
 * ```
 */
export function useEntitySearch(
  collectionId: string | null,
  request: EntitySearchRequest,
  enabled = true
) {
  return useQuery<EntitySearchResponse>({
    queryKey: graphKeys.entitySearch(collectionId ?? '', request),
    queryFn: async () => {
      if (!collectionId) {
        throw new Error('Collection ID is required');
      }
      const response = await graphApi.searchEntities(collectionId, request);
      return response.data;
    },
    enabled: enabled && !!collectionId,
    staleTime: 10000, // 10 seconds
  });
}

// ============================================================================
// Single Entity Hook
// ============================================================================

/**
 * Fetch a single entity by ID.
 *
 * @param collectionId - Collection UUID
 * @param entityId - Entity ID
 * @param enabled - Whether to enable the query
 * @returns Query result with entity data
 *
 * @example
 * ```tsx
 * const { data: entity } = useEntity(collectionId, selectedEntityId);
 * ```
 */
export function useEntity(
  collectionId: string | null,
  entityId: number | null,
  enabled = true
) {
  return useQuery<EntityResponse>({
    queryKey: graphKeys.entity(collectionId ?? '', entityId ?? 0),
    queryFn: async () => {
      if (!collectionId || entityId === null) {
        throw new Error('Collection ID and Entity ID are required');
      }
      const response = await graphApi.getEntity(collectionId, entityId);
      return response.data;
    },
    enabled: enabled && !!collectionId && entityId !== null,
    staleTime: 30000, // 30 seconds
  });
}

// ============================================================================
// Graph Traversal Hook
// ============================================================================

/**
 * Traverse the graph from a starting entity.
 *
 * @param collectionId - Collection UUID
 * @param entityId - Starting entity ID
 * @param maxHops - Maximum traversal depth (1-5, default 2)
 * @param enabled - Whether to enable the query
 * @returns Query result with graph data (nodes and edges)
 *
 * @example
 * ```tsx
 * const { data: graph } = useGraphTraversal(collectionId, entityId, 2);
 * // Use graph.nodes and graph.edges for visualization
 * ```
 */
export function useGraphTraversal(
  collectionId: string | null,
  entityId: number | null,
  maxHops = 2,
  enabled = true
) {
  return useQuery<GraphResponse>({
    queryKey: graphKeys.traversal(collectionId ?? '', entityId ?? 0, maxHops),
    queryFn: async () => {
      if (!collectionId || entityId === null) {
        throw new Error('Collection ID and Entity ID are required');
      }
      const response = await graphApi.traverseGraph(collectionId, {
        entity_id: entityId,
        max_hops: maxHops,
      });
      return response.data;
    },
    enabled: enabled && !!collectionId && entityId !== null,
    staleTime: 30000, // 30 seconds
  });
}

// ============================================================================
// Entity Types Hook
// ============================================================================

/**
 * Get entity types and counts for a collection.
 * Useful for building filter UI.
 *
 * @param collectionId - Collection UUID
 * @param enabled - Whether to enable the query
 * @returns Query result with entity type counts
 *
 * @example
 * ```tsx
 * const { data: entityTypes } = useEntityTypes(collectionId);
 * // entityTypes = { PERSON: 42, ORG: 15, ... }
 * ```
 */
export function useEntityTypes(collectionId: string | null, enabled = true) {
  return useQuery<Record<string, number>>({
    queryKey: graphKeys.entityTypes(collectionId ?? ''),
    queryFn: async () => {
      if (!collectionId) {
        throw new Error('Collection ID is required');
      }
      const response = await graphApi.getEntityTypes(collectionId);
      return response.data;
    },
    enabled: enabled && !!collectionId,
    staleTime: 60000, // 60 seconds
  });
}

// ============================================================================
// Relationship Types Hook
// ============================================================================

/**
 * Get relationship types and counts for a collection.
 * Useful for building filter UI.
 *
 * @param collectionId - Collection UUID
 * @param enabled - Whether to enable the query
 * @returns Query result with relationship type counts
 *
 * @example
 * ```tsx
 * const { data: relationshipTypes } = useRelationshipTypes(collectionId);
 * // relationshipTypes = { WORKS_FOR: 20, LOCATED_IN: 8, ... }
 * ```
 */
export function useRelationshipTypes(collectionId: string | null, enabled = true) {
  return useQuery<Record<string, number>>({
    queryKey: graphKeys.relationshipTypes(collectionId ?? ''),
    queryFn: async () => {
      if (!collectionId) {
        throw new Error('Collection ID is required');
      }
      const response = await graphApi.getRelationshipTypes(collectionId);
      return response.data;
    },
    enabled: enabled && !!collectionId,
    staleTime: 60000, // 60 seconds
  });
}

// ============================================================================
// Graph Explorer Mutation Hook
// ============================================================================

interface GraphExplorerParams {
  entityId: number;
  maxHops?: number;
  relationshipTypes?: string[];
}

/**
 * Mutation hook for interactive graph exploration.
 * Use this when you need imperative control over graph traversal
 * (e.g., when user clicks on a node to expand it).
 *
 * @param collectionId - Collection UUID
 * @returns Mutation object for graph traversal
 *
 * @example
 * ```tsx
 * const explorer = useGraphExplorer(collectionId);
 *
 * const handleNodeClick = (entityId: number) => {
 *   explorer.mutate({ entityId, maxHops: 1 });
 * };
 *
 * // Access result
 * if (explorer.data) {
 *   // Merge nodes and edges into visualization
 * }
 * ```
 */
export function useGraphExplorer(collectionId: string | null) {
  const queryClient = useQueryClient();

  return useMutation<GraphResponse, Error, GraphExplorerParams>({
    mutationFn: async ({ entityId, maxHops = 2, relationshipTypes }) => {
      if (!collectionId) {
        throw new Error('Collection ID is required');
      }
      const request: GraphTraversalRequest = {
        entity_id: entityId,
        max_hops: maxHops,
        relationship_types: relationshipTypes,
      };
      const response = await graphApi.traverseGraph(collectionId, request);
      return response.data;
    },
    onSuccess: (data, variables) => {
      // Cache the result for future queries
      if (collectionId) {
        queryClient.setQueryData(
          graphKeys.traversal(collectionId, variables.entityId, variables.maxHops ?? 2),
          data
        );
      }
    },
  });
}

// ============================================================================
// Cache Invalidation Utilities
// ============================================================================

/**
 * Hook to invalidate all graph-related caches for a collection.
 * Useful after data changes that affect the graph.
 *
 * @returns Function to invalidate graph caches
 *
 * @example
 * ```tsx
 * const invalidateGraph = useInvalidateGraphCache();
 *
 * // After reindexing with graph enabled
 * invalidateGraph(collectionId);
 * ```
 */
export function useInvalidateGraphCache() {
  const queryClient = useQueryClient();

  return (collectionId: string) => {
    queryClient.invalidateQueries({ queryKey: graphKeys.stats(collectionId) });
    queryClient.invalidateQueries({ queryKey: graphKeys.entitySearches(collectionId) });
    queryClient.invalidateQueries({ queryKey: graphKeys.entities(collectionId) });
    queryClient.invalidateQueries({ queryKey: graphKeys.traversals(collectionId) });
    queryClient.invalidateQueries({ queryKey: graphKeys.entityTypes(collectionId) });
    queryClient.invalidateQueries({ queryKey: graphKeys.relationshipTypes(collectionId) });
  };
}
