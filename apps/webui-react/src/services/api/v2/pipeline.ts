/**
 * API service for pipeline operations.
 */

import apiClient from './client';
import type { RoutePreviewResponse } from '@/types/routePreview';
import type { PipelineDAG } from '@/types/pipeline';

/**
 * A single predicate field available for routing.
 */
export interface PredicateField {
  /** Full field path (e.g., 'metadata.parsed.has_tables') */
  value: string;
  /** Human-readable label (e.g., 'Has Tables') */
  label: string;
  /** Field category for UI grouping */
  category: 'source' | 'detected' | 'parsed';
}

/**
 * Response containing available predicate fields for an edge.
 */
export interface AvailablePredicateFieldsResponse {
  fields: PredicateField[];
}

/**
 * Pipeline API service.
 */
export const pipelineApi = {
  /**
   * Preview how a file would be routed through a pipeline DAG.
   *
   * @param file - The file to test routing with
   * @param dag - The pipeline DAG configuration
   * @param includeParserMetadata - Whether to run parser and include metadata
   * @returns RoutePreviewResponse with detailed routing information
   */
  previewRoute: async (
    file: File,
    dag: PipelineDAG,
    includeParserMetadata: boolean = true
  ): Promise<RoutePreviewResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('dag', JSON.stringify(dag));
    formData.append('include_parser_metadata', String(includeParserMetadata));

    const response = await apiClient.post<RoutePreviewResponse>(
      '/api/v2/pipeline/preview-route',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  },

  /**
   * Get available predicate fields for an edge based on its source node.
   *
   * The available parsed.* fields depend on which parser is the source node:
   * - From _source: No parsed.* fields (parser hasn't run yet)
   * - From parser node: Only fields that parser emits
   *
   * @param dag - The pipeline DAG configuration
   * @param fromNode - The source node ID for the edge (e.g., '_source', 'text_parser')
   * @returns AvailablePredicateFieldsResponse with available fields
   */
  getAvailablePredicateFields: async (
    dag: PipelineDAG,
    fromNode: string
  ): Promise<AvailablePredicateFieldsResponse> => {
    const response = await apiClient.post<AvailablePredicateFieldsResponse>(
      '/api/v2/pipeline/available-predicate-fields',
      {
        dag,
        from_node: fromNode,
      }
    );
    return response.data;
  },
};

export default pipelineApi;
