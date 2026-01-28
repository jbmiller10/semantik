/**
 * API service for pipeline operations.
 */

import apiClient from './client';
import type { RoutePreviewResponse } from '@/types/routePreview';
import type { PipelineDAG } from '@/types/pipeline';

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
};

export default pipelineApi;
