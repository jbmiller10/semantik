/**
 * Hook for managing pipeline route preview state and operations.
 */

import { useState, useCallback } from 'react';
import { pipelineApi } from '@/services/api/v2/pipeline';
import type { RoutePreviewResponse, RoutePreviewState } from '@/types/routePreview';
import type { PipelineDAG } from '@/types/pipeline';

/**
 * Hook return type.
 */
export interface UseRoutePreviewReturn extends RoutePreviewState {
  /** Preview a file through the pipeline DAG */
  previewFile: (file: File, dag: PipelineDAG, includeParserMetadata?: boolean) => Promise<void>;
  /** Clear the preview result */
  clearPreview: () => void;
}

/**
 * Hook for managing pipeline route preview.
 *
 * @returns Preview state and actions
 *
 * @example
 * ```tsx
 * const { isLoading, error, result, previewFile, clearPreview } = useRoutePreview();
 *
 * const handleFileSelect = async (file: File) => {
 *   await previewFile(file, dag);
 * };
 * ```
 */
export function useRoutePreview(): UseRoutePreviewReturn {
  const [state, setState] = useState<RoutePreviewState>({
    isLoading: false,
    error: null,
    result: null,
    file: null,
  });

  const previewFile = useCallback(
    async (file: File, dag: PipelineDAG, includeParserMetadata: boolean = true) => {
      setState((prev) => ({
        ...prev,
        isLoading: true,
        error: null,
        file,
      }));

      try {
        const result: RoutePreviewResponse = await pipelineApi.previewRoute(
          file,
          dag,
          includeParserMetadata
        );

        setState((prev) => ({
          ...prev,
          isLoading: false,
          result,
        }));
      } catch (err) {
        // Log full error for debugging
        console.error('Route preview failed:', err);

        // Extract user-friendly error message with fallback chain
        let errorMessage = 'Preview failed';
        if (err instanceof Error) {
          errorMessage = err.message;
        } else if (typeof err === 'object' && err !== null) {
          // Handle axios error structure
          const axiosErr = err as {
            response?: { data?: { detail?: string; message?: string }; status?: number };
            message?: string;
          };
          if (axiosErr.response?.data?.detail) {
            errorMessage = axiosErr.response.data.detail;
          } else if (axiosErr.response?.data?.message) {
            errorMessage = axiosErr.response.data.message;
          } else if (axiosErr.response?.status) {
            errorMessage = `Request failed with status ${axiosErr.response.status}`;
          } else if (axiosErr.message) {
            errorMessage = axiosErr.message;
          }
        }

        setState((prev) => ({
          ...prev,
          isLoading: false,
          error: errorMessage,
        }));
      }
    },
    []
  );

  const clearPreview = useCallback(() => {
    setState({
      isLoading: false,
      error: null,
      result: null,
      file: null,
    });
  }, []);

  return {
    ...state,
    previewFile,
    clearPreview,
  };
}

export default useRoutePreview;
