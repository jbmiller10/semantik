/**
 * Hook for managing pipeline route preview state and operations.
 */

import { useState, useCallback } from 'react';
import { pipelineApi } from '@/services/api/v2/pipeline';
import type { RoutePreviewResponse, RoutePreviewState } from '@/types/routePreview';
import { INITIAL_ROUTE_PREVIEW_STATE } from '@/types/routePreview';
import type { PipelineDAG } from '@/types/pipeline';

/**
 * Hook return type.
 */
export interface UseRoutePreviewReturn {
  /** Current status of the preview */
  status: RoutePreviewState['status'];
  /** Whether a preview is in progress */
  isLoading: boolean;
  /** Error message if preview failed */
  error: string | null;
  /** Preview result if successful */
  result: RoutePreviewResponse | null;
  /** The file that was previewed */
  file: File | null;
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
  const [state, setState] = useState<RoutePreviewState>(INITIAL_ROUTE_PREVIEW_STATE);

  const previewFile = useCallback(
    async (file: File, dag: PipelineDAG, includeParserMetadata: boolean = true) => {
      setState({
        status: 'loading',
        file,
        error: null,
        result: null,
      });

      try {
        const result: RoutePreviewResponse = await pipelineApi.previewRoute(
          file,
          dag,
          includeParserMetadata
        );

        setState({
          status: 'success',
          file,
          error: null,
          result,
        });
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

        setState({
          status: 'error',
          file,
          error: errorMessage,
          result: null,
        });
      }
    },
    []
  );

  const clearPreview = useCallback(() => {
    setState(INITIAL_ROUTE_PREVIEW_STATE);
  }, []);

  return {
    status: state.status,
    isLoading: state.status === 'loading',
    error: state.error,
    result: state.result,
    file: state.file,
    previewFile,
    clearPreview,
  };
}

export default useRoutePreview;
