/**
 * React Query hooks for assisted flow session management.
 */

import { useMutation, useQueryClient } from '@tanstack/react-query';
import { assistedFlowApi, assistedFlowKeys } from '../services/api/v2/assisted-flow';
import { useUIStore } from '../stores/uiStore';
import { ApiErrorHandler } from '../utils/api-error-handler';
import type { StartFlowRequest, StartFlowResponse } from '../types/assisted-flow';

/**
 * Hook to start a new assisted flow session.
 */
export function useStartAssistedFlow() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async (data: StartFlowRequest) => {
      const response = await assistedFlowApi.startSession(data);
      return response.data;
    },
    onSuccess: (data) => {
      // Add to cache
      queryClient.setQueryData(assistedFlowKeys.session(data.session_id), data);
    },
    onError: (error) => {
      const errorMessage = ApiErrorHandler.getMessage(error);
      addToast({ type: 'error', message: `Failed to start assisted flow: ${errorMessage}` });
    },
  });
}

/**
 * Type for session state in the UI.
 */
export interface AssistedFlowSessionState {
  sessionId: string;
  sourceName: string;
  isActive: boolean;
}

/**
 * Convert API response to session state.
 */
export function toSessionState(response: StartFlowResponse): AssistedFlowSessionState {
  return {
    sessionId: response.session_id,
    sourceName: response.source_name,
    isActive: true,
  };
}
