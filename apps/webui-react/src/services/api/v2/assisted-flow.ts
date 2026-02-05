/**
 * Assisted flow API client.
 * Handles session lifecycle and provides stream URL for SSE.
 */

import apiClient from './client';
import { getApiBaseUrl } from '../baseUrl';
import type {
  StartFlowRequest,
  StartFlowResponse,
} from '../../../types/assisted-flow';

/**
 * Query key factory for assisted flow sessions.
 * Use for consistent React Query cache key generation.
 */
export const assistedFlowKeys = {
  all: ['assisted-flow'] as const,
  sessions: () => [...assistedFlowKeys.all, 'sessions'] as const,
  session: (id: string) => [...assistedFlowKeys.sessions(), id] as const,
};

/** Response from submitting an answer */
export interface SubmitAnswerResponse {
  success: boolean;
}

/**
 * Assisted flow API client.
 */
export const assistedFlowApi = {
  /**
   * Start a new assisted flow session for a source.
   */
  startSession: (data: StartFlowRequest) =>
    apiClient.post<StartFlowResponse>('/api/v2/assisted-flow/start', data),

  /**
   * Get the SSE stream URL for sending messages.
   * The actual streaming is handled by useAssistedFlowStream hook using fetch.
   */
  getStreamUrl: (sessionId: string) =>
    `${getApiBaseUrl()}/api/v2/assisted-flow/${sessionId}/messages/stream`,

  /**
   * Submit an answer to a pending question from AskUserQuestion.
   */
  submitAnswer: (sessionId: string, questionId: string, answers: Record<string, string>) =>
    apiClient.post<SubmitAnswerResponse>(`/api/v2/assisted-flow/${sessionId}/answer`, {
      question_id: questionId,
      answers,
    }),
};

export default assistedFlowApi;
