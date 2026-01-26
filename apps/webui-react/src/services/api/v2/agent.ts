/**
 * Agent conversation API client.
 * Handles conversation CRUD and provides stream URL for SSE.
 */

import apiClient from './client';
import { getApiBaseUrl } from '../baseUrl';
import type {
  Conversation,
  ConversationDetail,
  ConversationListResponse,
  CreateConversationRequest,
  ApplyPipelineRequest,
  ApplyPipelineResponse,
} from '../../../types/agent';

/**
 * Query key factory for agent conversations.
 * Use for consistent React Query cache key generation.
 */
export const agentKeys = {
  all: ['agent-conversations'] as const,
  lists: () => [...agentKeys.all, 'list'] as const,
  list: (filters?: { status?: string }) => [...agentKeys.lists(), filters] as const,
  details: () => [...agentKeys.all, 'detail'] as const,
  detail: (id: string) => [...agentKeys.details(), id] as const,
};

/**
 * Agent conversation API client.
 */
export const agentApi = {
  /**
   * Create a new agent conversation for a source.
   */
  createConversation: (data: CreateConversationRequest) =>
    apiClient.post<ConversationDetail>('/api/v2/agent/conversations', data),

  /**
   * Get conversation details by ID.
   */
  getConversation: (id: string) =>
    apiClient.get<ConversationDetail>(`/api/v2/agent/conversations/${id}`),

  /**
   * List all conversations for the current user.
   */
  listConversations: (params?: { status?: string; offset?: number; limit?: number }) =>
    apiClient.get<ConversationListResponse>('/api/v2/agent/conversations', { params }),

  /**
   * Apply the configured pipeline to create a collection.
   */
  applyPipeline: (id: string, data: ApplyPipelineRequest) =>
    apiClient.post<ApplyPipelineResponse>(`/api/v2/agent/conversations/${id}/apply`, data),

  /**
   * Abandon a conversation.
   */
  abandonConversation: (id: string) =>
    apiClient.patch<Conversation>(`/api/v2/agent/conversations/${id}/status`, {
      status: 'abandoned',
    }),

  /**
   * Answer a question from the agent.
   */
  answerQuestion: async (
    conversationId: string,
    questionId: string,
    optionId?: string,
    customResponse?: string
  ): Promise<{ success: boolean; message?: string }> => {
    const response = await apiClient.post(
      `/api/v2/agent/conversations/${conversationId}/answer`,
      {
        question_id: questionId,
        option_id: optionId,
        custom_response: customResponse,
      }
    );
    return response.data;
  },

  /**
   * Get the SSE stream URL for sending messages.
   * The actual streaming is handled by useAgentStream hook using fetch.
   */
  getStreamUrl: (conversationId: string) =>
    `${getApiBaseUrl()}/api/v2/agent/conversations/${conversationId}/messages/stream`,

  /**
   * Pause agent processing (switch to manual mode).
   */
  pauseAgent: async (
    conversationId: string
  ): Promise<{ success: boolean; is_paused: boolean; message?: string }> => {
    const response = await apiClient.post(
      `/api/v2/agent/conversations/${conversationId}/pause`
    );
    return response.data;
  },

  /**
   * Resume agent processing (switch to assisted mode).
   */
  resumeAgent: async (
    conversationId: string
  ): Promise<{ success: boolean; is_paused: boolean; message?: string }> => {
    const response = await apiClient.post(
      `/api/v2/agent/conversations/${conversationId}/resume`
    );
    return response.data;
  },

  /**
   * Send a message to the agent (non-streaming).
   */
  sendMessage: async (
    conversationId: string,
    message: string
  ): Promise<{
    response: string;
    pipeline_updated: boolean;
    uncertainties_added: Array<{
      id: string;
      severity: string;
      message: string;
      resolved: boolean;
      context?: Record<string, unknown>;
    }>;
    tool_calls: Array<{ name: string; arguments: Record<string, unknown> }>;
  }> => {
    const response = await apiClient.post(
      `/api/v2/agent/conversations/${conversationId}/messages`,
      { message }
    );
    return response.data;
  },
};

// Alias for consistency with other v2 API modules
export { agentApi as agentApiV2 };
export default agentApi;
