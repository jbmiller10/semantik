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
};

export default agentApi;
