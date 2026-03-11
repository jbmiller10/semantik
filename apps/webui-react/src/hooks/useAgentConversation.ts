/**
 * React Query hooks for agent conversation CRUD operations.
 *
 * @deprecated These hooks are for the legacy agent orchestrator which has been
 * replaced by the Claude Agent SDK-based assisted-flow API. New code should use
 * hooks from `useAssistedFlow.ts` instead.
 *
 * @see ./useAssistedFlow.ts for the new SDK-based hooks
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useState, useCallback } from 'react';
import { agentApi, agentKeys } from '../services/api/v2/agent';
import { useUIStore } from '../stores/uiStore';
import { ApiErrorHandler } from '../utils/api-error-handler';
import type {
  ConversationDetail,
  AgentMessage,
  CreateConversationRequest,
  ApplyPipelineRequest,
  Uncertainty,
  PipelineConfig,
} from '../types/agent';

/**
 * Hook to fetch and manage a single agent conversation.
 * Includes optimistic message management for streaming.
 */
export function useAgentConversation(conversationId: string) {
  const queryClient = useQueryClient();
  const [optimisticMessages, setOptimisticMessages] = useState<AgentMessage[]>([]);

  // Fetch conversation detail
  const query = useQuery({
    queryKey: agentKeys.detail(conversationId),
    queryFn: async () => {
      const response = await agentApi.getConversation(conversationId);
      return response.data;
    },
    enabled: !!conversationId,
    staleTime: 5000,
  });

  // Add an optimistic user message (before streaming)
  const addOptimisticUserMessage = useCallback((content: string) => {
    const message: AgentMessage = {
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    };
    setOptimisticMessages((prev) => [...prev, message]);
  }, []);

  // Add an optimistic assistant message (during/after streaming)
  const addOptimisticAssistantMessage = useCallback((content: string) => {
    const message: AgentMessage = {
      role: 'assistant',
      content,
      timestamp: new Date().toISOString(),
    };
    setOptimisticMessages((prev) => [...prev, message]);
  }, []);

  // Update the last assistant message (for streaming updates)
  const updateLastAssistantMessage = useCallback((content: string) => {
    setOptimisticMessages((prev) => {
      // Find the last assistant message index
      let lastIndex = -1;
      for (let i = prev.length - 1; i >= 0; i--) {
        if (prev[i].role === 'assistant') {
          lastIndex = i;
          break;
        }
      }
      if (lastIndex === -1) {
        // No assistant message, add one
        return [
          ...prev,
          {
            role: 'assistant' as const,
            content,
            timestamp: new Date().toISOString(),
          },
        ];
      }
      // Update existing
      return prev.map((m, i) => (i === lastIndex ? { ...m, content } : m));
    });
  }, []);

  // Sync with server (clear optimistic state and refetch)
  const syncWithServer = useCallback(async () => {
    setOptimisticMessages([]);
    await queryClient.invalidateQueries({ queryKey: agentKeys.detail(conversationId) });
  }, [queryClient, conversationId]);

  // Clear optimistic messages
  const clearOptimisticMessages = useCallback(() => {
    setOptimisticMessages([]);
  }, []);

  // Combine server messages with optimistic messages
  const allMessages = [
    ...(query.data?.messages || []),
    ...optimisticMessages,
  ];

  return {
    conversation: query.data,
    messages: allMessages,
    isLoading: query.isLoading,
    error: query.error,
    refetch: query.refetch,
    addOptimisticUserMessage,
    addOptimisticAssistantMessage,
    updateLastAssistantMessage,
    clearOptimisticMessages,
    syncWithServer,
  };
}

/**
 * Hook to list all agent conversations.
 */
export function useConversations(filters?: { status?: string }) {
  return useQuery({
    queryKey: agentKeys.list(filters),
    queryFn: async () => {
      const response = await agentApi.listConversations(filters);
      return response.data;
    },
    staleTime: 30000,
  });
}

/**
 * Hook to create a new agent conversation.
 */
export function useCreateConversation() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async (data: CreateConversationRequest) => {
      const response = await agentApi.createConversation(data);
      return response.data;
    },
    onSuccess: (data) => {
      // Add to cache
      queryClient.setQueryData(agentKeys.detail(data.id), data);
      // Invalidate list
      queryClient.invalidateQueries({ queryKey: agentKeys.lists() });
    },
    onError: (error) => {
      const errorMessage = ApiErrorHandler.getMessage(error);
      addToast({ type: 'error', message: `Failed to create conversation: ${errorMessage}` });
    },
  });
}

/**
 * Hook to apply a pipeline configuration.
 */
export function useApplyPipeline() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async ({
      conversationId,
      data,
    }: {
      conversationId: string;
      data: ApplyPipelineRequest;
    }) => {
      const response = await agentApi.applyPipeline(conversationId, data);
      return response.data;
    },
    onSuccess: (data, { conversationId }) => {
      // Update conversation status in cache
      queryClient.setQueryData<ConversationDetail>(
        agentKeys.detail(conversationId),
        (old) =>
          old
            ? {
                ...old,
                status: 'applied',
                collection_id: data.collection_id,
              }
            : old
      );
      // Invalidate collections list to show new collection
      queryClient.invalidateQueries({ queryKey: ['collections'] });

      addToast({
        type: 'success',
        message: `Collection "${data.collection_name}" created successfully`,
      });
    },
    onError: (error) => {
      const errorMessage = ApiErrorHandler.getMessage(error);
      addToast({ type: 'error', message: `Failed to apply pipeline: ${errorMessage}` });
    },
  });
}

/**
 * Hook to abandon a conversation.
 */
export function useAbandonConversation() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async (conversationId: string) => {
      const response = await agentApi.abandonConversation(conversationId);
      return response.data;
    },
    onSuccess: (_data, conversationId) => {
      // Update conversation status in cache
      queryClient.setQueryData<ConversationDetail>(
        agentKeys.detail(conversationId),
        (old) => (old ? { ...old, status: 'abandoned' } : old)
      );
      // Invalidate list
      queryClient.invalidateQueries({ queryKey: agentKeys.lists() });

      addToast({
        type: 'info',
        message: 'Conversation abandoned',
      });
    },
    onError: (error) => {
      const errorMessage = ApiErrorHandler.getMessage(error);
      addToast({ type: 'error', message: `Failed to abandon conversation: ${errorMessage}` });
    },
  });
}

/**
 * Simple hook to fetch conversation detail.
 * Wrapper around React Query for cleaner imports.
 */
export function useConversationDetail(conversationId: string) {
  return useQuery({
    queryKey: agentKeys.detail(conversationId),
    queryFn: async () => {
      const response = await agentApi.getConversation(conversationId);
      return response.data;
    },
    enabled: !!conversationId,
    staleTime: 5000,
  });
}

/**
 * Helper hook to update pipeline and uncertainties in cache.
 * Used by components that receive stream updates.
 */
export function useUpdateConversationCache() {
  const queryClient = useQueryClient();

  const updatePipeline = useCallback(
    (conversationId: string, pipeline: PipelineConfig) => {
      queryClient.setQueryData<ConversationDetail>(
        agentKeys.detail(conversationId),
        (old) => (old ? { ...old, current_pipeline: pipeline } : old)
      );
    },
    [queryClient]
  );

  const addUncertainty = useCallback(
    (conversationId: string, uncertainty: Uncertainty) => {
      queryClient.setQueryData<ConversationDetail>(
        agentKeys.detail(conversationId),
        (old) =>
          old
            ? {
                ...old,
                uncertainties: [...old.uncertainties, uncertainty],
              }
            : old
      );
    },
    [queryClient]
  );

  return { updatePipeline, addUncertainty };
}
