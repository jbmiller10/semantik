import { useQuery } from '@tanstack/react-query';
import { agentApi, agentKeys } from '../services/api/v2/agent';
import type { Conversation } from '../types/agent';

/**
 * Hook to fetch in-progress agent conversations.
 * These are conversations that were started but not completed.
 */
export function useInProgressConversations() {
  const query = useQuery({
    queryKey: agentKeys.list({ status: 'in_progress' }),
    queryFn: async () => {
      const response = await agentApi.listConversations({ status: 'in_progress' });
      return response.data.conversations;
    },
    staleTime: 30000, // 30 seconds
  });

  const resumeConversation = (conversationId: string) => {
    // This will be handled by opening the wizard with the conversation ID
    return conversationId;
  };

  return {
    conversations: query.data || [] as Conversation[],
    isLoading: query.isLoading,
    error: query.error,
    resumeConversation,
    refetch: query.refetch,
  };
}
