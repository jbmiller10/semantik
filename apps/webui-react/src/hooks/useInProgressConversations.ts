import type { Conversation } from '../types/agent';

/**
 * Hook to fetch in-progress (active) agent conversations.
 * These are conversations that were started but not completed.
 * Backend uses 'active' status for conversations that are in progress.
 *
 * NOTE: The legacy `/api/v2/agent/*` endpoints were removed as part of the
 * Claude Agent SDK migration. Assisted-flow sessions are currently in-memory
 * only (no persistence/list endpoint), so this hook is effectively disabled.
 */
export function useInProgressConversations() {
  const resumeConversation = (conversationId: string) => {
    // This will be handled by opening the wizard with the conversation ID
    return conversationId;
  };

  return {
    conversations: [] as Conversation[],
    isLoading: false,
    error: null,
    resumeConversation,
    refetch: async () => undefined,
  };
}
