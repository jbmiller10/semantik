/**
 * Main agent chat container component.
 * Two-column layout with chat on left and pipeline preview on right.
 */

import { useState, useCallback, useEffect } from 'react';
import { MessageList } from './MessageList';
import { PipelinePreview } from './PipelinePreview';
import {
  useAgentConversation,
  useApplyPipeline,
  useAbandonConversation,
  useUpdateConversationCache,
} from '../../hooks/useAgentConversation';
import { useAgentStream } from '../../hooks/useAgentStream';
import type { Uncertainty, PipelineConfig } from '../../types/agent';

interface AgentChatProps {
  conversationId: string;
  onClose: () => void;
  onApplySuccess?: (collectionId: string) => void;
}

export function AgentChat({
  conversationId,
  onClose,
  onApplySuccess,
}: AgentChatProps) {
  const [inputValue, setInputValue] = useState('');
  const [localPipeline, setLocalPipeline] = useState<PipelineConfig | null>(null);
  const [localUncertainties, setLocalUncertainties] = useState<Uncertainty[]>([]);

  // Conversation data and message management
  const {
    conversation,
    messages,
    isLoading,
    addOptimisticUserMessage,
    syncWithServer,
  } = useAgentConversation(conversationId);

  // Cache update helpers
  const { updatePipeline, addUncertainty } = useUpdateConversationCache();

  // Stream callbacks
  const handleContent = useCallback(
    (_text: string) => {
      // Accumulate content in the streaming assistant message
      // The hook already accumulates, but we need to trigger UI update
    },
    []
  );

  const handlePipelineUpdate = useCallback(
    (data: { pipeline: PipelineConfig }) => {
      setLocalPipeline(data.pipeline);
      updatePipeline(conversationId, data.pipeline);
    },
    [conversationId, updatePipeline]
  );

  const handleUncertainty = useCallback(
    (data: { id: string; severity: 'blocking' | 'notable' | 'info'; message: string; context?: Record<string, unknown> }) => {
      const uncertainty: Uncertainty = {
        id: data.id,
        severity: data.severity,
        message: data.message,
        resolved: false,
        context: data.context,
      };
      setLocalUncertainties((prev) => [...prev, uncertainty]);
      addUncertainty(conversationId, uncertainty);
    },
    [conversationId, addUncertainty]
  );

  const handleDone = useCallback(async () => {
    // Sync with server to get final state
    await syncWithServer();
  }, [syncWithServer]);

  const handleError = useCallback((error: string) => {
    console.error('Stream error:', error);
  }, []);

  // SSE streaming hook
  const {
    isStreaming,
    error: streamError,
    currentContent,
    toolCalls,
    subagents,
    sendMessage,
    cancel,
  } = useAgentStream(conversationId, {
    onContent: handleContent,
    onPipelineUpdate: handlePipelineUpdate,
    onUncertainty: handleUncertainty,
    onDone: handleDone,
    onError: handleError,
  });

  // Mutations
  const applyPipeline = useApplyPipeline();
  const abandonConversation = useAbandonConversation();

  // Sync local state with server data
  useEffect(() => {
    if (conversation) {
      if (conversation.current_pipeline) {
        setLocalPipeline(conversation.current_pipeline);
      }
      if (conversation.uncertainties) {
        setLocalUncertainties(conversation.uncertainties);
      }
    }
  }, [conversation]);

  // Handle sending a message
  const handleSend = useCallback(async () => {
    const message = inputValue.trim();
    if (!message || isStreaming) return;

    // Clear input and add optimistic user message
    setInputValue('');
    addOptimisticUserMessage(message);

    // Start streaming
    await sendMessage(message);
  }, [inputValue, isStreaming, addOptimisticUserMessage, sendMessage]);

  // Handle keyboard submission
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Handle apply pipeline
  const handleApply = useCallback(
    (collectionName: string) => {
      applyPipeline.mutate(
        {
          conversationId,
          data: { collection_name: collectionName },
        },
        {
          onSuccess: (data) => {
            onApplySuccess?.(data.collection_id);
          },
        }
      );
    },
    [conversationId, applyPipeline, onApplySuccess]
  );

  // Handle abandon
  const handleAbandon = useCallback(() => {
    if (confirm('Are you sure you want to abandon this conversation?')) {
      abandonConversation.mutate(conversationId, {
        onSuccess: () => {
          onClose();
        },
      });
    }
  }, [conversationId, abandonConversation, onClose]);

  // Combine uncertainties from server and streaming
  const allUncertainties = [
    ...localUncertainties,
    // Add any new ones from the stream that aren't already tracked
  ];

  // Loading state
  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-2 border-[var(--text-muted)] border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-[var(--text-muted)]">Loading conversation...</p>
        </div>
      </div>
    );
  }

  // Conversation applied or abandoned
  if (conversation?.status !== 'active') {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 bg-green-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg
              className="w-8 h-8 text-green-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 13l4 4L19 7"
              />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-[var(--text-primary)] mb-2">
            {conversation?.status === 'applied'
              ? 'Pipeline Applied'
              : 'Conversation Closed'}
          </h3>
          <p className="text-sm text-[var(--text-secondary)] mb-4">
            {conversation?.status === 'applied'
              ? 'Your collection is being created.'
              : 'This conversation has been abandoned.'}
          </p>
          <button onClick={onClose} className="btn-secondary">
            Close
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)] bg-[var(--bg-secondary)]">
        <div className="flex items-center gap-3">
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
            aria-label="Go back"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M10 19l-7-7m0 0l7-7m-7 7h18"
              />
            </svg>
          </button>
          <div>
            <h2 className="text-lg font-semibold text-[var(--text-primary)]">
              Pipeline Setup
            </h2>
            <p className="text-xs text-[var(--text-muted)]">
              Source #{conversation?.source_id}
            </p>
          </div>
        </div>

        <button
          onClick={handleAbandon}
          disabled={abandonConversation.isPending}
          className="btn-secondary text-red-400 hover:text-red-300 hover:bg-red-500/10"
        >
          {abandonConversation.isPending ? 'Abandoning...' : 'Abandon'}
        </button>
      </div>

      {/* Main content - two columns */}
      <div className="flex-1 flex overflow-hidden">
        {/* Chat section - left ~60% */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Messages */}
          <MessageList
            messages={messages}
            isStreaming={isStreaming}
            currentStreamContent={currentContent}
            toolCalls={toolCalls}
            subagents={subagents}
          />

          {/* Stream error */}
          {streamError && (
            <div className="px-4 py-2 bg-red-500/10 border-t border-red-500/30">
              <p className="text-sm text-red-400">{streamError}</p>
            </div>
          )}

          {/* Input area */}
          <div className="p-4 border-t border-[var(--border)] bg-[var(--bg-secondary)]">
            <div className="flex gap-2">
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type a message..."
                className="input-field flex-1 resize-none min-h-[44px] max-h-[120px]"
                rows={1}
                disabled={isStreaming}
              />
              <button
                onClick={isStreaming ? cancel : handleSend}
                disabled={!isStreaming && !inputValue.trim()}
                className={`btn-primary px-4 ${
                  isStreaming
                    ? 'bg-red-500/20 hover:bg-red-500/30 text-red-400'
                    : ''
                }`}
              >
                {isStreaming ? (
                  <svg
                    className="w-5 h-5"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                ) : (
                  <svg
                    className="w-5 h-5"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                    />
                  </svg>
                )}
              </button>
            </div>
            <p className="text-xs text-[var(--text-muted)] mt-2">
              Press Enter to send, Shift+Enter for new line
            </p>
          </div>
        </div>

        {/* Pipeline preview - right ~40% */}
        <div className="w-[360px] flex-shrink-0">
          <PipelinePreview
            pipeline={localPipeline}
            sourceAnalysis={conversation?.source_analysis || null}
            uncertainties={allUncertainties}
            onApply={handleApply}
            isApplying={applyPipeline.isPending}
            canApply={!!localPipeline}
          />
        </div>
      </div>
    </div>
  );
}

export default AgentChat;
