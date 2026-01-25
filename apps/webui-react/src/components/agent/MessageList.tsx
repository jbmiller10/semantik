/**
 * Message list component for agent chat.
 * Displays messages with auto-scroll and streaming indicators.
 */

import { useEffect, useRef } from 'react';
import { MessageBubble } from './MessageBubble';
import type { AgentMessage, ToolCallState, SubagentState } from '../../types/agent';

interface MessageListProps {
  messages: AgentMessage[];
  isStreaming: boolean;
  currentStreamContent: string;
  toolCalls: ToolCallState[];
  subagents: SubagentState[];
}

export function MessageList({
  messages,
  isStreaming,
  currentStreamContent,
  toolCalls,
  subagents,
}: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages or streaming content
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, currentStreamContent, toolCalls, subagents]);

  // Active (running) tool calls and subagents
  const activeToolCalls = toolCalls.filter((tc) => tc.status === 'running');
  const activeSubagents = subagents.filter((sa) => sa.status === 'running');

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-2">
      {/* Empty state */}
      {messages.length === 0 && !isStreaming && (
        <div className="flex flex-col items-center justify-center h-full text-center py-12">
          <div className="w-16 h-16 bg-[var(--bg-tertiary)] rounded-full flex items-center justify-center mb-4 border border-[var(--border)]">
            <svg
              className="w-8 h-8 text-[var(--text-muted)]"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
              />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-[var(--text-primary)] mb-2">
            Start a conversation
          </h3>
          <p className="text-sm text-[var(--text-secondary)] max-w-sm">
            Ask the assistant to help you configure the best pipeline settings for your documents.
          </p>
        </div>
      )}

      {/* Message bubbles */}
      {messages.map((message, index) => (
        <MessageBubble
          key={`${message.timestamp}-${index}`}
          message={message}
          isStreaming={false}
        />
      ))}

      {/* Streaming assistant message */}
      {isStreaming && currentStreamContent && (
        <MessageBubble
          message={{
            role: 'assistant',
            content: currentStreamContent,
            timestamp: new Date().toISOString(),
          }}
          isStreaming={true}
        />
      )}

      {/* Active tool calls indicator */}
      {activeToolCalls.length > 0 && (
        <div className="flex justify-start mb-4">
          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg px-4 py-3 max-w-[80%]">
            <div className="flex items-center gap-2 text-sm text-blue-400">
              <svg
                className="w-4 h-4 animate-spin"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                />
              </svg>
              <span>
                Running: {activeToolCalls.map((tc) => tc.tool).join(', ')}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Active subagents indicator */}
      {activeSubagents.length > 0 && (
        <div className="flex justify-start mb-4">
          <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg px-4 py-3 max-w-[80%]">
            <div className="flex items-center gap-2 text-sm text-amber-400">
              <svg
                className="w-4 h-4 animate-spin"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                />
              </svg>
              <span>
                Sub-agent: {activeSubagents.map((sa) => sa.name).join(', ')}
              </span>
            </div>
            {activeSubagents[0]?.task && (
              <p className="text-xs text-amber-300/80 mt-1">
                {activeSubagents[0].task}
              </p>
            )}
          </div>
        </div>
      )}

      {/* Scroll anchor */}
      <div ref={messagesEndRef} />
    </div>
  );
}

export default MessageList;
