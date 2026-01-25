/**
 * Message bubble component for agent chat.
 * Displays a single message with role-based styling.
 */

import type { AgentMessage } from '../../types/agent';

interface MessageBubbleProps {
  message: AgentMessage;
  isStreaming?: boolean;
}

export function MessageBubble({ message, isStreaming }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';
  const isTool = message.role === 'tool';
  const isSubagent = message.role === 'subagent';

  // Format timestamp
  const formattedTime = new Date(message.timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });

  // Role label for non-user messages
  const roleLabel = isTool
    ? 'Tool'
    : isSubagent
      ? 'Sub-agent'
      : isAssistant
        ? 'Assistant'
        : null;

  return (
    <div
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}
    >
      <div
        className={`max-w-[80%] rounded-lg px-4 py-3 ${
          isUser
            ? 'bg-[var(--bg-tertiary)] border border-[var(--border)]'
            : isTool || isSubagent
              ? 'bg-blue-500/10 border border-blue-500/30'
              : 'bg-[var(--bg-secondary)] border border-[var(--border-subtle)]'
        }`}
      >
        {/* Role label for non-user messages */}
        {roleLabel && (
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wide">
              {roleLabel}
            </span>
            {isStreaming && (
              <span className="inline-flex items-center">
                <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse" />
              </span>
            )}
          </div>
        )}

        {/* Message content */}
        <div className="text-sm text-[var(--text-primary)] whitespace-pre-wrap break-words">
          {message.content}
          {isStreaming && !message.content && (
            <span className="inline-flex items-center gap-1 text-[var(--text-muted)]">
              <span className="w-1 h-1 bg-current rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <span className="w-1 h-1 bg-current rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <span className="w-1 h-1 bg-current rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </span>
          )}
        </div>

        {/* Timestamp */}
        <div className="mt-1 text-xs text-[var(--text-muted)]">
          {formattedTime}
        </div>
      </div>
    </div>
  );
}

export default MessageBubble;
