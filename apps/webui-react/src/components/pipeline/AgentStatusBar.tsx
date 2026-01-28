import { useState, useCallback, useEffect } from 'react';
import { ChevronDown, ChevronUp, Square } from 'lucide-react';
import { ActivityLog } from './ActivityLog';
import { AgentInput } from './AgentInput';
import type { AgentPhase } from '@/types/agent';

export interface AgentStatusBarProps {
  status: {
    phase: AgentPhase;
    message: string;
    progress?: { current: number; total: number };
  } | null;
  activities: Array<{ message: string; timestamp: string }>;
  isStreaming: boolean;
  onSend: (message: string) => void;
  onStop: () => void;
  /** The agent's text response (markdown) */
  agentResponse?: string;
}

export function AgentStatusBar({
  status,
  activities,
  isStreaming,
  onSend,
  onStop,
  agentResponse,
}: AgentStatusBarProps) {
  const [expanded, setExpanded] = useState(false);
  const [hasAutoExpanded, setHasAutoExpanded] = useState(false);

  // Auto-expand when agent response arrives (first time only)
  useEffect(() => {
    if (agentResponse && !hasAutoExpanded && !isStreaming) {
      setExpanded(true);
      setHasAutoExpanded(true);
    }
  }, [agentResponse, hasAutoExpanded, isStreaming]);

  const toggleExpanded = useCallback(() => {
    setExpanded((prev) => !prev);
  }, []);

  // Determine indicator color based on phase
  const getIndicatorClass = (phase: AgentPhase | undefined) => {
    if (!phase || phase === 'idle') return 'bg-gray-400';
    if (phase === 'ready') return 'bg-green-500';
    return 'bg-amber-500';
  };

  const displayStatus = status ?? { phase: 'idle' as AgentPhase, message: 'Agent ready' };
  const showProgress = displayStatus.progress && displayStatus.phase !== 'idle';

  return (
    <div className="border-t border-[var(--border)] bg-[var(--bg-secondary)]">
      {/* Collapsed bar */}
      <div className="flex items-center justify-between px-4 py-2">
        <div className="flex items-center gap-3">
          {/* Status indicator */}
          <span
            data-testid="status-indicator"
            className={`w-2 h-2 rounded-full ${getIndicatorClass(displayStatus.phase)} ${
              isStreaming ? 'animate-pulse' : ''
            }`}
          />

          {/* Status message */}
          <span className="text-sm text-[var(--text-primary)]">
            {displayStatus.message}
          </span>

          {/* Progress */}
          {showProgress && (
            <span className="text-sm text-[var(--text-muted)]">
              ({displayStatus.progress!.current}/{displayStatus.progress!.total})
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Expand/Collapse button */}
          <button
            onClick={toggleExpanded}
            aria-label={expanded ? 'Collapse' : 'Expand'}
            className="p-1 rounded hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)]
                       focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white"
          >
            {expanded ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </button>

          {/* Stop button */}
          {isStreaming && (
            <button
              onClick={onStop}
              aria-label="Stop"
              className="px-3 py-1 rounded-md text-sm bg-red-500/10 text-red-400
                         border border-red-500/30 hover:bg-red-500/20
                         focus:outline-none focus:ring-2 focus:ring-red-400
                         transition-colors"
            >
              <Square className="w-3 h-3 inline mr-1" />
              Stop
            </button>
          )}
        </div>
      </div>

      {/* Expanded section */}
      {expanded && (
        <div className="border-t border-[var(--border)] px-4 py-3 space-y-3">
          {/* Agent response */}
          {agentResponse && (
            <div>
              <h4 className="text-xs font-medium text-[var(--text-muted)] mb-2 uppercase tracking-wide">
                Agent Response:
              </h4>
              <div
                className="text-sm text-[var(--text-primary)] bg-[var(--bg-tertiary)] rounded-lg p-3 max-h-48 overflow-y-auto whitespace-pre-wrap font-mono"
              >
                {agentResponse}
              </div>
            </div>
          )}

          {/* Activity log */}
          <div>
            <h4 className="text-xs font-medium text-[var(--text-muted)] mb-2 uppercase tracking-wide">
              Activity:
            </h4>
            <ActivityLog activities={activities} maxHeight="120px" />
          </div>

          {/* Agent input */}
          <div className="pt-2 border-t border-[var(--border-subtle)]">
            <AgentInput
              onSend={onSend}
              disabled={isStreaming}
            />
          </div>
        </div>
      )}
    </div>
  );
}
