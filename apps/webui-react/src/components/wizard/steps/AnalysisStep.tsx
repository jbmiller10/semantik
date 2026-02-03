// apps/webui-react/src/components/wizard/steps/AnalysisStep.tsx
import { useState, useCallback, useEffect, useMemo } from 'react';
import { useAssistedFlowStream } from '../../../hooks/useAssistedFlowStream';
import { PipelineVisualization, ConfigurationPanel } from '../../pipeline';
import type { PipelineDAG, DAGSelection, PipelineNode, PipelineEdge } from '../../../types/pipeline';

interface AnalysisStepProps {
  conversationId: string;
  dag: PipelineDAG;
  onDagChange: (dag: PipelineDAG) => void;
  onAgentComplete: () => void;
  onSwitchToManual: () => void;
  onSummaryChange?: (summary: string) => void;
}

export function AnalysisStep({
  conversationId,
  dag,
  onDagChange,
  onAgentComplete,
  onSwitchToManual,
  onSummaryChange,
}: AnalysisStepProps) {
  const [selection, setSelection] = useState<DAGSelection>({ type: 'none' });
  const [hasAutoStarted, setHasAutoStarted] = useState(false);
  const [userMessageInput, setUserMessageInput] = useState('');
  const [isComplete, setIsComplete] = useState(false);

  // Memoize callbacks to avoid re-creating on every render
  const streamCallbacks = useMemo(() => ({
    onDone: () => {
      setIsComplete(true);
      onAgentComplete();
    },
    onError: (errorMsg: string) => {
      console.error('Agent error:', errorMsg);
    },
  }), [onAgentComplete]);

  // Agent stream hook - using new assisted flow hook
  const {
    isStreaming,
    sendMessage,
    currentContent,
    toolCalls,
    error: streamError,
    reset: resetStream,
  } = useAssistedFlowStream(conversationId, streamCallbacks);

  // Auto-start agent analysis
  useEffect(() => {
    if (conversationId && !isStreaming && !hasAutoStarted) {
      setHasAutoStarted(true);
      sendMessage('Analyze my source and recommend a pipeline configuration.');
    }
  }, [conversationId, isStreaming, hasAutoStarted, sendMessage]);

  // Update summary when content changes
  useEffect(() => {
    if (currentContent && onSummaryChange) {
      onSummaryChange(currentContent);
    }
  }, [currentContent, onSummaryChange]);

  // Note: The new SDK-based backend uses tool calls instead of a separate pipeline object.
  // Pipeline updates come through tool_result events from the build_pipeline tool.
  // We best-effort parse those results and update the preview DAG.

  const extractPipelineFromToolResult = useCallback((result: unknown): PipelineDAG | null => {
    // Most commonly: result is a content array like [{ type: 'text', text: '{...json...}' }]
    const tryParseJson = (text: string): unknown => {
      try {
        return JSON.parse(text);
      } catch {
        return null;
      }
    };

    let payload: unknown = null;
    if (typeof result === 'string') {
      payload = tryParseJson(result);
    } else if (Array.isArray(result)) {
      const textBlock = result.find(
        (b) => b && typeof b === 'object' && 'text' in b && typeof (b as { text?: unknown }).text === 'string'
      ) as { text?: string } | undefined;
      if (textBlock?.text) payload = tryParseJson(textBlock.text);
    } else if (result && typeof result === 'object') {
      payload = result;
    }

    if (!payload || typeof payload !== 'object') return null;

    const maybe = payload as { pipeline?: unknown };
    const pipeline = maybe.pipeline && typeof maybe.pipeline === 'object' ? maybe.pipeline : null;
    if (!pipeline) return null;

    const dagCandidate = pipeline as Partial<PipelineDAG>;
    if (
      typeof dagCandidate.id === 'string' &&
      typeof dagCandidate.version === 'string' &&
      Array.isArray(dagCandidate.nodes) &&
      Array.isArray(dagCandidate.edges)
    ) {
      return dagCandidate as PipelineDAG;
    }

    return null;
  }, []);

  useEffect(() => {
    const latest = [...toolCalls]
      .reverse()
      .find((tc) => tc.tool_name === 'build_pipeline' && tc.status === 'success' && tc.result != null);
    if (!latest) return;

    const parsed = extractPipelineFromToolResult(latest.result);
    if (!parsed) return;

    // Avoid spamming updates if nothing changed.
    if (parsed.id === dag.id && parsed.version === dag.version && parsed.nodes.length === dag.nodes.length && parsed.edges.length === dag.edges.length) {
      return;
    }

    onDagChange(parsed);
  }, [toolCalls, extractPipelineFromToolResult, onDagChange, dag]);

  const handleNodeChange = useCallback((updatedNode: PipelineNode) => {
    onDagChange({
      ...dag,
      nodes: dag.nodes.map((n) => (n.id === updatedNode.id ? updatedNode : n)),
    });
  }, [dag, onDagChange]);

  const handleEdgeChange = useCallback((updatedEdge: PipelineEdge) => {
    onDagChange({
      ...dag,
      edges: dag.edges.map((e) =>
        e.from_node === updatedEdge.from_node && e.to_node === updatedEdge.to_node
          ? updatedEdge
          : e
      ),
    });
  }, [dag, onDagChange]);

  // Get status display based on streaming state and completion
  const getStatusDisplay = () => {
    if (streamError) {
      return { text: 'Analysis failed', color: 'red' };
    }
    if (isComplete) {
      return { text: 'Analysis complete', color: 'green' };
    }
    if (isStreaming) {
      // Show current tool activity if any
      const runningTool = toolCalls.find(tc => tc.status === 'running');
      if (runningTool) {
        return { text: `Running ${runningTool.tool_name}...`, color: 'amber' };
      }
      return { text: 'Analyzing...', color: 'amber' };
    }
    return { text: 'Ready', color: 'gray' };
  };

  const statusDisplay = getStatusDisplay();

  return (
    <div className="h-full flex flex-col">
      {/* Skip to manual banner */}
      <div className="px-4 py-2 bg-[var(--bg-tertiary)] border-b border-[var(--border)] flex items-center justify-between shrink-0">
        <span className="text-sm text-[var(--text-muted)]">
          The AI is analyzing your source to recommend optimal settings.
        </span>
        <button
          onClick={onSwitchToManual}
          className="text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] underline"
        >
          Skip to Manual
        </button>
      </div>

      {/* Two-column layout */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
        {/* Left: Agent Conversation (~40%) */}
        <div
          data-testid="agent-column"
          className="flex-1 lg:flex-[2] border-b lg:border-b-0 lg:border-r border-[var(--border)] flex flex-col min-h-[200px] lg:min-h-0"
        >
          {/* Status header */}
          <div className="px-4 py-3 border-b border-[var(--border)] bg-[var(--bg-secondary)] shrink-0">
            <div className="flex items-center gap-2">
              <div className={`
                w-2 h-2 rounded-full
                ${statusDisplay.color === 'amber' ? 'bg-amber-400 animate-pulse' : ''}
                ${statusDisplay.color === 'green' ? 'bg-green-400' : ''}
                ${statusDisplay.color === 'gray' ? 'bg-gray-400' : ''}
                ${statusDisplay.color === 'red' ? 'bg-red-400' : ''}
              `} />
              <span className="text-sm font-medium text-[var(--text-primary)]">
                {statusDisplay.text}
              </span>
              {toolCalls.length > 0 && (
                <span className="text-xs text-[var(--text-muted)]">
                  ({toolCalls.filter(tc => tc.status === 'success').length}/{toolCalls.length} tools)
                </span>
              )}
            </div>
          </div>

          {/* Agent response / activities */}
          <div className="flex-1 overflow-y-auto p-4">
            {/* Tool calls list */}
            {toolCalls.length > 0 && (
              <div className="space-y-2 mb-4">
                {toolCalls.map((toolCall) => (
                  <div key={toolCall.id} className="flex items-start gap-2 text-sm">
                    <span className={`
                      ${toolCall.status === 'running' ? 'text-amber-400' : ''}
                      ${toolCall.status === 'success' ? 'text-green-400' : ''}
                      ${toolCall.status === 'error' ? 'text-red-400' : ''}
                    `}>
                      {toolCall.status === 'running' ? '⟳' : toolCall.status === 'success' ? '✓' : '✗'}
                    </span>
                    <span className="text-[var(--text-secondary)]">{toolCall.tool_name}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Agent response text */}
            {currentContent && (
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <div className="whitespace-pre-wrap text-sm text-[var(--text-primary)]">
                  {currentContent}
                </div>
              </div>
            )}

            {/* Error state */}
            {streamError && (
              <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <svg className="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  <span className="text-sm font-medium text-red-400">Analysis failed</span>
                </div>
                <p className="text-sm text-red-300 mb-3">{streamError}</p>
                <button
                  onClick={() => {
                    resetStream();
                    setHasAutoStarted(false);
                  }}
                  className="text-sm text-red-400 hover:text-red-300 underline"
                >
                  Try again
                </button>
              </div>
            )}

            {/* Empty state */}
            {!currentContent && toolCalls.length === 0 && !streamError && (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[var(--text-muted)] mx-auto mb-4" />
                <p className="text-sm text-[var(--text-muted)]">Analyzing your source...</p>
              </div>
            )}
          </div>

          {/* User input (when not streaming and complete) */}
          {!isStreaming && isComplete && (
            <div className="border-t border-[var(--border)] p-4 shrink-0">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={userMessageInput}
                  onChange={(e) => setUserMessageInput(e.target.value)}
                  placeholder="Ask the agent something..."
                  className="flex-1 px-3 py-2 text-sm rounded-lg border border-[var(--border)] bg-[var(--bg-primary)]"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && userMessageInput.trim()) {
                      sendMessage(userMessageInput);
                      setUserMessageInput('');
                    }
                  }}
                />
                <button
                  onClick={() => {
                    if (userMessageInput.trim()) {
                      sendMessage(userMessageInput);
                      setUserMessageInput('');
                    }
                  }}
                  className="px-4 py-2 text-sm rounded-lg bg-gray-200 dark:bg-white text-gray-900 font-medium"
                >
                  Send
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Right: Pipeline Preview (~60%) */}
        <div
          data-testid="preview-column"
          className="flex-1 lg:flex-[3] flex flex-col overflow-hidden"
        >
          {/* Preview header */}
          <div className="px-4 py-3 border-b border-[var(--border)] bg-[var(--bg-secondary)] shrink-0">
            <span className="text-sm font-medium text-[var(--text-primary)]">
              Pipeline Preview
            </span>
            {dag.nodes.length > 0 && (
              <span className="ml-2 text-xs text-[var(--text-muted)]">
                {dag.nodes.length} nodes configured
              </span>
            )}
          </div>

          {/* Pipeline visualization */}
          <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
            {/* DAG visualization */}
            <div className="flex-1 lg:flex-[3] border-b lg:border-b-0 lg:border-r border-[var(--border)] overflow-auto p-4 min-h-[200px] lg:min-h-0">
              {dag.nodes.length > 0 ? (
                <PipelineVisualization
                  dag={dag}
                  selection={selection}
                  onSelectionChange={setSelection}
                />
              ) : (
                <div className="h-full flex items-center justify-center">
                  <div className="text-center">
                    <div className="w-12 h-12 rounded-full bg-[var(--bg-tertiary)] flex items-center justify-center mx-auto mb-4">
                      <svg className="w-6 h-6 text-[var(--text-muted)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                      </svg>
                    </div>
                    <p className="text-sm text-[var(--text-muted)]">
                      Pipeline will appear here as the agent makes recommendations
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Configuration panel */}
            <div className="flex-1 lg:flex-[2] overflow-auto">
              <ConfigurationPanel
                dag={dag}
                selection={selection}
                sourceAnalysis={null}
                onNodeChange={handleNodeChange}
                onEdgeChange={handleEdgeChange}
                readOnly={isStreaming}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
