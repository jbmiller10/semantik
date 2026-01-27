// apps/webui-react/src/components/wizard/steps/AnalysisStep.tsx
import { useState, useCallback, useEffect, useMemo } from 'react';
import { useAgentStream } from '../../../hooks/useAgentStream';
import { agentApi } from '../../../services/api/v2/agent';
import { PipelineVisualization, ConfigurationPanel } from '../../pipeline';
import type { PipelineDAG, DAGSelection, PipelineNode, PipelineEdge } from '../../../types/pipeline';
import type { AgentPhase } from '../../../types/agent';

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

  // Memoize callbacks to avoid re-creating on every render
  const streamCallbacks = useMemo(() => ({
    onDone: () => {
      onAgentComplete();
    },
    onError: (errorMsg: string) => {
      console.error('Agent error:', errorMsg);
    },
  }), [onAgentComplete]);

  // Agent stream hook
  const {
    status,
    activities,
    pendingQuestions,
    dismissQuestion,
    isStreaming,
    sendMessage,
    currentContent,
    pipeline,
  } = useAgentStream(conversationId, streamCallbacks);

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

  // Update DAG when pipeline changes from agent
  useEffect(() => {
    if (pipeline) {
      // Convert pipeline config to DAG format
      // This is a simplified conversion - the real one should be more comprehensive
      const newDag: PipelineDAG = {
        id: 'agent-recommended',
        version: '1',
        nodes: [
          { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
          {
            id: 'chunker1',
            type: 'chunker',
            plugin_id: (pipeline as Record<string, unknown>).chunking_strategy as string || 'semantic',
            config: (pipeline as Record<string, unknown>).chunking_config as Record<string, unknown> || {},
          },
          {
            id: 'embedder1',
            type: 'embedder',
            plugin_id: (pipeline as Record<string, unknown>).embedding_model as string || 'default',
            config: {},
          },
        ],
        edges: [
          { from_node: '_source', to_node: 'parser1', when: null },
          { from_node: 'parser1', to_node: 'chunker1', when: null },
          { from_node: 'chunker1', to_node: 'embedder1', when: null },
        ],
      };
      onDagChange(newDag);
    }
  }, [pipeline, onDagChange]);

  // Handle question answer
  const handleAnswer = useCallback(async (
    questionId: string,
    optionId?: string,
    customResponse?: string
  ) => {
    try {
      await agentApi.answerQuestion(conversationId, questionId, optionId, customResponse);
      dismissQuestion(questionId);
    } catch (err) {
      console.error('Failed to submit answer:', err);
    }
  }, [conversationId, dismissQuestion]);

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

  // Get status display
  const getStatusDisplay = () => {
    if (!status) return { text: 'Analyzing...', color: 'amber' };

    const phaseLabels: Record<AgentPhase, string> = {
      idle: 'Ready',
      analyzing: 'Analyzing source...',
      sampling: 'Sampling content...',
      building: 'Building pipeline...',
      validating: 'Validating configuration...',
      ready: 'Analysis complete',
    };

    const colors: Record<AgentPhase, string> = {
      idle: 'gray',
      analyzing: 'amber',
      sampling: 'amber',
      building: 'amber',
      validating: 'amber',
      ready: 'green',
    };

    return {
      text: status.message || phaseLabels[status.phase] || 'Working...',
      color: colors[status.phase] || 'amber',
    };
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
              `} />
              <span className="text-sm font-medium text-[var(--text-primary)]">
                {statusDisplay.text}
              </span>
              {status?.progress && (
                <span className="text-xs text-[var(--text-muted)]">
                  ({status.progress.current}/{status.progress.total})
                </span>
              )}
            </div>
          </div>

          {/* Agent response / activities */}
          <div className="flex-1 overflow-y-auto p-4">
            {/* Activities list */}
            {activities.length > 0 && (
              <div className="space-y-2 mb-4">
                {activities.map((activity, index) => (
                  <div key={index} className="flex items-start gap-2 text-sm">
                    <span className="text-[var(--text-muted)]">-</span>
                    <span className="text-[var(--text-secondary)]">{activity.message}</span>
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

            {/* Empty state */}
            {!currentContent && activities.length === 0 && (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[var(--text-muted)] mx-auto mb-4" />
                <p className="text-sm text-[var(--text-muted)]">Analyzing your source...</p>
              </div>
            )}
          </div>

          {/* Pending questions */}
          {pendingQuestions.length > 0 && (
            <div className="border-t border-[var(--border)] p-4 bg-[var(--bg-secondary)] shrink-0">
              {pendingQuestions.map((question) => (
                <div key={question.id} className="space-y-3">
                  <p className="text-sm font-medium text-[var(--text-primary)]">
                    {question.message}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {question.options.map((option) => (
                      <button
                        key={option.id}
                        onClick={() => handleAnswer(question.id, option.id)}
                        className="px-3 py-1.5 text-sm rounded-lg border border-[var(--border)] hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)]"
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                  {question.allowCustom && (
                    <input
                      type="text"
                      placeholder="Or type a custom response..."
                      className="w-full px-3 py-2 text-sm rounded-lg border border-[var(--border)] bg-[var(--bg-primary)]"
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          handleAnswer(question.id, undefined, (e.target as HTMLInputElement).value);
                        }
                      }}
                    />
                  )}
                </div>
              ))}
            </div>
          )}

          {/* User input (when not streaming and no pending questions) */}
          {!isStreaming && pendingQuestions.length === 0 && status?.phase === 'ready' && (
            <div className="border-t border-[var(--border)] p-4 shrink-0">
              <div className="flex gap-2">
                <input
                  type="text"
                  placeholder="Ask the agent something..."
                  className="flex-1 px-3 py-2 text-sm rounded-lg border border-[var(--border)] bg-[var(--bg-primary)]"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      sendMessage((e.target as HTMLInputElement).value);
                      (e.target as HTMLInputElement).value = '';
                    }
                  }}
                />
                <button
                  onClick={() => {
                    const input = document.querySelector('input[placeholder="Ask the agent something..."]') as HTMLInputElement;
                    if (input?.value) {
                      sendMessage(input.value);
                      input.value = '';
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
