// apps/webui-react/src/pages/PipelineBuilderPage.tsx
import { useState, useCallback, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useConversationDetail, useApplyPipeline } from '../hooks/useAgentConversation';
import { useAgentStream } from '../hooks/useAgentStream';
import { agentApiV2 } from '../services/api/v2/agent';
import { useUIStore } from '../stores/uiStore';
import {
  PipelineVisualization,
  ConfigurationPanel,
  AgentStatusBar,
  QuestionQueue,
} from '../components/pipeline';
import { PipelineBuilderHeader } from '../components/pipeline/PipelineBuilderHeader';
import { PipelineBuilderFooter } from '../components/pipeline/PipelineBuilderFooter';
import type { PipelineDAG, DAGSelection, PipelineNode, PipelineEdge } from '../types/pipeline';
import type { BuilderMode } from '../components/pipeline/PipelineBuilderHeader';

// Convert API pipeline config to DAG format
function pipelineConfigToDAG(config: Record<string, unknown> | null): PipelineDAG {
  // Default empty DAG
  if (!config) {
    return {
      id: 'default',
      version: '1',
      nodes: [],
      edges: [],
    };
  }

  // TODO: Convert actual pipeline config to DAG nodes/edges
  // For now, return a simple default structure
  return {
    id: 'pipeline',
    version: '1',
    nodes: [
      { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
      { id: 'chunker1', type: 'chunker', plugin_id: (config.chunking_strategy as string) || 'semantic', config: (config.chunking_config as Record<string, unknown>) || {} },
      { id: 'embedder1', type: 'embedder', plugin_id: (config.embedding_model as string) || 'default', config: {} },
    ],
    edges: [
      { from_node: '_source', to_node: 'parser1', when: null },
      { from_node: 'parser1', to_node: 'chunker1', when: null },
      { from_node: 'chunker1', to_node: 'embedder1', when: null },
    ],
  };
}

export function PipelineBuilderPage() {
  const { conversationId } = useParams<{ conversationId: string }>();
  const navigate = useNavigate();
  const { addToast } = useUIStore();

  // Fetch conversation details
  const { data: conversation, isLoading, error } = useConversationDetail(conversationId || '');
  const applyMutation = useApplyPipeline();

  // Local state
  const [mode, setMode] = useState<BuilderMode>('assisted');
  const [selection, setSelection] = useState<DAGSelection>({ type: 'none' });
  const [dag, setDag] = useState<PipelineDAG | null>(null);
  const [isValidating, setIsValidating] = useState(false);

  // Agent stream hook
  const {
    status,
    activities,
    pendingQuestions,
    dismissQuestion,
    isStreaming,
  } = useAgentStream(conversationId || '', {});

  // Initialize DAG from conversation
  useEffect(() => {
    if (conversation?.current_pipeline) {
      setDag(pipelineConfigToDAG(conversation.current_pipeline as Record<string, unknown>));
    }
  }, [conversation?.current_pipeline]);

  // Handle mode change
  const handleModeChange = useCallback(async (newMode: BuilderMode) => {
    if (!conversationId) return;

    try {
      if (newMode === 'manual') {
        await agentApiV2.pauseAgent(conversationId);
      } else {
        await agentApiV2.resumeAgent(conversationId);
      }
      setMode(newMode);
    } catch (err) {
      addToast({
        type: 'error',
        message: `Failed to switch mode: ${err instanceof Error ? err.message : 'Unknown error'}`,
      });
    }
  }, [conversationId, addToast]);

  // Handle close
  const handleClose = useCallback(() => {
    navigate('/');
  }, [navigate]);

  // Handle node change
  const handleNodeChange = useCallback((updatedNode: PipelineNode) => {
    if (!dag) return;
    setDag({
      ...dag,
      nodes: dag.nodes.map((n) => (n.id === updatedNode.id ? updatedNode : n)),
    });
  }, [dag]);

  // Handle edge change
  const handleEdgeChange = useCallback((updatedEdge: PipelineEdge) => {
    if (!dag) return;
    setDag({
      ...dag,
      edges: dag.edges.map((e) =>
        e.from_node === updatedEdge.from_node && e.to_node === updatedEdge.to_node
          ? updatedEdge
          : e
      ),
    });
  }, [dag]);

  // Handle question answer
  const handleAnswer = useCallback(async (
    questionId: string,
    optionId?: string,
    customResponse?: string
  ) => {
    if (!conversationId) return;

    try {
      await agentApiV2.answerQuestion(conversationId, questionId, optionId, customResponse);
      dismissQuestion(questionId);
    } catch (err) {
      addToast({
        type: 'error',
        message: `Failed to submit answer: ${err instanceof Error ? err.message : 'Unknown error'}`,
      });
    }
  }, [conversationId, dismissQuestion, addToast]);

  // Handle status bar send
  const handleSend = useCallback(async (message: string) => {
    if (!conversationId) return;

    try {
      await agentApiV2.sendMessage(conversationId, message);
    } catch (err) {
      addToast({
        type: 'error',
        message: `Failed to send message: ${err instanceof Error ? err.message : 'Unknown error'}`,
      });
    }
  }, [conversationId, addToast]);

  // Handle stop
  const handleStop = useCallback(async () => {
    if (!conversationId) return;
    await handleModeChange('manual');
  }, [conversationId, handleModeChange]);

  // Handle validate
  const handleValidate = useCallback(async () => {
    setIsValidating(true);
    // TODO: Implement validation
    setTimeout(() => setIsValidating(false), 2000);
  }, []);

  // Handle apply
  const handleApply = useCallback(async () => {
    if (!conversationId || !conversation) return;

    try {
      const result = await applyMutation.mutateAsync({
        conversationId,
        data: {
          collection_name: 'New Collection', // TODO: Get from user
        },
      });

      addToast({
        type: 'success',
        message: 'Pipeline applied! Collection created.',
      });

      navigate(`/collections/${result.collection_id}`);
    } catch (err) {
      addToast({
        type: 'error',
        message: `Failed to apply pipeline: ${err instanceof Error ? err.message : 'Unknown error'}`,
      });
    }
  }, [conversationId, conversation, applyMutation, addToast, navigate]);

  // Loading state
  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[var(--text-primary)] mx-auto mb-4" />
          <p className="text-[var(--text-muted)]">Loading pipeline builder...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (error || !conversationId) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-lg font-medium text-[var(--text-primary)] mb-2">
            Failed to load pipeline
          </h2>
          <p className="text-sm text-[var(--text-secondary)] mb-4">
            {error?.message || 'Invalid conversation ID'}
          </p>
          <button onClick={handleClose} className="px-4 py-2 border border-[var(--border)] rounded-lg hover:bg-[var(--bg-tertiary)]">
            Go Back
          </button>
        </div>
      </div>
    );
  }

  // Get source info
  const sourceInfo = conversation?.inline_source_config;
  const sourceName = (sourceInfo?.source_config?.path as string)
    || (sourceInfo?.source_config?.repo_url as string)
    || 'Unknown source';
  const sourceType = sourceInfo?.source_type || 'directory';

  // Check if pipeline is ready
  const isPipelineReady = dag && dag.nodes.length > 0 && dag.edges.length > 0;

  return (
    <div className="h-[calc(100vh-64px)] flex flex-col">
      {/* Header */}
      <PipelineBuilderHeader
        sourceName={sourceName}
        sourceType={sourceType}
        mode={mode}
        onModeChange={handleModeChange}
        onClose={handleClose}
      />

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Pipeline Visualization (~60%) */}
        <div className="flex-[3] border-r border-[var(--border)] overflow-auto p-4">
          {dag && (
            <PipelineVisualization
              dag={dag}
              selection={selection}
              onSelectionChange={setSelection}
            />
          )}
        </div>

        {/* Right: Configuration Panel (~40%) */}
        <div className="flex-[2] overflow-auto">
          {dag && (
            <ConfigurationPanel
              dag={dag}
              selection={selection}
              sourceAnalysis={conversation?.source_analysis ?? null}
              onNodeChange={handleNodeChange}
              onEdgeChange={handleEdgeChange}
              readOnly={mode === 'assisted' && isStreaming}
            />
          )}
        </div>
      </div>

      {/* Agent Status Bar (when in assisted mode) */}
      {mode === 'assisted' && (
        <AgentStatusBar
          status={status}
          activities={activities}
          isStreaming={isStreaming}
          onSend={handleSend}
          onStop={handleStop}
        />
      )}

      {/* Manual mode indicator */}
      {mode === 'manual' && (
        <div className="px-4 py-2 border-t border-[var(--border)] bg-[var(--bg-secondary)]">
          <div className="flex items-center justify-between">
            <span className="text-sm text-[var(--text-muted)]">
              Manual mode - You have full control over the pipeline configuration
            </span>
            <button
              onClick={() => handleModeChange('assisted')}
              className="text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] underline"
            >
              Resume agent
            </button>
          </div>
        </div>
      )}

      {/* Footer */}
      <PipelineBuilderFooter
        isReady={isPipelineReady ?? false}
        fileCount={conversation?.source_analysis?.total_files || 0}
        nodeCount={dag?.nodes.length || 0}
        onValidate={handleValidate}
        onApply={handleApply}
        isValidating={isValidating}
        isApplying={applyMutation.isPending}
      />

      {/* Question prompts overlay */}
      <QuestionQueue
        questions={pendingQuestions}
        onAnswer={handleAnswer}
        onDismiss={dismissQuestion}
      />
    </div>
  );
}

export default PipelineBuilderPage;
