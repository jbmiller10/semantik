// apps/webui-react/src/components/wizard/steps/ReviewStep.tsx
import { useState, useCallback } from 'react';
import { ChevronDown, ChevronRight, MessageSquare } from 'lucide-react';
import { PipelineVisualization, ConfigurationPanel } from '../../pipeline';
import type { PipelineDAG, DAGSelection, PipelineNode, PipelineEdge } from '../../../types/pipeline';

interface ReviewStepProps {
  dag: PipelineDAG;
  onDagChange: (dag: PipelineDAG) => void;
  agentSummary: string;
  conversationId?: string;
  onBackToAnalysis?: () => void;
}

export function ReviewStep({
  dag,
  onDagChange,
  agentSummary,
  conversationId,
  onBackToAnalysis,
}: ReviewStepProps) {
  const [selection, setSelection] = useState<DAGSelection>({ type: 'none' });
  const [showFullSummary, setShowFullSummary] = useState(false);
  const [modifiedNodes, setModifiedNodes] = useState<Set<string>>(new Set());

  // Suppress unused variable warning - conversationId is kept for future use
  void conversationId;

  const handleNodeChange = useCallback((updatedNode: PipelineNode) => {
    // Track that this node was modified
    setModifiedNodes(prev => new Set(prev).add(updatedNode.id));

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

  // Split summary into sentences for collapsible view
  const summarySentences = agentSummary.split(/(?<=[.!?])\s+/).filter(Boolean);
  const shortSummary = summarySentences.slice(0, 2).join(' ');
  const hasMoreSummary = summarySentences.length > 2;

  return (
    <div className="h-full flex flex-col lg:flex-row overflow-hidden">
      {/* Left: Agent Summary (~25%) */}
      <div
        data-testid="summary-column"
        className="lg:w-[300px] xl:w-[350px] border-b lg:border-b-0 lg:border-r border-[var(--border)] flex flex-col shrink-0"
      >
        {/* Header */}
        <div className="px-4 py-3 border-b border-[var(--border)] bg-[var(--bg-secondary)] shrink-0">
          <div className="flex items-center gap-2">
            <MessageSquare className="w-4 h-4 text-[var(--text-muted)]" />
            <span className="text-sm font-medium text-[var(--text-primary)]">
              Agent Recommendations
            </span>
          </div>
        </div>

        {/* Summary content */}
        <div className="flex-1 overflow-y-auto p-4">
          <div className="space-y-4">
            {/* Condensed summary */}
            <div className="space-y-2">
              <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
                {showFullSummary ? agentSummary : shortSummary}
                {hasMoreSummary && !showFullSummary && '...'}
              </p>

              {hasMoreSummary && (
                <button
                  onClick={() => setShowFullSummary(!showFullSummary)}
                  className="flex items-center gap-1 text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
                >
                  {showFullSummary ? (
                    <>
                      <ChevronDown className="w-3 h-3" />
                      Show less
                    </>
                  ) : (
                    <>
                      <ChevronRight className="w-3 h-3" />
                      View full reasoning
                    </>
                  )}
                </button>
              )}
            </div>

            {/* Node recommendations */}
            <div className="space-y-2">
              <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
                Configured Nodes
              </h4>
              {dag.nodes.map((node) => (
                <div
                  key={node.id}
                  className={`
                    p-2 rounded-lg text-sm
                    ${selection.type === 'node' && selection.nodeId === node.id
                      ? 'bg-gray-100 dark:bg-white/10 border border-gray-400 dark:border-white'
                      : 'bg-[var(--bg-tertiary)]'
                    }
                  `}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-[var(--text-primary)] capitalize">
                      {node.type}
                    </span>
                    {modifiedNodes.has(node.id) && (
                      <span className="text-xs text-amber-500">Modified</span>
                    )}
                  </div>
                  <span className="text-xs text-[var(--text-muted)]">
                    {node.plugin_id}
                  </span>
                </div>
              ))}
            </div>

            {/* Back to analysis link */}
            {onBackToAnalysis && (
              <button
                onClick={onBackToAnalysis}
                className="text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] underline"
              >
                Back to agent analysis
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Right: Full DAG Editor (~75%) */}
      <div
        data-testid="editor-column"
        className="flex-1 flex flex-col overflow-hidden"
      >
        {/* Editor header */}
        <div className="px-4 py-3 border-b border-[var(--border)] bg-[var(--bg-secondary)] shrink-0">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-[var(--text-primary)]">
              Pipeline Configuration
            </span>
            <span className="text-xs text-[var(--text-muted)]">
              All settings are editable
            </span>
          </div>
        </div>

        {/* DAG editor */}
        <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
          {/* Visualization */}
          <div className="flex-1 lg:flex-[3] border-b lg:border-b-0 lg:border-r border-[var(--border)] overflow-auto p-4 min-h-[200px] lg:min-h-0">
            <PipelineVisualization
              dag={dag}
              selection={selection}
              onSelectionChange={setSelection}
            />
          </div>

          {/* Configuration panel */}
          <div className="flex-1 lg:flex-[2] overflow-auto">
            <ConfigurationPanel
              dag={dag}
              selection={selection}
              sourceAnalysis={null}
              onNodeChange={handleNodeChange}
              onEdgeChange={handleEdgeChange}
              readOnly={false}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
