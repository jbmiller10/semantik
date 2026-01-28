// apps/webui-react/src/components/wizard/steps/ConfigureStep.tsx
import { useState, useCallback } from 'react';
import { PipelineVisualization, ConfigurationPanel } from '../../pipeline';
import type { PipelineDAG, DAGSelection, PipelineNode, PipelineEdge } from '../../../types/pipeline';
import type { SourceAnalysis } from '../../../types/agent';

interface ConfigureStepProps {
  dag: PipelineDAG;
  onDagChange: (dag: PipelineDAG) => void;
  sourceAnalysis: SourceAnalysis | null;
  onSwitchToAssisted?: () => void;
}

export function ConfigureStep({
  dag,
  onDagChange,
  sourceAnalysis,
  onSwitchToAssisted,
}: ConfigureStepProps) {
  const [selection, setSelection] = useState<DAGSelection>({ type: 'none' });

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

  return (
    <div className="h-full flex flex-col">
      {/* Assisted mode upsell banner */}
      {onSwitchToAssisted && (
        <div className="px-4 py-2 bg-blue-500/10 border-b border-blue-500/30 flex items-center justify-between">
          <span className="text-sm text-blue-400">
            Want help? Let AI analyze your source and recommend settings.
          </span>
          <button
            onClick={onSwitchToAssisted}
            className="text-sm text-blue-400 hover:text-blue-300 underline"
          >
            Switch to Assisted mode
          </button>
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
        {/* Left: Pipeline Visualization (~60%) */}
        <div className="flex-1 lg:flex-[3] border-b lg:border-b-0 lg:border-r border-[var(--border)] overflow-auto p-4 min-h-[300px] lg:min-h-0">
          <PipelineVisualization
            dag={dag}
            selection={selection}
            onSelectionChange={setSelection}
            onDagChange={onDagChange}
          />
        </div>

        {/* Right: Configuration Panel (~40%) */}
        <div className="flex-1 lg:flex-[2] overflow-auto">
          <ConfigurationPanel
            dag={dag}
            selection={selection}
            sourceAnalysis={sourceAnalysis}
            onNodeChange={handleNodeChange}
            onEdgeChange={handleEdgeChange}
            readOnly={false}
          />
        </div>
      </div>
    </div>
  );
}
