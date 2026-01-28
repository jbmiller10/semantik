/**
 * Context-sensitive configuration panel for pipeline builder.
 * Shows different content based on current selection:
 * - No selection: SourceAnalysisSummary
 * - Node selected: NodeConfigEditor
 * - Edge selected: EdgePredicateEditor
 * - Source node: SourceAnalysisSummary
 */

import { useMemo, useCallback } from 'react';
import type { PipelineDAG, PipelineNode, PipelineEdge, DAGSelection } from '@/types/pipeline';
import type { SourceAnalysis } from '@/types/agent';
import { SourceAnalysisSummary } from './SourceAnalysisSummary';
import { NodeConfigEditor } from './NodeConfigEditor';
import { EdgePredicateEditor } from './EdgePredicateEditor';
import { AlertCircle, Trash2 } from 'lucide-react';

interface ConfigurationPanelProps {
  dag: PipelineDAG;
  selection: DAGSelection;
  sourceAnalysis: SourceAnalysis | null;
  onNodeChange: (node: PipelineNode) => void;
  onEdgeChange: (edge: PipelineEdge) => void;
  onDelete?: () => void;
  readOnly?: boolean;
  className?: string;
}

/**
 * Get human-readable label for a node.
 */
function getNodeLabel(dag: PipelineDAG, nodeId: string): string {
  if (nodeId === '_source') return 'Source';
  const node = dag.nodes.find((n) => n.id === nodeId);
  if (!node) return nodeId;
  return node.plugin_id;
}

export function ConfigurationPanel({
  dag,
  selection,
  sourceAnalysis,
  onNodeChange,
  onEdgeChange,
  onDelete,
  readOnly = false,
  className = '',
}: ConfigurationPanelProps) {
  // Find selected node
  const selectedNode = useMemo(() => {
    if (selection.type !== 'node') return null;
    if (selection.nodeId === '_source') return null; // Source is not a real node
    return dag.nodes.find((n) => n.id === selection.nodeId) || null;
  }, [dag.nodes, selection]);

  // Find selected edge
  const selectedEdge = useMemo(() => {
    if (selection.type !== 'edge') return null;
    return (
      dag.edges.find(
        (e) => e.from_node === selection.fromNode && e.to_node === selection.toNode
      ) || null
    );
  }, [dag.edges, selection]);

  // Handle node change - update the node in the DAG
  const handleNodeChange = useCallback(
    (updatedNode: PipelineNode) => {
      onNodeChange(updatedNode);
    },
    [onNodeChange]
  );

  // Handle edge change - update the edge in the DAG
  const handleEdgeChange = useCallback(
    (updatedEdge: PipelineEdge) => {
      onEdgeChange(updatedEdge);
    },
    [onEdgeChange]
  );

  // Render content based on selection
  const renderContent = () => {
    // No selection or source selected -> show analysis
    if (selection.type === 'none') {
      return <SourceAnalysisSummary analysis={sourceAnalysis} />;
    }

    // Source node selected -> show analysis
    if (selection.type === 'node' && selection.nodeId === '_source') {
      return (
        <div className="p-4">
          <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-4">
            Data Source
          </h3>
          <SourceAnalysisSummary analysis={sourceAnalysis} />
        </div>
      );
    }

    // Node selected
    if (selection.type === 'node') {
      if (!selectedNode) {
        return (
          <div className="flex flex-col items-center justify-center h-full p-6 text-center">
            <AlertCircle className="w-12 h-12 text-amber-400 mb-3" />
            <p className="text-[var(--text-primary)]">Node not found</p>
            <p className="text-sm text-[var(--text-muted)] mt-1">
              The selected node "{selection.nodeId}" doesn't exist in the pipeline.
            </p>
          </div>
        );
      }

      return (
        <div className="flex flex-col h-full">
          {/* Header with delete button */}
          {!readOnly && onDelete && (
            <div className="flex justify-end p-2 border-b border-[var(--border)]">
              <button
                onClick={onDelete}
                className="p-2 rounded hover:bg-red-500/20 text-[var(--text-muted)] hover:text-red-400 transition-colors"
                title="Delete node"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          )}
          <div className="flex-1 overflow-y-auto">
            <NodeConfigEditor
              node={selectedNode}
              onChange={handleNodeChange}
              readOnly={readOnly}
            />
          </div>
        </div>
      );
    }

    // Edge selected
    if (selection.type === 'edge') {
      if (!selectedEdge) {
        return (
          <div className="flex flex-col items-center justify-center h-full p-6 text-center">
            <AlertCircle className="w-12 h-12 text-amber-400 mb-3" />
            <p className="text-[var(--text-primary)]">Edge not found</p>
            <p className="text-sm text-[var(--text-muted)] mt-1">
              The selected edge doesn't exist in the pipeline.
            </p>
          </div>
        );
      }

      return (
        <div className="flex flex-col h-full">
          {/* Header with delete button */}
          {!readOnly && onDelete && (
            <div className="flex justify-end p-2 border-b border-[var(--border)]">
              <button
                onClick={onDelete}
                className="p-2 rounded hover:bg-red-500/20 text-[var(--text-muted)] hover:text-red-400 transition-colors"
                title="Delete edge"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          )}
          <div className="flex-1 overflow-y-auto">
            <EdgePredicateEditor
              edge={selectedEdge}
              fromNodeLabel={getNodeLabel(dag, selectedEdge.from_node)}
              toNodeLabel={getNodeLabel(dag, selectedEdge.to_node)}
              onChange={handleEdgeChange}
              readOnly={readOnly}
            />
          </div>
        </div>
      );
    }

    // Fallback
    return <SourceAnalysisSummary analysis={sourceAnalysis} />;
  };

  return (
    <div
      className={`h-full overflow-y-auto bg-[var(--bg-secondary)] border-l border-[var(--border)] ${className}`}
    >
      {renderContent()}
    </div>
  );
}

export default ConfigurationPanel;
