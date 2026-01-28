/**
 * Tree view component for displaying edge evaluation results.
 * Shows which edges matched, which didn't, and why.
 */

import { useState } from 'react';
import { ChevronDown, ChevronRight, Check, X, SkipForward } from 'lucide-react';
import type { StageEvaluationResult, EdgeEvaluationResult, FieldEvaluationResult } from '@/types/routePreview';
import type { PipelineDAG, PipelineNode as PipelineNodeType } from '@/types/pipeline';

interface EdgeEvaluationTreeProps {
  /** Routing stages with edge evaluations */
  stages: StageEvaluationResult[];
  /** Pipeline DAG for node lookup */
  dag: PipelineDAG;
}

/**
 * Get status icon and color for an edge evaluation.
 */
function getStatusDisplay(status: 'matched' | 'not_matched' | 'skipped') {
  switch (status) {
    case 'matched':
      return {
        icon: <Check className="w-4 h-4" />,
        color: 'text-green-400',
        bgColor: 'bg-green-500/10',
        borderColor: 'border-green-500/30',
        label: 'Matched',
      };
    case 'not_matched':
      return {
        icon: <X className="w-4 h-4" />,
        color: 'text-red-400',
        bgColor: 'bg-red-500/10',
        borderColor: 'border-red-500/30',
        label: 'Not Matched',
      };
    case 'skipped':
      return {
        icon: <SkipForward className="w-4 h-4" />,
        color: 'text-[var(--text-muted)]',
        bgColor: 'bg-[var(--bg-secondary)]',
        borderColor: 'border-[var(--border)]',
        label: 'Skipped',
      };
  }
}

/**
 * Get display name for a node.
 */
function getNodeDisplayName(nodeId: string, nodes: PipelineNodeType[]): string {
  if (nodeId === '_source') {
    return 'Source';
  }
  const node = nodes.find((n) => n.id === nodeId);
  return node?.plugin_id || nodeId;
}

/**
 * Format a value for display.
 */
function formatValue(value: unknown): string {
  if (value === null) return 'null';
  if (value === undefined) return 'undefined';
  if (typeof value === 'string') return `"${value}"`;
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'number') return String(value);
  if (Array.isArray(value)) return `[${value.map(formatValue).join(', ')}]`;
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

interface FieldEvaluationRowProps {
  field: FieldEvaluationResult;
}

function FieldEvaluationRow({ field }: FieldEvaluationRowProps) {
  const matched = field.matched;

  return (
    <div className="flex items-start gap-2 py-1 px-2 text-xs font-mono">
      <span className={matched ? 'text-green-400' : 'text-red-400'}>
        {matched ? <Check className="w-3 h-3" /> : <X className="w-3 h-3" />}
      </span>
      <div className="flex-1 min-w-0">
        <span className="text-[var(--text-secondary)]">{field.field}</span>
        <span className="text-[var(--text-muted)]"> = </span>
        <span className={matched ? 'text-green-300' : 'text-red-300'}>
          {formatValue(field.value)}
        </span>
        <span className="text-[var(--text-muted)]"> vs </span>
        <span className="text-[var(--text-secondary)]">{formatValue(field.pattern)}</span>
      </div>
    </div>
  );
}

interface EdgeRowProps {
  edge: EdgeEvaluationResult;
  dag: PipelineDAG;
}

function EdgeRow({ edge, dag }: EdgeRowProps) {
  const [isExpanded, setIsExpanded] = useState(edge.status === 'not_matched');
  const status = getStatusDisplay(edge.status);
  const hasFieldEvaluations = edge.field_evaluations && edge.field_evaluations.length > 0;
  const toNodeName = getNodeDisplayName(edge.to_node, dag.nodes);

  return (
    <div className="border-l-2 border-[var(--border)] ml-2 pl-3">
      <button
        onClick={() => hasFieldEvaluations && setIsExpanded(!isExpanded)}
        className={`
          flex items-center gap-2 w-full text-left py-1.5 px-2 rounded
          hover:bg-[var(--bg-secondary)] transition-colors
          ${hasFieldEvaluations ? 'cursor-pointer' : 'cursor-default'}
        `}
        disabled={!hasFieldEvaluations}
      >
        {hasFieldEvaluations && (
          <span className="text-[var(--text-muted)]">
            {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          </span>
        )}

        <span className={`flex items-center gap-1 ${status.color}`}>
          {status.icon}
        </span>

        <span className="text-sm text-[var(--text-primary)]">
          → {toNodeName}
        </span>

        {edge.predicate ? (
          <span className="text-xs text-[var(--text-muted)] font-mono truncate">
            {JSON.stringify(edge.predicate)}
          </span>
        ) : (
          <span className="text-xs text-[var(--text-muted)] italic">catch-all (*)</span>
        )}
      </button>

      {isExpanded && hasFieldEvaluations && (
        <div className="ml-6 border-l border-[var(--border)] pl-2 mb-2">
          {edge.field_evaluations!.map((field, idx) => (
            <FieldEvaluationRow key={`${field.field}-${idx}`} field={field} />
          ))}
        </div>
      )}
    </div>
  );
}

interface StageRowProps {
  stage: StageEvaluationResult;
  dag: PipelineDAG;
}

function StageRow({ stage, dag }: StageRowProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  const fromNodeName = getNodeDisplayName(stage.from_node, dag.nodes);
  const hasEdges = stage.evaluated_edges.length > 0;
  const matchedEdge = stage.evaluated_edges.find((e) => e.status === 'matched');

  return (
    <div className="mb-3">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 w-full text-left py-2 px-3 rounded-lg bg-[var(--bg-secondary)] hover:bg-[var(--bg-tertiary)] transition-colors"
      >
        <span className="text-[var(--text-muted)]">
          {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        </span>

        <span className="text-sm font-medium text-[var(--text-primary)]">
          {fromNodeName}
        </span>

        <span className="text-xs text-[var(--text-muted)]">
          {stage.evaluated_edges.length} edge{stage.evaluated_edges.length !== 1 ? 's' : ''} evaluated
        </span>

        {matchedEdge && (
          <span className="ml-auto text-xs text-green-400 flex items-center gap-1">
            <Check className="w-3 h-3" />
            {getNodeDisplayName(matchedEdge.to_node, dag.nodes)}
          </span>
        )}
      </button>

      {isExpanded && hasEdges && (
        <div className="mt-1">
          {stage.evaluated_edges.map((edge, idx) => (
            <EdgeRow key={`${edge.from_node}-${edge.to_node}-${idx}`} edge={edge} dag={dag} />
          ))}
        </div>
      )}
    </div>
  );
}

export function EdgeEvaluationTree({ stages, dag }: EdgeEvaluationTreeProps) {
  if (stages.length === 0) {
    return (
      <div className="text-sm text-[var(--text-muted)] italic p-2">
        No routing stages to display
      </div>
    );
  }

  return (
    <div className="space-y-1">
      {stages.map((stage, idx) => (
        <StageRow key={`${stage.stage}-${idx}`} stage={stage} dag={dag} />
      ))}
    </div>
  );
}

export default EdgeEvaluationTree;
