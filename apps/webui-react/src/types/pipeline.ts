/**
 * TypeScript types for Pipeline DAG visualization.
 * Mirrors backend types from packages/shared/pipeline/types.py
 */

// =============================================================================
// Enums
// =============================================================================

export type NodeType = 'parser' | 'chunker' | 'extractor' | 'embedder';

// =============================================================================
// Core DAG Types
// =============================================================================

export interface PipelineNode {
  id: string;
  type: NodeType;
  plugin_id: string;
  config: Record<string, unknown>;
}

export interface PipelineEdge {
  from_node: string; // "_source" for entry edges
  to_node: string;
  when: Record<string, unknown> | null; // null = catch-all
}

export interface PipelineDAG {
  id: string;
  version: string;
  nodes: PipelineNode[];
  edges: PipelineEdge[];
}

// =============================================================================
// Visualization Types
// =============================================================================

/** Position of a node in the visualization */
export interface NodePosition {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Computed layout for the entire DAG */
export interface DAGLayout {
  nodes: Map<string, NodePosition>;
  width: number;
  height: number;
}

/** Selection state for the visualization */
export type DAGSelection =
  | { type: 'none' }
  | { type: 'node'; nodeId: string }
  | { type: 'edge'; fromNode: string; toNode: string };

// =============================================================================
// Component Props
// =============================================================================

export interface PipelineVisualizationProps {
  /** The pipeline DAG to render */
  dag: PipelineDAG;
  /** Currently selected element */
  selection?: DAGSelection;
  /** Callback when selection changes */
  onSelectionChange?: (selection: DAGSelection) => void;
  /** Whether the visualization is read-only */
  readOnly?: boolean;
  /** Optional CSS class name */
  className?: string;
}
