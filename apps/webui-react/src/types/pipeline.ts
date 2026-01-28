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
// Drag State Types
// =============================================================================

/** State for drag-to-connect interaction */
export type DragState =
  | { isDragging: false }
  | {
      isDragging: true;
      sourceNodeId: string;
      sourcePosition: { x: number; y: number };
      cursorPosition: { x: number; y: number };
    };

/** Initial drag state (not dragging) */
export const INITIAL_DRAG_STATE: DragState = { isDragging: false };

/** Position of a port on a node */
export interface PortPosition {
  nodeId: string;
  type: 'input' | 'output';
  x: number;
  y: number;
}

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
  /** Callback when the DAG is modified (new node/edge created) */
  onDagChange?: (dag: PipelineDAG) => void;
  /** Whether the visualization is read-only */
  readOnly?: boolean;
  /** Optional CSS class name */
  className?: string;
  /** Highlighted path for route preview (list of node IDs) */
  highlightedPath?: string[] | null;
}
