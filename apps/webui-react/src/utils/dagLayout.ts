/**
 * Computes layout positions for pipeline DAG visualization.
 *
 * Layout algorithm:
 * 1. Group nodes by type (parser, chunker, extractor, embedder)
 * 2. Assign columns: _source -> parsers -> chunkers -> extractors -> embedders
 * 3. Stack nodes of same type vertically
 * 4. Add padding and compute overall dimensions
 */

import type { PipelineDAG, DAGLayout, NodePosition, NodeType } from '@/types/pipeline';

// Layout constants
const NODE_WIDTH = 160;
const NODE_HEIGHT = 80;
const COLUMN_GAP = 100;
const ROW_GAP = 40;
const PADDING = 40;

// Column order for node types
const TYPE_COLUMNS: Record<NodeType | '_source', number> = {
  _source: 0,
  parser: 1,
  chunker: 2,
  extractor: 3,
  embedder: 4,
};

export function computeDAGLayout(dag: PipelineDAG): DAGLayout {
  const nodes = new Map<string, NodePosition>();

  // Group nodes by their column
  const columns: Map<number, string[]> = new Map();

  // Add _source pseudo-node
  const sourceCol = TYPE_COLUMNS._source;
  columns.set(sourceCol, ['_source']);

  // Group actual nodes by type -> column
  for (const node of dag.nodes) {
    const col = TYPE_COLUMNS[node.type];
    if (!columns.has(col)) {
      columns.set(col, []);
    }
    columns.get(col)!.push(node.id);
  }

  // Compute positions
  let maxX = 0;
  let maxY = 0;

  for (const [col, nodeIds] of columns) {
    const x = PADDING + col * (NODE_WIDTH + COLUMN_GAP);

    for (let row = 0; row < nodeIds.length; row++) {
      const y = PADDING + row * (NODE_HEIGHT + ROW_GAP);

      nodes.set(nodeIds[row], {
        x,
        y,
        width: NODE_WIDTH,
        height: NODE_HEIGHT,
      });

      maxX = Math.max(maxX, x + NODE_WIDTH);
      maxY = Math.max(maxY, y + NODE_HEIGHT);
    }
  }

  return {
    nodes,
    width: maxX + PADDING,
    height: maxY + PADDING,
  };
}

/**
 * Get the center point of a node for edge drawing.
 */
export function getNodeCenter(pos: NodePosition): { x: number; y: number } {
  return {
    x: pos.x + pos.width / 2,
    y: pos.y + pos.height / 2,
  };
}

/**
 * Get the right edge center of a node (for outgoing edges).
 */
export function getNodeRightCenter(pos: NodePosition): { x: number; y: number } {
  return {
    x: pos.x + pos.width,
    y: pos.y + pos.height / 2,
  };
}

/**
 * Get the left edge center of a node (for incoming edges).
 */
export function getNodeLeftCenter(pos: NodePosition): { x: number; y: number } {
  return {
    x: pos.x,
    y: pos.y + pos.height / 2,
  };
}
