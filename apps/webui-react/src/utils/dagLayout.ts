/**
 * Computes layout positions for pipeline DAG visualization.
 *
 * Layout algorithm (vertical flow, top-to-bottom):
 * 1. Group nodes by type (parser, chunker, extractor, embedder)
 * 2. Assign tiers: _source (0) -> parsers (1) -> chunkers (2) -> extractors (3) -> embedders (4)
 * 3. Spread nodes of same type horizontally within their tier
 * 4. Add padding and compute overall dimensions
 */

import type { PipelineDAG, DAGLayout, NodePosition, NodeType } from '@/types/pipeline';

// Layout constants
const NODE_WIDTH = 160;
const NODE_HEIGHT = 80;
const TIER_GAP = 100; // Vertical spacing between tiers
const NODE_GAP = 40; // Horizontal spacing within a tier
const PADDING = 40;

// Tier order for node types (vertical position)
const TYPE_TIERS: Record<NodeType | '_source', number> = {
  _source: 0,
  parser: 1,
  chunker: 2,
  extractor: 3,
  embedder: 4,
};

export function computeDAGLayout(dag: PipelineDAG): DAGLayout {
  const nodes = new Map<string, NodePosition>();

  // Group nodes by their tier (vertical position)
  const tiers: Map<number, string[]> = new Map();

  // Add _source pseudo-node
  const sourceTier = TYPE_TIERS._source;
  tiers.set(sourceTier, ['_source']);

  // Group actual nodes by type -> tier
  for (const node of dag.nodes) {
    const tier = TYPE_TIERS[node.type];
    if (!tiers.has(tier)) {
      tiers.set(tier, []);
    }
    tiers.get(tier)!.push(node.id);
  }

  // Find the maximum number of nodes in any tier (for centering)
  let maxNodesInTier = 0;
  for (const nodeIds of tiers.values()) {
    maxNodesInTier = Math.max(maxNodesInTier, nodeIds.length);
  }

  // Compute positions (vertical flow: Y based on tier, X based on position within tier)
  let maxX = 0;
  let maxY = 0;

  for (const [tier, nodeIds] of tiers) {
    const y = PADDING + tier * (NODE_HEIGHT + TIER_GAP);

    // Calculate total width of this tier for centering
    const tierWidth = nodeIds.length * NODE_WIDTH + (nodeIds.length - 1) * NODE_GAP;
    const maxTierWidth = maxNodesInTier * NODE_WIDTH + (maxNodesInTier - 1) * NODE_GAP;
    const tierOffset = (maxTierWidth - tierWidth) / 2;

    for (let posInTier = 0; posInTier < nodeIds.length; posInTier++) {
      const x = PADDING + tierOffset + posInTier * (NODE_WIDTH + NODE_GAP);

      nodes.set(nodeIds[posInTier], {
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
 * Get the bottom center of a node (for outgoing edges in vertical layout).
 */
export function getNodeBottomCenter(pos: NodePosition): { x: number; y: number } {
  return {
    x: pos.x + pos.width / 2,
    y: pos.y + pos.height,
  };
}

/**
 * Get the top center of a node (for incoming edges in vertical layout).
 */
export function getNodeTopCenter(pos: NodePosition): { x: number; y: number } {
  return {
    x: pos.x + pos.width / 2,
    y: pos.y,
  };
}
