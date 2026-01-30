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

// Layout constants - exported for use by other components
export const NODE_WIDTH = 160;
export const NODE_HEIGHT = 80;
export const TIER_GAP = 120; // Vertical spacing between tiers (was 100)
export const NODE_GAP = 40; // Horizontal spacing within a tier
export const PADDING = 40;

/**
 * Bounds for a tier's drop zone.
 */
export interface TierBounds {
  tier: NodeType;
  tierIndex: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

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

/**
 * All tier types in order (excluding _source which is implicit).
 */
export const ALL_TIERS: NodeType[] = ['parser', 'chunker', 'extractor', 'embedder'];

/**
 * Compute drop zone bounds for each tier.
 * Used for drag-to-connect interaction to show where new nodes can be added.
 */
export function computeTierBounds(
  dag: PipelineDAG,
  layout: DAGLayout,
  viewportWidth: number
): TierBounds[] {
  const bounds: TierBounds[] = [];
  const DROP_ZONE_HEIGHT = NODE_HEIGHT;
  const DROP_ZONE_PADDING = 20;

  // Find the rightmost edge of the layout
  let maxNodeRight = 0;
  for (const pos of layout.nodes.values()) {
    maxNodeRight = Math.max(maxNodeRight, pos.x + pos.width);
  }

  // Drop zone width spans from padding to the widest point (or viewport width)
  const dropZoneWidth = Math.max(
    maxNodeRight - PADDING + DROP_ZONE_PADDING * 2,
    viewportWidth - PADDING * 2
  );

  // Group existing nodes by tier
  const nodesByTier = new Map<number, string[]>();
  for (const node of dag.nodes) {
    const tier = TYPE_TIERS[node.type];
    if (!nodesByTier.has(tier)) {
      nodesByTier.set(tier, []);
    }
    nodesByTier.get(tier)!.push(node.id);
  }

  // Create bounds for each tier type
  for (const tierType of ALL_TIERS) {
    const tierIndex = TYPE_TIERS[tierType];
    const tierY = PADDING + tierIndex * (NODE_HEIGHT + TIER_GAP);

    // Check if tier has existing nodes
    const existingNodes = nodesByTier.get(tierIndex) || [];

    if (existingNodes.length > 0) {
      // Find rightmost node position in this tier
      let tierMaxX = 0;
      for (const nodeId of existingNodes) {
        const pos = layout.nodes.get(nodeId);
        if (pos) {
          tierMaxX = Math.max(tierMaxX, pos.x + pos.width);
        }
      }

      // Drop zone starts after existing nodes
      const dropZoneX = tierMaxX + NODE_GAP;
      bounds.push({
        tier: tierType,
        tierIndex,
        x: dropZoneX,
        y: tierY,
        width: Math.max(NODE_WIDTH + DROP_ZONE_PADDING * 2, dropZoneWidth - dropZoneX + PADDING),
        height: DROP_ZONE_HEIGHT,
      });
    } else {
      // Empty tier - drop zone spans the full width centered
      bounds.push({
        tier: tierType,
        tierIndex,
        x: PADDING,
        y: tierY,
        width: dropZoneWidth,
        height: DROP_ZONE_HEIGHT,
      });
    }
  }

  return bounds;
}

/**
 * Generate a unique node ID.
 */
export function generateNodeId(): string {
  return `node_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
}
