/**
 * Hook for managing drag-to-connect interactions in the pipeline visualization.
 * Handles drag state, valid target detection, and coordinate conversion.
 */

import { useState, useCallback } from 'react';
import type { PipelineDAG, DAGLayout, NodeType, DragState } from '@/types/pipeline';
import { INITIAL_DRAG_STATE } from '@/types/pipeline';
import type { TierBounds } from '@/utils/dagLayout';

interface UseDragToConnectOptions {
  dag: PipelineDAG;
  layout: DAGLayout;
  tierBounds?: TierBounds[];
  onConnect?: (fromNodeId: string, toNodeId: string) => void;
  onDropOnZone?: (fromNodeId: string, tier: NodeType, cursorPosition: { x: number; y: number }) => void;
}

interface UseDragToConnectResult {
  dragState: DragState;
  startDrag: (nodeId: string, position: { x: number; y: number }) => void;
  updateDrag: (cursorPosition: { x: number; y: number }) => void;
  endDrag: (cursorPosition?: { x: number; y: number }) => void;
  cancelDrag: () => void;
  getValidTargetTiers: (sourceNodeId: string) => NodeType[];
  isValidDropTarget: (targetNodeId: string) => boolean;
  getHoveredTier: () => NodeType | null;
}

/**
 * Get valid downstream tiers that a node can connect to.
 * Follows the pipeline tier order:
 *   source -> parser -> chunker -> (extractor -> embedder) OR (embedder directly)
 * Embedder is terminal and cannot connect to anything further.
 */
export function getValidTargetTiers(sourceNodeId: string, dag: PipelineDAG): NodeType[] {
  if (sourceNodeId === '_source') {
    return ['parser'];
  }

  const node = dag.nodes.find((n) => n.id === sourceNodeId);
  if (!node) return [];

  switch (node.type) {
    case 'parser':
      return ['chunker'];
    case 'chunker':
      return ['extractor', 'embedder'];
    case 'extractor':
      return ['embedder'];
    case 'embedder':
      return []; // Terminal node, cannot connect to anything
    default:
      return [];
  }
}

/**
 * Find which node (if any) the cursor is over.
 */
function findNodeAtPoint(
  point: { x: number; y: number },
  layout: DAGLayout,
  dag: PipelineDAG
): { id: string; type: NodeType } | null {
  for (const [nodeId, pos] of layout.nodes) {
    if (
      point.x >= pos.x &&
      point.x <= pos.x + pos.width &&
      point.y >= pos.y &&
      point.y <= pos.y + pos.height
    ) {
      // Return node info - handle _source specially
      if (nodeId === '_source') {
        return null; // Can't drop on source
      }
      const node = dag.nodes.find((n) => n.id === nodeId);
      if (node) {
        return { id: node.id, type: node.type };
      }
    }
  }
  return null;
}

/**
 * Find which tier drop zone (if any) the cursor is over.
 */
function findTierAtPoint(
  point: { x: number; y: number },
  tierBounds: TierBounds[]
): NodeType | null {
  for (const bounds of tierBounds) {
    if (
      point.x >= bounds.x &&
      point.x <= bounds.x + bounds.width &&
      point.y >= bounds.y &&
      point.y <= bounds.y + bounds.height
    ) {
      return bounds.tier;
    }
  }
  return null;
}

export function useDragToConnect({
  dag,
  layout,
  tierBounds = [],
  onConnect,
  onDropOnZone,
}: UseDragToConnectOptions): UseDragToConnectResult {
  const [dragState, setDragState] = useState<DragState>(INITIAL_DRAG_STATE);

  const startDrag = useCallback((nodeId: string, position: { x: number; y: number }) => {
    setDragState({
      isDragging: true,
      sourceNodeId: nodeId,
      sourcePosition: position,
      cursorPosition: position,
    });
  }, []);

  const updateDrag = useCallback((cursorPosition: { x: number; y: number }) => {
    setDragState((prev) => {
      if (!prev.isDragging) return prev;
      return {
        ...prev,
        cursorPosition,
      };
    });
  }, []);

  const endDrag = useCallback(
    (cursorPosition?: { x: number; y: number }) => {
      if (!dragState.isDragging) {
        setDragState(INITIAL_DRAG_STATE);
        return;
      }

      const dropPoint = cursorPosition || dragState.cursorPosition;

      const validTiers = getValidTargetTiers(dragState.sourceNodeId, dag);

      // Check if over an existing node's input port
      const targetNode = findNodeAtPoint(dropPoint, layout, dag);
      if (targetNode && validTiers.includes(targetNode.type)) {
        // Check if edge already exists
        const edgeExists = dag.edges.some(
          (e) => e.from_node === dragState.sourceNodeId && e.to_node === targetNode.id
        );
        if (!edgeExists && onConnect) {
          onConnect(dragState.sourceNodeId, targetNode.id);
        }
        setDragState(INITIAL_DRAG_STATE);
        return;
      }

      // Check if over a tier drop zone
      const targetTier = findTierAtPoint(dropPoint, tierBounds);
      if (targetTier && validTiers.includes(targetTier)) {
        if (onDropOnZone) {
          onDropOnZone(dragState.sourceNodeId, targetTier, dropPoint);
        }
        setDragState(INITIAL_DRAG_STATE);
        return;
      }

      // No valid drop target - cancel drag
      setDragState(INITIAL_DRAG_STATE);
    },
    [dragState, dag, layout, tierBounds, onConnect, onDropOnZone]
  );

  const cancelDrag = useCallback(() => {
    setDragState(INITIAL_DRAG_STATE);
  }, []);

  const getValidTargetTiersForSource = useCallback(
    (sourceNodeId: string) => {
      return getValidTargetTiers(sourceNodeId, dag);
    },
    [dag]
  );

  const isValidDropTarget = useCallback(
    (targetNodeId: string) => {
      if (!dragState.isDragging) return false;

      // Can't drop on self
      if (targetNodeId === dragState.sourceNodeId) return false;

      // Can't drop on source node
      if (targetNodeId === '_source') return false;

      // Check if target is in a valid tier
      const validTiers = getValidTargetTiers(dragState.sourceNodeId, dag);
      const targetNode = dag.nodes.find((n) => n.id === targetNodeId);

      if (!targetNode) return false;
      return validTiers.includes(targetNode.type);
    },
    [dragState, dag]
  );

  const getHoveredTier = useCallback((): NodeType | null => {
    if (!dragState.isDragging) {
      return null;
    }

    const validTiers = getValidTargetTiers(dragState.sourceNodeId, dag);
    const tier = findTierAtPoint(dragState.cursorPosition, tierBounds);

    if (tier && validTiers.includes(tier)) {
      return tier;
    }
    return null;
  }, [dragState, dag, tierBounds]);

  return {
    dragState,
    startDrag,
    updateDrag,
    endDrag,
    cancelDrag,
    getValidTargetTiers: getValidTargetTiersForSource,
    isValidDropTarget,
    getHoveredTier,
  };
}

export default useDragToConnect;
