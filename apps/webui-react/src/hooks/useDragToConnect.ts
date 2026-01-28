/**
 * Hook for managing drag-to-connect interactions in the pipeline visualization.
 * Handles drag state, valid target detection, and coordinate conversion.
 */

import { useState, useCallback } from 'react';
import type { PipelineDAG, DAGLayout, NodeType, DragState } from '@/types/pipeline';
import { INITIAL_DRAG_STATE } from '@/types/pipeline';

interface UseDragToConnectOptions {
  dag: PipelineDAG;
  layout: DAGLayout;
  onConnect?: (fromNodeId: string, toNodeId: string) => void;
  onDropOnZone?: (fromNodeId: string, tier: NodeType) => void;
}

interface UseDragToConnectResult {
  dragState: DragState;
  startDrag: (nodeId: string, position: { x: number; y: number }) => void;
  updateDrag: (cursorPosition: { x: number; y: number }) => void;
  endDrag: () => void;
  cancelDrag: () => void;
  getValidTargetTiers: (sourceNodeId: string) => NodeType[];
  isValidDropTarget: (targetNodeId: string) => boolean;
}

/**
 * Get valid downstream tiers that a node can connect to.
 * Follows the pipeline tier order: source -> parser -> chunker -> extractor/embedder -> embedder
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

export function useDragToConnect({
  dag,
  // Phase 3 will use these for drop handling
  layout: _layout,
  onConnect: _onConnect,
  onDropOnZone: _onDropOnZone,
}: UseDragToConnectOptions): UseDragToConnectResult {
  // Suppress unused variable warnings - these will be used in Phase 3
  void _layout;
  void _onConnect;
  void _onDropOnZone;
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

  const endDrag = useCallback(() => {
    // Phase 3 will add drop handling logic here
    // For now, just reset the state
    setDragState(INITIAL_DRAG_STATE);
  }, []);

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
      if (!dragState.isDragging || !dragState.sourceNodeId) return false;

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
    [dragState.isDragging, dragState.sourceNodeId, dag]
  );

  return {
    dragState,
    startDrag,
    updateDrag,
    endDrag,
    cancelDrag,
    getValidTargetTiers: getValidTargetTiersForSource,
    isValidDropTarget,
  };
}

export default useDragToConnect;
