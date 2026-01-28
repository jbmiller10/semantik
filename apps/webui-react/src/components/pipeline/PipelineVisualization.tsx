/**
 * Main pipeline visualization component.
 * Renders a DAG of pipeline nodes and edges in SVG.
 */

import { useMemo, useCallback, useRef, useEffect, useState } from 'react';
import type { PipelineVisualizationProps, NodeType, PipelineNode, PipelineEdge as PipelineEdgeType } from '@/types/pipeline';
import {
  computeDAGLayout,
  getNodeBottomCenter,
  computeTierBounds,
  generateNodeId,
  ALL_TIERS,
  NODE_HEIGHT,
  TIER_GAP,
  PADDING,
  NODE_WIDTH,
  NODE_GAP,
} from '@/utils/dagLayout';
import { findOrphanedNodes, findOrphanedNodesAfterEdgeDeletion } from '@/utils/dagUtils';
import { useDragToConnect } from '@/hooks/useDragToConnect';
import { useIsTouchDevice } from '@/hooks/useIsTouchDevice';
import { PipelineNodeComponent } from './PipelineNode';
import { PipelineEdgeComponent } from './PipelineEdge';
import { DragPreviewEdge } from './DragPreviewEdge';
import { TierDropZone } from './TierDropZone';
import { TierAddButton } from './TierAddButton';
import { NodePickerPopover } from './NodePickerPopover';
import { UpstreamNodePicker } from './UpstreamNodePicker';
import { DeleteConfirmationDialog } from './DeleteConfirmationDialog';

/**
 * Convert screen coordinates to SVG coordinates.
 * Handles any transforms applied to the SVG.
 */
function screenToSVG(
  svg: SVGSVGElement,
  screenX: number,
  screenY: number
): { x: number; y: number } {
  const point = svg.createSVGPoint();
  point.x = screenX;
  point.y = screenY;
  const ctm = svg.getScreenCTM();
  if (!ctm) {
    if (import.meta.env.DEV) {
      console.warn('screenToSVG: getScreenCTM() returned null, using raw coordinates');
    }
    return { x: screenX, y: screenY };
  }
  const transformed = point.matrixTransform(ctm.inverse());
  return { x: transformed.x, y: transformed.y };
}

/**
 * State for the node picker popover.
 */
type PopoverState =
  | { origin: 'button'; tier: NodeType; position: { x: number; y: number } }
  | { origin: 'drag'; tier: NodeType; sourceNodeId: string; position: { x: number; y: number } };

/**
 * State for the upstream node picker.
 */
interface UpstreamPickerState {
  tier: NodeType;
  pluginId: string;
  upstreamNodes: PipelineNode[];
  position: { x: number; y: number };
}

/**
 * State for the delete confirmation dialog.
 */
type DeleteConfirmationState =
  | { type: 'node'; nodeId: string; orphanedNodes: PipelineNode[] }
  | { type: 'edge'; fromNode: string; toNode: string; orphanedNodes: PipelineNode[] };

/**
 * Get the tier index for a node type.
 */
const TYPE_TIERS: Record<NodeType | '_source', number> = {
  _source: 0,
  parser: 1,
  chunker: 2,
  extractor: 3,
  embedder: 4,
};

/**
 * Get the tier type that comes before this one.
 */
function getPreviousTier(tier: NodeType): NodeType | '_source' | null {
  const tierIndex = TYPE_TIERS[tier];
  if (tierIndex <= 0) return null;

  // Find which tier has index tierIndex - 1
  for (const [key, value] of Object.entries(TYPE_TIERS)) {
    if (value === tierIndex - 1) {
      return key as NodeType | '_source';
    }
  }
  return null;
}

export function PipelineVisualization({
  dag,
  selection = { type: 'none' },
  onSelectionChange,
  onDagChange,
  readOnly = false,
  className = '',
  highlightedPath = null,
}: PipelineVisualizationProps) {
  // SVG ref for coordinate conversion
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Touch device detection - disable drag features on touch
  const isTouch = useIsTouchDevice();

  // Popover states
  const [popover, setPopover] = useState<PopoverState | null>(null);
  const [upstreamPicker, setUpstreamPicker] = useState<UpstreamPickerState | null>(null);
  const [deleteConfirmation, setDeleteConfirmation] = useState<DeleteConfirmationState | null>(null);

  // Track newly created nodes/edges for entrance animations
  const [newNodeIds, setNewNodeIds] = useState<Set<string>>(new Set());
  const [newEdgeKeys, setNewEdgeKeys] = useState<Set<string>>(new Set());

  // Compute layout
  const layout = useMemo(() => computeDAGLayout(dag), [dag]);

  // Container width for tier bounds
  const [containerWidth, setContainerWidth] = useState(600);
  useEffect(() => {
    if (!containerRef.current) return;
    // ResizeObserver may not be available in test environment
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width);
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Compute tier bounds for drop zones
  const tierBounds = useMemo(
    () => computeTierBounds(dag, layout, containerWidth),
    [dag, layout, containerWidth]
  );

  // Handle connection between nodes
  const handleConnect = useCallback(
    (fromNodeId: string, toNodeId: string) => {
      if (!onDagChange) return;

      // Create new edge
      const newEdge: PipelineEdgeType = {
        from_node: fromNodeId,
        to_node: toNodeId,
        when: null, // Default to catch-all
      };

      const newDag = {
        ...dag,
        edges: [...dag.edges, newEdge],
      };

      onDagChange(newDag);

      // Select the new edge
      if (onSelectionChange) {
        onSelectionChange({ type: 'edge', fromNode: fromNodeId, toNode: toNodeId });
      }
    },
    [dag, onDagChange, onSelectionChange]
  );

  // Handle drop on tier zone
  const handleDropOnZone = useCallback(
    (sourceNodeId: string, tier: NodeType, cursorPosition: { x: number; y: number }) => {
      // Convert SVG position to screen position for popover
      let screenPosition = cursorPosition;
      if (svgRef.current) {
        const ctm = svgRef.current.getScreenCTM();
        if (ctm) {
          const point = svgRef.current.createSVGPoint();
          point.x = cursorPosition.x;
          point.y = cursorPosition.y;
          const transformed = point.matrixTransform(ctm);
          screenPosition = { x: transformed.x, y: transformed.y };
        } else if (import.meta.env.DEV) {
          console.warn('handleDropOnZone: getScreenCTM() returned null, using raw SVG coordinates');
        }
      }

      setPopover({
        origin: 'drag',
        tier,
        sourceNodeId,
        position: screenPosition,
      });
    },
    []
  );

  // Drag-to-connect state and handlers
  const {
    dragState,
    startDrag,
    updateDrag,
    endDrag,
    cancelDrag,
    isValidDropTarget,
    getValidTargetTiers: getValidTiers,
    getHoveredTier,
  } = useDragToConnect({
    dag,
    layout,
    tierBounds,
    onConnect: handleConnect,
    onDropOnZone: handleDropOnZone,
  });

  // Get hovered tier for drop zone highlighting
  const hoveredTier = getHoveredTier();

  // Handle node click
  const handleNodeClick = useCallback(
    (nodeId: string) => {
      if (readOnly || !onSelectionChange) return;
      onSelectionChange({ type: 'node', nodeId });
    },
    [readOnly, onSelectionChange]
  );

  // Handle edge click
  const handleEdgeClick = useCallback(
    (fromNode: string, toNode: string) => {
      if (readOnly || !onSelectionChange) return;
      onSelectionChange({ type: 'edge', fromNode, toNode });
    },
    [readOnly, onSelectionChange]
  );

  // Handle background click (clear selection)
  const handleBackgroundClick = useCallback(
    (e: React.MouseEvent) => {
      // Only clear if clicking the SVG background itself
      if (e.target === e.currentTarget && onSelectionChange) {
        onSelectionChange({ type: 'none' });
      }
    },
    [onSelectionChange]
  );

  // Handle drag start from a node's output port
  const handleStartDrag = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    (nodeId: string, _screenPosition: { x: number; y: number }) => {
      if (readOnly || !svgRef.current) return;
      // Get the node's output port position in SVG coordinates
      const nodePosition = layout.nodes.get(nodeId);
      if (!nodePosition) return;
      const portPosition = getNodeBottomCenter(nodePosition);
      startDrag(nodeId, portPosition);
    },
    [readOnly, layout, startDrag]
  );

  // Handle mouse move during drag
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (!dragState.isDragging || !svgRef.current) return;
      const svgPosition = screenToSVG(svgRef.current, e.clientX, e.clientY);
      updateDrag(svgPosition);
    },
    [dragState.isDragging, updateDrag]
  );

  // Handle mouse up (end drag)
  const handleMouseUp = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (dragState.isDragging && svgRef.current) {
        const svgPosition = screenToSVG(svgRef.current, e.clientX, e.clientY);
        endDrag(svgPosition);
      }
    },
    [dragState.isDragging, endDrag]
  );

  // Handle mouse leave (cancel drag)
  const handleMouseLeave = useCallback(() => {
    if (dragState.isDragging) {
      cancelDrag();
    }
  }, [dragState.isDragging, cancelDrag]);

  // Handle Escape key to cancel drag
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && dragState.isDragging) {
        cancelDrag();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [dragState.isDragging, cancelDrag]);

  // Create node and connect from source(s)
  const createNodeAndConnect = useCallback(
    (pluginId: string, tier: NodeType, sourceNodeIds: string[]) => {
      if (!onDagChange) return;

      // Create new node
      const newNode: PipelineNode = {
        id: generateNodeId(),
        type: tier,
        plugin_id: pluginId,
        config: {},
      };

      // Create edges from all source nodes
      const newEdges: PipelineEdgeType[] = sourceNodeIds.map((sourceId) => ({
        from_node: sourceId,
        to_node: newNode.id,
        when: null, // Default to catch-all
      }));

      const newDag = {
        ...dag,
        nodes: [...dag.nodes, newNode],
        edges: [...dag.edges, ...newEdges],
      };

      onDagChange(newDag);

      // Track new node/edges for entrance animation
      setNewNodeIds((prev) => new Set([...prev, newNode.id]));
      setNewEdgeKeys((prev) => {
        const edgeKeys = newEdges.map((e) => `${e.from_node}-${e.to_node}`);
        return new Set([...prev, ...edgeKeys]);
      });

      // Select the new node
      if (onSelectionChange) {
        onSelectionChange({ type: 'node', nodeId: newNode.id });
      }
    },
    [dag, onDagChange, onSelectionChange]
  );

  // Handle plugin selection from popover
  const handlePluginSelect = useCallback(
    (pluginId: string) => {
      if (!popover || !onDagChange) {
        setPopover(null);
        return;
      }

      if (popover.origin === 'button') {
        // "+" button origin - need to determine upstream
        const prevTierType = getPreviousTier(popover.tier);
        let upstreamNodes: PipelineNode[] = [];

        if (prevTierType === '_source') {
          // Only source can connect to parser tier
          upstreamNodes = []; // Will connect from _source
        } else if (prevTierType) {
          // Find all nodes in the previous tier
          upstreamNodes = dag.nodes.filter((n) => n.type === prevTierType);
        }

        // Also check if chunker can connect to embedder (skip extractor)
        if (popover.tier === 'embedder') {
          const chunkers = dag.nodes.filter((n) => n.type === 'chunker');
          const extractors = dag.nodes.filter((n) => n.type === 'extractor');
          upstreamNodes = [...extractors, ...chunkers];
        }

        if (upstreamNodes.length > 1) {
          // Multiple upstream nodes - show picker
          setUpstreamPicker({
            tier: popover.tier,
            pluginId,
            upstreamNodes,
            position: popover.position,
          });
          setPopover(null);
          return;
        }

        // Single or no upstream node - proceed with creation
        const sourceNodeId =
          upstreamNodes.length === 1 ? upstreamNodes[0].id : '_source';

        createNodeAndConnect(pluginId, popover.tier, [sourceNodeId]);
      } else {
        // Drag-and-drop case - sourceNodeId is known
        createNodeAndConnect(pluginId, popover.tier, [popover.sourceNodeId]);
      }

      setPopover(null);
    },
    [popover, dag, onDagChange, createNodeAndConnect]
  );

  // Clear "new" status after animation completes
  useEffect(() => {
    if (newNodeIds.size === 0 && newEdgeKeys.size === 0) return;

    const timer = setTimeout(() => {
      setNewNodeIds(new Set());
      setNewEdgeKeys(new Set());
    }, 400); // Animation duration + buffer

    return () => clearTimeout(timer);
  }, [newNodeIds, newEdgeKeys]);

  // Handle upstream node selection
  const handleUpstreamSelect = useCallback(
    (nodeIds: string[]) => {
      if (!upstreamPicker) return;

      createNodeAndConnect(upstreamPicker.pluginId, upstreamPicker.tier, nodeIds);
      setUpstreamPicker(null);
    },
    [upstreamPicker, createNodeAndConnect]
  );

  // Handle "+" button click
  const handleAddButtonClick = useCallback(
    (tier: NodeType, position: { x: number; y: number }) => {
      // Convert SVG position to screen position for popover
      let screenPosition = position;
      if (svgRef.current) {
        const ctm = svgRef.current.getScreenCTM();
        if (ctm) {
          const point = svgRef.current.createSVGPoint();
          point.x = position.x;
          point.y = position.y;
          const transformed = point.matrixTransform(ctm);
          screenPosition = { x: transformed.x, y: transformed.y };
        } else if (import.meta.env.DEV) {
          console.warn('handleAddButtonClick: getScreenCTM() returned null, using raw SVG coordinates');
        }
      }

      setPopover({
        origin: 'button',
        tier,
        position: screenPosition,
      });
    },
    []
  );

  // Cancel popover
  const handlePopoverCancel = useCallback(() => {
    setPopover(null);
  }, []);

  // Cancel upstream picker
  const handleUpstreamCancel = useCallback(() => {
    setUpstreamPicker(null);
  }, []);

  // Execute node deletion
  const executeNodeDeletion = useCallback(
    (nodeId: string, includeOrphans: boolean = false) => {
      if (!onDagChange) return;

      let nodesToDelete = [nodeId];

      if (includeOrphans) {
        const orphanedNodes = findOrphanedNodes(dag, nodeId);
        nodesToDelete = [...nodesToDelete, ...orphanedNodes.map((n) => n.id)];
      }

      const newDag = {
        ...dag,
        nodes: dag.nodes.filter((n) => !nodesToDelete.includes(n.id)),
        edges: dag.edges.filter(
          (e) => !nodesToDelete.includes(e.from_node) && !nodesToDelete.includes(e.to_node)
        ),
      };

      onDagChange(newDag);
      onSelectionChange?.({ type: 'none' });
    },
    [dag, onDagChange, onSelectionChange]
  );

  // Execute edge deletion
  const executeEdgeDeletion = useCallback(
    (fromNode: string, toNode: string, deleteOrphan: boolean = false) => {
      if (!onDagChange) return;

      let newNodes = dag.nodes;
      let newEdges = dag.edges.filter(
        (e) => !(e.from_node === fromNode && e.to_node === toNode)
      );

      if (deleteOrphan) {
        // Find and remove orphaned nodes
        const orphanedNodes = findOrphanedNodesAfterEdgeDeletion(dag, fromNode, toNode);
        const orphanedIds = orphanedNodes.map((n) => n.id);
        newNodes = newNodes.filter((n) => !orphanedIds.includes(n.id));
        newEdges = newEdges.filter(
          (e) => !orphanedIds.includes(e.from_node) && !orphanedIds.includes(e.to_node)
        );
      }

      const newDag = { ...dag, nodes: newNodes, edges: newEdges };
      onDagChange(newDag);
      onSelectionChange?.({ type: 'none' });
    },
    [dag, onDagChange, onSelectionChange]
  );

  // Handle node deletion request
  const handleDeleteNode = useCallback(
    (nodeId: string) => {
      // Cannot delete source node
      if (nodeId === '_source') return;

      // Find orphaned nodes
      const orphanedNodes = findOrphanedNodes(dag, nodeId);

      if (orphanedNodes.length > 0) {
        // Show confirmation dialog
        setDeleteConfirmation({
          type: 'node',
          nodeId,
          orphanedNodes,
        });
      } else {
        // Delete immediately
        executeNodeDeletion(nodeId);
      }
    },
    [dag, executeNodeDeletion]
  );

  // Handle edge deletion request
  const handleDeleteEdge = useCallback(
    (fromNode: string, toNode: string) => {
      // Check if this would orphan the target node
      const orphanedNodes = findOrphanedNodesAfterEdgeDeletion(dag, fromNode, toNode);

      if (orphanedNodes.length > 0) {
        // Show confirmation
        setDeleteConfirmation({
          type: 'edge',
          fromNode,
          toNode,
          orphanedNodes,
        });
      } else {
        // Safe to delete
        executeEdgeDeletion(fromNode, toNode);
      }
    },
    [dag, executeEdgeDeletion]
  );

  // Handle delete action based on current selection
  const handleDelete = useCallback(() => {
    if (selection.type === 'node') {
      handleDeleteNode(selection.nodeId);
    } else if (selection.type === 'edge') {
      handleDeleteEdge(selection.fromNode, selection.toNode);
    }
  }, [selection, handleDeleteNode, handleDeleteEdge]);

  // Handle delete confirmation
  const handleDeleteConfirm = useCallback(
    (deleteOrphans: boolean) => {
      if (!deleteConfirmation) return;

      if (deleteConfirmation.type === 'node') {
        executeNodeDeletion(deleteConfirmation.nodeId, deleteOrphans);
      } else {
        executeEdgeDeletion(deleteConfirmation.fromNode, deleteConfirmation.toNode, deleteOrphans);
      }

      setDeleteConfirmation(null);
    },
    [deleteConfirmation, executeNodeDeletion, executeEdgeDeletion]
  );

  // Handle delete cancellation
  const handleDeleteCancel = useCallback(() => {
    setDeleteConfirmation(null);
  }, []);

  // Handle Delete/Backspace key to delete selected element
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.key === 'Delete' || e.key === 'Backspace') && selection.type !== 'none' && !readOnly) {
        // Don't trigger if user is typing in an input
        const target = e.target as HTMLElement;
        if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) {
          return;
        }
        e.preventDefault();
        handleDelete();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selection, readOnly, handleDelete]);

  // Check if node is selected
  const isNodeSelected = (nodeId: string) =>
    selection.type === 'node' && selection.nodeId === nodeId;

  // Check if edge is selected
  const isEdgeSelected = (fromNode: string, toNode: string) =>
    selection.type === 'edge' &&
    selection.fromNode === fromNode &&
    selection.toNode === toNode;

  // Check if node is in highlighted path
  const isNodeInPath = (nodeId: string) =>
    highlightedPath?.includes(nodeId) ?? false;

  // Check if edge is in highlighted path (both from and to nodes must be in path and consecutive)
  const isEdgeInPath = (fromNode: string, toNode: string) => {
    if (!highlightedPath || highlightedPath.length < 2) return false;
    for (let i = 0; i < highlightedPath.length - 1; i++) {
      if (highlightedPath[i] === fromNode && highlightedPath[i + 1] === toNode) {
        return true;
      }
    }
    return false;
  };

  // Get valid target tiers for current drag
  const validTargetTiers = useMemo(() => {
    if (!dragState.isDragging) return [];
    return getValidTiers(dragState.sourceNodeId);
  }, [dragState, getValidTiers]);

  // Compute "+" button positions for each tier
  const addButtonPositions = useMemo(() => {
    const positions: { tier: NodeType; x: number; y: number }[] = [];

    // Group existing nodes by tier
    const nodesByTier = new Map<number, string[]>();
    for (const node of dag.nodes) {
      const tier = TYPE_TIERS[node.type];
      if (!nodesByTier.has(tier)) {
        nodesByTier.set(tier, []);
      }
      nodesByTier.get(tier)!.push(node.id);
    }

    for (const tierType of ALL_TIERS) {
      const tierIndex = TYPE_TIERS[tierType];
      const tierY = PADDING + tierIndex * (NODE_HEIGHT + TIER_GAP) + NODE_HEIGHT / 2;

      const existingNodes = nodesByTier.get(tierIndex) || [];

      if (existingNodes.length > 0) {
        // Find rightmost node
        let maxX = 0;
        for (const nodeId of existingNodes) {
          const pos = layout.nodes.get(nodeId);
          if (pos) {
            maxX = Math.max(maxX, pos.x + pos.width);
          }
        }
        positions.push({
          tier: tierType,
          x: maxX + NODE_GAP + 14, // 14 is button radius
          y: tierY,
        });
      } else {
        // Empty tier - center the button
        positions.push({
          tier: tierType,
          x: PADDING + NODE_WIDTH / 2,
          y: tierY,
        });
      }
    }

    return positions;
  }, [dag.nodes, layout]);

  // Compute extended layout height to include all tiers
  const extendedHeight = useMemo(() => {
    // Ensure we have space for all tiers including embedder (tier 4)
    const maxTierIndex = 4; // embedder
    const neededHeight = PADDING + (maxTierIndex + 1) * (NODE_HEIGHT + TIER_GAP);
    return Math.max(layout.height, neededHeight);
  }, [layout.height]);

  // Empty state
  if (dag.nodes.length === 0) {
    return (
      <div className={`flex items-center justify-center h-full ${className}`}>
        <div className="text-center">
          <p className="text-[var(--text-muted)]">No pipeline configured</p>
          <p className="text-sm text-[var(--text-muted)] mt-1">
            Add nodes to build your pipeline
          </p>
        </div>
      </div>
    );
  }

  // Get source node position (always present when edges exist)
  const sourcePosition = layout.nodes.get('_source');

  return (
    <div ref={containerRef} className={`overflow-auto ${className}`}>
      <svg
        ref={svgRef}
        width={layout.width}
        height={extendedHeight}
        onClick={handleBackgroundClick}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        style={{ minWidth: '100%', minHeight: '100%' }}
      >
        {/* Arrow marker definition */}
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 10 3.5, 0 7" fill="var(--text-muted)" />
          </marker>
          <marker
            id="arrowhead-selected"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 10 3.5, 0 7" fill="var(--text-primary)" />
          </marker>
          <marker
            id="arrowhead-highlighted"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 10 3.5, 0 7" fill="rgb(34, 197, 94)" />
          </marker>
        </defs>

        {/* Render tier drop zones (during drag only) */}
        {dragState.isDragging && (
          <g className="tier-drop-zones">
            {tierBounds.map((bounds) => {
              const isActive = validTargetTiers.includes(bounds.tier);
              const isHovered = hoveredTier === bounds.tier;
              return (
                <TierDropZone
                  key={bounds.tier}
                  tier={bounds.tier}
                  bounds={bounds}
                  isActive={isActive}
                  isHovered={isHovered}
                />
              );
            })}
          </g>
        )}

        {/* Render edges first (below nodes) */}
        <g className="edges">
          {dag.edges.map((edge, index) => {
            const fromPos = layout.nodes.get(edge.from_node);
            const toPos = layout.nodes.get(edge.to_node);

            if (!fromPos || !toPos) return null;

            const edgeKey = `${edge.from_node}-${edge.to_node}`;
            const selected = isEdgeSelected(edge.from_node, edge.to_node);
            // Show catch-all (*) only for source edges
            const showCatchAll = edge.from_node === '_source' && edge.when === null;

            return (
              <PipelineEdgeComponent
                key={`${edgeKey}-${index}`}
                edge={edge}
                fromPosition={fromPos}
                toPosition={toPos}
                selected={selected}
                showCatchAll={showCatchAll}
                onClick={readOnly ? undefined : handleEdgeClick}
                isNew={newEdgeKeys.has(edgeKey)}
                isHighlighted={isEdgeInPath(edge.from_node, edge.to_node)}
              />
            );
          })}
        </g>

        {/* Render source node */}
        {sourcePosition && (
          <PipelineNodeComponent
            node={{ id: '_source', type: 'parser', plugin_id: 'source', config: {} }}
            position={sourcePosition}
            selected={isNodeSelected('_source')}
            isSource={true}
            onClick={readOnly ? undefined : handleNodeClick}
            onStartDrag={readOnly || isTouch ? undefined : handleStartDrag}
            showPorts={!isTouch && dragState.isDragging}
            isInPath={isNodeInPath('_source')}
          />
        )}

        {/* Render pipeline nodes */}
        <g className="nodes">
          {dag.nodes.map((node) => {
            const position = layout.nodes.get(node.id);
            if (!position) return null;

            return (
              <PipelineNodeComponent
                key={node.id}
                node={node}
                position={position}
                selected={isNodeSelected(node.id)}
                onClick={readOnly ? undefined : handleNodeClick}
                onStartDrag={readOnly || isTouch ? undefined : handleStartDrag}
                showPorts={!isTouch && dragState.isDragging}
                isValidDropTarget={!isTouch && isValidDropTarget(node.id)}
                isNew={newNodeIds.has(node.id)}
                isInPath={isNodeInPath(node.id)}
              />
            );
          })}
        </g>

        {/* Render "+" buttons (when not dragging and not read-only) */}
        {/* On touch devices, always visible; on desktop, visible when not dragging */}
        {!readOnly && (isTouch || !dragState.isDragging) && (
          <g className="tier-add-buttons">
            {addButtonPositions.map(({ tier, x, y }) => (
              <TierAddButton
                key={tier}
                tier={tier}
                position={{ x, y }}
                onClick={() => handleAddButtonClick(tier, { x, y })}
                isTouch={isTouch}
              />
            ))}
          </g>
        )}

        {/* Render preview edge during drag */}
        {dragState.isDragging && (
          <DragPreviewEdge from={dragState.sourcePosition} to={dragState.cursorPosition} />
        )}
      </svg>

      {/* Render node picker popover (outside SVG) */}
      {popover && (
        <NodePickerPopover
          tier={popover.tier}
          position={popover.position}
          onSelect={handlePluginSelect}
          onCancel={handlePopoverCancel}
        />
      )}

      {/* Render upstream node picker (outside SVG) */}
      {upstreamPicker && (
        <UpstreamNodePicker
          upstreamNodes={upstreamPicker.upstreamNodes}
          position={upstreamPicker.position}
          onSelect={handleUpstreamSelect}
          onCancel={handleUpstreamCancel}
        />
      )}

      {/* Render delete confirmation dialog */}
      {deleteConfirmation && (
        <DeleteConfirmationDialog
          type={deleteConfirmation.type}
          message={
            deleteConfirmation.type === 'node'
              ? `Delete the ${dag.nodes.find((n) => n.id === deleteConfirmation.nodeId)?.plugin_id || 'node'} node?`
              : `Delete the edge from ${deleteConfirmation.fromNode === '_source' ? 'Source' : dag.nodes.find((n) => n.id === deleteConfirmation.fromNode)?.plugin_id || deleteConfirmation.fromNode} to ${dag.nodes.find((n) => n.id === deleteConfirmation.toNode)?.plugin_id || deleteConfirmation.toNode}?`
          }
          orphanedNodes={deleteConfirmation.orphanedNodes}
          onConfirm={handleDeleteConfirm}
          onCancel={handleDeleteCancel}
        />
      )}
    </div>
  );
}

export default PipelineVisualization;
