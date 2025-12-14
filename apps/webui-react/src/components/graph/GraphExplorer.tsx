import { useState, useCallback, useMemo, useEffect } from 'react';
import {
  ReactFlow,
  Controls,
  MiniMap,
  Background,
  Panel,
  useNodesState,
  useEdgesState,
  MarkerType,
  BackgroundVariant,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Loader2, AlertCircle, Network, Circle } from 'lucide-react';

import { useGraphTraversal } from '@/hooks/useGraph';
import {
  getEntityColor,
  ENTITY_COLORS,
  type GraphNode,
  type GraphEdge,
  type EntityNode,
  type RelationshipEdge,
} from '@/types/graph';

// ============================================================================
// Types
// ============================================================================

export interface GraphExplorerProps {
  /** Collection UUID to explore */
  collectionId: string;
  /** Initial entity ID to center the graph on */
  initialEntityId?: number | null;
  /** Callback when an entity is selected (double-click) */
  onEntitySelect?: (entityId: number) => void;
  /** Additional CSS classes */
  className?: string;
  /** Maximum hops for graph traversal (1-5, default 2) */
  maxHops?: number;
}

// ============================================================================
// Layout Utilities
// ============================================================================

/**
 * Calculate node positions in concentric circles based on hop distance.
 * Center node (hop 0) is at origin, hop 1 nodes form first ring, etc.
 */
function calculateCircularLayout(
  nodes: GraphNode[],
  centerRadius: number = 150
): Map<number, { x: number; y: number }> {
  const positions = new Map<number, { x: number; y: number }>();

  // Group nodes by hop distance
  const nodesByHop = new Map<number, GraphNode[]>();
  nodes.forEach((node) => {
    const hop = node.hop;
    if (!nodesByHop.has(hop)) {
      nodesByHop.set(hop, []);
    }
    nodesByHop.get(hop)!.push(node);
  });

  // Position center node (hop 0)
  const centerNodes = nodesByHop.get(0) || [];
  centerNodes.forEach((node) => {
    positions.set(node.id, { x: 0, y: 0 });
  });

  // Position nodes in concentric circles for each hop level
  const maxHop = Math.max(...Array.from(nodesByHop.keys()));
  for (let hop = 1; hop <= maxHop; hop++) {
    const hopNodes = nodesByHop.get(hop) || [];
    if (hopNodes.length === 0) continue;

    const radius = centerRadius * hop;
    const angleStep = (2 * Math.PI) / hopNodes.length;
    const startAngle = -Math.PI / 2; // Start from top

    hopNodes.forEach((node, index) => {
      const angle = startAngle + index * angleStep;
      positions.set(node.id, {
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
      });
    });
  }

  return positions;
}

/**
 * Transform API graph data to React Flow elements.
 */
function transformToFlowElements(
  graphData: { nodes: GraphNode[]; edges: GraphEdge[] },
  centerId: number
): { nodes: EntityNode[]; edges: RelationshipEdge[] } {
  const positions = calculateCircularLayout(graphData.nodes);

  const nodes: EntityNode[] = graphData.nodes.map((node) => {
    const isCenter = node.id === centerId;
    const color = getEntityColor(node.type);
    const position = positions.get(node.id) || { x: 0, y: 0 };

    return {
      id: String(node.id),
      position,
      data: {
        label: node.name,
        entityType: node.type,
        hop: node.hop,
        entityId: node.id,
      },
      style: {
        background: color,
        color: 'white',
        border: 'none',
        borderRadius: '8px',
        padding: '8px 12px',
        fontSize: '12px',
        fontWeight: isCenter ? 700 : 500,
        boxShadow: isCenter
          ? '0 4px 12px rgba(0, 0, 0, 0.3)'
          : '0 2px 4px rgba(0, 0, 0, 0.1)',
        minWidth: '80px',
        textAlign: 'center' as const,
      },
    };
  });

  const edges: RelationshipEdge[] = graphData.edges.map((edge) => {
    const confidence = edge.confidence;
    // Edge width scales with confidence (1-4px)
    const strokeWidth = 1 + Math.floor(confidence * 3);
    // Animate low confidence edges
    const animated = confidence < 0.6;

    return {
      id: String(edge.id),
      source: String(edge.source),
      target: String(edge.target),
      label: edge.type,
      data: {
        relationshipType: edge.type,
        confidence: edge.confidence,
      },
      style: {
        strokeWidth,
        stroke: '#6B7280',
      },
      labelStyle: {
        fontSize: '10px',
        fill: '#4B5563',
        fontWeight: 500,
      },
      labelBgStyle: {
        fill: 'white',
        fillOpacity: 0.9,
      },
      labelBgPadding: [4, 2] as [number, number],
      labelBgBorderRadius: 4,
      animated,
      markerEnd: {
        type: MarkerType.ArrowClosed,
        width: 16,
        height: 16,
        color: '#6B7280',
      },
    };
  });

  return { nodes, edges };
}

// ============================================================================
// Sub-Components
// ============================================================================

/** Loading spinner */
function LoadingState() {
  return (
    <div className="flex flex-col items-center justify-center h-full">
      <Loader2 className="w-10 h-10 text-blue-500 animate-spin mb-4" />
      <p className="text-gray-600 font-medium">Loading graph...</p>
    </div>
  );
}

/** Error display */
function ErrorState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-full">
      <AlertCircle className="w-12 h-12 text-red-400 mb-4" />
      <h3 className="text-lg font-medium text-gray-900 mb-2">
        Error Loading Graph
      </h3>
      <p className="text-sm text-red-600 text-center max-w-sm">{message}</p>
    </div>
  );
}

/** Empty state when no entity is selected */
function EmptySelectionState() {
  return (
    <div className="flex flex-col items-center justify-center h-full">
      <Network className="w-12 h-12 text-gray-300 mb-4" />
      <h3 className="text-lg font-medium text-gray-900 mb-2">
        Select an Entity to Explore
      </h3>
      <p className="text-sm text-gray-500 text-center max-w-sm">
        Choose an entity from the browser to visualize its relationships
      </p>
    </div>
  );
}

/** Empty state when entity has no relationships */
function NoRelationshipsState() {
  return (
    <div className="flex flex-col items-center justify-center h-full">
      <Network className="w-12 h-12 text-gray-300 mb-4" />
      <h3 className="text-lg font-medium text-gray-900 mb-2">
        No Relationships Found
      </h3>
      <p className="text-sm text-gray-500 text-center max-w-sm">
        This entity has no connected relationships in the knowledge graph
      </p>
    </div>
  );
}

/** Legend panel showing entity type colors */
function LegendPanel({ visibleTypes }: { visibleTypes: string[] }) {
  // Only show types that are present in the current graph
  const typesToShow = visibleTypes.length > 0 ? visibleTypes : Object.keys(ENTITY_COLORS).slice(0, 8);

  return (
    <div className="bg-white rounded-lg shadow-lg p-3 border border-gray-200 max-w-xs">
      <h4 className="text-xs font-semibold text-gray-700 mb-2">Entity Types</h4>
      <div className="flex flex-wrap gap-2">
        {typesToShow.map((type) => {
          if (type === 'DEFAULT') return null;
          return (
            <div key={type} className="flex items-center gap-1.5">
              <Circle
                className="w-3 h-3 flex-shrink-0"
                fill={getEntityColor(type)}
                stroke="none"
              />
              <span className="text-xs text-gray-600">{type}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** Info panel showing node/edge counts */
function InfoPanel({ nodeCount, edgeCount }: { nodeCount: number; edgeCount: number }) {
  return (
    <div className="bg-white rounded-lg shadow-lg px-3 py-2 border border-gray-200">
      <div className="flex gap-4 text-xs">
        <span className="text-gray-600">
          <span className="font-semibold text-gray-800">{nodeCount}</span> nodes
        </span>
        <span className="text-gray-600">
          <span className="font-semibold text-gray-800">{edgeCount}</span> edges
        </span>
      </div>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

/**
 * GraphExplorer component for visualizing the knowledge graph using React Flow.
 *
 * Features:
 * - Circular layout positions nodes by hop distance from center
 * - Nodes colored by entity type with center node visually distinct
 * - Clicking a node re-centers the graph on that entity
 * - Double-clicking selects the entity without re-centering
 * - Edge width reflects confidence, low-confidence edges are animated
 * - Built-in controls, minimap, legend, and info panel
 * - Handles loading, error, and empty states
 */
export function GraphExplorer({
  collectionId,
  initialEntityId = null,
  onEntitySelect,
  className = '',
  maxHops = 2,
}: GraphExplorerProps) {
  // Currently centered entity ID (can change when user clicks nodes)
  const [centeredEntityId, setCenteredEntityId] = useState<number | null>(initialEntityId);

  // React Flow state
  const [nodes, setNodes, onNodesChange] = useNodesState<EntityNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<RelationshipEdge>([]);

  // Track visible entity types for legend
  const visibleTypes = useMemo(() => {
    const types = new Set<string>();
    nodes.forEach((node) => {
      if (node.data?.entityType) {
        types.add(node.data.entityType);
      }
    });
    return Array.from(types).sort();
  }, [nodes]);

  // Fetch graph data
  const {
    data: graphData,
    isLoading,
    error,
  } = useGraphTraversal(collectionId, centeredEntityId, maxHops);

  // Update centered entity when initialEntityId changes
  useEffect(() => {
    if (initialEntityId !== null) {
      setCenteredEntityId(initialEntityId);
    }
  }, [initialEntityId]);

  // Transform and set flow elements when graph data changes
  useEffect(() => {
    if (graphData && centeredEntityId !== null) {
      const { nodes: flowNodes, edges: flowEdges } = transformToFlowElements(
        graphData,
        centeredEntityId
      );
      setNodes(flowNodes);
      setEdges(flowEdges);
    }
  }, [graphData, centeredEntityId, setNodes, setEdges]);

  // Handle node click - re-center graph on clicked entity
  const handleNodeClick = useCallback(
    (_event: React.MouseEvent, node: EntityNode) => {
      const entityId = node.data?.entityId;
      if (entityId !== undefined && entityId !== centeredEntityId) {
        setCenteredEntityId(entityId);
      }
    },
    [centeredEntityId]
  );

  // Handle node double-click - select entity without re-centering
  const handleNodeDoubleClick = useCallback(
    (_event: React.MouseEvent, node: EntityNode) => {
      const entityId = node.data?.entityId;
      if (entityId !== undefined && onEntitySelect) {
        onEntitySelect(entityId);
      }
    },
    [onEntitySelect]
  );

  // Get minimap node color
  const getMinimapNodeColor = useCallback((node: EntityNode): string => {
    return node.data?.entityType ? getEntityColor(node.data.entityType) : '#9CA3AF';
  }, []);

  // Empty selection state
  if (centeredEntityId === null) {
    return (
      <div className={`relative ${className}`} style={{ minHeight: '400px' }}>
        <EmptySelectionState />
      </div>
    );
  }

  // Loading state
  if (isLoading) {
    return (
      <div className={`relative ${className}`} style={{ minHeight: '400px' }}>
        <LoadingState />
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className={`relative ${className}`} style={{ minHeight: '400px' }}>
        <ErrorState
          message={error instanceof Error ? error.message : 'An unexpected error occurred'}
        />
      </div>
    );
  }

  // No relationships state
  if (graphData && graphData.nodes.length <= 1) {
    return (
      <div className={`relative ${className}`} style={{ minHeight: '400px' }}>
        <NoRelationshipsState />
      </div>
    );
  }

  return (
    <div className={`relative ${className}`} style={{ minHeight: '400px' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={handleNodeClick}
        onNodeDoubleClick={handleNodeDoubleClick}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.1}
        maxZoom={2}
        attributionPosition="bottom-left"
        proOptions={{ hideAttribution: true }}
      >
        {/* Controls */}
        <Controls
          showInteractive={false}
          position="bottom-right"
        />

        {/* MiniMap */}
        <MiniMap
          nodeColor={getMinimapNodeColor}
          nodeStrokeWidth={2}
          zoomable
          pannable
          position="top-right"
        />

        {/* Background */}
        <Background
          variant={BackgroundVariant.Dots}
          gap={16}
          size={1}
          color="#E5E7EB"
        />

        {/* Legend Panel */}
        <Panel position="top-left">
          <LegendPanel visibleTypes={visibleTypes} />
        </Panel>

        {/* Info Panel */}
        <Panel position="bottom-left">
          <InfoPanel nodeCount={nodes.length} edgeCount={edges.length} />
        </Panel>
      </ReactFlow>
    </div>
  );
}

export default GraphExplorer;
