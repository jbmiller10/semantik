import type { Node, Edge } from '@xyflow/react';

// ============================================================================
// Entity Types (from API)
// ============================================================================

/** Entity response from the backend GraphRAG API */
export interface EntityResponse {
  id: number;
  name: string;
  entity_type: string;
  document_id: string;
  chunk_id: number | null;
  confidence: number;
  created_at: string;
}

// ============================================================================
// Entity Search Types
// ============================================================================

/** Request model for entity search */
export interface EntitySearchRequest {
  query?: string | null;
  entity_types?: string[] | null;
  limit?: number;
  offset?: number;
}

/** Response model for entity search */
export interface EntitySearchResponse {
  entities: EntityResponse[];
  total: number;
  has_more: boolean;
}

// ============================================================================
// Graph Traversal Types
// ============================================================================

/** Request model for graph traversal */
export interface GraphTraversalRequest {
  entity_id: number;
  max_hops?: number;
  relationship_types?: string[] | null;
}

/** Node in a graph traversal response with hop distance */
export interface GraphNode {
  id: number;
  name: string;
  type: string;
  hop: number;
}

/** Edge in a graph traversal response */
export interface GraphEdge {
  id: number;
  source: number;
  target: number;
  type: string;
  confidence: number;
}

/** Response from graph traversal API */
export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

// ============================================================================
// Graph Statistics Types
// ============================================================================

/** Graph statistics response for a collection */
export interface GraphStatsResponse {
  total_entities: number;
  entities_by_type: Record<string, number>;
  total_relationships: number;
  relationships_by_type: Record<string, number>;
  graph_enabled: boolean;
}

// ============================================================================
// React Flow Custom Types
// ============================================================================

/** Custom node data type for entity nodes */
export interface EntityNodeData extends Record<string, unknown> {
  label: string;
  entityType: string;
  hop: number;
  entityId: number;
  confidence?: number;
  documentId?: string;
  chunkId?: number | null;
}

/** Custom edge data type for relationship edges */
export interface RelationshipEdgeData extends Record<string, unknown> {
  relationshipType: string;
  confidence: number;
}

/** Typed React Flow node for entities */
export type EntityNode = Node<EntityNodeData>;

/** Typed React Flow edge for relationships */
export type RelationshipEdge = Edge<RelationshipEdgeData>;

// ============================================================================
// Entity Color Mapping
// ============================================================================

/** Standard entity type colors */
export const ENTITY_COLORS: Record<string, string> = {
  PERSON: '#3B82F6',      // Blue
  ORG: '#10B981',         // Green
  GPE: '#F59E0B',         // Amber (Geo-political entity)
  LOC: '#8B5CF6',         // Purple
  PRODUCT: '#EC4899',     // Pink
  EVENT: '#06B6D4',       // Cyan
  DATE: '#6B7280',        // Gray
  MONEY: '#84CC16',       // Lime
  PERCENT: '#F97316',     // Orange
  TIME: '#64748B',        // Slate
  QUANTITY: '#A855F7',    // Violet
  ORDINAL: '#78716C',     // Stone
  CARDINAL: '#71717A',    // Zinc
  FAC: '#EAB308',         // Yellow (Facility)
  NORP: '#14B8A6',        // Teal (Nationality/Religious/Political group)
  LAW: '#DC2626',         // Red
  LANGUAGE: '#2563EB',    // Blue
  WORK_OF_ART: '#DB2777', // Pink
  DEFAULT: '#9CA3AF',     // Light gray
};

/** Get color for an entity type */
export function getEntityColor(type: string): string {
  return ENTITY_COLORS[type.toUpperCase()] || ENTITY_COLORS.DEFAULT;
}

/** Get a lighter/background version of the entity color */
export function getEntityBackgroundColor(type: string): string {
  const color = getEntityColor(type);
  // Add transparency for background usage
  return `${color}20`;
}
