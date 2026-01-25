/**
 * Mapping utilities between pipeline NodeType and plugin PluginType.
 *
 * Pipeline nodes use slightly different naming than plugins:
 * - NodeType: 'parser' | 'chunker' | 'extractor' | 'embedder'
 * - PluginType: 'parser' | 'chunking' | 'extractor' | 'embedding' | ...
 */

import type { NodeType, PipelineNode } from '@/types/pipeline';
import type { PluginType } from '@/types/plugin';

/**
 * Map from pipeline NodeType to plugin PluginType.
 */
const NODE_TO_PLUGIN_TYPE: Record<NodeType, PluginType> = {
  parser: 'parser',
  chunker: 'chunking',
  extractor: 'extractor',
  embedder: 'embedding',
};

/**
 * Convert a pipeline node type to the corresponding plugin type.
 */
export function nodeTypeToPluginType(nodeType: NodeType): PluginType {
  return NODE_TO_PLUGIN_TYPE[nodeType];
}

/**
 * Get the plugin type for a pipeline node.
 * Returns null for source nodes (which don't have plugins).
 */
export function getPluginTypeForNode(
  node: PipelineNode,
  isSource: boolean = false
): PluginType | null {
  if (isSource || node.id === '_source') {
    return null;
  }
  return nodeTypeToPluginType(node.type);
}

/**
 * Human-readable label for a node type.
 */
export const NODE_TYPE_LABELS: Record<NodeType, string> = {
  parser: 'Parser',
  chunker: 'Chunker',
  extractor: 'Extractor',
  embedder: 'Embedder',
};
