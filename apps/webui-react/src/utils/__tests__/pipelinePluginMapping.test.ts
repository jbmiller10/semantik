import { describe, it, expect } from 'vitest';
import { nodeTypeToPluginType, getPluginTypeForNode } from '../pipelinePluginMapping';
import type { NodeType } from '@/types/pipeline';

describe('nodeTypeToPluginType', () => {
  it('maps parser to parser', () => {
    expect(nodeTypeToPluginType('parser')).toBe('parser');
  });

  it('maps chunker to chunking', () => {
    expect(nodeTypeToPluginType('chunker')).toBe('chunking');
  });

  it('maps extractor to extractor', () => {
    expect(nodeTypeToPluginType('extractor')).toBe('extractor');
  });

  it('maps embedder to embedding', () => {
    expect(nodeTypeToPluginType('embedder')).toBe('embedding');
  });
});

describe('getPluginTypeForNode', () => {
  it('returns correct plugin type for node', () => {
    const node = { id: 'test', type: 'chunker' as NodeType, plugin_id: 'recursive', config: {} };
    expect(getPluginTypeForNode(node)).toBe('chunking');
  });

  it('returns null for source node', () => {
    const node = { id: '_source', type: 'parser' as NodeType, plugin_id: 'source', config: {} };
    expect(getPluginTypeForNode(node, true)).toBeNull();
  });
});
