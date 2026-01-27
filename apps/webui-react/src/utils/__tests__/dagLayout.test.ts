import { describe, it, expect } from 'vitest';
import { computeDAGLayout, getNodeTopCenter, getNodeBottomCenter } from '../dagLayout';
import type { PipelineDAG, NodePosition } from '@/types/pipeline';

describe('computeDAGLayout', () => {
  it('positions nodes in tiers vertically by type (top-to-bottom flow)', () => {
    const dag: PipelineDAG = {
      id: 'test',
      version: '1',
      nodes: [
        { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
        { id: 'chunker1', type: 'chunker', plugin_id: 'recursive', config: {} },
        { id: 'embedder1', type: 'embedder', plugin_id: 'dense_local', config: {} },
      ],
      edges: [
        { from_node: '_source', to_node: 'parser1', when: null },
        { from_node: 'parser1', to_node: 'chunker1', when: null },
        { from_node: 'chunker1', to_node: 'embedder1', when: null },
      ],
    };

    const layout = computeDAGLayout(dag);

    const source = layout.nodes.get('_source');
    const parser = layout.nodes.get('parser1');
    const chunker = layout.nodes.get('chunker1');
    const embedder = layout.nodes.get('embedder1');

    expect(source).toBeDefined();
    expect(parser).toBeDefined();
    expect(chunker).toBeDefined();
    expect(embedder).toBeDefined();

    // Vertical flow: Y increases as we go down the pipeline
    expect(source!.y).toBeLessThan(parser!.y);
    expect(parser!.y).toBeLessThan(chunker!.y);
    expect(chunker!.y).toBeLessThan(embedder!.y);
  });

  it('spreads multiple nodes of same type horizontally within their tier', () => {
    const dag: PipelineDAG = {
      id: 'test',
      version: '1',
      nodes: [
        { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
        { id: 'parser2', type: 'parser', plugin_id: 'unstructured', config: {} },
        { id: 'chunker1', type: 'chunker', plugin_id: 'recursive', config: {} },
        { id: 'embedder1', type: 'embedder', plugin_id: 'dense_local', config: {} },
      ],
      edges: [
        { from_node: '_source', to_node: 'parser1', when: { mime_type: 'text/*' } },
        { from_node: '_source', to_node: 'parser2', when: null },
        { from_node: 'parser1', to_node: 'chunker1', when: null },
        { from_node: 'parser2', to_node: 'chunker1', when: null },
        { from_node: 'chunker1', to_node: 'embedder1', when: null },
      ],
    };

    const layout = computeDAGLayout(dag);

    const parser1 = layout.nodes.get('parser1');
    const parser2 = layout.nodes.get('parser2');

    // Same Y position (same tier)
    expect(parser1!.y).toBe(parser2!.y);
    // Different X positions (spread horizontally)
    expect(parser1!.x).not.toBe(parser2!.x);
  });

  it('includes source node in layout at top tier', () => {
    const dag: PipelineDAG = {
      id: 'test',
      version: '1',
      nodes: [
        { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
        { id: 'embedder1', type: 'embedder', plugin_id: 'dense_local', config: {} },
      ],
      edges: [
        { from_node: '_source', to_node: 'parser1', when: null },
        { from_node: 'parser1', to_node: 'embedder1', when: null },
      ],
    };

    const layout = computeDAGLayout(dag);

    const source = layout.nodes.get('_source');
    expect(source).toBeDefined();
    // Source is at the top (tier 0), parser below it
    const parser = layout.nodes.get('parser1');
    expect(source!.y).toBeLessThan(parser!.y);
  });

  it('computes overall width and height', () => {
    const dag: PipelineDAG = {
      id: 'test',
      version: '1',
      nodes: [
        { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
        { id: 'embedder1', type: 'embedder', plugin_id: 'dense_local', config: {} },
      ],
      edges: [
        { from_node: '_source', to_node: 'parser1', when: null },
        { from_node: 'parser1', to_node: 'embedder1', when: null },
      ],
    };

    const layout = computeDAGLayout(dag);

    expect(layout.width).toBeGreaterThan(0);
    expect(layout.height).toBeGreaterThan(0);
  });
});

describe('getNodeTopCenter', () => {
  it('returns center of top edge', () => {
    const pos: NodePosition = { x: 100, y: 50, width: 160, height: 80 };
    const result = getNodeTopCenter(pos);
    expect(result).toEqual({ x: 180, y: 50 }); // x + width/2, y
  });
});

describe('getNodeBottomCenter', () => {
  it('returns center of bottom edge', () => {
    const pos: NodePosition = { x: 100, y: 50, width: 160, height: 80 };
    const result = getNodeBottomCenter(pos);
    expect(result).toEqual({ x: 180, y: 130 }); // x + width/2, y + height
  });
});
