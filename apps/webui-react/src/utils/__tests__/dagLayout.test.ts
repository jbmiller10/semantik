import { describe, it, expect } from 'vitest';
import { computeDAGLayout } from '../dagLayout';
import type { PipelineDAG } from '@/types/pipeline';

describe('computeDAGLayout', () => {
  it('positions nodes in columns by type', () => {
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

    // Parser should be in first column (after source)
    const parser = layout.nodes.get('parser1');
    expect(parser).toBeDefined();

    // Chunker should be in second column
    const chunker = layout.nodes.get('chunker1');
    expect(chunker).toBeDefined();
    expect(chunker!.x).toBeGreaterThan(parser!.x);

    // Embedder should be in third column
    const embedder = layout.nodes.get('embedder1');
    expect(embedder).toBeDefined();
    expect(embedder!.x).toBeGreaterThan(chunker!.x);
  });

  it('stacks multiple nodes of same type vertically', () => {
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

    // Same x position (same column)
    expect(parser1!.x).toBe(parser2!.x);
    // Different y positions
    expect(parser1!.y).not.toBe(parser2!.y);
  });

  it('includes source node in layout', () => {
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
    // Source is in leftmost column (column 0) with padding
    const parser = layout.nodes.get('parser1');
    expect(source!.x).toBeLessThan(parser!.x);
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
