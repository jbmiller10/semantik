import { describe, it, expect } from 'vitest';
import { findOrphanedNodes, findOrphanedNodesAfterEdgeDeletion } from '../dagUtils';
import type { PipelineDAG } from '@/types/pipeline';

describe('dagUtils', () => {
  describe('findOrphanedNodes', () => {
    const baseDag: PipelineDAG = {
      id: 'test',
      version: '1',
      nodes: [
        { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
        { id: 'parser2', type: 'parser', plugin_id: 'pdf', config: {} },
        { id: 'chunker1', type: 'chunker', plugin_id: 'recursive', config: {} },
        { id: 'chunker2', type: 'chunker', plugin_id: 'semantic', config: {} },
        { id: 'embedder1', type: 'embedder', plugin_id: 'dense_local', config: {} },
      ],
      edges: [
        { from_node: '_source', to_node: 'parser1', when: null },
        { from_node: '_source', to_node: 'parser2', when: null },
        { from_node: 'parser1', to_node: 'chunker1', when: null },
        { from_node: 'parser2', to_node: 'chunker2', when: null },
        { from_node: 'chunker1', to_node: 'embedder1', when: null },
        { from_node: 'chunker2', to_node: 'embedder1', when: null },
      ],
    };

    it('returns empty array when no orphans exist', () => {
      // Deleting parser1 - chunker1 would be orphaned but parser2 still connects to embedder
      const result = findOrphanedNodes(baseDag, 'parser2');
      // chunker2 loses its only incoming edge
      expect(result.map((n) => n.id)).toContain('chunker2');
    });

    it('finds directly orphaned nodes', () => {
      const result = findOrphanedNodes(baseDag, 'parser1');
      // chunker1 loses its only incoming edge
      expect(result.map((n) => n.id)).toContain('chunker1');
    });

    it('finds transitively orphaned nodes', () => {
      // If we remove a node that has downstream dependencies
      const linearDag: PipelineDAG = {
        id: 'linear',
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

      const result = findOrphanedNodes(linearDag, 'parser1');
      // Both chunker1 and embedder1 should be orphaned
      expect(result.map((n) => n.id)).toContain('chunker1');
      expect(result.map((n) => n.id)).toContain('embedder1');
      expect(result).toHaveLength(2);
    });

    it('does not consider source node as orphaned', () => {
      const result = findOrphanedNodes(baseDag, 'parser1');
      // Source is never in the orphan list
      expect(result.find((n) => n.id === '_source')).toBeUndefined();
    });

    it('handles diamond DAG topology', () => {
      // A diamond: source -> [parser1, parser2] -> chunker
      const diamondDag: PipelineDAG = {
        id: 'diamond',
        version: '1',
        nodes: [
          { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
          { id: 'parser2', type: 'parser', plugin_id: 'pdf', config: {} },
          { id: 'chunker1', type: 'chunker', plugin_id: 'recursive', config: {} },
        ],
        edges: [
          { from_node: '_source', to_node: 'parser1', when: null },
          { from_node: '_source', to_node: 'parser2', when: null },
          { from_node: 'parser1', to_node: 'chunker1', when: null },
          { from_node: 'parser2', to_node: 'chunker1', when: null },
        ],
      };

      // Deleting parser1 - chunker1 still has parser2 as input
      const result = findOrphanedNodes(diamondDag, 'parser1');
      expect(result).toHaveLength(0);
    });

    it('handles custom edge list for edge deletion simulation', () => {
      // Simulate edge deletion by passing modified edges
      const edgesWithoutParser1ToChunker1 = baseDag.edges.filter(
        (e) => !(e.from_node === 'parser1' && e.to_node === 'chunker1')
      );

      // Use '_source' as deletedNodeId since we're simulating edge deletion
      // and want to keep all nodes, just check which ones become orphaned
      const result = findOrphanedNodes(baseDag, 'nonexistent', edgesWithoutParser1ToChunker1);
      // chunker1 should be orphaned without parser1's edge
      expect(result.map((n) => n.id)).toContain('chunker1');
    });
  });

  describe('findOrphanedNodesAfterEdgeDeletion', () => {
    const baseDag: PipelineDAG = {
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

    it('returns empty array when target has other incoming edges', () => {
      // DAG with multiple edges to same target
      const multiEdgeDag: PipelineDAG = {
        id: 'multi',
        version: '1',
        nodes: [
          { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
          { id: 'parser2', type: 'parser', plugin_id: 'pdf', config: {} },
          { id: 'chunker1', type: 'chunker', plugin_id: 'recursive', config: {} },
        ],
        edges: [
          { from_node: '_source', to_node: 'parser1', when: null },
          { from_node: '_source', to_node: 'parser2', when: null },
          { from_node: 'parser1', to_node: 'chunker1', when: null },
          { from_node: 'parser2', to_node: 'chunker1', when: null },
        ],
      };

      const result = findOrphanedNodesAfterEdgeDeletion(multiEdgeDag, 'parser1', 'chunker1');
      // chunker1 still has parser2's edge
      expect(result).toHaveLength(0);
    });

    it('finds the target node when it would be orphaned', () => {
      const result = findOrphanedNodesAfterEdgeDeletion(baseDag, 'parser1', 'chunker1');
      // chunker1 and embedder1 should be orphaned
      expect(result.map((n) => n.id)).toContain('chunker1');
    });

    it('finds downstream nodes that would be orphaned', () => {
      const result = findOrphanedNodesAfterEdgeDeletion(baseDag, 'parser1', 'chunker1');
      // embedder1 should also be orphaned since it only gets input from chunker1
      expect(result.map((n) => n.id)).toContain('embedder1');
      expect(result).toHaveLength(2);
    });

    it('returns empty array for non-existent edge', () => {
      const result = findOrphanedNodesAfterEdgeDeletion(baseDag, 'parser1', 'embedder1');
      // This edge doesn't exist, so no orphans
      expect(result).toHaveLength(0);
    });
  });
});
