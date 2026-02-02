import { describe, it, expect } from 'vitest';
import { ensurePathNames } from '../pipelineUtils';
import type { PipelineDAG } from '@/types/pipeline';

describe('ensurePathNames', () => {
  it('returns unchanged DAG when no parallel edges exist', () => {
    const dag: PipelineDAG = {
      id: 'test',
      version: '1',
      nodes: [
        { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
      ],
      edges: [
        { from_node: '_source', to_node: 'parser1', when: null },
      ],
    };

    const result = ensurePathNames(dag);

    expect(result.edges[0].path_name).toBeUndefined();
  });

  it('auto-generates path_name for parallel edge without one', () => {
    const dag: PipelineDAG = {
      id: 'test',
      version: '1',
      nodes: [
        { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
      ],
      edges: [
        { from_node: '_source', to_node: 'parser1', when: null, parallel: true },
      ],
    };

    const result = ensurePathNames(dag);

    expect(result.edges[0].path_name).toBe('path__source_0');
  });

  it('preserves existing path_name for parallel edge', () => {
    const dag: PipelineDAG = {
      id: 'test',
      version: '1',
      nodes: [
        { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
      ],
      edges: [
        { from_node: '_source', to_node: 'parser1', when: null, parallel: true, path_name: 'custom_path' },
      ],
    };

    const result = ensurePathNames(dag);

    expect(result.edges[0].path_name).toBe('custom_path');
  });

  it('generates unique path_names for multiple parallel edges from same node', () => {
    const dag: PipelineDAG = {
      id: 'test',
      version: '1',
      nodes: [
        { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
        { id: 'parser2', type: 'parser', plugin_id: 'pdf', config: {} },
      ],
      edges: [
        { from_node: '_source', to_node: 'parser1', when: { mime_type: 'text/plain' }, parallel: true },
        { from_node: '_source', to_node: 'parser2', when: { mime_type: 'application/pdf' }, parallel: true },
      ],
    };

    const result = ensurePathNames(dag);

    expect(result.edges[0].path_name).toBe('path__source_0');
    expect(result.edges[1].path_name).toBe('path__source_1');
    expect(result.edges[0].path_name).not.toBe(result.edges[1].path_name);
  });

  it('does not modify original DAG', () => {
    const dag: PipelineDAG = {
      id: 'test',
      version: '1',
      nodes: [],
      edges: [
        { from_node: '_source', to_node: 'parser1', when: null, parallel: true },
      ],
    };

    const result = ensurePathNames(dag);

    expect(dag.edges[0].path_name).toBeUndefined();
    expect(result.edges[0].path_name).toBe('path__source_0');
  });
});
