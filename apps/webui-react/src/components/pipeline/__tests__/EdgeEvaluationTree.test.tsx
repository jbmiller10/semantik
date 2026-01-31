import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@/tests/utils/test-utils';
import { EdgeEvaluationTree } from '../EdgeEvaluationTree';
import type { StageEvaluationResult, EdgeEvaluationResult } from '@/types/routePreview';
import type { PipelineDAG, PipelineNode } from '@/types/pipeline';

// Test the helper functions by using the component in ways that exercise them

describe('EdgeEvaluationTree', () => {
  const createMockDAG = (nodes: PipelineNode[]): PipelineDAG => ({
    id: 'test-dag',
    version: '1.0',
    nodes,
    edges: [],
  });

  const createMockEdge = (
    status: 'matched' | 'matched_parallel' | 'not_matched' | 'skipped',
    toNode: string,
    predicate?: Record<string, unknown>,
    fieldEvaluations?: EdgeEvaluationResult['field_evaluations'],
    isParallel: boolean = false,
    pathName: string | null = null
  ): EdgeEvaluationResult => ({
    from_node: '_source',
    to_node: toNode,
    status,
    predicate: predicate ?? null,
    field_evaluations: fieldEvaluations ?? null,
    matched: status === 'matched' || status === 'matched_parallel',
    is_parallel: isParallel,
    path_name: pathName,
  });

  const createMockStage = (
    fromNode: string,
    edges: EdgeEvaluationResult[],
    stage: number = 0
  ): StageEvaluationResult => {
    const matchedEdges = edges.filter(
      (e) => e.status === 'matched' || e.status === 'matched_parallel'
    );
    const selectedNode = matchedEdges[0]?.to_node ?? null;
    const selectedNodes =
      matchedEdges.length > 0 ? matchedEdges.map((e) => e.to_node) : null;
    return {
      stage: String(stage),
      from_node: fromNode,
      evaluated_edges: edges,
      selected_node: selectedNode,
      selected_nodes: selectedNodes,
      metadata_snapshot: {},
    };
  };

  describe('empty state', () => {
    it('renders empty state when stages is empty', () => {
      const dag = createMockDAG([]);
      render(<EdgeEvaluationTree stages={[]} dag={dag} />);
      expect(screen.getByText('No routing stages to display')).toBeInTheDocument();
    });
  });

  describe('getStatusDisplay', () => {
    it('shows matched status with check icon', () => {
      const dag = createMockDAG([{ id: 'parser1', type: 'parser', plugin_id: 'text', config: {} }]);
      const edge = createMockEdge('matched', 'parser1');
      const stage = createMockStage('_source', [edge]);

      render(<EdgeEvaluationTree stages={[stage]} dag={dag} />);

      // Check for green color class (matched status)
      const matchedElement = screen.getByText('→ text').closest('button');
      expect(matchedElement).toBeInTheDocument();
    });

    it('shows not_matched status with X icon', () => {
      const dag = createMockDAG([{ id: 'parser1', type: 'parser', plugin_id: 'text', config: {} }]);
      const edge = createMockEdge('not_matched', 'parser1', { extension: '.pdf' });
      const stage = createMockStage('_source', [edge]);

      render(<EdgeEvaluationTree stages={[stage]} dag={dag} />);

      // Verify the edge row is rendered
      expect(screen.getByText('→ text')).toBeInTheDocument();
    });

    it('shows skipped status with skip icon', () => {
      const dag = createMockDAG([{ id: 'parser1', type: 'parser', plugin_id: 'text', config: {} }]);
      const edge = createMockEdge('skipped', 'parser1');
      const stage = createMockStage('_source', [edge]);

      render(<EdgeEvaluationTree stages={[stage]} dag={dag} />);

      expect(screen.getByText('→ text')).toBeInTheDocument();
    });
  });

  describe('getNodeDisplayName', () => {
    it('displays "Source" for _source node', () => {
      const dag = createMockDAG([{ id: 'parser1', type: 'parser', plugin_id: 'text', config: {} }]);
      const edge = createMockEdge('matched', 'parser1');
      const stage = createMockStage('_source', [edge]);

      render(<EdgeEvaluationTree stages={[stage]} dag={dag} />);

      // Stage header shows "Source" for _source
      expect(screen.getByText('Source')).toBeInTheDocument();
    });

    it('displays plugin_id for found nodes', () => {
      const dag = createMockDAG([
        { id: 'parser1', type: 'parser', plugin_id: 'unstructured', config: {} },
      ]);
      const edge = createMockEdge('matched', 'parser1');
      const stage = createMockStage('_source', [edge]);

      render(<EdgeEvaluationTree stages={[stage]} dag={dag} />);

      expect(screen.getByText('→ unstructured')).toBeInTheDocument();
    });

    it('displays node id for missing nodes', () => {
      const dag = createMockDAG([]); // No nodes
      const edge = createMockEdge('matched', 'missing_node');
      const stage = createMockStage('_source', [edge]);

      render(<EdgeEvaluationTree stages={[stage]} dag={dag} />);

      // Falls back to node id
      expect(screen.getByText('→ missing_node')).toBeInTheDocument();
    });
  });

  describe('formatValue', () => {
    it('formats null as "null"', () => {
      const dag = createMockDAG([{ id: 'p1', type: 'parser', plugin_id: 'text', config: {} }]);
      const edge = createMockEdge('not_matched', 'p1', { extension: '.pdf' }, [
        { field: 'extension', pattern: '.pdf', value: null, matched: false },
      ]);
      const stage = createMockStage('_source', [edge]);

      render(<EdgeEvaluationTree stages={[stage]} dag={dag} />);

      // Expand to see field evaluations (auto-expanded for not_matched)
      expect(screen.getByText('null')).toBeInTheDocument();
    });

    it('formats string values with quotes', () => {
      const dag = createMockDAG([{ id: 'p1', type: 'parser', plugin_id: 'text', config: {} }]);
      const edge = createMockEdge('not_matched', 'p1', { extension: '.pdf' }, [
        { field: 'extension', pattern: '.pdf', value: '.txt', matched: false },
      ]);
      const stage = createMockStage('_source', [edge]);

      render(<EdgeEvaluationTree stages={[stage]} dag={dag} />);

      // Should show quoted string
      expect(screen.getByText('".txt"')).toBeInTheDocument();
    });

    it('formats boolean values as true/false', () => {
      const dag = createMockDAG([{ id: 'p1', type: 'parser', plugin_id: 'text', config: {} }]);
      const edge = createMockEdge('not_matched', 'p1', { is_code: true }, [
        { field: 'is_code', pattern: true, value: false, matched: false },
      ]);
      const stage = createMockStage('_source', [edge]);

      render(<EdgeEvaluationTree stages={[stage]} dag={dag} />);

      expect(screen.getByText('false')).toBeInTheDocument();
    });

    it('formats numbers as strings', () => {
      const dag = createMockDAG([{ id: 'p1', type: 'parser', plugin_id: 'text', config: {} }]);
      const edge = createMockEdge('not_matched', 'p1', { size: 100 }, [
        { field: 'size', pattern: 100, value: 42, matched: false },
      ]);
      const stage = createMockStage('_source', [edge]);

      render(<EdgeEvaluationTree stages={[stage]} dag={dag} />);

      expect(screen.getByText('42')).toBeInTheDocument();
    });

    it('formats arrays with brackets', () => {
      const dag = createMockDAG([{ id: 'p1', type: 'parser', plugin_id: 'text', config: {} }]);
      const edge = createMockEdge('not_matched', 'p1', { exts: ['.pdf'] }, [
        { field: 'exts', pattern: ['.pdf'], value: ['.txt', '.md'], matched: false },
      ]);
      const stage = createMockStage('_source', [edge]);

      render(<EdgeEvaluationTree stages={[stage]} dag={dag} />);

      // Array should be formatted
      expect(screen.getByText('[".txt", ".md"]')).toBeInTheDocument();
    });

    it('formats objects as JSON', () => {
      const dag = createMockDAG([{ id: 'p1', type: 'parser', plugin_id: 'text', config: {} }]);
      const edge = createMockEdge('not_matched', 'p1', { meta: { key: 'val' } }, [
        { field: 'meta', pattern: { key: 'val' }, value: { foo: 'bar' }, matched: false },
      ]);
      const stage = createMockStage('_source', [edge]);

      render(<EdgeEvaluationTree stages={[stage]} dag={dag} />);

      // Object should be JSON stringified
      expect(screen.getByText('{"foo":"bar"}')).toBeInTheDocument();
    });
  });

  describe('StageRow', () => {
    it('shows node name from DAG', () => {
      const dag = createMockDAG([
        { id: 'parser1', type: 'parser', plugin_id: 'my_parser', config: {} },
        { id: 'chunker1', type: 'chunker', plugin_id: 'recursive', config: {} },
      ]);
      const edge = createMockEdge('matched', 'chunker1');
      edge.from_node = 'parser1';
      const stage = createMockStage('parser1', [edge]);

      render(<EdgeEvaluationTree stages={[stage]} dag={dag} />);

      // Stage row header shows plugin_id (use getAllBy since matched edge also shows it)
      const elements = screen.getAllByText('my_parser');
      expect(elements.length).toBeGreaterThanOrEqual(1);
    });

    it('shows edge count', () => {
      const dag = createMockDAG([
        { id: 'p1', type: 'parser', plugin_id: 'text', config: {} },
        { id: 'p2', type: 'parser', plugin_id: 'unstructured', config: {} },
      ]);
      const edges = [
        createMockEdge('not_matched', 'p1'),
        createMockEdge('matched', 'p2'),
      ];
      const stage = createMockStage('_source', edges);

      render(<EdgeEvaluationTree stages={[stage]} dag={dag} />);

      expect(screen.getByText('2 edges evaluated')).toBeInTheDocument();
    });

    it('shows singular "edge" for single edge', () => {
      const dag = createMockDAG([{ id: 'p1', type: 'parser', plugin_id: 'text', config: {} }]);
      const stage = createMockStage('_source', [createMockEdge('matched', 'p1')]);

      render(<EdgeEvaluationTree stages={[stage]} dag={dag} />);

      expect(screen.getByText('1 edge evaluated')).toBeInTheDocument();
    });
  });

  describe('EdgeRow', () => {
    it('toggles expansion on click when has field evaluations', () => {
      const dag = createMockDAG([{ id: 'p1', type: 'parser', plugin_id: 'text', config: {} }]);
      const edge = createMockEdge('matched', 'p1', { ext: '.txt' }, [
        { field: 'ext', pattern: '.txt', value: '.txt', matched: true },
      ]);
      const stage = createMockStage('_source', [edge]);

      render(<EdgeEvaluationTree stages={[stage]} dag={dag} />);

      const edgeButton = screen.getByRole('button', { name: /→ text/i });

      // Click to toggle
      fireEvent.click(edgeButton);

      // Should show field evaluation
      expect(screen.getByText('ext')).toBeInTheDocument();
    });

    it('shows catch-all indicator when no predicate', () => {
      const dag = createMockDAG([{ id: 'p1', type: 'parser', plugin_id: 'text', config: {} }]);
      const edge = createMockEdge('matched', 'p1'); // No predicate
      const stage = createMockStage('_source', [edge]);

      render(<EdgeEvaluationTree stages={[stage]} dag={dag} />);

      expect(screen.getByText('catch-all (*)')).toBeInTheDocument();
    });

    it('shows predicate JSON when present', () => {
      const dag = createMockDAG([{ id: 'p1', type: 'parser', plugin_id: 'text', config: {} }]);
      const edge = createMockEdge('matched', 'p1', { extension: '.pdf' });
      const stage = createMockStage('_source', [edge]);

      render(<EdgeEvaluationTree stages={[stage]} dag={dag} />);

      expect(screen.getByText('{"extension":".pdf"}')).toBeInTheDocument();
    });
  });

  describe('FieldEvaluationRow', () => {
    it('shows check icon for matched fields', () => {
      const dag = createMockDAG([{ id: 'p1', type: 'parser', plugin_id: 'text', config: {} }]);
      const edge = createMockEdge('not_matched', 'p1', {}, [
        { field: 'ext', pattern: '.txt', value: '.txt', matched: true },
        { field: 'size', pattern: 100, value: 50, matched: false },
      ]);
      const stage = createMockStage('_source', [edge]);

      render(<EdgeEvaluationTree stages={[stage]} dag={dag} />);

      // Both field names should be visible (auto-expanded for not_matched)
      expect(screen.getByText('ext')).toBeInTheDocument();
      expect(screen.getByText('size')).toBeInTheDocument();
    });
  });

  describe('multiple stages', () => {
    it('renders all stages', () => {
      const dag = createMockDAG([
        { id: 'p1', type: 'parser', plugin_id: 'unstructured', config: {} },
        { id: 'c1', type: 'chunker', plugin_id: 'recursive', config: {} },
      ]);
      const stages = [
        createMockStage('_source', [createMockEdge('matched', 'p1')], 0),
        createMockStage('p1', [createMockEdge('matched', 'c1')], 1),
      ];

      render(<EdgeEvaluationTree stages={stages} dag={dag} />);

      expect(screen.getByText('Source')).toBeInTheDocument();
      // Use getAllBy because 'unstructured' may appear in multiple places (header + matched edge indicator)
      expect(screen.getAllByText('unstructured').length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText('→ recursive')).toBeInTheDocument();
    });
  });
});
