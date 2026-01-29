import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import { PipelineVisualization } from '../PipelineVisualization';
import type { PipelineDAG, DAGSelection } from '@/types/pipeline';

const mockDAG: PipelineDAG = {
  id: 'test-pipeline',
  version: '1',
  nodes: [
    { id: 'text_parser', type: 'parser', plugin_id: 'text', config: {} },
    { id: 'pdf_parser', type: 'parser', plugin_id: 'unstructured', config: { strategy: 'auto' } },
    { id: 'chunker', type: 'chunker', plugin_id: 'recursive', config: { max_tokens: 512 } },
    { id: 'embedder', type: 'embedder', plugin_id: 'dense_local', config: { model: 'bge-base' } },
  ],
  edges: [
    { from_node: '_source', to_node: 'text_parser', when: null },
    { from_node: '_source', to_node: 'pdf_parser', when: { mime_type: 'application/pdf' } },
    { from_node: 'text_parser', to_node: 'chunker', when: null },
    { from_node: 'pdf_parser', to_node: 'chunker', when: null },
    { from_node: 'chunker', to_node: 'embedder', when: null },
  ],
};

describe('PipelineVisualization', () => {
  it('renders all nodes from the DAG', () => {
    render(<PipelineVisualization dag={mockDAG} />);

    expect(screen.getByText('text')).toBeInTheDocument();
    expect(screen.getByText('unstructured')).toBeInTheDocument();
    expect(screen.getByText('recursive')).toBeInTheDocument();
    expect(screen.getByText('dense_local')).toBeInTheDocument();
  });

  it('renders source node', () => {
    render(<PipelineVisualization dag={mockDAG} />);

    expect(screen.getByText('Source')).toBeInTheDocument();
  });

  it('renders edges with indicator dots for predicates', () => {
    const { container } = render(<PipelineVisualization dag={mockDAG} />);

    // Conditional edge should have indicator dot instead of text label
    const indicatorDot = container.querySelector('circle.edge-indicator');
    expect(indicatorDot).toBeInTheDocument();
  });

  it('calls onSelectionChange when node is clicked', async () => {
    const user = userEvent.setup();
    const handleSelectionChange = vi.fn();

    render(
      <PipelineVisualization
        dag={mockDAG}
        onSelectionChange={handleSelectionChange}
      />
    );

    const parserNode = document.querySelector('g[data-node-id="text_parser"]');
    await user.click(parserNode!);

    expect(handleSelectionChange).toHaveBeenCalledWith({
      type: 'node',
      nodeId: 'text_parser',
    });
  });

  it('highlights selected node', () => {
    const selection: DAGSelection = { type: 'node', nodeId: 'chunker' };

    render(
      <PipelineVisualization
        dag={mockDAG}
        selection={selection}
      />
    );

    const chunkerNode = document.querySelector('g[data-node-id="chunker"] rect');
    expect(chunkerNode).toHaveAttribute('stroke-width', '2');
  });

  it('clears selection when clicking background', async () => {
    const user = userEvent.setup();
    const handleSelectionChange = vi.fn();
    const selection: DAGSelection = { type: 'node', nodeId: 'chunker' };

    render(
      <PipelineVisualization
        dag={mockDAG}
        selection={selection}
        onSelectionChange={handleSelectionChange}
      />
    );

    const svg = document.querySelector('svg');
    await user.click(svg!);

    expect(handleSelectionChange).toHaveBeenCalledWith({ type: 'none' });
  });

  it('renders empty state for empty DAG', () => {
    const emptyDAG: PipelineDAG = {
      id: 'empty',
      version: '1',
      nodes: [],
      edges: [],
    };

    render(<PipelineVisualization dag={emptyDAG} />);

    expect(screen.getByText(/no pipeline configured/i)).toBeInTheDocument();
  });

  it('applies custom className', () => {
    render(
      <PipelineVisualization dag={mockDAG} className="custom-class" />
    );

    const container = document.querySelector('.custom-class');
    expect(container).toBeInTheDocument();
  });

  describe('multiple path highlighting', () => {
    const dagWithParallelPaths: PipelineDAG = {
      id: 'parallel-dag',
      version: '1',
      nodes: [
        { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
        { id: 'chunker1', type: 'chunker', plugin_id: 'recursive', config: {} },
        { id: 'embedder1', type: 'embedder', plugin_id: 'dense', config: {} },
        { id: 'extractor1', type: 'extractor', plugin_id: 'keyword', config: {} },
      ],
      edges: [
        { from_node: '_source', to_node: 'parser1', when: null },
        { from_node: 'parser1', to_node: 'chunker1', when: null },
        { from_node: 'chunker1', to_node: 'embedder1', when: null },
        { from_node: 'chunker1', to_node: 'extractor1', when: null },
      ],
    };

    it('highlights all edges in multiple parallel paths', () => {
      const parallelPaths = [
        ['_source', 'parser1', 'chunker1', 'embedder1'],
        ['_source', 'parser1', 'chunker1', 'extractor1'],
      ];

      const { container } = render(
        <PipelineVisualization
          dag={dagWithParallelPaths}
          highlightedPaths={parallelPaths}
        />
      );

      // Both embedder and extractor edges should be highlighted
      const highlightedEdges = container.querySelectorAll('.pipeline-edge-highlighted');
      expect(highlightedEdges.length).toBeGreaterThanOrEqual(2);
    });

    it('uses different colors for primary vs secondary paths', () => {
      const parallelPaths = [
        ['_source', 'parser1', 'chunker1', 'embedder1'],
        ['_source', 'parser1', 'chunker1', 'extractor1'],
      ];

      const { container } = render(
        <PipelineVisualization
          dag={dagWithParallelPaths}
          highlightedPaths={parallelPaths}
        />
      );

      // Primary path edges should be green
      const greenEdges = container.querySelectorAll('path[stroke="rgb(34, 197, 94)"]');
      expect(greenEdges.length).toBeGreaterThan(0);

      // Secondary path edges should be blue
      const blueEdges = container.querySelectorAll('path[stroke="rgb(59, 130, 246)"]');
      expect(blueEdges.length).toBeGreaterThan(0);
    });

    it('supports backward-compatible single highlightedPath', () => {
      const singlePath = ['_source', 'parser1', 'chunker1', 'embedder1'];

      const { container } = render(
        <PipelineVisualization
          dag={dagWithParallelPaths}
          highlightedPath={singlePath}
        />
      );

      // Edges should be highlighted with default green
      const greenEdges = container.querySelectorAll('path[stroke="rgb(34, 197, 94)"]');
      expect(greenEdges.length).toBeGreaterThanOrEqual(3);
    });
  });
});
