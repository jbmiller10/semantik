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

  it('renders edges with predicates', () => {
    render(<PipelineVisualization dag={mockDAG} />);

    // PDF predicate should be visible
    expect(screen.getByText('pdf')).toBeInTheDocument();
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
});
