// apps/webui-react/src/components/wizard/steps/__tests__/ReviewStep.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ReviewStep } from '../ReviewStep';
import type { PipelineDAG } from '../../../../types/pipeline';

const mockDag: PipelineDAG = {
  id: 'test',
  version: '1',
  nodes: [
    { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
    { id: 'chunker1', type: 'chunker', plugin_id: 'semantic', config: { chunk_size: 512 } },
    { id: 'embedder1', type: 'embedder', plugin_id: 'default', config: {} },
  ],
  edges: [
    { from_node: '_source', to_node: 'parser1', when: null },
    { from_node: 'parser1', to_node: 'chunker1', when: null },
    { from_node: 'chunker1', to_node: 'embedder1', when: null },
  ],
};

describe('ReviewStep', () => {
  const defaultProps = {
    dag: mockDag,
    onDagChange: vi.fn(),
    agentSummary: 'The agent recommended semantic chunking for your markdown documentation.',
    conversationId: 'conv-123',
  };

  it('renders two-column layout with summary on left', () => {
    render(<ReviewStep {...defaultProps} />);

    // Should have summary column and editor column
    expect(screen.getByTestId('summary-column')).toBeInTheDocument();
    expect(screen.getByTestId('editor-column')).toBeInTheDocument();
  });

  it('displays agent summary', () => {
    render(<ReviewStep {...defaultProps} />);

    expect(screen.getByText(/semantic chunking/i)).toBeInTheDocument();
  });

  it('renders full DAG editor', () => {
    render(<ReviewStep {...defaultProps} />);

    // Should show nodes in the visualization
    expect(document.querySelector('svg')).toBeTruthy();
  });

  it('allows editing DAG', () => {
    const onDagChange = vi.fn();
    render(<ReviewStep {...defaultProps} onDagChange={onDagChange} />);

    // The editor should not be read-only
    expect(screen.queryByText(/read-only/i)).not.toBeInTheDocument();
  });
});
