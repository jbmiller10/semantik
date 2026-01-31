// apps/webui-react/src/components/wizard/steps/__tests__/ConfigureStep.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ConfigureStep } from '../ConfigureStep';
import type { PipelineDAG } from '../../../../types/pipeline';

const mockDag: PipelineDAG = {
  id: 'test',
  version: '1',
  nodes: [
    { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
    { id: 'chunker1', type: 'chunker', plugin_id: 'semantic', config: {} },
    { id: 'embedder1', type: 'embedder', plugin_id: 'default', config: {} },
  ],
  edges: [
    { from_node: '_source', to_node: 'parser1', when: null },
    { from_node: 'parser1', to_node: 'chunker1', when: null },
    { from_node: 'chunker1', to_node: 'embedder1', when: null },
  ],
};

const emptyDag: PipelineDAG = {
  id: 'empty',
  version: '1',
  nodes: [],
  edges: [],
};

describe('ConfigureStep', () => {
  const defaultProps = {
    dag: mockDag,
    onDagChange: vi.fn(),
    sourceAnalysis: null,
  };

  it('renders pipeline visualization when nodes exist', () => {
    render(<ConfigureStep {...defaultProps} />);
    // DAG visualization renders SVG
    const svg = document.querySelector('svg');
    expect(svg).toBeTruthy();
  });

  it('shows empty state when no nodes', () => {
    render(<ConfigureStep {...defaultProps} dag={emptyDag} />);
    expect(screen.getByText(/no pipeline configured/i)).toBeInTheDocument();
  });

  it('shows assisted mode upsell when callback provided', () => {
    render(<ConfigureStep {...defaultProps} onSwitchToAssisted={vi.fn()} />);
    expect(screen.getByText(/want help/i)).toBeInTheDocument();
  });

  it('does not show upsell when callback not provided', () => {
    render(<ConfigureStep {...defaultProps} />);
    expect(screen.queryByText(/want help/i)).not.toBeInTheDocument();
  });

  it('renders two-panel layout', () => {
    const { container } = render(<ConfigureStep {...defaultProps} />);
    // Should have a flex container with two child sections
    const flexContainer = container.querySelector('.flex-col.lg\\:flex-row');
    expect(flexContainer).toBeTruthy();
  });
});
