import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import { PipelineEdgeComponent } from '../PipelineEdge';
import type { PipelineEdge, NodePosition } from '@/types/pipeline';

describe('PipelineEdgeComponent', () => {
  // Vertical layout: source node above, target node below
  const fromPosition: NodePosition = { x: 100, y: 50, width: 160, height: 80 };
  const toPosition: NodePosition = { x: 100, y: 230, width: 160, height: 80 };

  const mockEdge: PipelineEdge = {
    from_node: 'parser1',
    to_node: 'chunker1',
    when: null,
  };

  it('renders a path between nodes', () => {
    render(
      <svg>
        <PipelineEdgeComponent
          edge={mockEdge}
          fromPosition={fromPosition}
          toPosition={toPosition}
          selected={false}
        />
      </svg>
    );

    const path = document.querySelector('path');
    expect(path).toBeInTheDocument();
    expect(path).toHaveAttribute('d');
  });

  it('renders predicate label when edge has when clause', () => {
    const edgeWithPredicate: PipelineEdge = {
      from_node: '_source',
      to_node: 'parser1',
      when: { mime_type: 'application/pdf' },
    };

    render(
      <svg>
        <PipelineEdgeComponent
          edge={edgeWithPredicate}
          fromPosition={fromPosition}
          toPosition={toPosition}
          selected={false}
        />
      </svg>
    );

    expect(screen.getByText(/pdf/i)).toBeInTheDocument();
  });

  it('shows catch-all indicator for null when clause', () => {
    render(
      <svg>
        <PipelineEdgeComponent
          edge={mockEdge}
          fromPosition={fromPosition}
          toPosition={toPosition}
          selected={false}
          showCatchAll={true}
        />
      </svg>
    );

    expect(screen.getByText('*')).toBeInTheDocument();
  });

  it('applies selected styling when selected', () => {
    render(
      <svg>
        <PipelineEdgeComponent
          edge={mockEdge}
          fromPosition={fromPosition}
          toPosition={toPosition}
          selected={true}
        />
      </svg>
    );

    const path = document.querySelector('path');
    expect(path).toHaveAttribute('stroke-width', '2');
  });

  it('calls onClick when clicked', async () => {
    const user = userEvent.setup();
    const handleClick = vi.fn();

    render(
      <svg>
        <PipelineEdgeComponent
          edge={mockEdge}
          fromPosition={fromPosition}
          toPosition={toPosition}
          selected={false}
          onClick={handleClick}
        />
      </svg>
    );

    const group = document.querySelector('g[data-edge-id]');
    await user.click(group!);

    expect(handleClick).toHaveBeenCalledWith('parser1', 'chunker1');
  });

  it('renders arrow marker at end of path', () => {
    render(
      <svg>
        <PipelineEdgeComponent
          edge={mockEdge}
          fromPosition={fromPosition}
          toPosition={toPosition}
          selected={false}
        />
      </svg>
    );

    const path = document.querySelector('path');
    expect(path).toHaveAttribute('marker-end');
  });
});
