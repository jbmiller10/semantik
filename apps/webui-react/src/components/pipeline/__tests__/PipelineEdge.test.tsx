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

  it('renders vertical bezier path from bottom of source to top of target', () => {
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
    const d = path?.getAttribute('d');

    // fromPosition: x=100, y=50, width=160, height=80
    // Bottom center of source: x=180 (100+160/2), y=130 (50+80)
    // toPosition: x=100, y=230, width=160, height=80
    // Top center of target: x=180 (100+160/2), y=230
    expect(d).toMatch(/^M 180 130/); // Starts at bottom center of source
    expect(d).toMatch(/180 230$/); // Ends at top center of target
  });

  it('renders indicator dot for edge with when clause', () => {
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

    const indicator = document.querySelector('circle.edge-indicator');
    expect(indicator).toBeInTheDocument();
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

  describe('indicator dots and tooltips', () => {
    it('renders indicator dot instead of text label for conditional edge', () => {
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

      // Should have indicator dot (circle with class 'edge-indicator')
      const indicator = document.querySelector('circle.edge-indicator');
      expect(indicator).toBeInTheDocument();

      // Should NOT have the old text label rect
      const labelRect = document.querySelector('rect');
      expect(labelRect).not.toBeInTheDocument();
    });

    it('renders tooltip title with formatted predicate on indicator', () => {
      const edgeWithPredicate: PipelineEdge = {
        from_node: '_source',
        to_node: 'parser1',
        when: { mime_type: ['application/pdf', 'application/vnd.*'] },
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

      // Should have title element with formatted text
      const title = document.querySelector('circle.edge-indicator title');
      expect(title).toBeInTheDocument();
      expect(title?.textContent).toContain('Mime type');
      expect(title?.textContent).toContain('pdf');
    });

    it('renders dashed edge styling for catch-all routes', () => {
      const catchAllEdge: PipelineEdge = {
        from_node: 'parser1',
        to_node: 'chunker1',
        when: null,
      };

      render(
        <svg>
          <PipelineEdgeComponent
            edge={catchAllEdge}
            fromPosition={fromPosition}
            toPosition={toPosition}
            selected={false}
            showCatchAll={true}
          />
        </svg>
      );

      const path = document.querySelector('path');
      expect(path).toHaveAttribute('stroke-dasharray');
    });

    it('shows catch-all indicator with asterisk for null when clause', () => {
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

      const indicator = document.querySelector('circle.edge-indicator-catchall');
      expect(indicator).toBeInTheDocument();
      expect(screen.getByText('*')).toBeInTheDocument();
    });
  });
});
