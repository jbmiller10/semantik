import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import { PipelineNodeComponent } from '../PipelineNode';
import type { PipelineNode } from '@/types/pipeline';

describe('PipelineNodeComponent', () => {
  const mockNode: PipelineNode = {
    id: 'parser1',
    type: 'parser',
    plugin_id: 'unstructured',
    config: { strategy: 'auto' },
  };

  const defaultPosition = { x: 100, y: 50, width: 160, height: 80 };

  it('renders node with plugin name', () => {
    render(
      <svg>
        <PipelineNodeComponent
          node={mockNode}
          position={defaultPosition}
          selected={false}
        />
      </svg>
    );

    expect(screen.getByText('unstructured')).toBeInTheDocument();
  });

  it('renders node type label', () => {
    render(
      <svg>
        <PipelineNodeComponent
          node={mockNode}
          position={defaultPosition}
          selected={false}
        />
      </svg>
    );

    expect(screen.getByText('PARSER')).toBeInTheDocument();
  });

  it('applies selected styling when selected', () => {
    render(
      <svg>
        <PipelineNodeComponent
          node={mockNode}
          position={defaultPosition}
          selected={true}
        />
      </svg>
    );

    const rect = document.querySelector('rect');
    expect(rect).toHaveAttribute('stroke-width', '2');
  });

  it('calls onClick when clicked', async () => {
    const user = userEvent.setup();
    const handleClick = vi.fn();

    render(
      <svg>
        <PipelineNodeComponent
          node={mockNode}
          position={defaultPosition}
          selected={false}
          onClick={handleClick}
        />
      </svg>
    );

    const group = document.querySelector('g[data-node-id="parser1"]');
    await user.click(group!);

    expect(handleClick).toHaveBeenCalledWith('parser1');
  });

  it('renders source node differently', () => {
    render(
      <svg>
        <PipelineNodeComponent
          node={{ id: '_source', type: 'parser', plugin_id: 'source', config: {} }}
          position={defaultPosition}
          selected={false}
          isSource={true}
        />
      </svg>
    );

    expect(screen.getByText('Source')).toBeInTheDocument();
  });

  it('uses different colors for different node types', () => {
    const { rerender } = render(
      <svg>
        <PipelineNodeComponent
          node={{ id: 'p1', type: 'parser', plugin_id: 'text', config: {} }}
          position={defaultPosition}
          selected={false}
        />
      </svg>
    );

    const parserRect = document.querySelector('rect');
    const parserFill = parserRect?.getAttribute('fill');

    rerender(
      <svg>
        <PipelineNodeComponent
          node={{ id: 'e1', type: 'embedder', plugin_id: 'dense', config: {} }}
          position={defaultPosition}
          selected={false}
        />
      </svg>
    );

    const embedderRect = document.querySelector('rect');
    const embedderFill = embedderRect?.getAttribute('fill');

    // Different types should have different fills
    expect(parserFill).not.toBe(embedderFill);
  });
});
