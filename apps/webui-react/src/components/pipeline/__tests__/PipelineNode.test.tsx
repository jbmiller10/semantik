import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@/tests/utils/test-utils';
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

  describe('ports', () => {
    describe('output port', () => {
      it('renders output port when showPorts is true', () => {
        render(
          <svg>
            <PipelineNodeComponent
              node={mockNode}
              position={defaultPosition}
              selected={false}
              showPorts={true}
            />
          </svg>
        );

        const outputPort = document.querySelector('.output-port');
        expect(outputPort).toBeInTheDocument();
      });

      it('output port is hidden by default', () => {
        render(
          <svg>
            <PipelineNodeComponent
              node={mockNode}
              position={defaultPosition}
              selected={false}
              showPorts={false}
            />
          </svg>
        );

        const outputPort = document.querySelector('.output-port');
        expect(outputPort).toBeInTheDocument();
        expect(outputPort).toHaveStyle({ opacity: '0' });
      });

      it('does not render output port on embedder nodes', () => {
        render(
          <svg>
            <PipelineNodeComponent
              node={{ id: 'emb1', type: 'embedder', plugin_id: 'dense', config: {} }}
              position={defaultPosition}
              selected={false}
              showPorts={true}
            />
          </svg>
        );

        const outputPort = document.querySelector('.output-port');
        expect(outputPort).not.toBeInTheDocument();
      });

      it('calls onStartDrag when mousedown on output port', () => {
        const onStartDrag = vi.fn();

        render(
          <svg>
            <PipelineNodeComponent
              node={mockNode}
              position={defaultPosition}
              selected={false}
              showPorts={true}
              onStartDrag={onStartDrag}
            />
          </svg>
        );

        const outputPort = document.querySelector('.output-port');
        fireEvent.mouseDown(outputPort!, { clientX: 180, clientY: 130 });

        expect(onStartDrag).toHaveBeenCalledWith('parser1', { x: 180, y: 130 });
      });

      it('renders output port at bottom center of node', () => {
        render(
          <svg>
            <PipelineNodeComponent
              node={mockNode}
              position={defaultPosition}
              selected={false}
              showPorts={true}
            />
          </svg>
        );

        const outputPort = document.querySelector('.output-port');
        expect(outputPort).toHaveAttribute('cx', String(defaultPosition.x + defaultPosition.width / 2));
        expect(outputPort).toHaveAttribute('cy', String(defaultPosition.y + defaultPosition.height));
      });
    });

    describe('input port', () => {
      it('renders input port when showPorts is true', () => {
        render(
          <svg>
            <PipelineNodeComponent
              node={mockNode}
              position={defaultPosition}
              selected={false}
              showPorts={true}
            />
          </svg>
        );

        const inputPort = document.querySelector('.input-port');
        expect(inputPort).toBeInTheDocument();
      });

      it('input port is hidden by default', () => {
        render(
          <svg>
            <PipelineNodeComponent
              node={mockNode}
              position={defaultPosition}
              selected={false}
              showPorts={false}
            />
          </svg>
        );

        const inputPort = document.querySelector('.input-port');
        expect(inputPort).toBeInTheDocument();
        expect(inputPort).toHaveStyle({ opacity: '0' });
      });

      it('does not render input port on source node', () => {
        render(
          <svg>
            <PipelineNodeComponent
              node={{ id: '_source', type: 'parser', plugin_id: 'source', config: {} }}
              position={defaultPosition}
              selected={false}
              isSource={true}
              showPorts={true}
            />
          </svg>
        );

        const inputPort = document.querySelector('.input-port');
        expect(inputPort).not.toBeInTheDocument();
      });

      it('renders input port at top center of node', () => {
        render(
          <svg>
            <PipelineNodeComponent
              node={mockNode}
              position={defaultPosition}
              selected={false}
              showPorts={true}
            />
          </svg>
        );

        const inputPort = document.querySelector('.input-port');
        expect(inputPort).toHaveAttribute('cx', String(defaultPosition.x + defaultPosition.width / 2));
        expect(inputPort).toHaveAttribute('cy', String(defaultPosition.y));
      });

      it('input port is visible when isValidDropTarget is true', () => {
        render(
          <svg>
            <PipelineNodeComponent
              node={mockNode}
              position={defaultPosition}
              selected={false}
              showPorts={false}
              isValidDropTarget={true}
            />
          </svg>
        );

        const inputPort = document.querySelector('.input-port');
        expect(inputPort).toHaveStyle({ opacity: '1' });
      });

      it('input port has highlight fill when isValidDropTarget', () => {
        render(
          <svg>
            <PipelineNodeComponent
              node={mockNode}
              position={defaultPosition}
              selected={false}
              isValidDropTarget={true}
            />
          </svg>
        );

        const inputPort = document.querySelector('.input-port');
        expect(inputPort).toHaveAttribute('fill', 'var(--text-primary)');
      });
    });
  });
});
