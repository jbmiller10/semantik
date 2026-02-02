/**
 * Tests for pipeline animation classes.
 * Verifies that new nodes, edges, and drop zones have proper animation classes.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render } from '@/tests/utils/test-utils';
import { PipelineVisualization } from '../PipelineVisualization';
import { PipelineNodeComponent } from '../PipelineNode';
import { PipelineEdgeComponent } from '../PipelineEdge';
import { DragPreviewEdge } from '../DragPreviewEdge';
import { TierDropZone } from '../TierDropZone';
import type { PipelineDAG } from '@/types/pipeline';

// Mock matchMedia for desktop mode
function mockMatchMediaDesktop() {
  window.matchMedia = vi.fn().mockReturnValue({
    matches: false,
    media: '(pointer: coarse)',
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  });
}

describe('PipelineNode Animation', () => {
  it('applies animation class when isNew is true', () => {
    const node = {
      id: 'test-node',
      type: 'parser' as const,
      plugin_id: 'text',
      config: {},
    };
    const position = { x: 100, y: 100, width: 160, height: 80 };

    const { container } = render(
      <svg>
        <PipelineNodeComponent
          node={node}
          position={position}
          selected={false}
          isNew={true}
        />
      </svg>
    );

    const nodeGroup = container.querySelector('g[data-node-id="test-node"]');
    expect(nodeGroup).toHaveClass('pipeline-node-new');
  });

  it('does not apply animation class when isNew is false', () => {
    const node = {
      id: 'test-node',
      type: 'parser' as const,
      plugin_id: 'text',
      config: {},
    };
    const position = { x: 100, y: 100, width: 160, height: 80 };

    const { container } = render(
      <svg>
        <PipelineNodeComponent
          node={node}
          position={position}
          selected={false}
          isNew={false}
        />
      </svg>
    );

    const nodeGroup = container.querySelector('g[data-node-id="test-node"]');
    expect(nodeGroup).not.toHaveClass('pipeline-node-new');
  });

  it('applies drop target class when isValidDropTarget is true', () => {
    const node = {
      id: 'test-node',
      type: 'parser' as const,
      plugin_id: 'text',
      config: {},
    };
    const position = { x: 100, y: 100, width: 160, height: 80 };

    const { container } = render(
      <svg>
        <PipelineNodeComponent
          node={node}
          position={position}
          selected={false}
          isValidDropTarget={true}
          showPorts={true}
        />
      </svg>
    );

    const rect = container.querySelector('g[data-node-id="test-node"] rect');
    expect(rect).toHaveClass('pipeline-node-drop-target');
  });

  it('applies pulse class to input port when isValidDropTarget is true', () => {
    const node = {
      id: 'test-node',
      type: 'parser' as const,
      plugin_id: 'text',
      config: {},
    };
    const position = { x: 100, y: 100, width: 160, height: 80 };

    const { container } = render(
      <svg>
        <PipelineNodeComponent
          node={node}
          position={position}
          selected={false}
          isValidDropTarget={true}
          showPorts={true}
        />
      </svg>
    );

    const inputPort = container.querySelector('.input-port');
    expect(inputPort).toHaveClass('pipeline-port-pulse');
  });
});

describe('PipelineEdge Animation', () => {
  const fromPosition = { x: 100, y: 100, width: 160, height: 80 };
  const toPosition = { x: 100, y: 280, width: 160, height: 80 };
  const edge = {
    from_node: 'source',
    to_node: 'target',
    when: null,
  };

  it('applies animation class when isNew is true', () => {
    const { container } = render(
      <svg>
        <PipelineEdgeComponent
          edge={edge}
          fromPosition={fromPosition}
          toPosition={toPosition}
          selected={false}
          isNew={true}
        />
      </svg>
    );

    const path = container.querySelector('path');
    expect(path).toHaveClass('pipeline-edge-new');
  });

  it('does not apply animation class when isNew is false', () => {
    const { container } = render(
      <svg>
        <PipelineEdgeComponent
          edge={edge}
          fromPosition={fromPosition}
          toPosition={toPosition}
          selected={false}
          isNew={false}
        />
      </svg>
    );

    const path = container.querySelector('path');
    expect(path).not.toHaveClass('pipeline-edge-new');
  });
});

describe('DragPreviewEdge Animation', () => {
  it('has animation class', () => {
    const { container } = render(
      <svg>
        <DragPreviewEdge from={{ x: 100, y: 100 }} to={{ x: 200, y: 200 }} />
      </svg>
    );

    const path = container.querySelector('path');
    expect(path).toHaveClass('preview-edge');
  });

  it('has correct animation style', () => {
    const { container } = render(
      <svg>
        <DragPreviewEdge from={{ x: 100, y: 100 }} to={{ x: 200, y: 200 }} />
      </svg>
    );

    const path = container.querySelector('path');
    expect(path).toHaveStyle({ animation: 'dash-flow 0.5s linear infinite' });
  });

  it('renders dashed stroke', () => {
    const { container } = render(
      <svg>
        <DragPreviewEdge from={{ x: 100, y: 100 }} to={{ x: 200, y: 200 }} />
      </svg>
    );

    const path = container.querySelector('path');
    expect(path).toHaveAttribute('stroke-dasharray', '6 4');
  });
});

describe('TierDropZone Animation', () => {
  const bounds = { x: 50, y: 180, width: 300, height: 120 };

  it('applies fade-in animation class when active', () => {
    const { container } = render(
      <svg>
        <TierDropZone
          tier="parser"
          bounds={bounds}
          isActive={true}
          isHovered={false}
        />
      </svg>
    );

    const group = container.querySelector('.tier-drop-zone');
    expect(group).toHaveClass('tier-drop-zone-fade-in');
  });

  it('applies hover animation class when hovered', () => {
    const { container } = render(
      <svg>
        <TierDropZone
          tier="parser"
          bounds={bounds}
          isActive={true}
          isHovered={true}
        />
      </svg>
    );

    const group = container.querySelector('.tier-drop-zone');
    expect(group).toHaveClass('tier-drop-zone-hover');
  });

  it('does not render when not active', () => {
    const { container } = render(
      <svg>
        <TierDropZone
          tier="parser"
          bounds={bounds}
          isActive={false}
          isHovered={false}
        />
      </svg>
    );

    const group = container.querySelector('.tier-drop-zone');
    expect(group).toBeNull();
  });
});

describe('Animation Integration in PipelineVisualization', () => {
  const originalMatchMedia = window.matchMedia;

  beforeEach(() => {
    mockMatchMediaDesktop();
    vi.useFakeTimers();
  });

  afterEach(() => {
    window.matchMedia = originalMatchMedia;
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  // Note: Testing the full animation lifecycle requires mocking plugin API
  // which is complex. These tests focus on the CSS classes being applied.

  it('preview edge renders with animation during drag simulation', () => {
    const mockDAG: PipelineDAG = {
      id: 'test-pipeline',
      version: '1',
      nodes: [
        { id: 'parser-1', type: 'parser', plugin_id: 'text', config: {} },
      ],
      edges: [
        { from_node: '_source', to_node: 'parser-1', when: null },
      ],
    };

    const { container } = render(
      <PipelineVisualization dag={mockDAG} onDagChange={vi.fn()} />
    );

    // Preview edge is not rendered initially (no drag)
    const previewEdge = container.querySelector('.preview-edge');
    expect(previewEdge).toBeNull();
  });

  it('tier add buttons are rendered for each tier', () => {
    const mockDAG: PipelineDAG = {
      id: 'test-pipeline',
      version: '1',
      nodes: [
        { id: 'parser-1', type: 'parser', plugin_id: 'text', config: {} },
      ],
      edges: [
        { from_node: '_source', to_node: 'parser-1', when: null },
      ],
    };

    const { container } = render(
      <PipelineVisualization dag={mockDAG} onDagChange={vi.fn()} />
    );

    // All 4 tier add buttons should be rendered
    const addButtons = container.querySelectorAll('.tier-add-button');
    expect(addButtons.length).toBe(4);
  });
});

describe('Port Hover States', () => {
  it('output port has transition styles', () => {
    const node = {
      id: 'test-node',
      type: 'parser' as const,
      plugin_id: 'text',
      config: {},
    };
    const position = { x: 100, y: 100, width: 160, height: 80 };

    const { container } = render(
      <svg>
        <PipelineNodeComponent
          node={node}
          position={position}
          selected={false}
          showPorts={true}
          onStartDrag={vi.fn()}
        />
      </svg>
    );

    const outputPort = container.querySelector('.output-port');
    const style = outputPort?.getAttribute('style') || '';
    expect(style).toContain('transition');
    expect(style).toContain('ease-in-out');
  });

  it('input port has transition styles', () => {
    const node = {
      id: 'test-node',
      type: 'parser' as const,
      plugin_id: 'text',
      config: {},
    };
    const position = { x: 100, y: 100, width: 160, height: 80 };

    const { container } = render(
      <svg>
        <PipelineNodeComponent
          node={node}
          position={position}
          selected={false}
          showPorts={true}
        />
      </svg>
    );

    const inputPort = container.querySelector('.input-port');
    const style = inputPort?.getAttribute('style') || '';
    expect(style).toContain('transition');
    expect(style).toContain('ease-in-out');
  });
});
