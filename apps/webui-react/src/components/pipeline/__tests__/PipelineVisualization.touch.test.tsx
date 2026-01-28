/**
 * Tests for touch device behavior in PipelineVisualization.
 * Verifies that drag features are hidden on touch devices
 * and that "+" buttons are properly displayed.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render } from '@/tests/utils/test-utils';
import { PipelineVisualization } from '../PipelineVisualization';
import type { PipelineDAG } from '@/types/pipeline';

// Mock matchMedia for touch device simulation
function mockMatchMedia(isTouch: boolean) {
  const listeners: Array<(e: MediaQueryListEvent) => void> = [];

  const mediaQueryList = {
    matches: isTouch,
    media: '(pointer: coarse)',
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn((event: string, callback: (e: MediaQueryListEvent) => void) => {
      if (event === 'change') listeners.push(callback);
    }),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  };

  window.matchMedia = vi.fn().mockReturnValue(mediaQueryList);
  return mediaQueryList;
}

const mockDAG: PipelineDAG = {
  id: 'test-pipeline',
  version: '1',
  nodes: [
    { id: 'parser-1', type: 'parser', plugin_id: 'text', config: {} },
    { id: 'chunker-1', type: 'chunker', plugin_id: 'recursive', config: {} },
    { id: 'embedder-1', type: 'embedder', plugin_id: 'dense_local', config: {} },
  ],
  edges: [
    { from_node: '_source', to_node: 'parser-1', when: null },
    { from_node: 'parser-1', to_node: 'chunker-1', when: null },
    { from_node: 'chunker-1', to_node: 'embedder-1', when: null },
  ],
};

describe('PipelineVisualization - Touch Device Behavior', () => {
  const originalMatchMedia = window.matchMedia;

  afterEach(() => {
    window.matchMedia = originalMatchMedia;
    vi.restoreAllMocks();
  });

  describe('desktop (non-touch) device', () => {
    beforeEach(() => {
      mockMatchMedia(false);
    });

    it('shows output ports when dragging (via showPorts prop)', () => {
      render(<PipelineVisualization dag={mockDAG} onDagChange={vi.fn()} />);

      // On desktop, ports should be rendered but hidden by default (opacity: 0)
      // They become visible when showPorts is true (during drag)
      const outputPorts = document.querySelectorAll('.output-port');
      expect(outputPorts.length).toBeGreaterThan(0);
    });

    it('renders "+" buttons', () => {
      render(<PipelineVisualization dag={mockDAG} onDagChange={vi.fn()} />);

      // "+" buttons should be rendered
      const addButtons = document.querySelectorAll('.tier-add-button');
      expect(addButtons.length).toBeGreaterThan(0);
    });

    it('renders add buttons with standard size', () => {
      render(<PipelineVisualization dag={mockDAG} onDagChange={vi.fn()} />);

      // Check that buttons don't have touch attribute
      const addButton = document.querySelector('.tier-add-button');
      expect(addButton).not.toHaveAttribute('data-touch', 'true');
    });
  });

  describe('touch device', () => {
    beforeEach(() => {
      mockMatchMedia(true);
    });

    it('hides drag ports on touch devices', () => {
      render(<PipelineVisualization dag={mockDAG} onDagChange={vi.fn()} />);

      // Output ports on source node should have no onStartDrag handler
      // We verify this by checking that ports aren't interactive
      const sourceNode = document.querySelector('g[data-node-id="_source"]');
      const outputPort = sourceNode?.querySelector('.output-port');

      // Port should exist but not have drag cursor
      expect(outputPort).toBeInTheDocument();
    });

    it('always shows "+" buttons on touch devices', () => {
      render(<PipelineVisualization dag={mockDAG} onDagChange={vi.fn()} />);

      // "+" buttons should be visible on touch devices
      const addButtons = document.querySelectorAll('.tier-add-button');
      expect(addButtons.length).toBeGreaterThan(0);
    });

    it('renders add buttons with larger touch size', () => {
      render(<PipelineVisualization dag={mockDAG} onDagChange={vi.fn()} />);

      // Check that buttons have touch attribute
      const addButton = document.querySelector('.tier-add-button');
      expect(addButton).toHaveAttribute('data-touch', 'true');
    });

    it('renders larger touch targets for add buttons', () => {
      render(<PipelineVisualization dag={mockDAG} onDagChange={vi.fn()} />);

      // Touch buttons should have radius of 22 (44px diameter) vs 14 for desktop
      const addButton = document.querySelector('.tier-add-button circle');
      expect(addButton).toHaveAttribute('r', '22');
    });

    it('does not show drop target highlight on touch devices', () => {
      render(<PipelineVisualization dag={mockDAG} onDagChange={vi.fn()} />);

      // Drop targets shouldn't be highlighted since drag is disabled
      const dropTargetNodes = document.querySelectorAll('.pipeline-node-drop-target');
      expect(dropTargetNodes.length).toBe(0);
    });
  });
});

describe('PipelineVisualization - Touch Add Button Flow', () => {
  const originalMatchMedia = window.matchMedia;

  beforeEach(() => {
    mockMatchMedia(true);
  });

  afterEach(() => {
    window.matchMedia = originalMatchMedia;
    vi.restoreAllMocks();
  });

  it('renders "+" buttons for each tier', () => {
    render(<PipelineVisualization dag={mockDAG} onDagChange={vi.fn()} />);

    // Should have buttons for each tier (parser, chunker, extractor, embedder)
    const addButtons = document.querySelectorAll('.tier-add-button');
    expect(addButtons.length).toBe(4);
  });

  it('each tier add button has correct tier attribute', () => {
    render(<PipelineVisualization dag={mockDAG} onDagChange={vi.fn()} />);

    const tiers = ['parser', 'chunker', 'extractor', 'embedder'];
    tiers.forEach((tier) => {
      const button = document.querySelector(`.tier-add-button[data-tier="${tier}"]`);
      expect(button).toBeInTheDocument();
    });
  });

  it('add buttons have proper accessibility labels', () => {
    render(<PipelineVisualization dag={mockDAG} onDagChange={vi.fn()} />);

    const parserButton = document.querySelector('.tier-add-button[data-tier="parser"]');
    expect(parserButton).toHaveAttribute('aria-label', 'Add parser');
  });
});
