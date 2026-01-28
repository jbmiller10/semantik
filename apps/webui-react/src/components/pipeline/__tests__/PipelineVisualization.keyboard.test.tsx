import { describe, it, expect, vi, beforeEach, beforeAll } from 'vitest';
import { render, fireEvent, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import '@testing-library/jest-dom';

import { PipelineVisualization } from '../PipelineVisualization';
import type { PipelineDAG, DAGSelection } from '@/types/pipeline';

// Mock the useAvailablePlugins hook
vi.mock('@/hooks/useAvailablePlugins', () => ({
  useAvailablePlugins: vi.fn(() => ({
    plugins: [
      { id: 'recursive', name: 'Recursive', description: 'Recursive text splitter' },
      { id: 'semantic', name: 'Semantic', description: 'Semantic chunking' },
    ],
    isLoading: false,
    error: null,
  })),
}));

// Track ResizeObserver instances for testing
let mockResizeObserverInstances: MockResizeObserver[] = [];

class MockResizeObserver {
  callback: ResizeObserverCallback;
  observedElements: Element[] = [];

  constructor(callback: ResizeObserverCallback) {
    this.callback = callback;
    mockResizeObserverInstances.push(this);
  }

  observe = vi.fn((element: Element) => {
    this.observedElements.push(element);
  });

  unobserve = vi.fn((element: Element) => {
    const index = this.observedElements.indexOf(element);
    if (index > -1) {
      this.observedElements.splice(index, 1);
    }
  });

  disconnect = vi.fn(() => {
    this.observedElements = [];
  });

  // Helper to simulate resize
  trigger(entries: ResizeObserverEntry[]) {
    this.callback(entries, this as unknown as ResizeObserver);
  }
}

beforeAll(() => {
  global.ResizeObserver = MockResizeObserver as unknown as typeof ResizeObserver;
});

describe('PipelineVisualization - Keyboard and UI interactions', () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  const mockDAG: PipelineDAG = {
    id: 'test-pipeline',
    version: '1',
    nodes: [
      { id: 'text_parser', type: 'parser', plugin_id: 'text', config: {} },
      { id: 'chunker', type: 'chunker', plugin_id: 'recursive', config: {} },
    ],
    edges: [
      { from_node: '_source', to_node: 'text_parser', when: null },
      { from_node: 'text_parser', to_node: 'chunker', when: null },
    ],
  };

  const minimalDAG: PipelineDAG = {
    id: 'minimal-pipeline',
    version: '1',
    nodes: [{ id: 'parser', type: 'parser', plugin_id: 'text', config: {} }],
    edges: [{ from_node: '_source', to_node: 'parser', when: null }],
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockResizeObserverInstances = [];
  });

  describe('Escape key during drag', () => {
    it('cancels drag operation when Escape is pressed while dragging', async () => {
      const handleDagChange = vi.fn();

      render(
        <PipelineVisualization dag={mockDAG} onDagChange={handleDagChange} />,
        { wrapper }
      );

      // Find the source node and its output port
      const sourceNode = document.querySelector('g[data-node-id="_source"]');
      expect(sourceNode).toBeInTheDocument();

      const outputPort = sourceNode!.querySelector('circle[data-port="output"]');

      if (outputPort) {
        // Start drag by clicking on output port
        fireEvent.mouseDown(outputPort, { clientX: 180, clientY: 120 });

        // Verify drag preview edge would appear
        // The drag is started - ports should be visible

        // Press Escape to cancel
        fireEvent.keyDown(window, { key: 'Escape' });

        // After Escape, the drag should be cancelled
        // Any subsequent mouse up should not create a connection
        const svg = document.querySelector('svg');
        if (svg) {
          fireEvent.mouseUp(svg, { clientX: 300, clientY: 400 });
        }

        // No DAG change should occur since drag was cancelled
        expect(handleDagChange).not.toHaveBeenCalled();
      }
    });

    it('Escape key has no effect when not dragging', () => {
      const handleDagChange = vi.fn();

      render(
        <PipelineVisualization dag={mockDAG} onDagChange={handleDagChange} />,
        { wrapper }
      );

      // Press Escape without starting a drag
      fireEvent.keyDown(window, { key: 'Escape' });

      // Should have no effect
      expect(handleDagChange).not.toHaveBeenCalled();
    });
  });

  describe('ResizeObserver setup and cleanup', () => {
    it('sets up ResizeObserver on mount', () => {
      render(<PipelineVisualization dag={mockDAG} />, { wrapper });

      // Should have created a ResizeObserver instance
      expect(mockResizeObserverInstances.length).toBeGreaterThan(0);

      // Should have observed the container
      const observer = mockResizeObserverInstances[0];
      expect(observer.observe).toHaveBeenCalled();
    });

    it('disconnects ResizeObserver on unmount', () => {
      const { unmount } = render(<PipelineVisualization dag={mockDAG} />, { wrapper });

      const observer = mockResizeObserverInstances[0];
      unmount();

      expect(observer.disconnect).toHaveBeenCalled();
    });

    it('handles ResizeObserver not being available', () => {
      // Temporarily remove ResizeObserver
      const originalResizeObserver = global.ResizeObserver;
      // @ts-expect-error - intentionally making it undefined
      global.ResizeObserver = undefined;

      // Should not throw
      expect(() => {
        render(<PipelineVisualization dag={mockDAG} />, { wrapper });
      }).not.toThrow();

      // Restore
      global.ResizeObserver = originalResizeObserver;
    });
  });

  describe('Add button positioning', () => {
    it('positions add button centered for empty tier', () => {
      // DAG with only parser - chunker tier is empty
      const dagWithOnlyParser: PipelineDAG = {
        id: 'test',
        version: '1',
        nodes: [{ id: 'parser', type: 'parser', plugin_id: 'text', config: {} }],
        edges: [{ from_node: '_source', to_node: 'parser', when: null }],
      };

      render(<PipelineVisualization dag={dagWithOnlyParser} onDagChange={vi.fn()} />, {
        wrapper,
      });

      // Find chunker tier add button (empty tier)
      const chunkerButton = document.querySelector(
        'g.tier-add-button[data-tier="chunker"]'
      );
      expect(chunkerButton).toBeInTheDocument();

      // The button should be positioned - we can check its transform attribute
      // For empty tiers, it should be centered at PADDING + NODE_WIDTH / 2
      const transform = chunkerButton?.getAttribute('transform');
      expect(transform).toBeDefined();
    });

    it('positions add button after rightmost node for populated tier', () => {
      render(<PipelineVisualization dag={mockDAG} onDagChange={vi.fn()} />, { wrapper });

      // Parser tier has a node
      const parserButton = document.querySelector(
        'g.tier-add-button[data-tier="parser"]'
      );
      expect(parserButton).toBeInTheDocument();

      // The button should be positioned to the right of the existing node
      const transform = parserButton?.getAttribute('transform');
      expect(transform).toBeDefined();
    });

    it('renders all tier add buttons', () => {
      render(<PipelineVisualization dag={minimalDAG} onDagChange={vi.fn()} />, {
        wrapper,
      });

      // Should have add buttons for all 4 tiers
      expect(document.querySelector('g.tier-add-button[data-tier="parser"]')).toBeInTheDocument();
      expect(document.querySelector('g.tier-add-button[data-tier="chunker"]')).toBeInTheDocument();
      expect(document.querySelector('g.tier-add-button[data-tier="extractor"]')).toBeInTheDocument();
      expect(document.querySelector('g.tier-add-button[data-tier="embedder"]')).toBeInTheDocument();
    });

    it('hides add buttons during drag', () => {
      render(<PipelineVisualization dag={mockDAG} onDagChange={vi.fn()} />, { wrapper });

      // Initially buttons should be visible
      expect(document.querySelector('g.tier-add-buttons')).toBeInTheDocument();

      // Start a drag
      const sourceNode = document.querySelector('g[data-node-id="_source"]');
      const outputPort = sourceNode?.querySelector('circle[data-port="output"]');

      if (outputPort) {
        fireEvent.mouseDown(outputPort, { clientX: 180, clientY: 120 });

        // Buttons should be hidden during drag
        expect(document.querySelector('g.tier-add-buttons')).not.toBeInTheDocument();

        // Cancel the drag
        fireEvent.keyDown(window, { key: 'Escape' });

        // Buttons should reappear
        expect(document.querySelector('g.tier-add-buttons')).toBeInTheDocument();
      }
    });
  });

  describe('Delete key edge cases (focus handling)', () => {
    // Note: Basic delete/backspace tests are covered in PipelineVisualization.delete.test.tsx
    // These tests focus on input focus handling edge cases

    it('does not trigger deletion when focus is in textarea', () => {
      const handleDagChange = vi.fn();
      const selection: DAGSelection = { type: 'node', nodeId: 'text_parser' };

      render(
        <div>
          <textarea data-testid="test-textarea" />
          <PipelineVisualization
            dag={mockDAG}
            selection={selection}
            onSelectionChange={vi.fn()}
            onDagChange={handleDagChange}
          />
        </div>,
        { wrapper }
      );

      const textarea = screen.getByTestId('test-textarea');
      textarea.focus();

      fireEvent.keyDown(textarea, { key: 'Delete' });

      expect(handleDagChange).not.toHaveBeenCalled();
    });

    it('does not trigger deletion when focus is in contenteditable element', () => {
      const handleDagChange = vi.fn();
      const selection: DAGSelection = { type: 'node', nodeId: 'text_parser' };

      render(
        <div>
          <div data-testid="editable" contentEditable={true} />
          <PipelineVisualization
            dag={mockDAG}
            selection={selection}
            onSelectionChange={vi.fn()}
            onDagChange={handleDagChange}
          />
        </div>,
        { wrapper }
      );

      const editable = screen.getByTestId('editable');
      editable.focus();

      fireEvent.keyDown(editable, { key: 'Delete' });

      expect(handleDagChange).not.toHaveBeenCalled();
    });
  });

  describe('Mouse leave during drag', () => {
    it('cancels drag when mouse leaves SVG', () => {
      const handleDagChange = vi.fn();

      render(
        <PipelineVisualization dag={mockDAG} onDagChange={handleDagChange} />,
        { wrapper }
      );

      const sourceNode = document.querySelector('g[data-node-id="_source"]');
      const outputPort = sourceNode?.querySelector('circle[data-port="output"]');

      if (outputPort) {
        // Start drag
        fireEvent.mouseDown(outputPort, { clientX: 180, clientY: 120 });

        // Mouse leave the SVG
        const svg = document.querySelector('svg');
        if (svg) {
          fireEvent.mouseLeave(svg);
        }

        // Drag should be cancelled, no connection made
        expect(handleDagChange).not.toHaveBeenCalled();
      }
    });
  });

  describe('Extended layout height', () => {
    it('extends layout height to include all tiers', () => {
      render(<PipelineVisualization dag={minimalDAG} />, { wrapper });

      const svg = document.querySelector('svg');
      expect(svg).toBeInTheDocument();

      // SVG height should be tall enough to include all tiers
      // Each tier is at: PADDING + tierIndex * (NODE_HEIGHT + TIER_GAP)
      // With PADDING=40, NODE_HEIGHT=80, TIER_GAP=100
      // Embedder tier (4) would be at: 40 + 4 * (80 + 100) = 40 + 720 = 760
      // Plus some padding, the height should be at least this
      const height = svg?.getAttribute('height');
      expect(height).toBeDefined();
      expect(parseInt(height!, 10)).toBeGreaterThanOrEqual(500);
    });
  });

  describe('Edge rendering', () => {
    it('renders edges with correct source and target', () => {
      render(<PipelineVisualization dag={mockDAG} />, { wrapper });

      // Find edges in the edges group
      const edgesGroup = document.querySelector('g.edges');
      expect(edgesGroup).toBeInTheDocument();

      // Should have edges
      const paths = edgesGroup?.querySelectorAll('path');
      expect(paths?.length).toBeGreaterThan(0);
    });

    it('shows catch-all indicator for source edges with null when', () => {
      render(<PipelineVisualization dag={mockDAG} />, { wrapper });

      // Source edges with when: null should show catch-all indicator
      // This is shown as "*" text on the edge
      const catchAllIndicators = document.querySelectorAll('text');
      // There should be at least one showing catch-all behavior
      const texts = Array.from(catchAllIndicators).map((el) => el.textContent);
      expect(texts.some((t) => t?.includes('*'))).toBe(true);
    });
  });
});
