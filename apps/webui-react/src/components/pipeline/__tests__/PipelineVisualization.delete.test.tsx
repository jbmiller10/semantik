import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import { PipelineVisualization } from '../PipelineVisualization';
import type { PipelineDAG, DAGSelection } from '@/types/pipeline';

// Mock the useAvailablePlugins hook
vi.mock('@/hooks/useAvailablePlugins', () => ({
  useAvailablePlugins: vi.fn(() => ({
    plugins: [],
    isLoading: false,
    error: null,
  })),
}));

// Mock ResizeObserver
beforeEach(() => {
  vi.stubGlobal('ResizeObserver', class {
    observe() {}
    unobserve() {}
    disconnect() {}
  });
});

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

// Linear DAG for simpler orphan testing
const linearDAG: PipelineDAG = {
  id: 'linear-pipeline',
  version: '1',
  nodes: [
    { id: 'parser', type: 'parser', plugin_id: 'text', config: {} },
    { id: 'chunker', type: 'chunker', plugin_id: 'recursive', config: {} },
    { id: 'embedder', type: 'embedder', plugin_id: 'dense_local', config: {} },
  ],
  edges: [
    { from_node: '_source', to_node: 'parser', when: null },
    { from_node: 'parser', to_node: 'chunker', when: null },
    { from_node: 'chunker', to_node: 'embedder', when: null },
  ],
};

describe('PipelineVisualization - Deletion', () => {
  describe('Keyboard shortcuts', () => {
    it('deletes selected node when Delete key is pressed', async () => {
      const handleDagChange = vi.fn();
      const handleSelectionChange = vi.fn();
      const selection: DAGSelection = { type: 'node', nodeId: 'text_parser' };

      render(
        <PipelineVisualization
          dag={mockDAG}
          selection={selection}
          onSelectionChange={handleSelectionChange}
          onDagChange={handleDagChange}
        />
      );

      fireEvent.keyDown(window, { key: 'Delete' });

      await waitFor(() => {
        expect(handleDagChange).toHaveBeenCalled();
      });

      const newDag = handleDagChange.mock.calls[0][0];
      expect(newDag.nodes.find((n: { id: string }) => n.id === 'text_parser')).toBeUndefined();
    });

    it('deletes selected node when Backspace key is pressed', async () => {
      const handleDagChange = vi.fn();
      const selection: DAGSelection = { type: 'node', nodeId: 'text_parser' };

      render(
        <PipelineVisualization
          dag={mockDAG}
          selection={selection}
          onSelectionChange={vi.fn()}
          onDagChange={handleDagChange}
        />
      );

      fireEvent.keyDown(window, { key: 'Backspace' });

      await waitFor(() => {
        expect(handleDagChange).toHaveBeenCalled();
      });
    });

    it('deletes selected edge when Delete key is pressed', async () => {
      const handleDagChange = vi.fn();
      const selection: DAGSelection = { type: 'edge', fromNode: 'text_parser', toNode: 'chunker' };

      render(
        <PipelineVisualization
          dag={mockDAG}
          selection={selection}
          onSelectionChange={vi.fn()}
          onDagChange={handleDagChange}
        />
      );

      fireEvent.keyDown(window, { key: 'Delete' });

      await waitFor(() => {
        expect(handleDagChange).toHaveBeenCalled();
      });

      const newDag = handleDagChange.mock.calls[0][0];
      const deletedEdge = newDag.edges.find(
        (e: { from_node: string; to_node: string }) =>
          e.from_node === 'text_parser' && e.to_node === 'chunker'
      );
      expect(deletedEdge).toBeUndefined();
    });

    it('does not delete when nothing is selected', () => {
      const handleDagChange = vi.fn();

      render(
        <PipelineVisualization
          dag={mockDAG}
          selection={{ type: 'none' }}
          onSelectionChange={vi.fn()}
          onDagChange={handleDagChange}
        />
      );

      fireEvent.keyDown(window, { key: 'Delete' });

      expect(handleDagChange).not.toHaveBeenCalled();
    });

    it('does not delete in read-only mode', () => {
      const handleDagChange = vi.fn();
      const selection: DAGSelection = { type: 'node', nodeId: 'text_parser' };

      render(
        <PipelineVisualization
          dag={mockDAG}
          selection={selection}
          onSelectionChange={vi.fn()}
          onDagChange={handleDagChange}
          readOnly={true}
        />
      );

      fireEvent.keyDown(window, { key: 'Delete' });

      expect(handleDagChange).not.toHaveBeenCalled();
    });

    it('does not delete when user is typing in an input', () => {
      const handleDagChange = vi.fn();
      const selection: DAGSelection = { type: 'node', nodeId: 'text_parser' };

      const { container } = render(
        <>
          <input data-testid="text-input" />
          <PipelineVisualization
            dag={mockDAG}
            selection={selection}
            onSelectionChange={vi.fn()}
            onDagChange={handleDagChange}
          />
        </>
      );

      const input = container.querySelector('input')!;
      input.focus();
      fireEvent.keyDown(input, { key: 'Delete' });

      expect(handleDagChange).not.toHaveBeenCalled();
    });
  });

  describe('Source node protection', () => {
    it('cannot delete the source node', () => {
      const handleDagChange = vi.fn();
      const selection: DAGSelection = { type: 'node', nodeId: '_source' };

      render(
        <PipelineVisualization
          dag={mockDAG}
          selection={selection}
          onSelectionChange={vi.fn()}
          onDagChange={handleDagChange}
        />
      );

      fireEvent.keyDown(window, { key: 'Delete' });

      expect(handleDagChange).not.toHaveBeenCalled();
    });
  });

  describe('Orphan handling', () => {
    it('shows confirmation dialog when deletion would orphan nodes', async () => {
      const selection: DAGSelection = { type: 'node', nodeId: 'parser' };

      render(
        <PipelineVisualization
          dag={linearDAG}
          selection={selection}
          onSelectionChange={vi.fn()}
          onDagChange={vi.fn()}
        />
      );

      fireEvent.keyDown(window, { key: 'Delete' });

      // Wait for the confirmation dialog to appear
      await waitFor(() => {
        expect(screen.getByRole('dialog')).toBeInTheDocument();
      });

      // Should show orphaned nodes warning
      expect(screen.getByText(/orphaned node/i)).toBeInTheDocument();
    });

    it('deletes orphaned nodes when confirmed', async () => {
      const user = userEvent.setup();
      const handleDagChange = vi.fn();
      const selection: DAGSelection = { type: 'node', nodeId: 'parser' };

      render(
        <PipelineVisualization
          dag={linearDAG}
          selection={selection}
          onSelectionChange={vi.fn()}
          onDagChange={handleDagChange}
        />
      );

      fireEvent.keyDown(window, { key: 'Delete' });

      // Wait for dialog and click Delete
      await waitFor(() => {
        expect(screen.getByRole('dialog')).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /delete/i }));

      expect(handleDagChange).toHaveBeenCalled();
      const newDag = handleDagChange.mock.calls[0][0];
      // All downstream nodes should be deleted
      expect(newDag.nodes.find((n: { id: string }) => n.id === 'parser')).toBeUndefined();
      expect(newDag.nodes.find((n: { id: string }) => n.id === 'chunker')).toBeUndefined();
      expect(newDag.nodes.find((n: { id: string }) => n.id === 'embedder')).toBeUndefined();
    });

    it('cancels deletion when Cancel is clicked', async () => {
      const user = userEvent.setup();
      const handleDagChange = vi.fn();
      const selection: DAGSelection = { type: 'node', nodeId: 'parser' };

      render(
        <PipelineVisualization
          dag={linearDAG}
          selection={selection}
          onSelectionChange={vi.fn()}
          onDagChange={handleDagChange}
        />
      );

      fireEvent.keyDown(window, { key: 'Delete' });

      await waitFor(() => {
        expect(screen.getByRole('dialog')).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /cancel/i }));

      expect(handleDagChange).not.toHaveBeenCalled();
      // Dialog should be closed
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });
  });

  describe('Selection clearing', () => {
    it('clears selection after successful deletion', async () => {
      const handleSelectionChange = vi.fn();
      const handleDagChange = vi.fn();
      const selection: DAGSelection = { type: 'node', nodeId: 'text_parser' };

      render(
        <PipelineVisualization
          dag={mockDAG}
          selection={selection}
          onSelectionChange={handleSelectionChange}
          onDagChange={handleDagChange}
        />
      );

      fireEvent.keyDown(window, { key: 'Delete' });

      await waitFor(() => {
        expect(handleSelectionChange).toHaveBeenCalledWith({ type: 'none' });
      });
    });
  });

  describe('Edge deletion with orphan handling', () => {
    it('shows confirmation when edge deletion would orphan target node', async () => {
      const selection: DAGSelection = { type: 'edge', fromNode: 'parser', toNode: 'chunker' };

      render(
        <PipelineVisualization
          dag={linearDAG}
          selection={selection}
          onSelectionChange={vi.fn()}
          onDagChange={vi.fn()}
        />
      );

      fireEvent.keyDown(window, { key: 'Delete' });

      await waitFor(() => {
        expect(screen.getByRole('dialog')).toBeInTheDocument();
      });
    });

    it('deletes edge without confirmation when target has other incoming edges', async () => {
      const handleDagChange = vi.fn();
      const selection: DAGSelection = { type: 'edge', fromNode: 'text_parser', toNode: 'chunker' };

      render(
        <PipelineVisualization
          dag={mockDAG}
          selection={selection}
          onSelectionChange={vi.fn()}
          onDagChange={handleDagChange}
        />
      );

      fireEvent.keyDown(window, { key: 'Delete' });

      // Should delete immediately without dialog
      await waitFor(() => {
        expect(handleDagChange).toHaveBeenCalled();
      });

      // No dialog should appear
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });
  });
});
