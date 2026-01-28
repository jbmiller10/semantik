import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import { DeleteConfirmationDialog } from '../DeleteConfirmationDialog';
import type { PipelineNode } from '@/types/pipeline';

describe('DeleteConfirmationDialog', () => {
  const mockOrphanedNodes: PipelineNode[] = [
    { id: 'chunker1', type: 'chunker', plugin_id: 'recursive', config: {} },
    { id: 'embedder1', type: 'embedder', plugin_id: 'dense_local', config: {} },
  ];

  it('renders the dialog with message', () => {
    render(
      <DeleteConfirmationDialog
        type="node"
        message="Delete the text node?"
        orphanedNodes={[]}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />
    );

    expect(screen.getByText(/delete node\?/i)).toBeInTheDocument();
    expect(screen.getByText('Delete the text node?')).toBeInTheDocument();
  });

  it('shows orphaned nodes warning', () => {
    render(
      <DeleteConfirmationDialog
        type="node"
        message="Delete the parser node?"
        orphanedNodes={mockOrphanedNodes}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />
    );

    expect(screen.getByText(/this will also delete 2 orphaned nodes/i)).toBeInTheDocument();
    expect(screen.getByText('recursive')).toBeInTheDocument();
    expect(screen.getByText('dense_local')).toBeInTheDocument();
  });

  it('shows singular orphan message for one node', () => {
    render(
      <DeleteConfirmationDialog
        type="node"
        message="Delete the parser node?"
        orphanedNodes={[mockOrphanedNodes[0]]}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />
    );

    expect(screen.getByText(/this will also delete 1 orphaned node:/i)).toBeInTheDocument();
  });

  it('calls onConfirm with true when Delete button is clicked', async () => {
    const user = userEvent.setup();
    const handleConfirm = vi.fn();

    render(
      <DeleteConfirmationDialog
        type="node"
        message="Delete?"
        orphanedNodes={mockOrphanedNodes}
        onConfirm={handleConfirm}
        onCancel={vi.fn()}
      />
    );

    await user.click(screen.getByRole('button', { name: /delete/i }));

    expect(handleConfirm).toHaveBeenCalledWith(true);
  });

  it('calls onCancel when Cancel button is clicked', async () => {
    const user = userEvent.setup();
    const handleCancel = vi.fn();

    render(
      <DeleteConfirmationDialog
        type="node"
        message="Delete?"
        orphanedNodes={[]}
        onConfirm={vi.fn()}
        onCancel={handleCancel}
      />
    );

    await user.click(screen.getByRole('button', { name: /cancel/i }));

    expect(handleCancel).toHaveBeenCalled();
  });

  it('calls onCancel when Escape key is pressed', () => {
    const handleCancel = vi.fn();

    render(
      <DeleteConfirmationDialog
        type="node"
        message="Delete?"
        orphanedNodes={[]}
        onConfirm={vi.fn()}
        onCancel={handleCancel}
      />
    );

    fireEvent.keyDown(window, { key: 'Escape' });

    expect(handleCancel).toHaveBeenCalled();
  });

  it('calls onCancel when clicking outside dialog', async () => {
    const user = userEvent.setup();
    const handleCancel = vi.fn();

    render(
      <DeleteConfirmationDialog
        type="node"
        message="Delete?"
        orphanedNodes={[]}
        onConfirm={vi.fn()}
        onCancel={handleCancel}
      />
    );

    // Click the backdrop (the outer div with role="dialog")
    const dialog = screen.getByRole('dialog');
    await user.click(dialog);

    expect(handleCancel).toHaveBeenCalled();
  });

  it('does not call onCancel when clicking inside dialog', async () => {
    const user = userEvent.setup();
    const handleCancel = vi.fn();

    render(
      <DeleteConfirmationDialog
        type="node"
        message="Delete?"
        orphanedNodes={[]}
        onConfirm={vi.fn()}
        onCancel={handleCancel}
      />
    );

    // Click the message text inside the dialog
    await user.click(screen.getByText('Delete?'));

    expect(handleCancel).not.toHaveBeenCalled();
  });

  it('shows edge type in title when type is edge', () => {
    render(
      <DeleteConfirmationDialog
        type="edge"
        message="Delete the edge?"
        orphanedNodes={[]}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />
    );

    expect(screen.getByText(/delete edge\?/i)).toBeInTheDocument();
  });

  it('calls onCancel when close button is clicked', async () => {
    const user = userEvent.setup();
    const handleCancel = vi.fn();

    render(
      <DeleteConfirmationDialog
        type="node"
        message="Delete?"
        orphanedNodes={[]}
        onConfirm={vi.fn()}
        onCancel={handleCancel}
      />
    );

    // Find the close button by aria-label
    const closeButton = screen.getByRole('button', { name: /close/i });
    await user.click(closeButton);

    expect(handleCancel).toHaveBeenCalled();
  });
});
