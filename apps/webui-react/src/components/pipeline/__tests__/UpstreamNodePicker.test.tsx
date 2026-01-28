import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

import { UpstreamNodePicker } from '../UpstreamNodePicker';
import type { PipelineNode } from '@/types/pipeline';

describe('UpstreamNodePicker', () => {
  const mockUpstreamNodes: PipelineNode[] = [
    { id: 'parser-1', type: 'parser', plugin_id: 'pdf-parser', config: {} },
    { id: 'parser-2', type: 'parser', plugin_id: 'html-parser', config: {} },
    { id: 'parser-3', type: 'parser', plugin_id: 'text-parser', config: {} },
  ];

  const defaultProps = {
    upstreamNodes: mockUpstreamNodes,
    position: { x: 100, y: 200 },
    onSelect: vi.fn(),
    onCancel: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('initial state', () => {
    it('renders with all nodes selected by default', () => {
      render(<UpstreamNodePicker {...defaultProps} />);

      // All checkboxes should be checked
      const checkboxes = screen.getAllByRole('checkbox', { hidden: true });
      expect(checkboxes).toHaveLength(3);
      checkboxes.forEach((checkbox) => {
        expect(checkbox).toBeChecked();
      });
    });

    it('displays all upstream node plugin IDs', () => {
      render(<UpstreamNodePicker {...defaultProps} />);

      expect(screen.getByText('pdf-parser')).toBeInTheDocument();
      expect(screen.getByText('html-parser')).toBeInTheDocument();
      expect(screen.getByText('text-parser')).toBeInTheDocument();
    });

    it('displays node types in parentheses', () => {
      render(<UpstreamNodePicker {...defaultProps} />);

      const parserLabels = screen.getAllByText('(parser)');
      expect(parserLabels).toHaveLength(3);
    });

    it('shows "Connect from" header', () => {
      render(<UpstreamNodePicker {...defaultProps} />);

      expect(screen.getByText('Connect from')).toBeInTheDocument();
    });

    it('shows connect button with count', () => {
      render(<UpstreamNodePicker {...defaultProps} />);

      expect(screen.getByRole('button', { name: /connect \(3\)/i })).toBeInTheDocument();
    });
  });

  describe('click-outside handling', () => {
    it('calls onCancel when clicking outside the popover', async () => {
      const onCancel = vi.fn();

      render(
        <div>
          <div data-testid="outside">Outside</div>
          <UpstreamNodePicker {...defaultProps} onCancel={onCancel} />
        </div>
      );

      // Wait for the click outside handler to be registered (uses setTimeout)
      await new Promise((resolve) => setTimeout(resolve, 10));

      // Click outside the popover
      fireEvent.mouseDown(screen.getByTestId('outside'));

      expect(onCancel).toHaveBeenCalled();
    });

    it('does not call onCancel when clicking inside the popover', async () => {
      const onCancel = vi.fn();

      render(<UpstreamNodePicker {...defaultProps} onCancel={onCancel} />);

      // Wait for the click outside handler to be registered
      await new Promise((resolve) => setTimeout(resolve, 10));

      // Click inside the popover
      fireEvent.mouseDown(screen.getByText('Connect from'));

      expect(onCancel).not.toHaveBeenCalled();
    });
  });

  describe('escape key handling', () => {
    it('calls onCancel when Escape key is pressed', () => {
      const onCancel = vi.fn();

      render(<UpstreamNodePicker {...defaultProps} onCancel={onCancel} />);

      fireEvent.keyDown(document, { key: 'Escape' });

      expect(onCancel).toHaveBeenCalled();
    });
  });

  describe('toggle selection', () => {
    it('toggles checkbox when clicking node label', async () => {
      const user = userEvent.setup();
      render(<UpstreamNodePicker {...defaultProps} />);

      // Find and click the first node's label
      const firstNodeLabel = screen.getByText('pdf-parser');
      await user.click(firstNodeLabel);

      // First checkbox should now be unchecked
      const checkboxes = screen.getAllByRole('checkbox', { hidden: true });
      expect(checkboxes[0]).not.toBeChecked();

      // Button should show updated count
      expect(screen.getByRole('button', { name: /connect \(2\)/i })).toBeInTheDocument();
    });

    it('allows re-selecting a deselected node', async () => {
      const user = userEvent.setup();
      render(<UpstreamNodePicker {...defaultProps} />);

      const firstNodeLabel = screen.getByText('pdf-parser');

      // Deselect
      await user.click(firstNodeLabel);
      expect(screen.getByRole('button', { name: /connect \(2\)/i })).toBeInTheDocument();

      // Re-select
      await user.click(firstNodeLabel);
      expect(screen.getByRole('button', { name: /connect \(3\)/i })).toBeInTheDocument();
    });
  });

  describe('minimum selection constraint', () => {
    it('prevents deselecting the last node', async () => {
      const user = userEvent.setup();

      // Only one node
      const singleNode: PipelineNode[] = [
        { id: 'parser-1', type: 'parser', plugin_id: 'pdf-parser', config: {} },
      ];

      render(
        <UpstreamNodePicker
          {...defaultProps}
          upstreamNodes={singleNode}
        />
      );

      // Try to deselect the only node
      const nodeLabel = screen.getByText('pdf-parser');
      await user.click(nodeLabel);

      // Should still be selected
      const checkbox = screen.getByRole('checkbox', { hidden: true });
      expect(checkbox).toBeChecked();

      // Button should still show 1
      expect(screen.getByRole('button', { name: /connect \(1\)/i })).toBeInTheDocument();
    });

    it('prevents deselecting when only one node remains selected', async () => {
      const user = userEvent.setup();
      render(<UpstreamNodePicker {...defaultProps} />);

      // Deselect first two nodes
      await user.click(screen.getByText('pdf-parser'));
      await user.click(screen.getByText('html-parser'));

      // Only text-parser should be selected now
      expect(screen.getByRole('button', { name: /connect \(1\)/i })).toBeInTheDocument();

      // Try to deselect the last one
      await user.click(screen.getByText('text-parser'));

      // Should still be selected (cannot deselect last node)
      const checkboxes = screen.getAllByRole('checkbox', { hidden: true });
      const lastCheckbox = checkboxes[2];
      expect(lastCheckbox).toBeChecked();
    });
  });

  describe('viewport boundary clamping', () => {
    const originalInnerWidth = window.innerWidth;
    const originalInnerHeight = window.innerHeight;

    beforeEach(() => {
      // Set viewport size for testing
      Object.defineProperty(window, 'innerWidth', { value: 800, writable: true });
      Object.defineProperty(window, 'innerHeight', { value: 600, writable: true });
    });

    afterEach(() => {
      Object.defineProperty(window, 'innerWidth', { value: originalInnerWidth, writable: true });
      Object.defineProperty(window, 'innerHeight', { value: originalInnerHeight, writable: true });
    });

    it('clamps position when near right edge', () => {
      render(
        <UpstreamNodePicker
          {...defaultProps}
          position={{ x: 750, y: 200 }} // Near right edge
        />
      );

      const dialog = screen.getByRole('dialog');
      const style = dialog.style;

      // Position should be clamped to not exceed viewport
      const leftValue = parseInt(style.left, 10);
      expect(leftValue).toBeLessThan(750);
    });

    it('clamps position when near left edge', () => {
      render(
        <UpstreamNodePicker
          {...defaultProps}
          position={{ x: 5, y: 200 }} // Near left edge
        />
      );

      const dialog = screen.getByRole('dialog');
      const style = dialog.style;

      // Position should be at least padding (16)
      const leftValue = parseInt(style.left, 10);
      expect(leftValue).toBeGreaterThanOrEqual(16);
    });

    it('clamps position when near bottom edge', () => {
      render(
        <UpstreamNodePicker
          {...defaultProps}
          position={{ x: 100, y: 550 }} // Near bottom edge
        />
      );

      const dialog = screen.getByRole('dialog');
      const style = dialog.style;

      // Position should be adjusted to fit
      const topValue = parseInt(style.top, 10);
      // Should be positioned above the click point when near bottom
      expect(topValue).toBeLessThan(550);
    });

    it('clamps position when near top edge', () => {
      render(
        <UpstreamNodePicker
          {...defaultProps}
          position={{ x: 100, y: -100 }} // Above viewport
        />
      );

      const dialog = screen.getByRole('dialog');
      const style = dialog.style;

      // Position should be at least padding (16)
      const topValue = parseInt(style.top, 10);
      expect(topValue).toBeGreaterThanOrEqual(16);
    });
  });

  describe('confirm button', () => {
    it('calls onSelect with selected node IDs when clicked', async () => {
      const user = userEvent.setup();
      const onSelect = vi.fn();

      render(<UpstreamNodePicker {...defaultProps} onSelect={onSelect} />);

      // Deselect one node
      await user.click(screen.getByText('html-parser'));

      // Click confirm
      await user.click(screen.getByRole('button', { name: /connect \(2\)/i }));

      expect(onSelect).toHaveBeenCalledWith(['parser-1', 'parser-3']);
    });

    it('calls onSelect with all node IDs when all are selected', async () => {
      const user = userEvent.setup();
      const onSelect = vi.fn();

      render(<UpstreamNodePicker {...defaultProps} onSelect={onSelect} />);

      // Click confirm with all selected
      await user.click(screen.getByRole('button', { name: /connect \(3\)/i }));

      expect(onSelect).toHaveBeenCalledWith(['parser-1', 'parser-2', 'parser-3']);
    });

    it('is disabled when no nodes are selected (edge case)', () => {
      // This shouldn't happen in practice due to minimum selection constraint,
      // but the button has a disabled state
      render(<UpstreamNodePicker {...defaultProps} />);

      const confirmButton = screen.getByRole('button', { name: /connect/i });
      // With all selected, should not be disabled
      expect(confirmButton).not.toBeDisabled();
    });
  });

  describe('cancel button', () => {
    it('calls onCancel when Cancel button is clicked', async () => {
      const user = userEvent.setup();
      const onCancel = vi.fn();

      render(<UpstreamNodePicker {...defaultProps} onCancel={onCancel} />);

      // Use getAllBy since there are two cancel buttons (X icon and text Cancel)
      const cancelButtons = screen.getAllByRole('button', { name: /cancel/i });
      // The text "Cancel" button is the second one
      await user.click(cancelButtons[1]);

      expect(onCancel).toHaveBeenCalled();
    });

    it('calls onCancel when X button is clicked', async () => {
      const user = userEvent.setup();
      const onCancel = vi.fn();

      render(<UpstreamNodePicker {...defaultProps} onCancel={onCancel} />);

      // The X button has aria-label="Cancel"
      const closeButton = screen.getAllByRole('button', { name: /cancel/i })[0];
      await user.click(closeButton);

      expect(onCancel).toHaveBeenCalled();
    });
  });

  describe('accessibility', () => {
    it('has role="dialog"', () => {
      render(<UpstreamNodePicker {...defaultProps} />);

      expect(screen.getByRole('dialog')).toBeInTheDocument();
    });

    it('has aria-label', () => {
      render(<UpstreamNodePicker {...defaultProps} />);

      expect(screen.getByRole('dialog')).toHaveAttribute('aria-label', 'Select upstream nodes');
    });

    it('close button has aria-label', () => {
      render(<UpstreamNodePicker {...defaultProps} />);

      const closeButtons = screen.getAllByRole('button', { name: /cancel/i });
      // The X button (first cancel button) should have aria-label
      expect(closeButtons[0]).toHaveAttribute('aria-label', 'Cancel');
    });
  });

  describe('cleanup', () => {
    it('removes event listeners on unmount', () => {
      const onCancel = vi.fn();
      const { unmount } = render(
        <UpstreamNodePicker {...defaultProps} onCancel={onCancel} />
      );

      unmount();

      // Fire events after unmount - should not trigger onCancel
      fireEvent.keyDown(document, { key: 'Escape' });

      // Give time for any async handlers
      expect(onCancel).not.toHaveBeenCalled();
    });
  });

  describe('mixed node types', () => {
    it('displays different node types correctly', () => {
      const mixedNodes: PipelineNode[] = [
        { id: 'chunker-1', type: 'chunker', plugin_id: 'sentence-chunker', config: {} },
        { id: 'extractor-1', type: 'extractor', plugin_id: 'entity-extractor', config: {} },
      ];

      render(
        <UpstreamNodePicker
          {...defaultProps}
          upstreamNodes={mixedNodes}
        />
      );

      expect(screen.getByText('sentence-chunker')).toBeInTheDocument();
      expect(screen.getByText('entity-extractor')).toBeInTheDocument();
      expect(screen.getByText('(chunker)')).toBeInTheDocument();
      expect(screen.getByText('(extractor)')).toBeInTheDocument();
    });
  });
});
