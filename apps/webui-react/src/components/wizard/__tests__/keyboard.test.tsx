import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '../../../tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import { CollectionWizard } from '../CollectionWizard';

describe('Wizard Keyboard Navigation', () => {
  it('closes wizard on Escape', async () => {
    const onClose = vi.fn();
    render(<CollectionWizard onClose={onClose} onSuccess={vi.fn()} />);

    fireEvent.keyDown(document, { key: 'Escape' });

    expect(onClose).toHaveBeenCalled();
  });

  it('focuses first input on open', async () => {
    render(<CollectionWizard onClose={vi.fn()} onSuccess={vi.fn()} />);

    // Wait for render to complete
    await waitFor(() => {
      const nameInput = screen.getByLabelText(/collection name/i);
      expect(document.activeElement).toBe(nameInput);
    });
  });

  it('navigates forward with Tab through focusable elements', async () => {
    const user = userEvent.setup();
    render(<CollectionWizard onClose={vi.fn()} onSuccess={vi.fn()} />);

    // Tab through multiple times
    await user.tab();
    await user.tab();
    await user.tab();

    // Focus should be somewhere in the modal
    const modal = screen.getByRole('dialog');
    expect(modal.contains(document.activeElement)).toBe(true);
  });

  it('traps focus within modal', async () => {
    const user = userEvent.setup();
    render(<CollectionWizard onClose={vi.fn()} onSuccess={vi.fn()} />);

    const modal = screen.getByRole('dialog');

    // Tab through all elements
    for (let i = 0; i < 20; i++) {
      await user.tab();
    }

    // Focus should still be within the modal
    expect(modal.contains(document.activeElement)).toBe(true);
  });

  it('advances step with Cmd/Ctrl+Enter when valid', async () => {
    const user = userEvent.setup();
    render(<CollectionWizard onClose={vi.fn()} onSuccess={vi.fn()} />);

    // Fill in required field
    await user.type(screen.getByLabelText(/collection name/i), 'Test');

    // Cmd+Enter to advance
    fireEvent.keyDown(document, { key: 'Enter', metaKey: true });

    // Should be on step 2 (mode selection)
    await waitFor(() => {
      expect(screen.getByText(/choose how you want to configure/i)).toBeInTheDocument();
    });
  });
});
