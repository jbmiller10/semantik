// apps/webui-react/src/components/wizard/__tests__/CollectionWizard.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CollectionWizard } from '../CollectionWizard';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: false } },
});

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <BrowserRouter>
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  </BrowserRouter>
);

describe('CollectionWizard', () => {
  it('renders with step 1 visible initially', () => {
    render(<CollectionWizard onClose={vi.fn()} onSuccess={vi.fn()} />, { wrapper });

    // Step 1 content should be visible
    expect(screen.getByLabelText(/collection name/i)).toBeInTheDocument();
  });

  it('shows step progress indicator', () => {
    render(<CollectionWizard onClose={vi.fn()} onSuccess={vi.fn()} />, { wrapper });

    // Should see step labels
    expect(screen.getByText(/basics/i)).toBeInTheDocument();
    expect(screen.getByText(/mode/i)).toBeInTheDocument();
  });

  it('starts with compact modal size', () => {
    render(<CollectionWizard onClose={vi.fn()} onSuccess={vi.fn()} />, { wrapper });

    // Modal container should have compact sizing class
    const modal = screen.getByRole('dialog');
    expect(modal.className).toContain('max-w-2xl');
  });

  it('calls onClose when close button clicked', async () => {
    const onClose = vi.fn();
    render(<CollectionWizard onClose={onClose} onSuccess={vi.fn()} />, { wrapper });

    const closeButton = screen.getByRole('button', { name: /close/i });
    await userEvent.click(closeButton);

    expect(onClose).toHaveBeenCalled();
  });
});
