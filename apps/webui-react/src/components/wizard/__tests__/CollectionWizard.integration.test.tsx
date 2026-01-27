// apps/webui-react/src/components/wizard/__tests__/CollectionWizard.integration.test.tsx
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CollectionWizard } from '../CollectionWizard';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';

// Mock the API hooks
vi.mock('../../../hooks/useCollections', () => ({
  useCreateCollection: () => ({
    mutateAsync: vi.fn().mockResolvedValue({ id: 'new-collection-123' }),
    isPending: false,
  }),
}));

vi.mock('../../../hooks/useCollectionOperations', () => ({
  useAddSource: () => ({
    mutateAsync: vi.fn().mockResolvedValue({}),
    isPending: false,
  }),
}));

vi.mock('../../../hooks/useConnectors', () => ({
  useConnectorCatalog: () => ({
    data: {
      directory: {
        name: 'Directory',
        fields: [{ name: 'path', label: 'Path', type: 'string', required: true }],
        secrets: [],
      },
    },
    isLoading: false,
  }),
  useGitPreview: () => ({ mutateAsync: vi.fn() }),
  useImapPreview: () => ({ mutateAsync: vi.fn() }),
}));

vi.mock('../../../stores/uiStore', () => ({
  useUIStore: () => ({
    addToast: vi.fn(),
  }),
}));

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

describe('CollectionWizard Manual Flow Integration', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    queryClient.clear();
  });

  it('completes manual flow: basics -> mode -> configure -> create', async () => {
    const onSuccess = vi.fn();
    render(<CollectionWizard onClose={vi.fn()} onSuccess={onSuccess} />, { wrapper });

    // Step 1: Fill basics
    const nameInput = screen.getByLabelText(/collection name/i);
    await user.type(nameInput, 'Test Collection');

    // Click Next
    const nextButton = screen.getByRole('button', { name: /next/i });
    await user.click(nextButton);

    // Step 2: Mode selection - should default to manual
    await waitFor(() => {
      expect(screen.getByText(/manual/i)).toBeInTheDocument();
    });

    // Click Next to go to step 3
    await user.click(screen.getByRole('button', { name: /next/i }));

    // Step 3: Configure - modal should be expanded
    await waitFor(() => {
      const modal = screen.getByRole('dialog');
      expect(modal.className).toContain('90vw');
    });

    // Create Collection button should be visible
    const createButton = screen.getByRole('button', { name: /create collection/i });
    expect(createButton).toBeInTheDocument();
  });

  it('expands modal when reaching step 3', async () => {
    render(<CollectionWizard onClose={vi.fn()} onSuccess={vi.fn()} />, { wrapper });

    // Fill step 1
    await user.type(screen.getByLabelText(/collection name/i), 'Test');
    await user.click(screen.getByRole('button', { name: /next/i }));

    // Step 2: Click next
    await waitFor(() => expect(screen.getByText(/manual/i)).toBeInTheDocument());
    await user.click(screen.getByRole('button', { name: /next/i }));

    // Modal should now be expanded
    const modal = screen.getByRole('dialog');
    await waitFor(() => {
      expect(modal.className).toContain('90vw');
    });
  });

  it('allows going back through steps', async () => {
    render(<CollectionWizard onClose={vi.fn()} onSuccess={vi.fn()} />, { wrapper });

    // Navigate to step 2
    await user.type(screen.getByLabelText(/collection name/i), 'Test');
    await user.click(screen.getByRole('button', { name: /next/i }));

    await waitFor(() => expect(screen.getByText(/manual/i)).toBeInTheDocument());

    // Go back
    await user.click(screen.getByRole('button', { name: /back/i }));

    // Should see step 1 content
    await waitFor(() => {
      expect(screen.getByLabelText(/collection name/i)).toBeInTheDocument();
    });
  });

  it('disables Next button when name is empty', () => {
    render(<CollectionWizard onClose={vi.fn()} onSuccess={vi.fn()} />, { wrapper });

    const nextButton = screen.getByRole('button', { name: /next/i });
    expect(nextButton).toBeDisabled();
  });

  it('enables Next button when name is provided', async () => {
    render(<CollectionWizard onClose={vi.fn()} onSuccess={vi.fn()} />, { wrapper });

    await user.type(screen.getByLabelText(/collection name/i), 'Test');

    const nextButton = screen.getByRole('button', { name: /next/i });
    expect(nextButton).not.toBeDisabled();
  });
});
