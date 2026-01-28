// apps/webui-react/src/components/wizard/__tests__/CollectionWizard.assisted.test.tsx
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CollectionWizard } from '../CollectionWizard';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';

// Mock hooks
vi.mock('../../../hooks/useAgentConversation', () => ({
  useCreateConversation: () => ({
    mutateAsync: vi.fn().mockResolvedValue({ id: 'conv-123' }),
    isPending: false,
  }),
}));

vi.mock('../../../hooks/useAgentStream', () => ({
  useAgentStream: () => ({
    status: { phase: 'ready', message: 'Analysis complete' },
    activities: [{ message: 'Scanned 10 files', timestamp: new Date().toISOString() }],
    pendingQuestions: [],
    dismissQuestion: vi.fn(),
    isStreaming: false,
    sendMessage: vi.fn(),
    currentContent: 'I recommend semantic chunking for your documentation.',
    pipeline: { chunking_strategy: 'semantic', embedding_model: 'default' },
  }),
}));

vi.mock('../../../hooks/useConnectors', () => ({
  useConnectorCatalog: () => ({
    data: {
      none: {
        name: 'No Source',
        fields: [],
        secrets: [],
      },
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

vi.mock('../../../hooks/useCollections', () => ({
  useCreateCollection: () => ({
    mutateAsync: vi.fn().mockResolvedValue({ id: 'collection-123' }),
    isPending: false,
  }),
}));

vi.mock('../../../hooks/useCollectionOperations', () => ({
  useAddSource: () => ({
    mutateAsync: vi.fn().mockResolvedValue({}),
    isPending: false,
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

describe('CollectionWizard Assisted Flow', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    queryClient.clear();
    vi.clearAllMocks();
  });

  it('creates conversation when entering step 3 in assisted mode', async () => {
    render(<CollectionWizard onClose={vi.fn()} onSuccess={vi.fn()} />, { wrapper });

    // Step 1: Fill basics with source
    await user.type(screen.getByLabelText(/collection name/i), 'Test Collection');

    // Select directory source
    const dirButton = screen.getByText(/directory/i);
    await user.click(dirButton);

    // Fill in path
    const pathInput = await screen.findByLabelText(/path/i);
    await user.type(pathInput, '/home/user/docs');

    await user.click(screen.getByRole('button', { name: /next/i }));

    // Step 2: Select Assisted mode
    await waitFor(() => expect(screen.getByText(/assisted/i)).toBeInTheDocument());
    const assistedOption = screen.getByText(/assisted/i).closest('button');
    if (assistedOption) await user.click(assistedOption);

    await user.click(screen.getByRole('button', { name: /next/i }));

    // Step 3: Should see analysis UI
    await waitFor(() => {
      expect(screen.getByTestId('agent-column')).toBeInTheDocument();
    });
  });

  it('shows review step after clicking next from analysis', async () => {
    render(<CollectionWizard onClose={vi.fn()} onSuccess={vi.fn()} />, { wrapper });

    // Step 1: Fill basics with source
    await user.type(screen.getByLabelText(/collection name/i), 'Test');

    // Select directory source
    const dirButton = screen.getByText(/directory/i);
    await user.click(dirButton);

    // Fill in path
    const pathInput = await screen.findByLabelText(/path/i);
    await user.type(pathInput, '/home/user/docs');

    await user.click(screen.getByRole('button', { name: /next/i }));

    // Step 2: Select Assisted mode
    await waitFor(() => expect(screen.getByText(/assisted/i)).toBeInTheDocument());
    const assistedOption = screen.getByText(/assisted/i).closest('button');
    if (assistedOption) await user.click(assistedOption);

    await user.click(screen.getByRole('button', { name: /next/i }));

    // Should be on analysis step
    await waitFor(() => expect(screen.getByTestId('agent-column')).toBeInTheDocument());

    // Click "Next" to go to review step
    await user.click(screen.getByRole('button', { name: /next/i }));

    // Should see review step
    await waitFor(() => {
      expect(screen.getByTestId('summary-column')).toBeInTheDocument();
    });
  });
});
