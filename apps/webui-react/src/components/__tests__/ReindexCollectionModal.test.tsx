import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, useNavigate } from 'react-router-dom';
import { AxiosError } from 'axios';
import ReindexCollectionModal from '../ReindexCollectionModal';
import { useReindexCollection } from '../../hooks/useCollectionOperations';
import { useUIStore } from '../../stores/uiStore';
import type { Collection } from '../../types/collection';

// Mock dependencies
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: vi.fn(),
  };
});

vi.mock('../../hooks/useCollectionOperations', () => ({
  useReindexCollection: vi.fn(),
}));

vi.mock('../../stores/uiStore', () => ({
  useUIStore: vi.fn(),
}));

const mockCollection: Collection = {
  id: 'test-collection-id',
  name: 'Test Collection',
  description: 'Test description',
  owner_id: 1,
  vector_store_name: 'test_collection_vectors',
  embedding_model: 'text-embedding-ada-002',
  quantization: 'float32',
  chunk_size: 1000,
  chunk_overlap: 200,
  is_public: false,
  status: 'ready',
  document_count: 150,
  vector_count: 2500,
  total_size_bytes: 1048576,
  created_at: '2025-01-01T00:00:00Z',
  updated_at: '2025-01-01T00:00:00Z',
};

const mockConfigChanges = {
  embedding_model: 'text-embedding-3-small',
  chunk_size: 500,
  chunk_overlap: 100,
};

describe('ReindexCollectionModal', () => {
  let queryClient: QueryClient;
  let mockNavigate: ReturnType<typeof vi.fn>;
  let mockAddToast: ReturnType<typeof vi.fn>;
  let mockMutateAsync: ReturnType<typeof vi.fn>;
  let mockOnClose: ReturnType<typeof vi.fn>;
  let mockOnSuccess: ReturnType<typeof vi.fn>;

  const renderComponent = (props = {}) => {
    return render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter>
          <ReindexCollectionModal
            collection={mockCollection}
            configChanges={mockConfigChanges}
            onClose={mockOnClose}
            onSuccess={mockOnSuccess}
            {...props}
          />
        </MemoryRouter>
      </QueryClientProvider>
    );
  };

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    
    mockNavigate = vi.fn();
    mockAddToast = vi.fn();
    mockMutateAsync = vi.fn();
    mockOnClose = vi.fn();
    mockOnSuccess = vi.fn();

    (useNavigate as ReturnType<typeof vi.fn>).mockReturnValue(mockNavigate);
    (useUIStore as ReturnType<typeof vi.fn>).mockReturnValue({ addToast: mockAddToast });
    (useReindexCollection as ReturnType<typeof vi.fn>).mockReturnValue({
      mutateAsync: mockMutateAsync,
      isPending: false,
      isError: false,
    });
  });

  it('renders modal with collection data and configuration changes', () => {
    renderComponent();

    // Check modal title
    expect(screen.getByText(`Re-index Collection: ${mockCollection.name}`)).toBeInTheDocument();

    // Check warning message
    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText('Warning: This action cannot be undone')).toBeInTheDocument();

    // Check impact details
    expect(screen.getByText(`Delete all existing vectors (${mockCollection.vector_count} vectors)`)).toBeInTheDocument();
    expect(screen.getByText(`Re-process all documents (${mockCollection.document_count} documents) with new settings`)).toBeInTheDocument();
    expect(screen.getByText('Make the collection unavailable during processing')).toBeInTheDocument();

    // Check configuration changes section
    expect(screen.getByText('Configuration Changes (3 changes):')).toBeInTheDocument();

    // Check embedding model change
    expect(screen.getByText('Embedding Model:')).toBeInTheDocument();
    expect(screen.getByText(mockCollection.embedding_model)).toBeInTheDocument();
    expect(screen.getByText(mockConfigChanges.embedding_model!)).toBeInTheDocument();

    // Check chunk size change
    expect(screen.getByText('Chunk Size:')).toBeInTheDocument();
    expect(screen.getByText(mockCollection.chunk_size.toString())).toBeInTheDocument();
    expect(screen.getByText(mockConfigChanges.chunk_size!.toString())).toBeInTheDocument();

    // Check chunk overlap change
    expect(screen.getByText('Chunk Overlap:')).toBeInTheDocument();
    expect(screen.getByText(mockCollection.chunk_overlap.toString())).toBeInTheDocument();
    expect(screen.getByText(mockConfigChanges.chunk_overlap!.toString())).toBeInTheDocument();

    // Check estimated impact
    expect(screen.getByText('Estimated impact:')).toBeInTheDocument();
    expect(screen.getByText(`Processing time: ~${Math.ceil(mockCollection.document_count / 100)} minutes (estimate)`)).toBeInTheDocument();
  });

  it('renders with partial configuration changes', () => {
    const partialChanges = { chunk_size: 750 };
    renderComponent({ configChanges: partialChanges });

    expect(screen.getByText('Configuration Changes (1 change):')).toBeInTheDocument();
    expect(screen.getByText('Chunk Size:')).toBeInTheDocument();
    expect(screen.queryByText('Embedding Model:')).not.toBeInTheDocument();
    expect(screen.queryByText('Chunk Overlap:')).not.toBeInTheDocument();
  });

  it('highlights model change warning', () => {
    renderComponent();

    // Should show special warning for model changes
    expect(screen.getByText('Change the embedding model (requires complete re-embedding)')).toBeInTheDocument();
    expect(screen.getByText('Model change may significantly affect search results')).toBeInTheDocument();
  });

  it('requires correct confirmation text to enable submit button', async () => {
    const user = userEvent.setup();
    renderComponent();

    const submitButton = screen.getByRole('button', { name: /re-index collection/i });
    const confirmInput = screen.getByLabelText('Confirmation text for re-indexing');

    // Initially disabled
    expect(submitButton).toBeDisabled();

    // Type incorrect text
    await user.type(confirmInput, 'wrong text');
    expect(submitButton).toBeDisabled();

    // Clear and type correct text
    await user.clear(confirmInput);
    await user.type(confirmInput, `reindex ${mockCollection.name}`);
    expect(submitButton).toBeEnabled();
  });

  it('handles successful reindex submission', async () => {
    const user = userEvent.setup();
    mockMutateAsync.mockResolvedValueOnce({});
    
    renderComponent();

    // Type confirmation text
    const confirmInput = screen.getByLabelText('Confirmation text for re-indexing');
    await user.type(confirmInput, `reindex ${mockCollection.name}`);

    // Submit form
    const submitButton = screen.getByRole('button', { name: /re-index collection/i });
    await user.click(submitButton);

    await waitFor(() => {
      // Check that mutation was called with correct parameters
      expect(mockMutateAsync).toHaveBeenCalledWith({
        collectionId: mockCollection.id,
        config: {
          embedding_model: mockConfigChanges.embedding_model,
          chunk_size: mockConfigChanges.chunk_size,
          chunk_overlap: mockConfigChanges.chunk_overlap,
        },
      });

      // Check navigation
      expect(mockNavigate).toHaveBeenCalledWith(`/collections/${mockCollection.id}`);

      // Check success callback
      expect(mockOnSuccess).toHaveBeenCalled();
    });
  });

  it('handles API errors appropriately', async () => {
    const user = userEvent.setup();
    
    // Test 403 Forbidden error
    const error403 = new AxiosError('Forbidden', '403', undefined, undefined, {
      status: 403,
      data: {},
      statusText: 'Forbidden',
      headers: {},
      config: {} as any,
    });
    
    mockMutateAsync.mockRejectedValueOnce(error403);
    
    renderComponent();

    const confirmInput = screen.getByLabelText('Confirmation text for re-indexing');
    await user.type(confirmInput, `reindex ${mockCollection.name}`);
    await user.click(screen.getByRole('button', { name: /re-index collection/i }));

    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'error',
        message: 'You do not have permission to re-index this collection.',
      });
    });
  });

  it('handles 404 Not Found error', async () => {
    const user = userEvent.setup();
    
    const error404 = new AxiosError('Not Found', '404', undefined, undefined, {
      status: 404,
      data: {},
      statusText: 'Not Found',
      headers: {},
      config: {} as any,
    });
    
    mockMutateAsync.mockRejectedValueOnce(error404);
    
    renderComponent();

    const confirmInput = screen.getByLabelText('Confirmation text for re-indexing');
    await user.type(confirmInput, `reindex ${mockCollection.name}`);
    await user.click(screen.getByRole('button', { name: /re-index collection/i }));

    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'error',
        message: 'Collection not found. It may have been deleted.',
      });
    });
  });

  it('handles 409 Conflict error', async () => {
    const user = userEvent.setup();
    
    const error409 = new AxiosError('Conflict', '409', undefined, undefined, {
      status: 409,
      data: {},
      statusText: 'Conflict',
      headers: {},
      config: {} as any,
    });
    
    mockMutateAsync.mockRejectedValueOnce(error409);
    
    renderComponent();

    const confirmInput = screen.getByLabelText('Confirmation text for re-indexing');
    await user.type(confirmInput, `reindex ${mockCollection.name}`);
    await user.click(screen.getByRole('button', { name: /re-index collection/i }));

    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'error',
        message: 'Another operation is already in progress for this collection.',
      });
    });
  });

  it('handles custom error detail from API response', async () => {
    const user = userEvent.setup();
    
    const customError = new AxiosError('Bad Request', '400', undefined, undefined, {
      status: 400,
      data: { detail: 'Custom error message from API' },
      statusText: 'Bad Request',
      headers: {},
      config: {} as any,
    });
    
    mockMutateAsync.mockRejectedValueOnce(customError);
    
    renderComponent();

    const confirmInput = screen.getByLabelText('Confirmation text for re-indexing');
    await user.type(confirmInput, `reindex ${mockCollection.name}`);
    await user.click(screen.getByRole('button', { name: /re-index collection/i }));

    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'error',
        message: 'Custom error message from API',
      });
    });
  });

  it('handles generic error', async () => {
    const user = userEvent.setup();
    
    const genericError = new Error('Something went wrong');
    mockMutateAsync.mockRejectedValueOnce(genericError);
    
    renderComponent();

    const confirmInput = screen.getByLabelText('Confirmation text for re-indexing');
    await user.type(confirmInput, `reindex ${mockCollection.name}`);
    await user.click(screen.getByRole('button', { name: /re-index collection/i }));

    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'error',
        message: 'Re-indexing failed: Something went wrong',
      });
    });
  });

  it('disables buttons and shows loading state during submission', async () => {
    const user = userEvent.setup();
    
    // Create a promise that we can control
    let resolvePromise: () => void;
    const pendingPromise = new Promise((resolve) => {
      resolvePromise = resolve;
    });
    
    mockMutateAsync.mockReturnValueOnce(pendingPromise);
    
    renderComponent();

    const confirmInput = screen.getByLabelText('Confirmation text for re-indexing');
    await user.type(confirmInput, `reindex ${mockCollection.name}`);

    const submitButton = screen.getByRole('button', { name: /re-index collection/i });
    const cancelButton = screen.getByRole('button', { name: /cancel/i });

    await user.click(submitButton);

    // Check loading state
    expect(screen.getByRole('button', { name: /starting re-index/i })).toBeDisabled();
    expect(cancelButton).toBeDisabled();

    // Resolve the promise
    resolvePromise!();
    
    await waitFor(() => {
      expect(mockOnSuccess).toHaveBeenCalled();
    });
  });

  it('closes modal when cancel button is clicked', async () => {
    const user = userEvent.setup();
    renderComponent();

    const cancelButton = screen.getByRole('button', { name: /cancel/i });
    await user.click(cancelButton);

    expect(mockOnClose).toHaveBeenCalled();
  });

  it('closes modal when clicking overlay', async () => {
    const user = userEvent.setup();
    renderComponent();

    // Find the overlay by its class
    const overlay = document.querySelector('.fixed.inset-0.bg-black.bg-opacity-50');
    expect(overlay).toBeInTheDocument();

    await user.click(overlay!);

    expect(mockOnClose).toHaveBeenCalled();
  });

  it('closes modal when Escape key is pressed', () => {
    renderComponent();

    const modalContent = document.querySelector('.relative.bg-white.rounded-lg');
    expect(modalContent).toBeInTheDocument();

    fireEvent.keyDown(modalContent!, { key: 'Escape' });

    expect(mockOnClose).toHaveBeenCalled();
  });

  it('does not close modal when submitting', async () => {
    const user = userEvent.setup();
    
    // Mock mutation to be pending
    (useReindexCollection as ReturnType<typeof vi.fn>).mockReturnValue({
      mutateAsync: mockMutateAsync,
      isPending: true,
      isError: false,
    });
    
    renderComponent();

    // Modal should not have click handler on overlay during submission
    // The component doesn't implement backdrop click to close, so we just verify the cancel button is disabled
    const cancelButton = screen.getByRole('button', { name: /cancel/i });
    expect(cancelButton).toBeDisabled();

    // Cancel button should be disabled during submission
    await user.click(cancelButton);
    expect(mockOnClose).not.toHaveBeenCalled();
  });

  it('handles form submission with enter key', async () => {
    const user = userEvent.setup();
    mockMutateAsync.mockResolvedValueOnce({});
    
    renderComponent();

    const confirmInput = screen.getByLabelText('Confirmation text for re-indexing');
    await user.type(confirmInput, `reindex ${mockCollection.name}`);
    
    // Press Enter in the input field
    await user.keyboard('{Enter}');

    await waitFor(() => {
      expect(mockMutateAsync).toHaveBeenCalled();
    });
  });

  it('ensures accessibility attributes are properly set', () => {
    renderComponent();

    // Check ARIA attributes
    const warningAlert = screen.getByRole('alert');
    expect(warningAlert).toBeInTheDocument();

    // Check form labels
    const confirmInput = screen.getByLabelText('Confirmation text for re-indexing');
    expect(confirmInput).toHaveAttribute('aria-describedby', 'confirm-help-text');

    // Check screen reader text
    expect(screen.getByText('Type the exact phrase shown above to confirm the re-index operation')).toHaveClass('sr-only');

    // Check icon aria-hidden
    const svgIcons = document.querySelectorAll('svg[aria-hidden="true"]');
    expect(svgIcons.length).toBeGreaterThan(0);
  });

  it('focuses confirmation input on mount', () => {
    renderComponent();

    const confirmInput = screen.getByLabelText('Confirmation text for re-indexing');
    expect(document.activeElement).toBe(confirmInput);
  });

  it('prevents form submission when confirmation text is invalid', async () => {
    const user = userEvent.setup();
    renderComponent();

    const confirmInput = screen.getByLabelText('Confirmation text for re-indexing');
    await user.type(confirmInput, 'invalid text');

    // Try to submit with Enter key
    await user.keyboard('{Enter}');

    expect(mockMutateAsync).not.toHaveBeenCalled();
  });

  it('handles mutation error state from hook', async () => {
    const user = userEvent.setup();
    
    // Mock the hook to indicate an error has already been handled
    (useReindexCollection as ReturnType<typeof vi.fn>).mockReturnValue({
      mutateAsync: mockMutateAsync,
      isPending: false,
      isError: true, // Indicates error was already handled by mutation
    });
    
    mockMutateAsync.mockRejectedValueOnce(new Error('Already handled'));
    
    renderComponent();

    const confirmInput = screen.getByLabelText('Confirmation text for re-indexing');
    await user.type(confirmInput, `reindex ${mockCollection.name}`);
    await user.click(screen.getByRole('button', { name: /re-index collection/i }));

    await waitFor(() => {
      // Should not show error toast since mutation already handled it
      expect(mockAddToast).not.toHaveBeenCalled();
    });
  });

  it('calculates correct processing time estimate', () => {
    // Test with different document counts
    const collectionWithManyDocs = {
      ...mockCollection,
      document_count: 523,
    };
    
    renderComponent({ collection: collectionWithManyDocs });
    
    expect(screen.getByText(`Processing time: ~${Math.ceil(523 / 100)} minutes (estimate)`)).toBeInTheDocument();
  });

  it('shows correct singular/plural for single change', () => {
    const singleChange = { chunk_size: 750 };
    renderComponent({ configChanges: singleChange });

    expect(screen.getByText('Configuration Changes (1 change):')).toBeInTheDocument();
  });

  it('handles empty configuration changes gracefully', () => {
    renderComponent({ configChanges: {} });

    expect(screen.getByText('Configuration Changes (0 changes):')).toBeInTheDocument();
    
    // Configuration changes section should still be visible but empty
    const configSection = screen.getByText('Configuration Changes (0 changes):').parentElement;
    expect(configSection).toBeInTheDocument();
  });
});