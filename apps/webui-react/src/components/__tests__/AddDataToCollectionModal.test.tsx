import { describe, it, expect, vi, beforeEach, type Mock } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import AddDataToCollectionModal from '../AddDataToCollectionModal';
import { useAddSource } from '../../hooks/useCollectionOperations';
import { useConnectorCatalog, useGitPreview, useImapPreview } from '../../hooks/useConnectors';
import { useUIStore } from '../../stores/uiStore';
import type { Collection } from '../../types/collection';
import type { ConnectorCatalog } from '../../types/connector';

// Mock dependencies
vi.mock('react-router-dom', () => ({
  useNavigate: vi.fn(),
}));

vi.mock('../../hooks/useCollectionOperations', () => ({
  useAddSource: vi.fn(),
}));

vi.mock('../../hooks/useConnectors', () => ({
  useConnectorCatalog: vi.fn(),
  useGitPreview: vi.fn(),
  useImapPreview: vi.fn(),
}));

vi.mock('../../stores/uiStore', () => ({
  useUIStore: vi.fn(),
}));

// Mock connector catalog
const mockCatalog: ConnectorCatalog = {
  directory: {
    name: 'Local Directory',
    description: 'Index files from a local directory path',
    icon: 'folder',
    fields: [
      {
        name: 'path',
        type: 'text',
        label: 'Directory Path',
        description: 'Absolute path to the directory',
        required: true,
        placeholder: '/path/to/documents',
      },
    ],
    secrets: [],
    supports_sync: true,
  },
  git: {
    name: 'Git Repository',
    description: 'Clone and index a Git repository',
    icon: 'git-branch',
    fields: [
      {
        name: 'repo_url',
        type: 'text',
        label: 'Repository URL',
        required: true,
        placeholder: 'https://github.com/user/repo.git',
      },
      {
        name: 'ref',
        type: 'text',
        label: 'Branch/Tag',
        required: false,
        default: 'main',
      },
      {
        name: 'auth_method',
        type: 'select',
        label: 'Authentication',
        required: false,
        default: 'none',
        options: [
          { value: 'none', label: 'None (Public)' },
          { value: 'https_token', label: 'HTTPS Token' },
          { value: 'ssh_key', label: 'SSH Key' },
        ],
      },
    ],
    secrets: [
      {
        name: 'token',
        label: 'Access Token',
        required: false,
        show_when: { field: 'auth_method', equals: 'https_token' },
      },
    ],
    supports_sync: true,
    preview_endpoint: '/api/v2/connectors/preview/git',
  },
  imap: {
    name: 'Email (IMAP)',
    description: 'Index emails from an IMAP mailbox',
    icon: 'mail',
    fields: [
      {
        name: 'host',
        type: 'text',
        label: 'IMAP Server',
        required: true,
        placeholder: 'imap.gmail.com',
      },
      {
        name: 'port',
        type: 'number',
        label: 'Port',
        required: true,
        default: 993,
      },
      {
        name: 'use_ssl',
        type: 'boolean',
        label: 'Use SSL',
        required: false,
        default: true,
      },
      {
        name: 'username',
        type: 'text',
        label: 'Username',
        required: true,
      },
    ],
    secrets: [
      {
        name: 'password',
        label: 'Password',
        required: true,
      },
    ],
    supports_sync: true,
    preview_endpoint: '/api/v2/connectors/preview/imap',
  },
};

// Mock collection data
const mockCollection: Collection = {
  id: 'test-collection-id',
  name: 'Test Collection',
  description: 'Test description',
  owner_id: 1,
  vector_store_name: 'test_vector_store',
  embedding_model: 'text-embedding-ada-002',
  quantization: 'float32',
  chunk_size: 1000,
  chunk_overlap: 200,
  is_public: false,
  status: 'ready',
  document_count: 42,
  vector_count: 100,
  total_size_bytes: 1024000,
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
  sync_mode: 'one_time',
};

describe('AddDataToCollectionModal', () => {
  let queryClient: QueryClient;
  let mockNavigate: Mock;
  let mockAddToast: Mock;
  let mockMutateAsync: Mock;
  let mockOnClose: Mock;
  let mockOnSuccess: Mock;

  beforeEach(() => {
    // Reset all mocks
    vi.clearAllMocks();

    // Create a new QueryClient for each test
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });

    // Setup mock functions
    mockNavigate = vi.fn();
    mockAddToast = vi.fn();
    mockMutateAsync = vi.fn();
    mockOnClose = vi.fn();
    mockOnSuccess = vi.fn();

    // Mock useNavigate
    (useNavigate as Mock).mockReturnValue(mockNavigate);

    // Mock useUIStore
    (useUIStore as unknown as Mock).mockReturnValue({
      addToast: mockAddToast,
    });

    // Mock useAddSource
    (useAddSource as Mock).mockReturnValue({
      mutateAsync: mockMutateAsync,
      isPending: false,
      isError: false,
    });

    // Mock useConnectorCatalog - return loaded catalog by default
    (useConnectorCatalog as Mock).mockReturnValue({
      data: mockCatalog,
      isLoading: false,
      isError: false,
    });

    // Mock preview hooks
    (useGitPreview as Mock).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false,
    });

    (useImapPreview as Mock).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false,
    });
  });

  const renderComponent = (props = {}) => {
    return render(
      <QueryClientProvider client={queryClient}>
        <AddDataToCollectionModal
          collection={mockCollection}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          {...props}
        />
      </QueryClientProvider>
    );
  };

  describe('Modal Rendering', () => {
    it('should render the modal with collection data', () => {
      renderComponent();

      // Check modal title
      expect(screen.getByText('Add Data to Collection')).toBeInTheDocument();

      // Check collection name in subtitle
      expect(screen.getByText(/Add new documents to "Test Collection"/)).toBeInTheDocument();

      // Check connector type selector
      expect(screen.getByText('Select Source Type')).toBeInTheDocument();
      expect(screen.getByText('Local Directory')).toBeInTheDocument();
      expect(screen.getByText('Git Repository')).toBeInTheDocument();
      expect(screen.getByText('Email (IMAP)')).toBeInTheDocument();

      // Check form elements for directory (default selected)
      expect(screen.getByLabelText(/Directory Path/)).toBeInTheDocument();

      // Check collection settings display
      expect(screen.getByText('Collection Settings')).toBeInTheDocument();
      expect(screen.getByText('text-embedding-ada-002')).toBeInTheDocument();
      expect(screen.getByText('1000 characters')).toBeInTheDocument();
      expect(screen.getByText('200 characters')).toBeInTheDocument();

      // Check buttons
      expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Add Source' })).toBeInTheDocument();
    });

    it('should display loading state while catalog is loading', () => {
      (useConnectorCatalog as Mock).mockReturnValue({
        data: undefined,
        isLoading: true,
        isError: false,
      });

      renderComponent();

      expect(screen.getByText('Loading connectors...')).toBeInTheDocument();
    });

    it('should display error state when catalog fails to load', () => {
      (useConnectorCatalog as Mock).mockReturnValue({
        data: undefined,
        isLoading: false,
        isError: true,
      });

      renderComponent();

      expect(screen.getByText('Failed to load connector catalog')).toBeInTheDocument();
    });

    it('should display the info message about duplicate content', () => {
      renderComponent();

      expect(screen.getByText(/Duplicate content will be automatically skipped/)).toBeInTheDocument();
    });
  });

  describe('Connector Type Selection', () => {
    it('should switch to Git connector when selected', async () => {
      renderComponent();

      // Click on Git Repository card
      const gitButton = screen.getByText('Git Repository').closest('button');
      await userEvent.click(gitButton!);

      // Check that Git fields are shown
      expect(screen.getByLabelText(/Repository URL/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Branch\/Tag/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Authentication/)).toBeInTheDocument();

      // Directory Path should no longer be visible
      expect(screen.queryByLabelText('Directory Path')).not.toBeInTheDocument();
    });

    it('should switch to IMAP connector when selected', async () => {
      renderComponent();

      // Click on Email (IMAP) card
      const imapButton = screen.getByText('Email (IMAP)').closest('button');
      await userEvent.click(imapButton!);

      // Check that IMAP fields are shown
      expect(screen.getByLabelText(/IMAP Server/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Port/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Use SSL/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Username/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Password/)).toBeInTheDocument();
    });

    it('should show Test Connection button for Git and IMAP connectors', async () => {
      renderComponent();

      // Directory should not have Test Connection
      expect(screen.queryByRole('button', { name: 'Test Connection' })).not.toBeInTheDocument();

      // Switch to Git
      const gitButton = screen.getByText('Git Repository').closest('button');
      await userEvent.click(gitButton!);
      expect(screen.getByRole('button', { name: 'Test Connection' })).toBeInTheDocument();

      // Switch to IMAP
      const imapButton = screen.getByText('Email (IMAP)').closest('button');
      await userEvent.click(imapButton!);
      expect(screen.getByRole('button', { name: 'Test Connection' })).toBeInTheDocument();
    });
  });

  describe('Path Input Validation', () => {
    it('should show error toast when submitting empty path', async () => {
      renderComponent();

      // Submit the form with empty path
      const submitButton = screen.getByRole('button', { name: 'Add Source' });
      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          type: 'error',
          message: 'Please fill in all required fields',
        });
      });
      expect(mockMutateAsync).not.toHaveBeenCalled();
    });

    it('should validate required fields before submission', async () => {
      renderComponent();

      const input = screen.getByLabelText(/Directory Path/);
      await userEvent.type(input, '/valid/path');

      const submitButton = screen.getByRole('button', { name: 'Add Source' });
      await userEvent.click(submitButton);

      // Should not show validation error
      expect(mockMutateAsync).toHaveBeenCalled();
    });
  });

  describe('Source Addition Flow', () => {
    it('should successfully add a directory source and navigate to collection detail', async () => {
      renderComponent();
      mockMutateAsync.mockResolvedValue({ data: { id: 'operation-id' } });

      const input = screen.getByLabelText(/Directory Path/);
      await userEvent.type(input, '/valid/path');

      const submitButton = screen.getByRole('button', { name: 'Add Source' });
      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalledWith({
          collectionId: 'test-collection-id',
          sourceType: 'directory',
          sourceConfig: { path: '/valid/path' },
          secrets: undefined,
          sourcePath: '/valid/path',
          config: {
            chunk_size: 1000,
            chunk_overlap: 200,
          },
        });
      });

      expect(mockNavigate).toHaveBeenCalledWith('/collections/test-collection-id');
      expect(mockOnSuccess).toHaveBeenCalled();
    });

    it('should handle API errors from mutation', async () => {
      renderComponent();
      const error = new Error('Failed to add source');
      mockMutateAsync.mockRejectedValue(error);

      const input = screen.getByLabelText(/Directory Path/);
      await userEvent.type(input, '/invalid/path');

      const submitButton = screen.getByRole('button', { name: 'Add Source' });
      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Failed to add source',
          type: 'error',
        });
      });

      expect(mockNavigate).not.toHaveBeenCalled();
      expect(mockOnSuccess).not.toHaveBeenCalled();
    });

    it('should handle non-Error exceptions', async () => {
      renderComponent();
      mockMutateAsync.mockRejectedValue('Unknown error');

      const input = screen.getByLabelText(/Directory Path/);
      await userEvent.type(input, '/path');

      const submitButton = screen.getByRole('button', { name: 'Add Source' });
      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Failed to add data source',
          type: 'error',
        });
      });
    });

    it('should not show duplicate error toasts when mutation handles error', async () => {
      // Mock mutation to indicate it already handled the error
      (useAddSource as Mock).mockReturnValue({
        mutateAsync: mockMutateAsync,
        isPending: false,
        isError: true, // Mutation already handled error
      });

      renderComponent();
      mockMutateAsync.mockRejectedValue(new Error('API Error'));

      const input = screen.getByLabelText(/Directory Path/);
      await userEvent.type(input, '/path');

      const submitButton = screen.getByRole('button', { name: 'Add Source' });
      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalled();
      });

      // Should not call addToast since mutation already handled the error
      expect(mockAddToast).not.toHaveBeenCalled();
    });
  });

  describe('Loading States', () => {
    it('should disable form elements while submitting', async () => {
      renderComponent();
      mockMutateAsync.mockImplementation(() => new Promise(() => {})); // Never resolves

      const input = screen.getByLabelText(/Directory Path/);
      await userEvent.type(input, '/path');

      const submitButton = screen.getByRole('button', { name: 'Add Source' });
      const cancelButton = screen.getByRole('button', { name: 'Cancel' });

      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(submitButton).toBeDisabled();
        expect(submitButton).toHaveTextContent('Adding Source...');
        expect(cancelButton).toBeDisabled();
      });
    });

    it('should show loading state when mutation is pending', () => {
      (useAddSource as Mock).mockReturnValue({
        mutateAsync: mockMutateAsync,
        isPending: true,
        isError: false,
      });

      renderComponent();

      const submitButton = screen.getByRole('button', { name: 'Adding Source...' });
      const cancelButton = screen.getByRole('button', { name: 'Cancel' });

      expect(submitButton).toBeDisabled();
      expect(cancelButton).toBeDisabled();
    });

    it('should re-enable form after successful submission', async () => {
      renderComponent();
      mockMutateAsync.mockResolvedValue({ data: { id: 'operation-id' } });

      const input = screen.getByLabelText(/Directory Path/);
      await userEvent.type(input, '/path');

      const submitButton = screen.getByRole('button', { name: 'Add Source' });
      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(mockOnSuccess).toHaveBeenCalled();
      });

      // Form should be re-enabled after success (though modal would typically close)
      expect(submitButton).not.toBeDisabled();
      expect(submitButton).toHaveTextContent('Add Source');
    });

    it('should re-enable form after error', async () => {
      renderComponent();
      mockMutateAsync.mockRejectedValue(new Error('Failed'));

      const input = screen.getByLabelText(/Directory Path/);
      await userEvent.type(input, '/path');

      const submitButton = screen.getByRole('button', { name: 'Add Source' });
      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalled();
      });

      expect(submitButton).not.toBeDisabled();
      expect(submitButton).toHaveTextContent('Add Source');
    });
  });

  describe('Modal Close Functionality', () => {
    it('should call onClose when clicking Cancel button', async () => {
      renderComponent();

      const cancelButton = screen.getByRole('button', { name: 'Cancel' });
      await userEvent.click(cancelButton);

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    it('should call onClose when clicking backdrop', async () => {
      renderComponent();

      // Find the backdrop - uses bg-black/50 and dark:bg-black/80
      const backdrop = document.querySelector('.fixed.inset-0.bg-black\\/50');
      expect(backdrop).toBeInTheDocument();

      fireEvent.click(backdrop!);

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    it('should not close modal when clicking inside modal content', async () => {
      renderComponent();

      const modalContent = screen.getByText('Add Data to Collection').closest('div');
      fireEvent.click(modalContent!);

      expect(mockOnClose).not.toHaveBeenCalled();
    });
  });

  describe('Form Submission', () => {
    it('should prevent default form submission', async () => {
      renderComponent();

      const form = screen.getByLabelText(/Directory Path/).closest('form');
      expect(form).toBeInTheDocument();

      const submitEvent = new Event('submit', { bubbles: true, cancelable: true });
      const preventDefaultSpy = vi.spyOn(submitEvent, 'preventDefault');

      fireEvent(form!, submitEvent);

      expect(preventDefaultSpy).toHaveBeenCalled();
    });

    it('should submit form on Enter key in input field', async () => {
      renderComponent();
      mockMutateAsync.mockResolvedValue({ data: { id: 'operation-id' } });

      const input = screen.getByLabelText(/Directory Path/);
      await userEvent.type(input, '/path/to/docs{Enter}');

      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalled();
      });
    });
  });

  describe('Collection Status Variations', () => {
    it('should handle collections with missing optional fields', () => {
      const minimalCollection = {
        ...mockCollection,
        description: undefined,
        status_message: undefined,
        metadata: undefined,
        total_size_bytes: undefined,
      };

      renderComponent({ collection: minimalCollection });

      // Should still render without errors
      expect(screen.getByText('Add Data to Collection')).toBeInTheDocument();
    });
  });

  describe('Configuration Display', () => {
    it('should display custom chunk configuration', () => {
      const customCollection = {
        ...mockCollection,
        chunk_size: 2000,
        chunk_overlap: 500,
      };

      renderComponent({ collection: customCollection });

      expect(screen.getByText('2000 characters')).toBeInTheDocument();
      expect(screen.getByText('500 characters')).toBeInTheDocument();
    });

    it('should display correct embedding model configuration', () => {
      const customCollection = {
        ...mockCollection,
        embedding_model: 'text-embedding-3-small',
      };

      renderComponent({ collection: customCollection });

      expect(screen.getByText('text-embedding-3-small')).toBeInTheDocument();
    });
  });

  describe('Error Recovery', () => {
    it('should allow retry after failed submission', async () => {
      renderComponent();

      // First attempt fails
      mockMutateAsync.mockRejectedValueOnce(new Error('Network error'));

      const input = screen.getByLabelText(/Directory Path/);
      await userEvent.type(input, '/path/to/docs');

      const submitButton = screen.getByRole('button', { name: 'Add Source' });
      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Network error',
          type: 'error',
        });
      });

      // Second attempt succeeds
      mockMutateAsync.mockResolvedValueOnce({ data: { id: 'operation-id' } });

      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(mockNavigate).toHaveBeenCalledWith('/collections/test-collection-id');
        expect(mockOnSuccess).toHaveBeenCalled();
      });
    });
  });

  describe('Edge Cases', () => {
    it('should handle very long collection names gracefully', () => {
      const longNameCollection = {
        ...mockCollection,
        name: 'A'.repeat(100),
      };

      renderComponent({ collection: longNameCollection });

      // Check that the long name is displayed (might be truncated with CSS)
      expect(screen.getByText(new RegExp(`Add new documents to "${longNameCollection.name}"`)))
        .toBeInTheDocument();
    });

    it('should handle special characters in paths', async () => {
      renderComponent();
      mockMutateAsync.mockResolvedValue({ data: { id: 'operation-id' } });

      const input = screen.getByLabelText(/Directory Path/);
      const specialPath = '/path/with spaces/and-special@chars#test';
      await userEvent.type(input, specialPath);

      const submitButton = screen.getByRole('button', { name: 'Add Source' });
      await userEvent.click(submitButton);

      expect(mockMutateAsync).toHaveBeenCalledWith({
        collectionId: 'test-collection-id',
        sourceType: 'directory',
        sourceConfig: { path: specialPath },
        secrets: undefined,
        sourcePath: specialPath,
        config: {
          chunk_size: 1000,
          chunk_overlap: 200,
        },
      });
    });

    it('should handle rapid form submissions', async () => {
      renderComponent();

      // Mock a slow mutation to test protection against multiple submissions
      let resolvePromise: () => void;
      const slowPromise = new Promise<{ data: { id: string } }>((resolve) => {
        resolvePromise = () => resolve({ data: { id: 'operation-id' } });
      });
      mockMutateAsync.mockReturnValue(slowPromise);

      const input = screen.getByLabelText(/Directory Path/);
      await userEvent.type(input, '/path');

      const submitButton = screen.getByRole('button', { name: 'Add Source' });

      // First click starts the submission
      await userEvent.click(submitButton);

      // Button should now be disabled and show loading state
      expect(submitButton).toBeDisabled();
      expect(submitButton).toHaveTextContent('Adding Source...');

      // Try clicking multiple times while it's disabled
      await userEvent.click(submitButton);
      await userEvent.click(submitButton);

      // Should only call mutate once due to isSubmitting state
      expect(mockMutateAsync).toHaveBeenCalledTimes(1);

      // Resolve the promise to complete the test
      resolvePromise!();

      await waitFor(() => {
        expect(mockOnSuccess).toHaveBeenCalled();
      });
    });
  });

  describe('Git Connector', () => {
    it('should submit Git source with correct parameters', async () => {
      renderComponent();
      mockMutateAsync.mockResolvedValue({ data: { id: 'operation-id' } });

      // Switch to Git connector
      const gitButton = screen.getByText('Git Repository').closest('button');
      await userEvent.click(gitButton!);

      // Fill in Git fields
      const repoUrlInput = screen.getByLabelText(/Repository URL/);
      await userEvent.type(repoUrlInput, 'https://github.com/test/repo.git');

      const submitButton = screen.getByRole('button', { name: 'Add Source' });
      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalledWith(
          expect.objectContaining({
            collectionId: 'test-collection-id',
            sourceType: 'git',
            sourceConfig: expect.objectContaining({
              repo_url: 'https://github.com/test/repo.git',
              ref: 'main',
              auth_method: 'none',
            }),
            sourcePath: 'https://github.com/test/repo.git',
          })
        );
      });
    });

    it('should call preview for Git connector', async () => {
      const mockGitPreviewMutate = vi.fn().mockResolvedValue({
        valid: true,
        refs_found: ['main', 'develop'],
      });

      (useGitPreview as Mock).mockReturnValue({
        mutateAsync: mockGitPreviewMutate,
        isPending: false,
      });

      renderComponent();

      // Switch to Git connector
      const gitButton = screen.getByText('Git Repository').closest('button');
      await userEvent.click(gitButton!);

      // Fill in required field
      const repoUrlInput = screen.getByLabelText(/Repository URL/);
      await userEvent.type(repoUrlInput, 'https://github.com/test/repo.git');

      // Click Test Connection
      const testButton = screen.getByRole('button', { name: 'Test Connection' });
      await userEvent.click(testButton);

      await waitFor(() => {
        expect(mockGitPreviewMutate).toHaveBeenCalled();
      });
    });
  });

  describe('IMAP Connector', () => {
    it('should submit IMAP source with correct parameters including secrets', async () => {
      renderComponent();
      mockMutateAsync.mockResolvedValue({ data: { id: 'operation-id' } });

      // Switch to IMAP connector
      const imapButton = screen.getByText('Email (IMAP)').closest('button');
      await userEvent.click(imapButton!);

      // Fill in IMAP fields
      await userEvent.type(screen.getByLabelText(/IMAP Server/), 'imap.example.com');
      await userEvent.type(screen.getByLabelText(/Username/), 'user@example.com');
      await userEvent.type(screen.getByLabelText(/Password/), 'secret123');

      const submitButton = screen.getByRole('button', { name: 'Add Source' });
      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalledWith(
          expect.objectContaining({
            collectionId: 'test-collection-id',
            sourceType: 'imap',
            sourceConfig: expect.objectContaining({
              host: 'imap.example.com',
              port: 993,
              use_ssl: true,
              username: 'user@example.com',
            }),
            secrets: { password: 'secret123' },
            sourcePath: 'user@example.com@imap.example.com',
          })
        );
      });
    });
  });
});
