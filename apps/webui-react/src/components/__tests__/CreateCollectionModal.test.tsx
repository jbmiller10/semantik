import React from 'react';
import { render, screen, waitFor, within, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';
import { CreateCollectionModal } from '../CreateCollectionModal';
import { TestWrapper } from '../../tests/utils/TestWrapper';

// Mock react-router-dom
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

// Mock hooks and stores
const mockCreateCollectionMutation = {
  mutateAsync: vi.fn(),
  isError: false,
  isPending: false,
};

const mockAddSourceMutation = {
  mutateAsync: vi.fn(),
  isError: false,
  isPending: false,
};

const mockAddToast = vi.fn();

vi.mock('../../hooks/useCollections', () => ({
  useCreateCollection: () => mockCreateCollectionMutation,
}));

vi.mock('../../hooks/useCollectionOperations', () => ({
  useAddSource: () => mockAddSourceMutation,
}));

vi.mock('../../stores/uiStore', () => ({
  useUIStore: () => ({
    addToast: mockAddToast,
  }),
}));

const mockStartScan = vi.fn();
const mockResetScan = vi.fn();
const mockDirectoryScanState = {
  scanning: false,
  scanResult: null,
  error: null,
  startScan: mockStartScan,
  reset: mockResetScan,
};

vi.mock('../../hooks/useDirectoryScan', () => ({
  useDirectoryScan: () => mockDirectoryScanState,
}));

const mockOperationProgressState = {
  sendMessage: vi.fn(),
  readyState: WebSocket.CLOSED,
  isConnected: false,
  onComplete: undefined as (() => void) | undefined,
  onError: undefined as ((error: Error) => void) | undefined,
};

vi.mock('../../hooks/useOperationProgress', () => ({
  useOperationProgress: vi.fn((operationId, options) => {
    if (operationId && options?.onComplete) {
      mockOperationProgressState.onComplete = options.onComplete;
    }
    if (operationId && options?.onError) {
      mockOperationProgressState.onError = options.onError;
    }
    return mockOperationProgressState;
  }),
}));

// Helper function to render with wrapper
const renderCreateCollectionModal = (props = {}) => {
  return render(
    <CreateCollectionModal {...props} />,
    { wrapper: TestWrapper }
  );
};

describe('CreateCollectionModal', () => {
  const mockOnClose = vi.fn();
  const mockOnSuccess = vi.fn();

  const defaultProps = {
    onClose: mockOnClose,
    onSuccess: mockOnSuccess,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    
    // Reset mutation mocks
    mockCreateCollectionMutation.mutateAsync.mockReset();
    mockAddSourceMutation.mutateAsync.mockReset();
    mockCreateCollectionMutation.isError = false;
    mockCreateCollectionMutation.isPending = false;
    mockAddSourceMutation.isError = false;
    mockAddSourceMutation.isPending = false;
    
    // Reset other mocks
    mockAddToast.mockReset();
    mockNavigate.mockReset();
    mockStartScan.mockReset();
    mockResetScan.mockReset();
    
    // Reset directory scan state
    mockDirectoryScanState.scanning = false;
    mockDirectoryScanState.scanResult = null;
    mockDirectoryScanState.error = null;
    
    // Reset operation progress state
    mockOperationProgressState.onComplete = undefined;
    mockOperationProgressState.onError = undefined;
  });

  describe('Initial Render', () => {
    it('should render with default values', () => {
      renderCreateCollectionModal(defaultProps);

      // Check modal is visible
      expect(screen.getByRole('heading', { name: /create new collection/i })).toBeInTheDocument();

      // Check required fields
      expect(screen.getByLabelText(/collection name/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/collection name/i)).toHaveValue('');
      expect(screen.getByText('*')).toBeInTheDocument(); // Required indicator

      // Check optional fields
      expect(screen.getByLabelText(/description/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/initial source directory/i)).toBeInTheDocument();

      // Check default embedding model
      expect(screen.getByLabelText(/embedding model/i)).toHaveValue('Qwen/Qwen3-Embedding-0.6B');

      // Check default quantization
      expect(screen.getByLabelText(/model quantization/i)).toHaveValue('float16');

      // Check advanced settings is collapsed
      expect(screen.getByText(/advanced settings/i)).toBeInTheDocument();
      expect(screen.queryByLabelText(/chunk size/i)).not.toBeInTheDocument();

      // Check buttons
      expect(screen.getByRole('button', { name: /cancel/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /create collection/i })).toBeInTheDocument();
    });

    it('should have proper ARIA labels for accessibility', () => {
      renderCreateCollectionModal(defaultProps);

      // Check all inputs have labels
      expect(screen.getByLabelText(/collection name/i)).toHaveAttribute('id', 'name');
      expect(screen.getByLabelText(/description/i)).toHaveAttribute('id', 'description');
      expect(screen.getByLabelText(/initial source directory/i)).toHaveAttribute('id', 'sourcePath');
      expect(screen.getByLabelText(/embedding model/i)).toHaveAttribute('id', 'embedding_model');
      expect(screen.getByLabelText(/model quantization/i)).toHaveAttribute('id', 'quantization');
    });
  });

  describe('Form Validation', () => {
    it('should show validation error for empty collection name', async () => {
      const user = userEvent.setup();
      renderCreateCollectionModal(defaultProps);

      // Try to submit without filling required field
      await user.click(screen.getByRole('button', { name: /create collection/i }));

      await waitFor(() => {
        // Check for the error in the input field's error message
        const nameInput = screen.getByLabelText(/collection name/i);
        const errorMessage = nameInput.parentElement?.querySelector('.text-red-600');
        expect(errorMessage).toHaveTextContent(/collection name is required/i);
      });

      // Form should not be submitted
      expect(mockCreateCollectionMutation.mutateAsync).not.toHaveBeenCalled();
    });

    it('should show validation error for collection name exceeding 100 characters', async () => {
      const user = userEvent.setup();
      renderCreateCollectionModal(defaultProps);

      const longName = 'a'.repeat(101);
      await user.type(screen.getByLabelText(/collection name/i), longName);
      await user.click(screen.getByRole('button', { name: /create collection/i }));

      await waitFor(() => {
        const nameInput = screen.getByLabelText(/collection name/i);
        const errorMessage = nameInput.parentElement?.querySelector('.text-red-600');
        expect(errorMessage).toHaveTextContent(/collection name must be 100 characters or less/i);
      });

      expect(mockCreateCollectionMutation.mutateAsync).not.toHaveBeenCalled();
    });

    it('should show validation error for description exceeding 500 characters', async () => {
      const user = userEvent.setup();
      renderCreateCollectionModal(defaultProps);

      await user.type(screen.getByLabelText(/collection name/i), 'Test Collection');
      const longDescription = 'a'.repeat(501);
      await user.type(screen.getByLabelText(/description/i), longDescription);
      await user.click(screen.getByRole('button', { name: /create collection/i }));

      await waitFor(() => {
        const descInput = screen.getByLabelText(/description/i);
        const errorMessage = descInput.parentElement?.querySelector('.text-red-600');
        expect(errorMessage).toHaveTextContent(/description must be 500 characters or less/i);
      });

      expect(mockCreateCollectionMutation.mutateAsync).not.toHaveBeenCalled();
    });

    it('should clear validation errors when fields are corrected', async () => {
      const user = userEvent.setup();
      renderCreateCollectionModal(defaultProps);

      // Submit without required field
      await user.click(screen.getByRole('button', { name: /create collection/i }));

      await waitFor(() => {
        const nameInput = screen.getByLabelText(/collection name/i);
        const errorMessage = nameInput.parentElement?.querySelector('.text-red-600');
        expect(errorMessage).toHaveTextContent(/collection name is required/i);
      });

      // Type in the field
      await user.type(screen.getByLabelText(/collection name/i), 'Test');

      // Error should disappear
      await waitFor(() => {
        const nameInput = screen.getByLabelText(/collection name/i);
        const errorMessage = nameInput.parentElement?.querySelector('.text-red-600');
        expect(errorMessage).not.toBeInTheDocument();
      });
    });
  });

  describe('Form Submission', () => {
    it('should successfully create collection without source', async () => {
      const user = userEvent.setup();
      mockCreateCollectionMutation.mutateAsync.mockResolvedValue({
        id: 'test-id',
        name: 'Test Collection',
        initial_operation_id: null,
      });

      renderCreateCollectionModal(defaultProps);

      // Fill form
      await user.type(screen.getByLabelText(/collection name/i), 'Test Collection');
      await user.type(screen.getByLabelText(/description/i), 'Test description');

      // Submit
      await user.click(screen.getByRole('button', { name: /create collection/i }));

      await waitFor(() => {
        expect(mockCreateCollectionMutation.mutateAsync).toHaveBeenCalledWith({
          name: 'Test Collection',
          description: 'Test description',
          embedding_model: 'Qwen/Qwen3-Embedding-0.6B',
          quantization: 'float16',
          chunking_strategy: 'recursive',
          chunking_config: {
            chunk_size: 600,
            chunk_overlap: 100,
            preserve_sentences: true,
          },
          is_public: false,
        });
      });

      expect(mockAddToast).toHaveBeenCalledWith({
        message: 'Collection created successfully!',
        type: 'success',
      });

      expect(mockOnSuccess).toHaveBeenCalled();
    });

    it('should show loading state during submission', async () => {
      const user = userEvent.setup();
      mockCreateCollectionMutation.mutateAsync.mockImplementation(() => 
        new Promise(resolve => setTimeout(() => resolve({ id: 'test-id' }), 100))
      );

      renderCreateCollectionModal(defaultProps);

      await user.type(screen.getByLabelText(/collection name/i), 'Test Collection');
      
      const submitButton = screen.getByRole('button', { name: /create collection/i });
      await user.click(submitButton);

      // Check loading state
      expect(submitButton).toBeDisabled();
      expect(submitButton).toHaveTextContent(/creating/i);
      expect(screen.getByText(/creating collection/i)).toBeInTheDocument();

      // Check loading overlay exists
      const loadingText = screen.getByText(/creating collection/i);
      expect(loadingText).toBeInTheDocument();
      expect(loadingText.parentElement).toHaveClass('text-center');
    });

    it('should disable form during submission', async () => {
      const user = userEvent.setup();
      mockCreateCollectionMutation.mutateAsync.mockImplementation(() => 
        new Promise(resolve => setTimeout(() => resolve({ id: 'test-id' }), 100))
      );

      renderCreateCollectionModal(defaultProps);

      await user.type(screen.getByLabelText(/collection name/i), 'Test Collection');
      await user.click(screen.getByRole('button', { name: /create collection/i }));

      // Check all inputs are disabled
      expect(screen.getByLabelText(/collection name/i)).toBeDisabled();
      expect(screen.getByLabelText(/description/i)).toBeDisabled();
      expect(screen.getByLabelText(/initial source directory/i)).toBeDisabled();
      expect(screen.getByRole('button', { name: /cancel/i })).toBeDisabled();
    });
  });

  describe('Two-Step Creation Process', () => {
    it('should handle collection creation with source path', async () => {
      const user = userEvent.setup();
      
      // Mock successful collection creation with INDEX operation
      mockCreateCollectionMutation.mutateAsync.mockResolvedValue({
        id: 'test-collection-id',
        name: 'Test Collection',
        initial_operation_id: 'index-op-123',
      });

      // Mock successful source addition
      mockAddSourceMutation.mutateAsync.mockResolvedValue({});

      renderCreateCollectionModal(defaultProps);

      // Fill form with source
      await user.type(screen.getByLabelText(/collection name/i), 'Test Collection');
      await user.type(screen.getByLabelText(/initial source directory/i), '/data/documents');

      // Submit
      await user.click(screen.getByRole('button', { name: /create collection/i }));

      // Should show waiting message
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Collection created! Waiting for initialization before adding source...',
          type: 'info',
        });
      });

      // Simulate INDEX operation completion
      expect(mockOperationProgressState.onComplete).toBeDefined();
      await act(async () => {
        await mockOperationProgressState.onComplete!();
      });

      // Should add source
      await waitFor(() => {
        expect(mockAddSourceMutation.mutateAsync).toHaveBeenCalledWith({
          collectionId: 'test-collection-id',
          sourcePath: '/data/documents',
          config: {
            chunking_strategy: 'recursive',
            chunking_config: {
              chunk_size: 600,
              chunk_overlap: 100,
              preserve_sentences: true,
            },
          },
        });
      });

      // Should show success and navigate
      expect(mockAddToast).toHaveBeenCalledWith({
        message: 'Collection created and source added successfully! Navigating to collection...',
        type: 'success',
      });

      expect(mockOnSuccess).toHaveBeenCalled();

      // Should navigate after delay
      await act(async () => {
        // Wait for the setTimeout in the component
        await new Promise(resolve => setTimeout(resolve, 1100));
      });
      
      expect(mockNavigate).toHaveBeenCalledWith('/collections/test-collection-id');
    });

    it('should handle source addition failure after collection creation', async () => {
      const user = userEvent.setup();
      
      mockCreateCollectionMutation.mutateAsync.mockResolvedValue({
        id: 'test-collection-id',
        name: 'Test Collection',
        initial_operation_id: 'index-op-123',
      });

      // Mock source addition failure
      mockAddSourceMutation.mutateAsync.mockRejectedValue(new Error('Failed to add source'));

      renderCreateCollectionModal(defaultProps);

      await user.type(screen.getByLabelText(/collection name/i), 'Test Collection');
      await user.type(screen.getByLabelText(/initial source directory/i), '/data/documents');
      await user.click(screen.getByRole('button', { name: /create collection/i }));

      // Wait for initial toast
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Collection created! Waiting for initialization before adding source...',
          type: 'info',
        });
      });

      // Simulate INDEX completion
      await act(async () => {
        await mockOperationProgressState.onComplete!();
      });

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Collection created but failed to add source: Failed to add source',
          type: 'warning',
        });
      });

      // Should still call onSuccess since collection was created
      expect(mockOnSuccess).toHaveBeenCalled();
    });
  });

  describe('Advanced Settings', () => {
    it('should toggle advanced settings visibility', async () => {
      const user = userEvent.setup();
      renderCreateCollectionModal(defaultProps);

      // Chunking strategy is always visible (not in advanced settings anymore)
      expect(screen.getByText(/chunking strategy/i)).toBeInTheDocument();
      
      // Initially hidden - only public checkbox is in advanced settings
      expect(screen.queryByLabelText(/make this collection public/i)).not.toBeInTheDocument();

      // Click to expand
      const advancedButton = screen.getByText(/advanced settings/i);
      await user.click(advancedButton);

      // Should show advanced fields (public checkbox)
      expect(screen.getByLabelText(/make this collection public/i)).toBeInTheDocument();
      
      // Chunking strategy should still be visible (it's always visible)
      expect(screen.getByText(/chunking strategy/i)).toBeInTheDocument();
      
      // Verify public toggle default value
      expect(screen.getByLabelText(/make this collection public/i)).not.toBeChecked();

      // Click to collapse
      await user.click(advancedButton);

      // Should hide advanced fields
      await waitFor(() => {
        expect(screen.queryByLabelText(/chunk size/i)).not.toBeInTheDocument();
      });
    });
  });

  describe('Directory Scanning', () => {
    it('should scan directory when clicking scan button', async () => {
      const user = userEvent.setup();
      renderCreateCollectionModal(defaultProps);

      // Scan button should be disabled without path
      const scanButton = screen.getByRole('button', { name: /scan/i });
      expect(scanButton).toBeDisabled();

      // Enter path
      await user.type(screen.getByLabelText(/initial source directory/i), '/data/documents');

      // Scan button should be enabled
      expect(scanButton).not.toBeDisabled();

      // Click scan
      await user.click(scanButton);

      expect(mockStartScan).toHaveBeenCalledWith('/data/documents');
    });

    it('should display scan results', () => {
      // Mock scan result
      mockDirectoryScanState.scanResult = {
        total_files: 150,
        total_size: 1024 * 1024 * 50, // 50 MB
      };

      renderCreateCollectionModal(defaultProps);

      // Should show scan results
      expect(screen.getByText(/found 150 files/i)).toBeInTheDocument();
      expect(screen.getByText(/50\.0 MB/i)).toBeInTheDocument();
    });

    it('should show warning for large directories', () => {
      // Mock large scan result
      mockDirectoryScanState.scanResult = {
        total_files: 15000,
        total_size: 1024 * 1024 * 1024 * 10, // 10 GB
      };

      renderCreateCollectionModal(defaultProps);

      // Should show warning
      expect(screen.getByText(/found 15,000 files/i)).toBeInTheDocument();
      expect(screen.getByText(/warning: large directory detected/i)).toBeInTheDocument();
      expect(screen.getByText(/indexing may take a significant amount of time/i)).toBeInTheDocument();
    });

    it('should show scanning state', async () => {
      const user = userEvent.setup();
      
      // Mock scanning state
      mockDirectoryScanState.scanning = true;

      renderCreateCollectionModal(defaultProps);

      await user.type(screen.getByLabelText(/initial source directory/i), '/data/documents');

      const scanButton = screen.getByRole('button', { name: /scanning/i });
      expect(scanButton).toBeDisabled();
      expect(scanButton).toHaveTextContent(/scanning/i);
    });

    it('should display scan errors', () => {
      // Mock scan error
      mockDirectoryScanState.error = 'Directory not found';

      renderCreateCollectionModal(defaultProps);

      expect(screen.getByText(/directory not found/i)).toBeInTheDocument();
    });

    it('should reset scan results when path changes', async () => {
      const user = userEvent.setup();
      
      // Start with scan result
      mockDirectoryScanState.scanResult = {
        total_files: 100,
        total_size: 1024 * 1024,
      };

      renderCreateCollectionModal(defaultProps);

      // Should show results
      expect(screen.getByText(/found 100 files/i)).toBeInTheDocument();

      // Change path
      await user.type(screen.getByLabelText(/initial source directory/i), '/new/path');

      // Should reset scan
      expect(mockResetScan).toHaveBeenCalled();
    });
  });

  describe('Modal Lifecycle', () => {
    it('should handle escape key to close modal', async () => {
      const user = userEvent.setup();
      renderCreateCollectionModal(defaultProps);

      // Press escape
      await user.keyboard('{Escape}');

      expect(mockOnClose).toHaveBeenCalled();
    });

    it('should not close on escape during submission', async () => {
      const user = userEvent.setup();
      
      mockCreateCollectionMutation.mutateAsync.mockImplementation(() => 
        new Promise(resolve => setTimeout(() => resolve({ id: 'test-id' }), 100))
      );

      renderCreateCollectionModal(defaultProps);

      await user.type(screen.getByLabelText(/collection name/i), 'Test');
      await user.click(screen.getByRole('button', { name: /create collection/i }));

      // Press escape during submission
      await user.keyboard('{Escape}');

      // Should not close
      expect(mockOnClose).not.toHaveBeenCalled();
    });

    it('should handle cancel button click', async () => {
      const user = userEvent.setup();
      renderCreateCollectionModal(defaultProps);

      await user.click(screen.getByRole('button', { name: /cancel/i }));

      expect(mockOnClose).toHaveBeenCalled();
    });

    it('should prevent form submission on enter key in input fields', async () => {
      const user = userEvent.setup();
      renderCreateCollectionModal(defaultProps);

      // Type and press enter in name field
      const nameInput = screen.getByLabelText(/collection name/i);
      await user.type(nameInput, 'Test Collection');
      await user.keyboard('{Enter}');

      // Should not submit
      expect(mockCreateCollectionMutation.mutateAsync).not.toHaveBeenCalled();

      // Form should still be open
      expect(screen.getByRole('heading', { name: /create new collection/i })).toBeInTheDocument();
    });
  });

  describe('Form State Preservation', () => {
    it('should preserve form data on submission errors', async () => {
      const user = userEvent.setup();
      
      // Mock submission failure
      mockCreateCollectionMutation.mutateAsync.mockRejectedValue(new Error('Network error'));

      renderCreateCollectionModal(defaultProps);

      // Fill form
      const formData = {
        name: 'Test Collection',
        description: 'Test description',
        sourcePath: '/data/documents',
      };

      await user.type(screen.getByLabelText(/collection name/i), formData.name);
      await user.type(screen.getByLabelText(/description/i), formData.description);
      await user.type(screen.getByLabelText(/initial source directory/i), formData.sourcePath);

      // Submit
      await user.click(screen.getByRole('button', { name: /create collection/i }));

      // Wait for error
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Network error',
          type: 'error',
        });
      });

      // Form data should be preserved
      expect(screen.getByLabelText(/collection name/i)).toHaveValue(formData.name);
      expect(screen.getByLabelText(/description/i)).toHaveValue(formData.description);
      expect(screen.getByLabelText(/initial source directory/i)).toHaveValue(formData.sourcePath);

      // Form should remain open
      expect(screen.getByRole('heading', { name: /create new collection/i })).toBeInTheDocument();
    });
  });

  describe('Embedding Model Selection', () => {
    it('should allow selecting different embedding models', async () => {
      const user = userEvent.setup();
      renderCreateCollectionModal(defaultProps);

      const modelSelect = screen.getByLabelText(/embedding model/i);
      
      // Check available options
      const options = within(modelSelect).getAllByRole('option');
      expect(options).toHaveLength(3);
      expect(options[0]).toHaveTextContent(/qwen3-embedding/i);
      expect(options[1]).toHaveTextContent(/e5-base-v2/i);
      expect(options[2]).toHaveTextContent(/all-minilm/i);

      // Select different model
      await user.selectOptions(modelSelect, 'intfloat/e5-base-v2');
      expect(modelSelect).toHaveValue('intfloat/e5-base-v2');

      // Submit with different model
      await user.type(screen.getByLabelText(/collection name/i), 'Test');
      await user.click(screen.getByRole('button', { name: /create collection/i }));

      mockCreateCollectionMutation.mutateAsync.mockResolvedValue({ id: 'test-id' });

      await waitFor(() => {
        expect(mockCreateCollectionMutation.mutateAsync).toHaveBeenCalledWith(
          expect.objectContaining({
            embedding_model: 'intfloat/e5-base-v2',
          })
        );
      });
    });
  });

  describe('Quantization Selection', () => {
    it('should allow selecting different quantization levels', async () => {
      const user = userEvent.setup();
      renderCreateCollectionModal(defaultProps);

      const quantSelect = screen.getByLabelText(/model quantization/i);
      
      // Check available options
      const options = within(quantSelect).getAllByRole('option');
      expect(options).toHaveLength(3);
      expect(options[0]).toHaveTextContent(/float32.*highest precision/i);
      expect(options[1]).toHaveTextContent(/float16.*balanced/i);
      expect(options[2]).toHaveTextContent(/int8.*lowest memory/i);

      // Select different quantization
      await user.selectOptions(quantSelect, 'int8');
      expect(quantSelect).toHaveValue('int8');

      // Submit with different quantization
      await user.type(screen.getByLabelText(/collection name/i), 'Test');
      await user.click(screen.getByRole('button', { name: /create collection/i }));

      mockCreateCollectionMutation.mutateAsync.mockResolvedValue({ id: 'test-id' });

      await waitFor(() => {
        expect(mockCreateCollectionMutation.mutateAsync).toHaveBeenCalledWith(
          expect.objectContaining({
            quantization: 'int8',
          })
        );
      });
    });
  });

  describe('Edge Cases', () => {
    it('should handle missing initial_operation_id when source is provided', async () => {
      const user = userEvent.setup();
      
      // Mock collection creation without operation ID
      mockCreateCollectionMutation.mutateAsync.mockResolvedValue({
        id: 'test-id',
        name: 'Test Collection',
        initial_operation_id: null,
      });

      renderCreateCollectionModal(defaultProps);

      await user.type(screen.getByLabelText(/collection name/i), 'Test Collection');
      await user.type(screen.getByLabelText(/initial source directory/i), '/data/documents');

      await user.click(screen.getByRole('button', { name: /create collection/i }));

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Collection created! Please add the source manually.',
          type: 'warning',
        });
      });

      expect(mockOnSuccess).toHaveBeenCalled();
      expect(mockAddSourceMutation.mutateAsync).not.toHaveBeenCalled();
    });

    it('should handle rapid submit clicks', async () => {
      const user = userEvent.setup();
      mockCreateCollectionMutation.mutateAsync.mockResolvedValue({ id: 'test-id' });

      renderCreateCollectionModal(defaultProps);

      await user.type(screen.getByLabelText(/collection name/i), 'Test');

      const submitButton = screen.getByRole('button', { name: /create collection/i });
      
      // Rapid clicks
      await user.click(submitButton);
      await user.click(submitButton);
      await user.click(submitButton);

      // Should only submit once
      await waitFor(() => {
        expect(mockCreateCollectionMutation.mutateAsync).toHaveBeenCalledTimes(1);
      });
    });

    it('should handle whitespace-only collection name', async () => {
      const user = userEvent.setup();
      renderCreateCollectionModal(defaultProps);

      await user.type(screen.getByLabelText(/collection name/i), '   ');
      await user.click(screen.getByRole('button', { name: /create collection/i }));

      await waitFor(() => {
        const nameInput = screen.getByLabelText(/collection name/i);
        const errorMessage = nameInput.parentElement?.querySelector('.text-red-600');
        expect(errorMessage).toHaveTextContent(/collection name is required/i);
      });

      expect(mockCreateCollectionMutation.mutateAsync).not.toHaveBeenCalled();
    });
  });
});