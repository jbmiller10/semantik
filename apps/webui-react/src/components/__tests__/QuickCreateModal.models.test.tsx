import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';
import { QuickCreateModal } from '../QuickCreateModal';
import { TestWrapper } from '../../tests/utils/TestWrapper';

// Mock hooks
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

vi.mock('../../hooks/useDirectoryScan', () => ({
  useDirectoryScan: () => ({
    scanning: false,
    scanResult: null,
    error: null,
    startScan: vi.fn(),
    reset: vi.fn(),
  }),
}));

vi.mock('../../hooks/useOperationProgress', () => ({
  useOperationProgress: vi.fn(() => ({
    sendMessage: vi.fn(),
    readyState: WebSocket.CLOSED,
    isConnected: false,
  })),
}));

const renderQuickCreateModal = (props = {}) => {
  return render(
    <QuickCreateModal onClose={vi.fn()} onSuccess={vi.fn()} {...props} />,
    { wrapper: TestWrapper }
  );
};

describe('QuickCreateModal - Dynamic Model Loading', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockCreateCollectionMutation.mutateAsync.mockReset();
    mockAddSourceMutation.mutateAsync.mockReset();
  });

  it('should render models from API including plugin models', async () => {
    renderQuickCreateModal();

    // Wait for models to load from MSW mock
    await waitFor(() => {
      const modelSelect = screen.getByLabelText(/embedding model/i);
      expect(modelSelect).not.toHaveTextContent('Loading');
    });

    const modelSelect = screen.getByLabelText(/embedding model/i);
    const options = within(modelSelect).getAllByRole('option');

    // MSW mock returns 4 models including test-plugin/model-v1
    expect(options.length).toBeGreaterThanOrEqual(4);
  });

  it('should display plugin model with provider indicator', async () => {
    renderQuickCreateModal();

    await waitFor(() => {
      const modelSelect = screen.getByLabelText(/embedding model/i);
      expect(modelSelect).not.toHaveTextContent('Loading');
    });

    // Plugin model should show provider name
    const modelSelect = screen.getByLabelText(/embedding model/i);
    expect(modelSelect).toHaveTextContent('test_plugin');
  });

  it('should sort models alphabetically', async () => {
    renderQuickCreateModal();

    await waitFor(() => {
      const modelSelect = screen.getByLabelText(/embedding model/i);
      expect(modelSelect).not.toHaveTextContent('Loading');
    });

    const modelSelect = screen.getByLabelText(/embedding model/i);
    const options = within(modelSelect).getAllByRole('option');

    // Models should be in alphabetical order
    // BAAI/bge-large-en-v1.5 should come before Qwen/Qwen3-Embedding-0.6B
    const modelNames = options.map((opt) => opt.getAttribute('value'));
    const sortedNames = [...modelNames].sort();
    expect(modelNames).toEqual(sortedNames);
  });

  it('should display current device information', async () => {
    renderQuickCreateModal();

    await waitFor(() => {
      // MSW mock returns current_device: 'cuda:0'
      expect(screen.getByText(/cuda:0/i)).toBeInTheDocument();
    });
  });

  it('should use default model when models are loading', () => {
    // This test checks the initial loading state
    renderQuickCreateModal();

    const modelSelect = screen.getByLabelText(/embedding model/i);
    // During loading, the select shows "Loading models..."
    expect(modelSelect).toHaveTextContent('Loading');
  });

  it('should allow selecting a plugin model', async () => {
    const user = userEvent.setup();
    const mockOnSuccess = vi.fn();
    mockCreateCollectionMutation.mutateAsync.mockResolvedValue({ id: 'test-id' });

    renderQuickCreateModal({ onSuccess: mockOnSuccess });

    await waitFor(() => {
      const modelSelect = screen.getByLabelText(/embedding model/i);
      expect(modelSelect).not.toHaveTextContent('Loading');
    });

    // Select the plugin model
    const modelSelect = screen.getByLabelText(/embedding model/i);
    await user.selectOptions(modelSelect, 'test-plugin/model-v1');
    expect(modelSelect).toHaveValue('test-plugin/model-v1');

    // Fill required fields and submit
    await user.type(screen.getByLabelText(/collection name/i), 'Test Collection');
    await user.click(screen.getByRole('button', { name: /create collection/i }));

    await waitFor(() => {
      expect(mockCreateCollectionMutation.mutateAsync).toHaveBeenCalledWith(
        expect.objectContaining({
          embedding_model: 'test-plugin/model-v1',
        })
      );
    });
  });

  it('should not show provider indicator for built-in dense_local models', async () => {
    renderQuickCreateModal();

    await waitFor(() => {
      const modelSelect = screen.getByLabelText(/embedding model/i);
      expect(modelSelect).not.toHaveTextContent('Loading');
    });

    const modelSelect = screen.getByLabelText(/embedding model/i);
    const options = within(modelSelect).getAllByRole('option');

    // Find the Qwen model option (which is dense_local)
    const qwenOption = options.find(
      (opt) => opt.getAttribute('value') === 'Qwen/Qwen3-Embedding-0.6B'
    );
    expect(qwenOption).toBeDefined();
    // Should NOT contain (dense_local) since that's the default provider
    expect(qwenOption?.textContent).not.toContain('dense_local');
  });
});
