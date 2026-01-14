import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import CollectionDefaultsSettings from '../CollectionDefaultsSettings';
import * as usePreferencesModule from '@/hooks/usePreferences';
import * as useModelsModule from '@/hooks/useModels';
import type { UserPreferencesResponse } from '@/types/preferences';

// Mock the hooks
vi.mock('@/hooks/usePreferences', () => ({
  usePreferences: vi.fn(),
  useUpdatePreferences: vi.fn(),
  useResetCollectionDefaults: vi.fn(),
}));

vi.mock('@/hooks/useModels', () => ({
  useEmbeddingModels: vi.fn(),
}));

// Mock data
const mockPreferences: UserPreferencesResponse = {
  collection_defaults: {
    embedding_model: 'Qwen/Qwen3-Embedding-0.6B',
    quantization: 'float16',
    chunking_strategy: 'recursive',
    chunk_size: 1024,
    chunk_overlap: 200,
    enable_sparse: false,
    sparse_type: 'bm25',
    enable_hybrid: false,
  },
  search: {
    top_k: 10,
    mode: 'dense',
    use_reranker: false,
    rrf_k: 60,
    similarity_threshold: null,
  },
  interface: {
    data_refresh_interval_ms: 30000,
    visualization_sample_limit: 200000,
    animation_enabled: true,
  },
  created_at: '2025-01-01T00:00:00Z',
  updated_at: '2025-01-01T00:00:00Z',
};

const mockModels = {
  models: {
    'Qwen/Qwen3-Embedding-0.6B': {
      model_name: 'Qwen/Qwen3-Embedding-0.6B',
      dimension: 1024,
      description: 'Default model',
      provider: 'dense_local',
      supports_quantization: true,
      recommended_quantization: 'float16',
      is_asymmetric: true,
    },
    'BAAI/bge-large-en-v1.5': {
      model_name: 'BAAI/bge-large-en-v1.5',
      dimension: 1024,
      description: 'High quality',
      provider: 'dense_local',
      supports_quantization: true,
      recommended_quantization: 'float16',
      is_asymmetric: true,
    },
  },
  current_device: 'cuda:0',
  using_real_embeddings: true,
};

describe('CollectionDefaultsSettings', () => {
  const mockUpdateMutateAsync = vi.fn();
  const mockResetMutateAsync = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();

    // Default mock implementations
    vi.mocked(usePreferencesModule.usePreferences).mockReturnValue({
      data: mockPreferences,
      isLoading: false,
      error: null,
    } as unknown as ReturnType<typeof usePreferencesModule.usePreferences>);

    vi.mocked(usePreferencesModule.useUpdatePreferences).mockReturnValue({
      mutateAsync: mockUpdateMutateAsync,
      isPending: false,
    } as unknown as ReturnType<typeof usePreferencesModule.useUpdatePreferences>);

    vi.mocked(usePreferencesModule.useResetCollectionDefaults).mockReturnValue({
      mutateAsync: mockResetMutateAsync,
      isPending: false,
    } as unknown as ReturnType<typeof usePreferencesModule.useResetCollectionDefaults>);

    vi.mocked(useModelsModule.useEmbeddingModels).mockReturnValue({
      data: mockModels,
      isLoading: false,
      error: null,
    } as unknown as ReturnType<typeof useModelsModule.useEmbeddingModels>);
  });

  describe('loading state', () => {
    it('shows loading spinner when preferences are loading', () => {
      vi.mocked(usePreferencesModule.usePreferences).mockReturnValue({
        data: undefined,
        isLoading: true,
        error: null,
      } as unknown as ReturnType<typeof usePreferencesModule.usePreferences>);

      render(<CollectionDefaultsSettings />);

      expect(screen.getByText('Loading collection defaults...')).toBeInTheDocument();
    });

    it('shows loading spinner when models are loading', () => {
      vi.mocked(useModelsModule.useEmbeddingModels).mockReturnValue({
        data: undefined,
        isLoading: true,
        error: null,
      } as unknown as ReturnType<typeof useModelsModule.useEmbeddingModels>);

      render(<CollectionDefaultsSettings />);

      expect(screen.getByText('Loading collection defaults...')).toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('shows error message when loading fails', () => {
      vi.mocked(usePreferencesModule.usePreferences).mockReturnValue({
        data: undefined,
        isLoading: false,
        error: new Error('Failed to load preferences'),
      } as unknown as ReturnType<typeof usePreferencesModule.usePreferences>);

      render(<CollectionDefaultsSettings />);

      expect(screen.getByText('Error loading preferences')).toBeInTheDocument();
      expect(screen.getByText('Failed to load preferences')).toBeInTheDocument();
    });
  });

  describe('form rendering', () => {
    it('renders the header and description', () => {
      render(<CollectionDefaultsSettings />);

      expect(screen.getByText('Collection Defaults')).toBeInTheDocument();
      expect(screen.getByText(/Configure default settings applied when creating new collections/)).toBeInTheDocument();
    });

    it('renders the info box', () => {
      render(<CollectionDefaultsSettings />);

      expect(screen.getByText(/These defaults will pre-populate the collection creation form/)).toBeInTheDocument();
    });

    it('renders embedding model dropdown with options', () => {
      render(<CollectionDefaultsSettings />);

      expect(screen.getByText('Default Embedding Model')).toBeInTheDocument();
      const comboboxes = screen.getAllByRole('combobox');
      expect(comboboxes.length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText('Use system default')).toBeInTheDocument();
    });

    it('renders quantization buttons', () => {
      render(<CollectionDefaultsSettings />);

      expect(screen.getByText('Model Precision')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'float32' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'float16' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'int8' })).toBeInTheDocument();
    });

    it('renders chunking strategy dropdown', () => {
      render(<CollectionDefaultsSettings />);

      expect(screen.getByText('Chunking Strategy')).toBeInTheDocument();
      const selects = screen.getAllByRole('combobox');
      expect(selects.length).toBeGreaterThanOrEqual(2);
    });

    it('renders chunk size and overlap inputs', () => {
      render(<CollectionDefaultsSettings />);

      expect(screen.getByText('Chunk Size')).toBeInTheDocument();
      expect(screen.getByText('Chunk Overlap')).toBeInTheDocument();
    });

    it('renders sparse indexing checkbox', () => {
      render(<CollectionDefaultsSettings />);

      expect(screen.getByText('Enable Sparse Indexing')).toBeInTheDocument();
      const checkboxes = screen.getAllByRole('checkbox');
      expect(checkboxes.length).toBeGreaterThanOrEqual(1);
    });

    it('renders action buttons', () => {
      render(<CollectionDefaultsSettings />);

      expect(screen.getByRole('button', { name: 'Reset to System Defaults' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Save Defaults' })).toBeInTheDocument();
    });
  });

  describe('form interactions', () => {
    it('changes quantization when clicking buttons', async () => {
      const user = userEvent.setup();
      render(<CollectionDefaultsSettings />);

      const float32Button = screen.getByRole('button', { name: 'float32' });
      await user.click(float32Button);

      // Check that float32 button is now selected (has selected styles)
      expect(float32Button).toHaveClass('bg-blue-100');
    });

    it('enables sparse type selection when sparse indexing is enabled', async () => {
      const user = userEvent.setup();
      render(<CollectionDefaultsSettings />);

      // First checkbox is sparse indexing, second is hybrid
      const checkboxes = screen.getAllByRole('checkbox');
      const sparseCheckbox = checkboxes[0];
      await user.click(sparseCheckbox);

      // Now sparse type buttons should be visible
      expect(screen.getByRole('button', { name: 'BM25' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'SPLADE' })).toBeInTheDocument();
    });

    it('disables hybrid checkbox when sparse is disabled', () => {
      render(<CollectionDefaultsSettings />);

      // First checkbox is sparse indexing, second is hybrid
      const checkboxes = screen.getAllByRole('checkbox');
      const hybridCheckbox = checkboxes[1];
      expect(hybridCheckbox).toBeDisabled();
    });

    it('enables hybrid checkbox when sparse is enabled', async () => {
      const user = userEvent.setup();
      render(<CollectionDefaultsSettings />);

      // First checkbox is sparse indexing, second is hybrid
      const checkboxes = screen.getAllByRole('checkbox');
      const sparseCheckbox = checkboxes[0];
      await user.click(sparseCheckbox);

      // Re-query after click to get updated state
      const updatedCheckboxes = screen.getAllByRole('checkbox');
      const hybridCheckbox = updatedCheckboxes[1];
      expect(hybridCheckbox).not.toBeDisabled();
    });
  });

  describe('save mutation', () => {
    it('calls updateMutation.mutateAsync when save button is clicked', async () => {
      const user = userEvent.setup();
      mockUpdateMutateAsync.mockResolvedValueOnce(mockPreferences);

      render(<CollectionDefaultsSettings />);

      const saveButton = screen.getByRole('button', { name: 'Save Defaults' });
      await user.click(saveButton);

      await waitFor(() => {
        expect(mockUpdateMutateAsync).toHaveBeenCalledWith({
          collection_defaults: expect.objectContaining({
            embedding_model: 'Qwen/Qwen3-Embedding-0.6B',
            quantization: 'float16',
            chunking_strategy: 'recursive',
          }),
        });
      });
    });

    it('shows saving state during mutation', () => {
      vi.mocked(usePreferencesModule.useUpdatePreferences).mockReturnValue({
        mutateAsync: mockUpdateMutateAsync,
        isPending: true,
      } as unknown as ReturnType<typeof usePreferencesModule.useUpdatePreferences>);

      render(<CollectionDefaultsSettings />);

      expect(screen.getByText('Saving...')).toBeInTheDocument();
    });
  });

  describe('reset mutation', () => {
    it('calls resetMutation.mutateAsync when reset button is clicked', async () => {
      const user = userEvent.setup();
      mockResetMutateAsync.mockResolvedValueOnce(mockPreferences);

      render(<CollectionDefaultsSettings />);

      const resetButton = screen.getByRole('button', { name: 'Reset to System Defaults' });
      await user.click(resetButton);

      await waitFor(() => {
        expect(mockResetMutateAsync).toHaveBeenCalled();
      });
    });

    it('shows resetting state during mutation', () => {
      vi.mocked(usePreferencesModule.useResetCollectionDefaults).mockReturnValue({
        mutateAsync: mockResetMutateAsync,
        isPending: true,
      } as unknown as ReturnType<typeof usePreferencesModule.useResetCollectionDefaults>);

      render(<CollectionDefaultsSettings />);

      expect(screen.getByText('Resetting...')).toBeInTheDocument();
    });
  });

  describe('validation', () => {
    it('shows validation error when hybrid is enabled without sparse', async () => {
      // Render with sparse enabled + hybrid enabled initially
      const prefsWithHybridNoSparse = {
        ...mockPreferences,
        collection_defaults: {
          ...mockPreferences.collection_defaults,
          enable_sparse: false,
          enable_hybrid: true,
        },
      };

      vi.mocked(usePreferencesModule.usePreferences).mockReturnValue({
        data: prefsWithHybridNoSparse,
        isLoading: false,
        error: null,
      } as unknown as ReturnType<typeof usePreferencesModule.usePreferences>);

      render(<CollectionDefaultsSettings />);

      await waitFor(() => {
        expect(screen.getByText('Hybrid search requires sparse indexing to be enabled')).toBeInTheDocument();
      });
    });

    it('disables save button when there is a validation error', async () => {
      const prefsWithHybridNoSparse = {
        ...mockPreferences,
        collection_defaults: {
          ...mockPreferences.collection_defaults,
          enable_sparse: false,
          enable_hybrid: true,
        },
      };

      vi.mocked(usePreferencesModule.usePreferences).mockReturnValue({
        data: prefsWithHybridNoSparse,
        isLoading: false,
        error: null,
      } as unknown as ReturnType<typeof usePreferencesModule.usePreferences>);

      render(<CollectionDefaultsSettings />);

      await waitFor(() => {
        const saveButton = screen.getByRole('button', { name: 'Save Defaults' });
        expect(saveButton).toBeDisabled();
      });
    });
  });
});
