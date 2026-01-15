import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import SearchPreferencesSettings from '../SearchPreferencesSettings';
import * as usePreferencesModule from '@/hooks/usePreferences';
import * as useSystemInfoModule from '@/hooks/useSystemInfo';
import type { UserPreferencesResponse } from '@/types/preferences';

// Mock the hooks
vi.mock('@/hooks/usePreferences', () => ({
  usePreferences: vi.fn(),
  useUpdatePreferences: vi.fn(),
  useResetSearchPreferences: vi.fn(),
}));

vi.mock('@/hooks/useSystemInfo', () => ({
  useSystemStatus: vi.fn(),
}));

// Mock data
const mockPreferences: UserPreferencesResponse = {
  collection_defaults: {
    embedding_model: null,
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
    use_hyde: false,
    hyde_quality_tier: 'low',
    hyde_timeout_seconds: 10,
  },
  interface: {
    data_refresh_interval_ms: 30000,
    visualization_sample_limit: 200000,
    animation_enabled: true,
  },
  created_at: '2025-01-01T00:00:00Z',
  updated_at: '2025-01-01T00:00:00Z',
};

const mockSystemStatus = {
  healthy: true,
  reranking_available: true,
  gpu_available: true,
  gpu_memory_mb: 8192,
};

describe('SearchPreferencesSettings', () => {
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

    vi.mocked(usePreferencesModule.useResetSearchPreferences).mockReturnValue({
      mutateAsync: mockResetMutateAsync,
      isPending: false,
    } as unknown as ReturnType<typeof usePreferencesModule.useResetSearchPreferences>);

    vi.mocked(useSystemInfoModule.useSystemStatus).mockReturnValue({
      data: mockSystemStatus,
      isLoading: false,
      error: null,
    } as unknown as ReturnType<typeof useSystemInfoModule.useSystemStatus>);
  });

  describe('loading state', () => {
    it('shows loading spinner when preferences are loading', () => {
      vi.mocked(usePreferencesModule.usePreferences).mockReturnValue({
        data: undefined,
        isLoading: true,
        error: null,
      } as unknown as ReturnType<typeof usePreferencesModule.usePreferences>);

      render(<SearchPreferencesSettings />);

      expect(screen.getByText('Loading search preferences...')).toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('shows error message when loading fails', () => {
      vi.mocked(usePreferencesModule.usePreferences).mockReturnValue({
        data: undefined,
        isLoading: false,
        error: new Error('Failed to load preferences'),
      } as unknown as ReturnType<typeof usePreferencesModule.usePreferences>);

      render(<SearchPreferencesSettings />);

      expect(screen.getByText('Error loading preferences')).toBeInTheDocument();
      expect(screen.getByText('Failed to load preferences')).toBeInTheDocument();
    });
  });

  describe('form rendering', () => {
    it('renders the header and description', () => {
      render(<SearchPreferencesSettings />);

      expect(screen.getByText('Search Preferences')).toBeInTheDocument();
      expect(screen.getByText(/Configure default search behavior/)).toBeInTheDocument();
    });

    it('renders the info box', () => {
      render(<SearchPreferencesSettings />);

      expect(screen.getByText(/These defaults will be applied when you open the search interface/)).toBeInTheDocument();
    });

    it('renders top_k input', () => {
      render(<SearchPreferencesSettings />);

      expect(screen.getByText('Default Results Count')).toBeInTheDocument();
      const inputs = screen.getAllByRole('spinbutton');
      expect(inputs[0]).toHaveValue(10);
    });

    it('renders search mode buttons', () => {
      render(<SearchPreferencesSettings />);

      expect(screen.getByText('Search Mode')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Dense' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Sparse' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Hybrid' })).toBeInTheDocument();
    });

    it('renders reranker checkbox', () => {
      render(<SearchPreferencesSettings />);

      expect(screen.getByText('Use Reranker')).toBeInTheDocument();
    });

    it('renders similarity threshold input', () => {
      render(<SearchPreferencesSettings />);

      expect(screen.getByText('Similarity Threshold')).toBeInTheDocument();
    });

    it('renders action buttons', () => {
      render(<SearchPreferencesSettings />);

      expect(screen.getByRole('button', { name: 'Reset to Defaults' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Save Preferences' })).toBeInTheDocument();
    });
  });

  describe('conditional rendering', () => {
    it('does not show RRF field when mode is not hybrid', () => {
      render(<SearchPreferencesSettings />);

      expect(screen.queryByText('RRF Constant (k)')).not.toBeInTheDocument();
    });

    it('shows RRF field when mode is hybrid', async () => {
      const user = userEvent.setup();
      render(<SearchPreferencesSettings />);

      const hybridButton = screen.getByRole('button', { name: 'Hybrid' });
      await user.click(hybridButton);

      expect(screen.getByText('RRF Constant (k)')).toBeInTheDocument();
    });
  });

  describe('reranker availability', () => {
    it('disables reranker checkbox when not available', () => {
      vi.mocked(useSystemInfoModule.useSystemStatus).mockReturnValue({
        data: { ...mockSystemStatus, reranking_available: false },
        isLoading: false,
        error: null,
      } as unknown as ReturnType<typeof useSystemInfoModule.useSystemStatus>);

      render(<SearchPreferencesSettings />);

      // Get all checkboxes and find the reranker one (first checkbox in the form)
      const checkboxes = screen.getAllByRole('checkbox');
      const rerankerCheckbox = checkboxes[0]; // Reranker is the first checkbox
      expect(rerankerCheckbox).toBeDisabled();
      expect(screen.getByText('(not available)')).toBeInTheDocument();
    });

    it('enables reranker checkbox when available', () => {
      render(<SearchPreferencesSettings />);

      // Get all checkboxes and find the reranker one (first checkbox in the form)
      const checkboxes = screen.getAllByRole('checkbox');
      const rerankerCheckbox = checkboxes[0]; // Reranker is the first checkbox
      expect(rerankerCheckbox).not.toBeDisabled();
    });
  });

  describe('form interactions', () => {
    it('changes search mode when clicking buttons', async () => {
      const user = userEvent.setup();
      render(<SearchPreferencesSettings />);

      const sparseButton = screen.getByRole('button', { name: 'Sparse' });
      await user.click(sparseButton);

      expect(sparseButton).toHaveClass('bg-blue-100');
    });

    it('toggles reranker checkbox', async () => {
      const user = userEvent.setup();
      render(<SearchPreferencesSettings />);

      // Get all checkboxes and find the reranker one (first checkbox in the form)
      const checkboxes = screen.getAllByRole('checkbox');
      const rerankerCheckbox = checkboxes[0]; // Reranker is the first checkbox
      expect(rerankerCheckbox).not.toBeChecked();

      await user.click(rerankerCheckbox);
      expect(rerankerCheckbox).toBeChecked();
    });

    it('updates top_k value', async () => {
      const user = userEvent.setup();
      render(<SearchPreferencesSettings />);

      const inputs = screen.getAllByRole('spinbutton');
      await user.tripleClick(inputs[0]);
      await user.keyboard('25');

      expect(inputs[0]).toHaveValue(25);
    });
  });

  describe('save mutation', () => {
    it('calls updateMutation.mutateAsync when save button is clicked', async () => {
      const user = userEvent.setup();
      mockUpdateMutateAsync.mockResolvedValueOnce(mockPreferences);

      render(<SearchPreferencesSettings />);

      const saveButton = screen.getByRole('button', { name: 'Save Preferences' });
      await user.click(saveButton);

      await waitFor(() => {
        expect(mockUpdateMutateAsync).toHaveBeenCalledWith({
          search: expect.objectContaining({
            top_k: 10,
            mode: 'dense',
            use_reranker: false,
          }),
        });
      });
    });

    it('shows saving state during mutation', () => {
      vi.mocked(usePreferencesModule.useUpdatePreferences).mockReturnValue({
        mutateAsync: mockUpdateMutateAsync,
        isPending: true,
      } as unknown as ReturnType<typeof usePreferencesModule.useUpdatePreferences>);

      render(<SearchPreferencesSettings />);

      expect(screen.getByText('Saving...')).toBeInTheDocument();
    });
  });

  describe('reset mutation', () => {
    it('calls resetMutation.mutateAsync when reset button is clicked', async () => {
      const user = userEvent.setup();
      mockResetMutateAsync.mockResolvedValueOnce(mockPreferences);

      render(<SearchPreferencesSettings />);

      const resetButton = screen.getByRole('button', { name: 'Reset to Defaults' });
      await user.click(resetButton);

      await waitFor(() => {
        expect(mockResetMutateAsync).toHaveBeenCalled();
      });
    });

    it('shows resetting state during mutation', () => {
      vi.mocked(usePreferencesModule.useResetSearchPreferences).mockReturnValue({
        mutateAsync: mockResetMutateAsync,
        isPending: true,
      } as unknown as ReturnType<typeof usePreferencesModule.useResetSearchPreferences>);

      render(<SearchPreferencesSettings />);

      expect(screen.getByText('Resetting...')).toBeInTheDocument();
    });
  });

  describe('similarity threshold handling', () => {
    it('renders with empty similarity threshold as empty string', () => {
      render(<SearchPreferencesSettings />);

      const thresholdInput = screen.getByPlaceholderText('No threshold');
      expect(thresholdInput).toHaveValue(null);
    });

    it('renders with similarity threshold value when set', () => {
      const prefsWithThreshold = {
        ...mockPreferences,
        search: {
          ...mockPreferences.search,
          similarity_threshold: 0.5,
        },
      };

      vi.mocked(usePreferencesModule.usePreferences).mockReturnValue({
        data: prefsWithThreshold,
        isLoading: false,
        error: null,
      } as unknown as ReturnType<typeof usePreferencesModule.usePreferences>);

      render(<SearchPreferencesSettings />);

      const thresholdInput = screen.getByPlaceholderText('No threshold');
      expect(thresholdInput).toHaveValue(0.5);
    });
  });
});
