import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import SearchRerankSettings from '../SearchRerankSettings';
import * as useSystemSettingsModule from '@/hooks/useSystemSettings';

// Mock the hooks
vi.mock('@/hooks/useSystemSettings', () => ({
  useEffectiveSettings: vi.fn(),
  useUpdateSystemSettings: vi.fn(),
  useResetSettingsToDefaults: vi.fn(),
}));

// Mock data
const mockEffectiveSettings = {
  settings: {
    rerank_candidate_multiplier: 5,
    rerank_min_candidates: 20,
    rerank_max_candidates: 200,
    rerank_hybrid_weight: 0.3,
    // Other settings
    gpu_memory_max_percent: 0.9,
    cpu_memory_max_percent: 0.5,
    enable_cpu_offload: true,
    eviction_idle_threshold_seconds: 120,
    max_collections_per_user: 10,
    max_storage_gb_per_user: 50,
    max_document_size_mb: 100,
    cache_ttl_seconds: 300,
    model_unload_timeout_seconds: 300,
  },
};

describe('SearchRerankSettings', () => {
  const mockUpdateMutateAsync = vi.fn();
  const mockResetMutateAsync = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();

    // Default mock implementations
    vi.mocked(useSystemSettingsModule.useEffectiveSettings).mockReturnValue({
      data: mockEffectiveSettings,
      isLoading: false,
      error: null,
    } as unknown as ReturnType<typeof useSystemSettingsModule.useEffectiveSettings>);

    vi.mocked(useSystemSettingsModule.useUpdateSystemSettings).mockReturnValue({
      mutateAsync: mockUpdateMutateAsync,
      isPending: false,
    } as unknown as ReturnType<typeof useSystemSettingsModule.useUpdateSystemSettings>);

    vi.mocked(useSystemSettingsModule.useResetSettingsToDefaults).mockReturnValue({
      mutateAsync: mockResetMutateAsync,
      isPending: false,
    } as unknown as ReturnType<typeof useSystemSettingsModule.useResetSettingsToDefaults>);
  });

  describe('loading state', () => {
    it('shows loading spinner when settings are loading', () => {
      vi.mocked(useSystemSettingsModule.useEffectiveSettings).mockReturnValue({
        data: undefined,
        isLoading: true,
        error: null,
      } as unknown as ReturnType<typeof useSystemSettingsModule.useEffectiveSettings>);

      render(<SearchRerankSettings />);

      expect(screen.getByText('Loading search settings...')).toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('shows error message when loading fails', () => {
      vi.mocked(useSystemSettingsModule.useEffectiveSettings).mockReturnValue({
        data: undefined,
        isLoading: false,
        error: new Error('Failed to load settings'),
      } as unknown as ReturnType<typeof useSystemSettingsModule.useEffectiveSettings>);

      render(<SearchRerankSettings />);

      expect(screen.getByText('Error loading settings')).toBeInTheDocument();
      expect(screen.getByText('Failed to load settings')).toBeInTheDocument();
    });
  });

  describe('form rendering', () => {
    it('renders the info box', () => {
      render(<SearchRerankSettings />);

      expect(screen.getByText(/These settings control how the reranking model processes search results/)).toBeInTheDocument();
    });

    it('renders candidate multiplier input', () => {
      render(<SearchRerankSettings />);

      expect(screen.getByText('Rerank Candidate Multiplier')).toBeInTheDocument();
      const inputs = screen.getAllByRole('spinbutton');
      expect(inputs[0]).toHaveValue(5);
    });

    it('renders minimum candidates input', () => {
      render(<SearchRerankSettings />);

      expect(screen.getByText('Minimum Candidates')).toBeInTheDocument();
    });

    it('renders maximum candidates input', () => {
      render(<SearchRerankSettings />);

      expect(screen.getByText('Maximum Candidates')).toBeInTheDocument();
    });

    it('renders hybrid weight slider', () => {
      render(<SearchRerankSettings />);

      expect(screen.getByText(/Hybrid Weight/)).toBeInTheDocument();
      expect(screen.getByRole('slider')).toBeInTheDocument();
    });

    it('displays hybrid weight labels', () => {
      render(<SearchRerankSettings />);

      expect(screen.getByText('Dense only (0.0)')).toBeInTheDocument();
      expect(screen.getByText('Balanced (0.5)')).toBeInTheDocument();
      expect(screen.getByText('Sparse only (1.0)')).toBeInTheDocument();
    });

    it('renders action buttons', () => {
      render(<SearchRerankSettings />);

      expect(screen.getByRole('button', { name: 'Reset to Defaults' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Save Settings' })).toBeInTheDocument();
    });
  });

  describe('form interactions', () => {
    it('updates candidate multiplier input', async () => {
      const user = userEvent.setup();
      render(<SearchRerankSettings />);

      const inputs = screen.getAllByRole('spinbutton');
      await user.tripleClick(inputs[0]);
      await user.keyboard('10');

      expect(inputs[0]).toHaveValue(10);
    });

    it('updates minimum candidates input', async () => {
      const user = userEvent.setup();
      render(<SearchRerankSettings />);

      const inputs = screen.getAllByRole('spinbutton');
      await user.tripleClick(inputs[1]);
      await user.keyboard('30');

      expect(inputs[1]).toHaveValue(30);
    });

    it('updates maximum candidates input', async () => {
      const user = userEvent.setup();
      render(<SearchRerankSettings />);

      const inputs = screen.getAllByRole('spinbutton');
      await user.tripleClick(inputs[2]);
      await user.keyboard('300');

      expect(inputs[2]).toHaveValue(300);
    });
  });

  describe('save mutation', () => {
    it('calls updateMutation.mutateAsync when save button is clicked', async () => {
      const user = userEvent.setup();
      mockUpdateMutateAsync.mockResolvedValueOnce({ updated: ['rerank_candidate_multiplier'] });

      render(<SearchRerankSettings />);

      const saveButton = screen.getByRole('button', { name: 'Save Settings' });
      await user.click(saveButton);

      await waitFor(() => {
        expect(mockUpdateMutateAsync).toHaveBeenCalledWith({
          settings: expect.objectContaining({
            rerank_candidate_multiplier: 5,
            rerank_min_candidates: 20,
            rerank_max_candidates: 200,
            rerank_hybrid_weight: 0.3,
          }),
        });
      });
    });

    it('shows saving state during mutation', () => {
      vi.mocked(useSystemSettingsModule.useUpdateSystemSettings).mockReturnValue({
        mutateAsync: mockUpdateMutateAsync,
        isPending: true,
      } as unknown as ReturnType<typeof useSystemSettingsModule.useUpdateSystemSettings>);

      render(<SearchRerankSettings />);

      expect(screen.getByText('Saving...')).toBeInTheDocument();
    });
  });

  describe('reset mutation', () => {
    it('calls resetMutation.mutateAsync when reset button is clicked', async () => {
      const user = userEvent.setup();
      mockResetMutateAsync.mockResolvedValueOnce({ updated: [] });

      render(<SearchRerankSettings />);

      const resetButton = screen.getByRole('button', { name: 'Reset to Defaults' });
      await user.click(resetButton);

      await waitFor(() => {
        expect(mockResetMutateAsync).toHaveBeenCalledWith([
          'rerank_candidate_multiplier',
          'rerank_min_candidates',
          'rerank_max_candidates',
          'rerank_hybrid_weight',
        ]);
      });
    });

    it('shows resetting state during mutation', () => {
      vi.mocked(useSystemSettingsModule.useResetSettingsToDefaults).mockReturnValue({
        mutateAsync: mockResetMutateAsync,
        isPending: true,
      } as unknown as ReturnType<typeof useSystemSettingsModule.useResetSettingsToDefaults>);

      render(<SearchRerankSettings />);

      expect(screen.getByText('Resetting...')).toBeInTheDocument();
    });
  });
});
