import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import PerformanceSettings from '../PerformanceSettings';
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
    cache_ttl_seconds: 300,
    model_unload_timeout_seconds: 300,
    // Other settings
    gpu_memory_max_percent: 0.9,
    cpu_memory_max_percent: 0.5,
    enable_cpu_offload: true,
    eviction_idle_threshold_seconds: 120,
    rerank_candidate_multiplier: 5,
    rerank_min_candidates: 20,
    rerank_max_candidates: 200,
    rerank_hybrid_weight: 0.3,
    max_collections_per_user: 10,
    max_storage_gb_per_user: 50,
    max_document_size_mb: 100,
  },
};

describe('PerformanceSettings', () => {
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

      render(<PerformanceSettings />);

      expect(screen.getByText('Loading performance settings...')).toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('shows error message when loading fails', () => {
      vi.mocked(useSystemSettingsModule.useEffectiveSettings).mockReturnValue({
        data: undefined,
        isLoading: false,
        error: new Error('Failed to load settings'),
      } as unknown as ReturnType<typeof useSystemSettingsModule.useEffectiveSettings>);

      render(<PerformanceSettings />);

      expect(screen.getByText('Error loading settings')).toBeInTheDocument();
      expect(screen.getByText('Failed to load settings')).toBeInTheDocument();
    });
  });

  describe('form rendering', () => {
    it('renders the warning info box', () => {
      render(<PerformanceSettings />);

      expect(screen.getByText(/Changing these settings may impact system performance/)).toBeInTheDocument();
    });

    it('renders cache TTL input', () => {
      render(<PerformanceSettings />);

      expect(screen.getByText('Cache TTL (seconds)')).toBeInTheDocument();
      const inputs = screen.getAllByRole('spinbutton');
      expect(inputs[0]).toHaveValue(300);
    });

    it('renders model unload timeout input', () => {
      render(<PerformanceSettings />);

      expect(screen.getByText('Model Unload Timeout (seconds)')).toBeInTheDocument();
    });

    it('renders action buttons', () => {
      render(<PerformanceSettings />);

      expect(screen.getByRole('button', { name: 'Reset to Defaults' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Save Settings' })).toBeInTheDocument();
    });
  });

  describe('form interactions', () => {
    it('updates cache TTL input', async () => {
      const user = userEvent.setup();
      render(<PerformanceSettings />);

      const inputs = screen.getAllByRole('spinbutton');
      await user.tripleClick(inputs[0]);
      await user.keyboard('600');

      expect(inputs[0]).toHaveValue(600);
    });

    it('updates model unload timeout input', async () => {
      const user = userEvent.setup();
      render(<PerformanceSettings />);

      const inputs = screen.getAllByRole('spinbutton');
      await user.tripleClick(inputs[1]);
      await user.keyboard('600');

      expect(inputs[1]).toHaveValue(600);
    });
  });

  describe('save mutation', () => {
    it('calls updateMutation.mutateAsync when save button is clicked', async () => {
      const user = userEvent.setup();
      mockUpdateMutateAsync.mockResolvedValueOnce({ updated: ['cache_ttl_seconds'] });

      render(<PerformanceSettings />);

      const saveButton = screen.getByRole('button', { name: 'Save Settings' });
      await user.click(saveButton);

      await waitFor(() => {
        expect(mockUpdateMutateAsync).toHaveBeenCalledWith({
          settings: expect.objectContaining({
            cache_ttl_seconds: 300,
            model_unload_timeout_seconds: 300,
          }),
        });
      });
    });

    it('shows saving state during mutation', () => {
      vi.mocked(useSystemSettingsModule.useUpdateSystemSettings).mockReturnValue({
        mutateAsync: mockUpdateMutateAsync,
        isPending: true,
      } as unknown as ReturnType<typeof useSystemSettingsModule.useUpdateSystemSettings>);

      render(<PerformanceSettings />);

      expect(screen.getByText('Saving...')).toBeInTheDocument();
    });
  });

  describe('reset mutation', () => {
    it('calls resetMutation.mutateAsync when reset button is clicked', async () => {
      const user = userEvent.setup();
      mockResetMutateAsync.mockResolvedValueOnce({ updated: [] });

      render(<PerformanceSettings />);

      const resetButton = screen.getByRole('button', { name: 'Reset to Defaults' });
      await user.click(resetButton);

      await waitFor(() => {
        expect(mockResetMutateAsync).toHaveBeenCalledWith([
          'cache_ttl_seconds',
          'model_unload_timeout_seconds',
        ]);
      });
    });

    it('shows resetting state during mutation', () => {
      vi.mocked(useSystemSettingsModule.useResetSettingsToDefaults).mockReturnValue({
        mutateAsync: mockResetMutateAsync,
        isPending: true,
      } as unknown as ReturnType<typeof useSystemSettingsModule.useResetSettingsToDefaults>);

      render(<PerformanceSettings />);

      expect(screen.getByText('Resetting...')).toBeInTheDocument();
    });
  });
});
