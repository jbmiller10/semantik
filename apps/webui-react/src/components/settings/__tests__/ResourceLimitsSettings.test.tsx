import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import ResourceLimitsSettings from '../ResourceLimitsSettings';
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
    max_collections_per_user: 10,
    max_storage_gb_per_user: 50,
    max_document_size_mb: 100,
    // Other settings
    gpu_memory_max_percent: 0.9,
    cpu_memory_max_percent: 0.5,
    enable_cpu_offload: true,
    eviction_idle_threshold_seconds: 120,
    rerank_candidate_multiplier: 5,
    rerank_min_candidates: 20,
    rerank_max_candidates: 200,
    rerank_hybrid_weight: 0.3,
    cache_ttl_seconds: 300,
    model_unload_timeout_seconds: 300,
  },
};

describe('ResourceLimitsSettings', () => {
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

      render(<ResourceLimitsSettings />);

      expect(screen.getByText('Loading resource limits...')).toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('shows error message when loading fails', () => {
      vi.mocked(useSystemSettingsModule.useEffectiveSettings).mockReturnValue({
        data: undefined,
        isLoading: false,
        error: new Error('Failed to load settings'),
      } as unknown as ReturnType<typeof useSystemSettingsModule.useEffectiveSettings>);

      render(<ResourceLimitsSettings />);

      expect(screen.getByText('Error loading settings')).toBeInTheDocument();
      expect(screen.getByText('Failed to load settings')).toBeInTheDocument();
    });
  });

  describe('form rendering', () => {
    it('renders the info box', () => {
      render(<ResourceLimitsSettings />);

      expect(screen.getByText(/These limits apply to all users/)).toBeInTheDocument();
    });

    it('renders max collections input', () => {
      render(<ResourceLimitsSettings />);

      expect(screen.getByText('Max Collections per User')).toBeInTheDocument();
      const inputs = screen.getAllByRole('spinbutton');
      expect(inputs[0]).toHaveValue(10);
    });

    it('renders max storage input', () => {
      render(<ResourceLimitsSettings />);

      expect(screen.getByText('Max Storage per User (GB)')).toBeInTheDocument();
    });

    it('renders max document size input', () => {
      render(<ResourceLimitsSettings />);

      expect(screen.getByText('Max Document Size (MB)')).toBeInTheDocument();
    });

    it('renders action buttons', () => {
      render(<ResourceLimitsSettings />);

      expect(screen.getByRole('button', { name: 'Reset to Defaults' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Save Settings' })).toBeInTheDocument();
    });
  });

  describe('form interactions', () => {
    it('updates max collections input', async () => {
      const user = userEvent.setup();
      render(<ResourceLimitsSettings />);

      const inputs = screen.getAllByRole('spinbutton');
      await user.tripleClick(inputs[0]);
      await user.keyboard('20');

      expect(inputs[0]).toHaveValue(20);
    });

    it('updates max storage input', async () => {
      const user = userEvent.setup();
      render(<ResourceLimitsSettings />);

      const inputs = screen.getAllByRole('spinbutton');
      await user.tripleClick(inputs[1]);
      await user.keyboard('100');

      expect(inputs[1]).toHaveValue(100);
    });

    it('updates max document size input', async () => {
      const user = userEvent.setup();
      render(<ResourceLimitsSettings />);

      const inputs = screen.getAllByRole('spinbutton');
      await user.tripleClick(inputs[2]);
      await user.keyboard('200');

      expect(inputs[2]).toHaveValue(200);
    });
  });

  describe('save mutation', () => {
    it('calls updateMutation.mutateAsync when save button is clicked', async () => {
      const user = userEvent.setup();
      mockUpdateMutateAsync.mockResolvedValueOnce({ updated: ['max_collections_per_user'] });

      render(<ResourceLimitsSettings />);

      const saveButton = screen.getByRole('button', { name: 'Save Settings' });
      await user.click(saveButton);

      await waitFor(() => {
        expect(mockUpdateMutateAsync).toHaveBeenCalledWith({
          settings: expect.objectContaining({
            max_collections_per_user: 10,
            max_storage_gb_per_user: 50,
            max_document_size_mb: 100,
          }),
        });
      });
    });

    it('shows saving state during mutation', () => {
      vi.mocked(useSystemSettingsModule.useUpdateSystemSettings).mockReturnValue({
        mutateAsync: mockUpdateMutateAsync,
        isPending: true,
      } as unknown as ReturnType<typeof useSystemSettingsModule.useUpdateSystemSettings>);

      render(<ResourceLimitsSettings />);

      expect(screen.getByText('Saving...')).toBeInTheDocument();
    });
  });

  describe('reset mutation', () => {
    it('calls resetMutation.mutateAsync when reset button is clicked', async () => {
      const user = userEvent.setup();
      mockResetMutateAsync.mockResolvedValueOnce({ updated: [] });

      render(<ResourceLimitsSettings />);

      const resetButton = screen.getByRole('button', { name: 'Reset to Defaults' });
      await user.click(resetButton);

      await waitFor(() => {
        expect(mockResetMutateAsync).toHaveBeenCalledWith([
          'max_collections_per_user',
          'max_storage_gb_per_user',
          'max_document_size_mb',
        ]);
      });
    });

    it('shows resetting state during mutation', () => {
      vi.mocked(useSystemSettingsModule.useResetSettingsToDefaults).mockReturnValue({
        mutateAsync: mockResetMutateAsync,
        isPending: true,
      } as unknown as ReturnType<typeof useSystemSettingsModule.useResetSettingsToDefaults>);

      render(<ResourceLimitsSettings />);

      expect(screen.getByText('Resetting...')).toBeInTheDocument();
    });
  });
});
