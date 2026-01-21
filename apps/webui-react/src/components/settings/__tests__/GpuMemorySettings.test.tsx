import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import GpuMemorySettings from '../GpuMemorySettings';
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
    gpu_memory_max_percent: 0.9,
    cpu_memory_max_percent: 0.5,
    enable_cpu_offload: true,
    eviction_idle_threshold_seconds: 120,
    // Other settings that might be present
    rerank_candidate_multiplier: 5,
    rerank_min_candidates: 20,
    rerank_max_candidates: 200,
    rerank_hybrid_weight: 0.3,
    max_collections_per_user: 10,
    max_storage_gb_per_user: 50,
    max_document_size_mb: 100,
    cache_ttl_seconds: 300,
    model_unload_timeout_seconds: 300,
  },
};

describe('GpuMemorySettings', () => {
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

      render(<GpuMemorySettings />);

      expect(screen.getByText('Loading GPU/memory settings...')).toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('shows error message when loading fails', () => {
      vi.mocked(useSystemSettingsModule.useEffectiveSettings).mockReturnValue({
        data: undefined,
        isLoading: false,
        error: new Error('Failed to load settings'),
      } as unknown as ReturnType<typeof useSystemSettingsModule.useEffectiveSettings>);

      render(<GpuMemorySettings />);

      expect(screen.getByText('Error loading settings')).toBeInTheDocument();
      expect(screen.getByText('Failed to load settings')).toBeInTheDocument();
    });
  });

  describe('form rendering', () => {
    it('renders the caution warning box', () => {
      render(<GpuMemorySettings />);

      expect(screen.getByText(/Incorrect memory settings can cause out-of-memory errors/)).toBeInTheDocument();
    });

    it('renders GPU Memory section', () => {
      render(<GpuMemorySettings />);

      expect(screen.getByText('GPU Memory')).toBeInTheDocument();
      expect(screen.getByText(/GPU Memory Limit/)).toBeInTheDocument();
    });

    it('renders CPU Memory section', () => {
      render(<GpuMemorySettings />);

      expect(screen.getByText('CPU Memory (Warm Models)')).toBeInTheDocument();
      expect(screen.getByText(/CPU Memory Limit/)).toBeInTheDocument();
    });

    it('renders Offloading & Eviction section', () => {
      render(<GpuMemorySettings />);

      expect(screen.getByText('Offloading & Eviction')).toBeInTheDocument();
      expect(screen.getByText('Enable CPU Offload')).toBeInTheDocument();
      expect(screen.getByText('Eviction Idle Threshold (seconds)')).toBeInTheDocument();
    });

    it('renders action buttons', () => {
      render(<GpuMemorySettings />);

      expect(screen.getByRole('button', { name: 'Reset to Defaults' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Save Settings' })).toBeInTheDocument();
    });

    it('displays current GPU memory percentage', () => {
      render(<GpuMemorySettings />);

      // Should show 90% since default is 0.9
      expect(screen.getByText(/GPU Memory Limit \(90%\)/)).toBeInTheDocument();
    });

    it('displays current CPU memory percentage', () => {
      render(<GpuMemorySettings />);

      // Should show 50% since default is 0.5
      expect(screen.getByText(/CPU Memory Limit \(50%\)/)).toBeInTheDocument();
    });
  });

  describe('form interactions', () => {
    it('toggles CPU offload checkbox', async () => {
      const user = userEvent.setup();
      render(<GpuMemorySettings />);

      const checkbox = screen.getByRole('checkbox');
      expect(checkbox).toBeChecked(); // Initially enabled

      await user.click(checkbox);
      expect(checkbox).not.toBeChecked();
    });

    it('updates eviction threshold input', async () => {
      const user = userEvent.setup();
      render(<GpuMemorySettings />);

      const input = screen.getByRole('spinbutton');
      await user.tripleClick(input);
      await user.keyboard('180');

      expect(input).toHaveValue(180);
    });
  });

  describe('save mutation', () => {
    it('calls updateMutation.mutateAsync when save button is clicked', async () => {
      const user = userEvent.setup();
      mockUpdateMutateAsync.mockResolvedValueOnce({ updated: ['gpu_memory_max_percent'] });

      render(<GpuMemorySettings />);

      const saveButton = screen.getByRole('button', { name: 'Save Settings' });
      await user.click(saveButton);

      await waitFor(() => {
        expect(mockUpdateMutateAsync).toHaveBeenCalledWith({
          settings: expect.objectContaining({
            gpu_memory_max_percent: 0.9,
            cpu_memory_max_percent: 0.5,
            enable_cpu_offload: true,
            eviction_idle_threshold_seconds: 120,
          }),
        });
      });
    });

    it('shows saving state during mutation', () => {
      vi.mocked(useSystemSettingsModule.useUpdateSystemSettings).mockReturnValue({
        mutateAsync: mockUpdateMutateAsync,
        isPending: true,
      } as unknown as ReturnType<typeof useSystemSettingsModule.useUpdateSystemSettings>);

      render(<GpuMemorySettings />);

      expect(screen.getByText('Saving...')).toBeInTheDocument();
    });
  });

  describe('reset mutation', () => {
    it('calls resetMutation.mutateAsync when reset button is clicked', async () => {
      const user = userEvent.setup();
      mockResetMutateAsync.mockResolvedValueOnce({ updated: [] });

      render(<GpuMemorySettings />);

      const resetButton = screen.getByRole('button', { name: 'Reset to Defaults' });
      await user.click(resetButton);

      await waitFor(() => {
        expect(mockResetMutateAsync).toHaveBeenCalledWith([
          'gpu_memory_max_percent',
          'cpu_memory_max_percent',
          'enable_cpu_offload',
          'eviction_idle_threshold_seconds',
        ]);
      });
    });

    it('shows resetting state during mutation', () => {
      vi.mocked(useSystemSettingsModule.useResetSettingsToDefaults).mockReturnValue({
        mutateAsync: mockResetMutateAsync,
        isPending: true,
      } as unknown as ReturnType<typeof useSystemSettingsModule.useResetSettingsToDefaults>);

      render(<GpuMemorySettings />);

      expect(screen.getByText('Resetting...')).toBeInTheDocument();
    });
  });
});
