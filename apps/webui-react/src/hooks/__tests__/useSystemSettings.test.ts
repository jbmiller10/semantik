import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createElement, type ReactNode } from 'react';
import {
  useEffectiveSettings,
  useSystemSettingsDefaults,
  useUpdateSystemSettings,
  useResetSettingsToDefaults,
  systemSettingsKeys,
} from '../useSystemSettings';
import { systemSettingsApi } from '@/services/api/v2/system-settings';
import { useUIStore } from '@/stores/uiStore';

// Mock the API and store
vi.mock('@/services/api/v2/system-settings', () => ({
  systemSettingsApi: {
    getEffective: vi.fn(),
    getDefaults: vi.fn(),
    updateSettings: vi.fn(),
  },
}));

vi.mock('@/stores/uiStore', () => ({
  useUIStore: vi.fn(),
}));

// Mock data
const mockEffectiveSettings = {
  settings: {
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

const mockDefaults = {
  defaults: {
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

describe('useSystemSettings hooks', () => {
  let queryClient: QueryClient;
  const mockAddToast = vi.fn();

  const wrapper = ({ children }: { children: ReactNode }) =>
    createElement(QueryClientProvider, { client: queryClient }, children);

  beforeEach(() => {
    vi.clearAllMocks();
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
        mutations: {
          retry: false,
        },
      },
    });

    vi.mocked(useUIStore).mockReturnValue({
      addToast: mockAddToast,
    } as unknown as ReturnType<typeof useUIStore>);
  });

  describe('systemSettingsKeys', () => {
    it('returns correct query keys', () => {
      expect(systemSettingsKeys.all).toEqual(['system-settings']);
      expect(systemSettingsKeys.settings()).toEqual(['system-settings', 'settings']);
      expect(systemSettingsKeys.effective()).toEqual(['system-settings', 'effective']);
      expect(systemSettingsKeys.defaults()).toEqual(['system-settings', 'defaults']);
    });
  });

  describe('useEffectiveSettings', () => {
    it('fetches effective settings successfully', async () => {
      vi.mocked(systemSettingsApi.getEffective).mockResolvedValueOnce({
        data: mockEffectiveSettings,
        status: 200,
        statusText: 'OK',
        headers: {},
        config: {} as never,
      });

      const { result } = renderHook(() => useEffectiveSettings(), { wrapper });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockEffectiveSettings);
      expect(systemSettingsApi.getEffective).toHaveBeenCalled();
    });

    it('handles error when fetching effective settings', async () => {
      vi.mocked(systemSettingsApi.getEffective).mockRejectedValueOnce(
        new Error('Failed to fetch')
      );

      const { result } = renderHook(() => useEffectiveSettings(), { wrapper });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error?.message).toBe('Failed to fetch');
    });
  });

  describe('useSystemSettingsDefaults', () => {
    it('fetches defaults successfully', async () => {
      vi.mocked(systemSettingsApi.getDefaults).mockResolvedValueOnce({
        data: mockDefaults,
        status: 200,
        statusText: 'OK',
        headers: {},
        config: {} as never,
      });

      const { result } = renderHook(() => useSystemSettingsDefaults(), { wrapper });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockDefaults);
      expect(systemSettingsApi.getDefaults).toHaveBeenCalled();
    });
  });

  describe('useUpdateSystemSettings', () => {
    it('updates settings successfully', async () => {
      const updateResponse = {
        updated: ['gpu_memory_max_percent'],
        settings: { gpu_memory_max_percent: { value: 0.8, updated_at: '2025-01-01', updated_by: 1 } },
      };

      vi.mocked(systemSettingsApi.updateSettings).mockResolvedValueOnce({
        data: updateResponse,
        status: 200,
        statusText: 'OK',
        headers: {},
        config: {} as never,
      });

      const { result } = renderHook(() => useUpdateSystemSettings(), { wrapper });

      await result.current.mutateAsync({ settings: { gpu_memory_max_percent: 0.8 } });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(systemSettingsApi.updateSettings).toHaveBeenCalledWith({
        settings: { gpu_memory_max_percent: 0.8 },
      });
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'success',
        message: '1 setting saved successfully',
      });
    });

    it('shows plural message for multiple settings', async () => {
      const updateResponse = {
        updated: ['gpu_memory_max_percent', 'cpu_memory_max_percent'],
        settings: {},
      };

      vi.mocked(systemSettingsApi.updateSettings).mockResolvedValueOnce({
        data: updateResponse,
        status: 200,
        statusText: 'OK',
        headers: {},
        config: {} as never,
      });

      const { result } = renderHook(() => useUpdateSystemSettings(), { wrapper });

      await result.current.mutateAsync({
        settings: { gpu_memory_max_percent: 0.8, cpu_memory_max_percent: 0.6 },
      });

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          type: 'success',
          message: '2 settings saved successfully',
        });
      });
    });

    it('shows error toast on failure', async () => {
      vi.mocked(systemSettingsApi.updateSettings).mockRejectedValueOnce(
        new Error('Update failed')
      );

      const { result } = renderHook(() => useUpdateSystemSettings(), { wrapper });

      try {
        await result.current.mutateAsync({ settings: { gpu_memory_max_percent: 0.8 } });
      } catch {
        // Expected to throw
      }

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          type: 'error',
          message: 'Update failed',
        });
      });
    });
  });

  describe('useResetSettingsToDefaults', () => {
    it('resets settings to defaults successfully', async () => {
      vi.mocked(systemSettingsApi.getDefaults).mockResolvedValueOnce({
        data: mockDefaults,
        status: 200,
        statusText: 'OK',
        headers: {},
        config: {} as never,
      });

      const updateResponse = {
        updated: ['gpu_memory_max_percent'],
        settings: {},
      };

      vi.mocked(systemSettingsApi.updateSettings).mockResolvedValueOnce({
        data: updateResponse,
        status: 200,
        statusText: 'OK',
        headers: {},
        config: {} as never,
      });

      const { result } = renderHook(() => useResetSettingsToDefaults(), { wrapper });

      await result.current.mutateAsync(['gpu_memory_max_percent']);

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(systemSettingsApi.getDefaults).toHaveBeenCalled();
      expect(systemSettingsApi.updateSettings).toHaveBeenCalledWith({
        settings: { gpu_memory_max_percent: 0.9 },
      });
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'success',
        message: 'Settings reset to defaults',
      });
    });

    it('shows error toast on failure', async () => {
      vi.mocked(systemSettingsApi.getDefaults).mockRejectedValueOnce(
        new Error('Reset failed')
      );

      const { result } = renderHook(() => useResetSettingsToDefaults(), { wrapper });

      try {
        await result.current.mutateAsync(['gpu_memory_max_percent']);
      } catch {
        // Expected to throw
      }

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          type: 'error',
          message: 'Reset failed',
        });
      });
    });
  });
});
