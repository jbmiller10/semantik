import { renderHook, waitFor, act } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import {
  usePlugins,
  usePlugin,
  usePluginManifest,
  usePluginConfigSchema,
  usePluginHealth,
  useEnablePlugin,
  useDisablePlugin,
  useUpdatePluginConfig,
  useRefreshPluginHealth,
  pluginKeys,
} from '../usePlugins';
import { pluginsApi } from '../../services/api/v2/plugins';
import type {
  PluginInfo,
  PluginManifest,
  PluginConfigSchema,
  PluginStatusResponse,
  PluginHealthResponse,
  PluginType,
  HealthStatus,
} from '../../types/plugin';
import type { MockAxiosResponse } from '../../tests/types/test-types';

// Mock the API module
vi.mock('../../services/api/v2/plugins', () => ({
  pluginsApi: {
    list: vi.fn(),
    get: vi.fn(),
    getManifest: vi.fn(),
    getConfigSchema: vi.fn(),
    checkHealth: vi.fn(),
    enable: vi.fn(),
    disable: vi.fn(),
    updateConfig: vi.fn(),
  },
}));

// Test helpers
const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        staleTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  });

const createWrapper = (queryClient: QueryClient) => {
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

// Mock data
const mockPlugin: PluginInfo = {
  id: 'test-plugin',
  type: 'embedding' as PluginType,
  version: '1.0.0',
  manifest: {
    id: 'test-plugin',
    type: 'embedding' as PluginType,
    version: '1.0.0',
    display_name: 'Test Plugin',
    description: 'A test plugin',
    author: 'Test Author',
    license: 'MIT',
    homepage: null,
    requires: [],
    capabilities: {},
  },
  enabled: true,
  config: { api_key: 'test-key' },
  health_status: 'healthy' as HealthStatus,
  last_health_check: '2025-01-01T00:00:00Z',
  error_message: null,
  requires_restart: false,
};

const mockPlugins: PluginInfo[] = [
  mockPlugin,
  {
    ...mockPlugin,
    id: 'test-plugin-2',
    manifest: { ...mockPlugin.manifest, id: 'test-plugin-2', display_name: 'Test Plugin 2' },
    enabled: false,
  },
];

const mockManifest: PluginManifest = mockPlugin.manifest;

const mockSchema: PluginConfigSchema = {
  type: 'object',
  properties: {
    api_key: { type: 'string', title: 'API Key' },
    timeout: { type: 'number', title: 'Timeout' },
  },
  required: ['api_key'],
};

const mockHealthResponse: PluginHealthResponse = {
  plugin_id: 'test-plugin',
  health_status: 'healthy',
  last_health_check: '2025-01-01T00:00:00Z',
  error_message: null,
};

const mockStatusResponse: PluginStatusResponse = {
  plugin_id: 'test-plugin',
  enabled: true,
  requires_restart: true,
};

describe('usePlugins hooks', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    vi.clearAllMocks();
    queryClient = createTestQueryClient();
  });

  describe('pluginKeys', () => {
    it('generates correct query keys', () => {
      expect(pluginKeys.all).toEqual(['plugins']);
      expect(pluginKeys.list()).toEqual(['plugins', 'list', undefined]);
      expect(pluginKeys.list({ type: 'embedding' })).toEqual([
        'plugins',
        'list',
        { type: 'embedding' },
      ]);
      expect(pluginKeys.detail('test-plugin')).toEqual(['plugins', 'detail', 'test-plugin']);
      expect(pluginKeys.manifest('test-plugin')).toEqual(['plugins', 'manifest', 'test-plugin']);
      expect(pluginKeys.configSchema('test-plugin')).toEqual([
        'plugins',
        'config-schema',
        'test-plugin',
      ]);
      expect(pluginKeys.health('test-plugin')).toEqual(['plugins', 'health', 'test-plugin']);
    });
  });

  describe('usePlugins', () => {
    it('fetches and returns plugin list', async () => {
      vi.mocked(pluginsApi.list).mockResolvedValue({
        data: { plugins: mockPlugins },
      } as MockAxiosResponse<{ plugins: PluginInfo[] }>);

      const { result } = renderHook(() => usePlugins(), {
        wrapper: createWrapper(queryClient),
      });

      expect(result.current.isLoading).toBe(true);

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.data).toEqual(mockPlugins);
      expect(pluginsApi.list).toHaveBeenCalledWith(undefined);
    });

    it('passes filters to the API', async () => {
      vi.mocked(pluginsApi.list).mockResolvedValue({
        data: { plugins: [mockPlugin] },
      } as MockAxiosResponse<{ plugins: PluginInfo[] }>);

      const filters = { type: 'embedding' as PluginType, enabled: true };

      const { result } = renderHook(() => usePlugins(filters), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(pluginsApi.list).toHaveBeenCalledWith(filters);
    });

    it('handles fetch error', async () => {
      const error = new Error('Network error');
      vi.mocked(pluginsApi.list).mockRejectedValue(error);

      const { result } = renderHook(() => usePlugins(), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error).toBe(error);
    });
  });

  describe('usePlugin', () => {
    it('fetches a single plugin by id', async () => {
      vi.mocked(pluginsApi.get).mockResolvedValue({
        data: mockPlugin,
      } as MockAxiosResponse<PluginInfo>);

      const { result } = renderHook(() => usePlugin('test-plugin'), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockPlugin);
      expect(pluginsApi.get).toHaveBeenCalledWith('test-plugin');
    });

    it('does not fetch when id is empty', () => {
      const { result } = renderHook(() => usePlugin(''), {
        wrapper: createWrapper(queryClient),
      });

      expect(result.current.fetchStatus).toBe('idle');
      expect(pluginsApi.get).not.toHaveBeenCalled();
    });
  });

  describe('usePluginManifest', () => {
    it('fetches plugin manifest', async () => {
      vi.mocked(pluginsApi.getManifest).mockResolvedValue({
        data: mockManifest,
      } as MockAxiosResponse<PluginManifest>);

      const { result } = renderHook(() => usePluginManifest('test-plugin'), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockManifest);
      expect(pluginsApi.getManifest).toHaveBeenCalledWith('test-plugin');
    });

    it('does not fetch when id is empty', () => {
      const { result } = renderHook(() => usePluginManifest(''), {
        wrapper: createWrapper(queryClient),
      });

      expect(result.current.fetchStatus).toBe('idle');
      expect(pluginsApi.getManifest).not.toHaveBeenCalled();
    });
  });

  describe('usePluginConfigSchema', () => {
    it('fetches plugin config schema', async () => {
      vi.mocked(pluginsApi.getConfigSchema).mockResolvedValue({
        data: mockSchema,
      } as MockAxiosResponse<PluginConfigSchema>);

      const { result } = renderHook(() => usePluginConfigSchema('test-plugin'), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockSchema);
      expect(pluginsApi.getConfigSchema).toHaveBeenCalledWith('test-plugin');
    });

    it('does not fetch when id is empty', () => {
      const { result } = renderHook(() => usePluginConfigSchema(''), {
        wrapper: createWrapper(queryClient),
      });

      expect(result.current.fetchStatus).toBe('idle');
      expect(pluginsApi.getConfigSchema).not.toHaveBeenCalled();
    });
  });

  describe('usePluginHealth', () => {
    it('fetches plugin health status', async () => {
      vi.mocked(pluginsApi.checkHealth).mockResolvedValue({
        data: mockHealthResponse,
      } as MockAxiosResponse<PluginHealthResponse>);

      const { result } = renderHook(() => usePluginHealth('test-plugin'), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockHealthResponse);
      expect(pluginsApi.checkHealth).toHaveBeenCalledWith('test-plugin');
    });

    it('does not fetch when id is empty', () => {
      const { result } = renderHook(() => usePluginHealth(''), {
        wrapper: createWrapper(queryClient),
      });

      expect(result.current.fetchStatus).toBe('idle');
      expect(pluginsApi.checkHealth).not.toHaveBeenCalled();
    });
  });

  describe('useEnablePlugin', () => {
    it('enables a plugin and invalidates queries', async () => {
      vi.mocked(pluginsApi.enable).mockResolvedValue({
        data: mockStatusResponse,
      } as MockAxiosResponse<PluginStatusResponse>);

      const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

      const { result } = renderHook(() => useEnablePlugin(), {
        wrapper: createWrapper(queryClient),
      });

      let response: PluginStatusResponse | undefined;
      await act(async () => {
        response = await result.current.mutateAsync('test-plugin');
      });

      expect(response).toEqual(mockStatusResponse);
      expect(pluginsApi.enable).toHaveBeenCalledWith('test-plugin');
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: pluginKeys.all });
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: pluginKeys.detail('test-plugin') });
    });

    it('handles enable error', async () => {
      const error = new Error('Failed to enable');
      vi.mocked(pluginsApi.enable).mockRejectedValue(error);

      const { result } = renderHook(() => useEnablePlugin(), {
        wrapper: createWrapper(queryClient),
      });

      await expect(
        act(async () => {
          await result.current.mutateAsync('test-plugin');
        })
      ).rejects.toThrow('Failed to enable');
    });
  });

  describe('useDisablePlugin', () => {
    it('disables a plugin and invalidates queries', async () => {
      vi.mocked(pluginsApi.disable).mockResolvedValue({
        data: { ...mockStatusResponse, enabled: false },
      } as MockAxiosResponse<PluginStatusResponse>);

      const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

      const { result } = renderHook(() => useDisablePlugin(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        await result.current.mutateAsync('test-plugin');
      });

      expect(pluginsApi.disable).toHaveBeenCalledWith('test-plugin');
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: pluginKeys.all });
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: pluginKeys.detail('test-plugin') });
    });
  });

  describe('useUpdatePluginConfig', () => {
    it('updates plugin config and updates cache', async () => {
      const updatedPlugin = { ...mockPlugin, config: { api_key: 'new-key' } };
      vi.mocked(pluginsApi.updateConfig).mockResolvedValue({
        data: updatedPlugin,
      } as MockAxiosResponse<PluginInfo>);

      const setQueryDataSpy = vi.spyOn(queryClient, 'setQueryData');
      const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

      const { result } = renderHook(() => useUpdatePluginConfig(), {
        wrapper: createWrapper(queryClient),
      });

      let response: PluginInfo | undefined;
      await act(async () => {
        response = await result.current.mutateAsync({
          pluginId: 'test-plugin',
          config: { api_key: 'new-key' },
        });
      });

      expect(response).toEqual(updatedPlugin);
      expect(pluginsApi.updateConfig).toHaveBeenCalledWith('test-plugin', { api_key: 'new-key' });
      expect(setQueryDataSpy).toHaveBeenCalledWith(
        pluginKeys.detail('test-plugin'),
        updatedPlugin
      );
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: pluginKeys.list() });
    });

    it('handles update error', async () => {
      const error = new Error('Failed to update config');
      vi.mocked(pluginsApi.updateConfig).mockRejectedValue(error);

      const { result } = renderHook(() => useUpdatePluginConfig(), {
        wrapper: createWrapper(queryClient),
      });

      await expect(
        act(async () => {
          await result.current.mutateAsync({
            pluginId: 'test-plugin',
            config: { api_key: 'new-key' },
          });
        })
      ).rejects.toThrow('Failed to update config');
    });
  });

  describe('useRefreshPluginHealth', () => {
    it('refreshes health and updates caches', async () => {
      vi.mocked(pluginsApi.checkHealth).mockResolvedValue({
        data: mockHealthResponse,
      } as MockAxiosResponse<PluginHealthResponse>);

      const setQueryDataSpy = vi.spyOn(queryClient, 'setQueryData');
      const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

      const { result } = renderHook(() => useRefreshPluginHealth(), {
        wrapper: createWrapper(queryClient),
      });

      let response: PluginHealthResponse | undefined;
      await act(async () => {
        response = await result.current.mutateAsync('test-plugin');
      });

      expect(response).toEqual(mockHealthResponse);
      expect(pluginsApi.checkHealth).toHaveBeenCalledWith('test-plugin');
      expect(setQueryDataSpy).toHaveBeenCalledWith(
        pluginKeys.health('test-plugin'),
        mockHealthResponse
      );
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: pluginKeys.detail('test-plugin') });
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: pluginKeys.list() });
    });
  });
});
