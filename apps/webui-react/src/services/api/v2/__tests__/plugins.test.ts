import { describe, it, expect, vi, beforeEach } from 'vitest';
import { pluginsApi } from '../plugins';
import type {
  PluginInfo,
  PluginListResponse,
  PluginManifest,
  PluginConfigSchema,
  PluginStatusResponse,
  PluginHealthResponse,
  PluginType,
  HealthStatus,
} from '../../../../types/plugin';

// Mock apiClient
vi.mock('../client', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
  },
}));

import apiClient from '../client';

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

const mockManifest: PluginManifest = mockPlugin.manifest;

const mockSchema: PluginConfigSchema = {
  type: 'object',
  properties: {
    api_key: { type: 'string', title: 'API Key' },
  },
  required: ['api_key'],
};

const mockStatusResponse: PluginStatusResponse = {
  plugin_id: 'test-plugin',
  enabled: true,
  requires_restart: true,
};

const mockHealthResponse: PluginHealthResponse = {
  plugin_id: 'test-plugin',
  health_status: 'healthy',
  last_health_check: '2025-01-01T00:00:00Z',
  error_message: null,
};

describe('pluginsApi', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('list', () => {
    it('should list all plugins without filters', async () => {
      const mockResponse: PluginListResponse = { plugins: [mockPlugin] };
      vi.mocked(apiClient.get).mockResolvedValueOnce({ data: mockResponse });

      const result = await pluginsApi.list();

      expect(apiClient.get).toHaveBeenCalledWith('/api/v2/plugins', {
        params: undefined,
      });
      expect(result.data).toEqual(mockResponse);
    });

    it('should list plugins with type filter', async () => {
      const mockResponse: PluginListResponse = { plugins: [mockPlugin] };
      vi.mocked(apiClient.get).mockResolvedValueOnce({ data: mockResponse });

      const filters = { type: 'embedding' as PluginType };
      const result = await pluginsApi.list(filters);

      expect(apiClient.get).toHaveBeenCalledWith('/api/v2/plugins', {
        params: filters,
      });
      expect(result.data).toEqual(mockResponse);
    });

    it('should list plugins with enabled filter', async () => {
      const mockResponse: PluginListResponse = { plugins: [mockPlugin] };
      vi.mocked(apiClient.get).mockResolvedValueOnce({ data: mockResponse });

      const filters = { enabled: true };
      const result = await pluginsApi.list(filters);

      expect(apiClient.get).toHaveBeenCalledWith('/api/v2/plugins', {
        params: filters,
      });
      expect(result.data).toEqual(mockResponse);
    });

    it('should list plugins with include_health filter', async () => {
      const mockResponse: PluginListResponse = { plugins: [mockPlugin] };
      vi.mocked(apiClient.get).mockResolvedValueOnce({ data: mockResponse });

      const filters = { include_health: true };
      const result = await pluginsApi.list(filters);

      expect(apiClient.get).toHaveBeenCalledWith('/api/v2/plugins', {
        params: filters,
      });
      expect(result.data).toEqual(mockResponse);
    });

    it('should list plugins with multiple filters', async () => {
      const mockResponse: PluginListResponse = { plugins: [mockPlugin] };
      vi.mocked(apiClient.get).mockResolvedValueOnce({ data: mockResponse });

      const filters = {
        type: 'embedding' as PluginType,
        enabled: true,
        include_health: true,
      };
      const result = await pluginsApi.list(filters);

      expect(apiClient.get).toHaveBeenCalledWith('/api/v2/plugins', {
        params: filters,
      });
      expect(result.data).toEqual(mockResponse);
    });
  });

  describe('get', () => {
    it('should get a specific plugin by id', async () => {
      vi.mocked(apiClient.get).mockResolvedValueOnce({ data: mockPlugin });

      const result = await pluginsApi.get('test-plugin');

      expect(apiClient.get).toHaveBeenCalledWith('/api/v2/plugins/test-plugin');
      expect(result.data).toEqual(mockPlugin);
    });
  });

  describe('getManifest', () => {
    it('should get plugin manifest', async () => {
      vi.mocked(apiClient.get).mockResolvedValueOnce({ data: mockManifest });

      const result = await pluginsApi.getManifest('test-plugin');

      expect(apiClient.get).toHaveBeenCalledWith(
        '/api/v2/plugins/test-plugin/manifest'
      );
      expect(result.data).toEqual(mockManifest);
    });
  });

  describe('getConfigSchema', () => {
    it('should get plugin config schema', async () => {
      vi.mocked(apiClient.get).mockResolvedValueOnce({ data: mockSchema });

      const result = await pluginsApi.getConfigSchema('test-plugin');

      expect(apiClient.get).toHaveBeenCalledWith(
        '/api/v2/plugins/test-plugin/config-schema'
      );
      expect(result.data).toEqual(mockSchema);
    });

    it('should return null when plugin has no config', async () => {
      vi.mocked(apiClient.get).mockResolvedValueOnce({ data: null });

      const result = await pluginsApi.getConfigSchema('no-config-plugin');

      expect(result.data).toBeNull();
    });
  });

  describe('enable', () => {
    it('should enable a plugin', async () => {
      vi.mocked(apiClient.post).mockResolvedValueOnce({ data: mockStatusResponse });

      const result = await pluginsApi.enable('test-plugin');

      expect(apiClient.post).toHaveBeenCalledWith(
        '/api/v2/plugins/test-plugin/enable'
      );
      expect(result.data).toEqual(mockStatusResponse);
    });
  });

  describe('disable', () => {
    it('should disable a plugin', async () => {
      const disabledResponse = { ...mockStatusResponse, enabled: false };
      vi.mocked(apiClient.post).mockResolvedValueOnce({ data: disabledResponse });

      const result = await pluginsApi.disable('test-plugin');

      expect(apiClient.post).toHaveBeenCalledWith(
        '/api/v2/plugins/test-plugin/disable'
      );
      expect(result.data).toEqual(disabledResponse);
    });
  });

  describe('updateConfig', () => {
    it('should update plugin configuration', async () => {
      const newConfig = { api_key: 'new-key' };
      const updatedPlugin = { ...mockPlugin, config: newConfig };
      vi.mocked(apiClient.put).mockResolvedValueOnce({ data: updatedPlugin });

      const result = await pluginsApi.updateConfig('test-plugin', newConfig);

      expect(apiClient.put).toHaveBeenCalledWith(
        '/api/v2/plugins/test-plugin/config',
        { config: newConfig }
      );
      expect(result.data).toEqual(updatedPlugin);
    });

    it('should update with complex configuration', async () => {
      const complexConfig = {
        api_key: 'my-key',
        timeout: 30,
        options: {
          model: 'gpt-4',
          temperature: 0.7,
        },
        tags: ['production', 'v2'],
      };
      const updatedPlugin = { ...mockPlugin, config: complexConfig };
      vi.mocked(apiClient.put).mockResolvedValueOnce({ data: updatedPlugin });

      const result = await pluginsApi.updateConfig('test-plugin', complexConfig);

      expect(apiClient.put).toHaveBeenCalledWith(
        '/api/v2/plugins/test-plugin/config',
        { config: complexConfig }
      );
      expect(result.data).toEqual(updatedPlugin);
    });
  });

  describe('checkHealth', () => {
    it('should check plugin health', async () => {
      vi.mocked(apiClient.get).mockResolvedValueOnce({ data: mockHealthResponse });

      const result = await pluginsApi.checkHealth('test-plugin');

      expect(apiClient.get).toHaveBeenCalledWith(
        '/api/v2/plugins/test-plugin/health'
      );
      expect(result.data).toEqual(mockHealthResponse);
    });

    it('should return unhealthy status with error message', async () => {
      const unhealthyResponse: PluginHealthResponse = {
        plugin_id: 'test-plugin',
        health_status: 'unhealthy',
        last_health_check: '2025-01-01T00:00:00Z',
        error_message: 'Connection refused',
      };
      vi.mocked(apiClient.get).mockResolvedValueOnce({ data: unhealthyResponse });

      const result = await pluginsApi.checkHealth('test-plugin');

      expect(result.data).toEqual(unhealthyResponse);
      expect(result.data.health_status).toBe('unhealthy');
      expect(result.data.error_message).toBe('Connection refused');
    });
  });

  describe('error handling', () => {
    it('should propagate errors from API calls', async () => {
      const error = new Error('Network error');
      vi.mocked(apiClient.get).mockRejectedValueOnce(error);

      await expect(pluginsApi.list()).rejects.toThrow('Network error');
    });

    it('should propagate 404 errors', async () => {
      const error = Object.assign(new Error('Not Found'), {
        response: { status: 404, data: { detail: 'Plugin not found' } },
      });
      vi.mocked(apiClient.get).mockRejectedValueOnce(error);

      await expect(pluginsApi.get('non-existent')).rejects.toThrow('Not Found');
    });

    it('should propagate validation errors on config update', async () => {
      const error = Object.assign(new Error('Validation Error'), {
        response: {
          status: 422,
          data: { detail: 'api_key is required' },
        },
      });
      vi.mocked(apiClient.put).mockRejectedValueOnce(error);

      await expect(
        pluginsApi.updateConfig('test-plugin', {})
      ).rejects.toThrow('Validation Error');
    });
  });
});
