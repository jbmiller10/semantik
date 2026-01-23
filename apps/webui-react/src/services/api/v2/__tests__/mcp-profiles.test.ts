import { describe, it, expect, vi, beforeEach } from 'vitest';
import { AxiosError, type AxiosResponse } from 'axios';
import { mcpProfilesApi } from '../mcp-profiles';
import type { MCPProfile, MCPProfileListResponse, MCPClientConfig } from '../../../../types/mcp-profile';

// Mock apiClient
vi.mock('../client', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
  },
}));

import apiClient from '../client';

// Mock data
const mockProfile: MCPProfile = {
  id: 'profile-123',
  name: 'test-profile',
  description: 'A test profile',
  enabled: true,
  search_type: 'semantic',
  result_count: 10,
  use_reranker: true,
  score_threshold: null,
  hybrid_alpha: null,
  search_mode: 'dense',
  rrf_k: null,
  use_hyde: false,
  collections: [
    { id: 'col-1', name: 'Collection 1' },
    { id: 'col-2', name: 'Collection 2' },
  ],
  created_at: '2025-01-01T00:00:00Z',
  updated_at: '2025-01-01T00:00:00Z',
};

const mockListResponse: MCPProfileListResponse = {
  profiles: [mockProfile],
  total: 1,
};

const mockConfig: MCPClientConfig = {
  transport: 'stdio',
  server_name: 'semantik-test-profile',
  command: 'semantik-mcp',
  args: ['serve', '--profile', 'test-profile'],
  env: {
    SEMANTIK_WEBUI_URL: 'http://localhost:8080',
    SEMANTIK_AUTH_TOKEN: '<your-access-token-or-api-key>',
  },
};

describe('mcpProfilesApi', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('list', () => {
    it('should list all profiles without filter', async () => {
      vi.mocked(apiClient.get).mockResolvedValueOnce({
        data: mockListResponse,
      } as AxiosResponse<MCPProfileListResponse>);

      const response = await mcpProfilesApi.list();

      expect(apiClient.get).toHaveBeenCalledWith('/api/v2/mcp/profiles', {
        params: undefined,
      });
      expect(response.data).toEqual(mockListResponse);
    });

    it('should list profiles with enabled filter set to true', async () => {
      vi.mocked(apiClient.get).mockResolvedValueOnce({
        data: mockListResponse,
      } as AxiosResponse<MCPProfileListResponse>);

      const response = await mcpProfilesApi.list(true);

      expect(apiClient.get).toHaveBeenCalledWith('/api/v2/mcp/profiles', {
        params: { enabled: true },
      });
      expect(response.data).toEqual(mockListResponse);
    });

    it('should list profiles with enabled filter set to false', async () => {
      vi.mocked(apiClient.get).mockResolvedValueOnce({
        data: mockListResponse,
      } as AxiosResponse<MCPProfileListResponse>);

      const response = await mcpProfilesApi.list(false);

      expect(apiClient.get).toHaveBeenCalledWith('/api/v2/mcp/profiles', {
        params: { enabled: false },
      });
      expect(response.data).toEqual(mockListResponse);
    });

    it('should handle API errors on list', async () => {
      const error = new AxiosError('Network Error');
      vi.mocked(apiClient.get).mockRejectedValueOnce(error);

      await expect(mcpProfilesApi.list()).rejects.toThrow('Network Error');
    });
  });

  describe('get', () => {
    it('should get a specific profile by ID', async () => {
      vi.mocked(apiClient.get).mockResolvedValueOnce({
        data: mockProfile,
      } as AxiosResponse<MCPProfile>);

      const response = await mcpProfilesApi.get('profile-123');

      expect(apiClient.get).toHaveBeenCalledWith('/api/v2/mcp/profiles/profile-123');
      expect(response.data).toEqual(mockProfile);
    });

    it('should handle 404 error for non-existent profile', async () => {
      const error = new AxiosError('Not Found', '404', undefined, undefined, {
        status: 404,
        statusText: 'Not Found',
        data: { detail: 'Profile not found' },
        headers: {},
        config: {} as AxiosResponse['config'],
      } as AxiosResponse);
      vi.mocked(apiClient.get).mockRejectedValueOnce(error);

      await expect(mcpProfilesApi.get('non-existent')).rejects.toThrow();
      expect(apiClient.get).toHaveBeenCalledWith('/api/v2/mcp/profiles/non-existent');
    });
  });

  describe('create', () => {
    it('should create a new profile', async () => {
      const createData = {
        name: 'new-profile',
        description: 'A new profile',
        collection_ids: ['col-1'],
        enabled: true,
        search_type: 'semantic' as const,
        result_count: 10,
        use_reranker: true,
      };

      vi.mocked(apiClient.post).mockResolvedValueOnce({
        data: { ...mockProfile, name: 'new-profile' },
      } as AxiosResponse<MCPProfile>);

      const response = await mcpProfilesApi.create(createData);

      expect(apiClient.post).toHaveBeenCalledWith('/api/v2/mcp/profiles', createData);
      expect(response.data.name).toBe('new-profile');
    });

    it('should handle 409 conflict error for duplicate name', async () => {
      const createData = {
        name: 'existing-profile',
        description: 'Duplicate',
        collection_ids: ['col-1'],
      };

      const error = new AxiosError('Conflict', '409', undefined, undefined, {
        status: 409,
        statusText: 'Conflict',
        data: { detail: 'Profile with this name already exists' },
        headers: {},
        config: {} as AxiosResponse['config'],
      } as AxiosResponse);
      vi.mocked(apiClient.post).mockRejectedValueOnce(error);

      await expect(mcpProfilesApi.create(createData)).rejects.toThrow();
    });
  });

  describe('update', () => {
    it('should update an existing profile', async () => {
      const updateData = {
        description: 'Updated description',
        result_count: 20,
      };

      vi.mocked(apiClient.put).mockResolvedValueOnce({
        data: { ...mockProfile, ...updateData },
      } as AxiosResponse<MCPProfile>);

      const response = await mcpProfilesApi.update('profile-123', updateData);

      expect(apiClient.put).toHaveBeenCalledWith(
        '/api/v2/mcp/profiles/profile-123',
        updateData
      );
      expect(response.data.description).toBe('Updated description');
      expect(response.data.result_count).toBe(20);
    });

    it('should handle 404 error when updating non-existent profile', async () => {
      const error = new AxiosError('Not Found', '404', undefined, undefined, {
        status: 404,
        statusText: 'Not Found',
        data: { detail: 'Profile not found' },
        headers: {},
        config: {} as AxiosResponse['config'],
      } as AxiosResponse);
      vi.mocked(apiClient.put).mockRejectedValueOnce(error);

      await expect(
        mcpProfilesApi.update('non-existent', { description: 'test' })
      ).rejects.toThrow();
    });

    it('should handle 409 conflict when name already exists', async () => {
      const error = new AxiosError('Conflict', '409', undefined, undefined, {
        status: 409,
        statusText: 'Conflict',
        data: { detail: 'Profile name already in use' },
        headers: {},
        config: {} as AxiosResponse['config'],
      } as AxiosResponse);
      vi.mocked(apiClient.put).mockRejectedValueOnce(error);

      await expect(
        mcpProfilesApi.update('profile-123', { name: 'existing-name' })
      ).rejects.toThrow();
    });
  });

  describe('delete', () => {
    it('should delete a profile', async () => {
      vi.mocked(apiClient.delete).mockResolvedValueOnce({
        data: undefined,
        status: 204,
      } as AxiosResponse<void>);

      await mcpProfilesApi.delete('profile-123');

      expect(apiClient.delete).toHaveBeenCalledWith('/api/v2/mcp/profiles/profile-123');
    });

    it('should handle 404 error when deleting non-existent profile', async () => {
      const error = new AxiosError('Not Found', '404', undefined, undefined, {
        status: 404,
        statusText: 'Not Found',
        data: { detail: 'Profile not found' },
        headers: {},
        config: {} as AxiosResponse['config'],
      } as AxiosResponse);
      vi.mocked(apiClient.delete).mockRejectedValueOnce(error);

      await expect(mcpProfilesApi.delete('non-existent')).rejects.toThrow();
    });
  });

  describe('getConfig', () => {
    it('should get config with default stdio transport', async () => {
      vi.mocked(apiClient.get).mockResolvedValueOnce({
        data: mockConfig,
      } as AxiosResponse<MCPClientConfig>);

      const response = await mcpProfilesApi.getConfig('profile-123');

      expect(apiClient.get).toHaveBeenCalledWith('/api/v2/mcp/profiles/profile-123/config', {
        params: { transport: 'stdio' },
      });
      expect(response.data).toEqual(mockConfig);
    });

    it('should get config with http transport', async () => {
      const httpConfig: MCPClientConfig = {
        transport: 'http',
        server_name: 'semantik-test-profile',
        url: 'http://localhost:9090/mcp',
      };

      vi.mocked(apiClient.get).mockResolvedValueOnce({
        data: httpConfig,
      } as AxiosResponse<MCPClientConfig>);

      const response = await mcpProfilesApi.getConfig('profile-123', 'http');

      expect(apiClient.get).toHaveBeenCalledWith('/api/v2/mcp/profiles/profile-123/config', {
        params: { transport: 'http' },
      });
      expect(response.data).toEqual(httpConfig);
      expect(response.data.transport).toBe('http');
    });

    it('should handle 404 error when profile not found', async () => {
      const error = new AxiosError('Not Found', '404', undefined, undefined, {
        status: 404,
        statusText: 'Not Found',
        data: { detail: 'Profile not found' },
        headers: {},
        config: {} as AxiosResponse['config'],
      } as AxiosResponse);
      vi.mocked(apiClient.get).mockRejectedValueOnce(error);

      await expect(mcpProfilesApi.getConfig('non-existent')).rejects.toThrow();
    });
  });
});
