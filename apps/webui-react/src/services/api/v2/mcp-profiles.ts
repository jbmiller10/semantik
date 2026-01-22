import apiClient from './client';
import type {
  MCPProfile,
  MCPProfileListResponse,
  MCPProfileCreate,
  MCPProfileUpdate,
  MCPClientConfig,
  MCPTransport,
} from '../../../types/mcp-profile';

/**
 * V2 MCP Profiles API client
 * Provides endpoints for managing MCP search profiles
 */
export const mcpProfilesApi = {
  /**
   * List all MCP profiles for the current user
   * @param enabled Optional filter by enabled state
   */
  list: (enabled?: boolean) =>
    apiClient.get<MCPProfileListResponse>('/api/v2/mcp/profiles', {
      params: enabled !== undefined ? { enabled } : undefined,
    }),

  /**
   * Get a specific MCP profile
   * @param profileId The profile UUID
   */
  get: (profileId: string) =>
    apiClient.get<MCPProfile>(`/api/v2/mcp/profiles/${profileId}`),

  /**
   * Create a new MCP profile
   * @param data The profile data
   */
  create: (data: MCPProfileCreate) =>
    apiClient.post<MCPProfile>('/api/v2/mcp/profiles', data),

  /**
   * Update an existing MCP profile
   * @param profileId The profile UUID
   * @param data The fields to update
   */
  update: (profileId: string, data: MCPProfileUpdate) =>
    apiClient.put<MCPProfile>(`/api/v2/mcp/profiles/${profileId}`, data),

  /**
   * Delete an MCP profile
   * @param profileId The profile UUID
   */
  delete: (profileId: string) =>
    apiClient.delete(`/api/v2/mcp/profiles/${profileId}`),

  /**
   * Get the MCP client configuration snippet for a profile
   * @param profileId The profile UUID
   * @param transport Transport type: 'stdio' for local, 'http' for Docker
   */
  getConfig: (profileId: string, transport: MCPTransport = 'stdio') =>
    apiClient.get<MCPClientConfig>(`/api/v2/mcp/profiles/${profileId}/config`, {
      params: { transport },
    }),
};
