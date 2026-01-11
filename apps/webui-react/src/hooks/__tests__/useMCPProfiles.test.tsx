import { renderHook, waitFor, act } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import {
  useMCPProfiles,
  useMCPProfile,
  useMCPProfileConfig,
  useCreateMCPProfile,
  useUpdateMCPProfile,
  useDeleteMCPProfile,
  useToggleMCPProfileEnabled,
  mcpProfileKeys,
} from '../useMCPProfiles';
import { mcpProfilesApi } from '../../services/api/v2/mcp-profiles';
import type {
  MCPProfile,
  MCPProfileListResponse,
  MCPClientConfig,
} from '../../types/mcp-profile';

// Mock the API module
vi.mock('../../services/api/v2/mcp-profiles', () => ({
  mcpProfilesApi: {
    list: vi.fn(),
    get: vi.fn(),
    getConfig: vi.fn(),
    create: vi.fn(),
    update: vi.fn(),
    delete: vi.fn(),
  },
}));

// Mock the UI store
vi.mock('../../stores/uiStore', () => ({
  useUIStore: () => ({
    addToast: vi.fn(),
  }),
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
const mockProfile: MCPProfile = {
  id: 'profile-1',
  name: 'test-profile',
  description: 'A test profile for searching',
  enabled: true,
  search_type: 'semantic',
  result_count: 10,
  use_reranker: true,
  score_threshold: null,
  hybrid_alpha: null,
  collections: [
    { id: 'col-1', name: 'Collection 1' },
    { id: 'col-2', name: 'Collection 2' },
  ],
  created_at: '2025-01-01T00:00:00Z',
  updated_at: '2025-01-01T00:00:00Z',
};

const mockProfiles: MCPProfile[] = [
  mockProfile,
  {
    ...mockProfile,
    id: 'profile-2',
    name: 'another-profile',
    enabled: false,
  },
];

const mockConfig: MCPClientConfig = {
  server_name: 'semantik-test-profile',
  command: 'npx',
  args: ['-y', '@anthropic/mcp-server-semantik', '--profile', 'test-profile'],
  env: {
    SEMANTIK_API_URL: 'http://localhost:8000',
    SEMANTIK_API_KEY: '<your-access-token-or-api-key>',
  },
};

describe('useMCPProfiles hooks', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('mcpProfileKeys', () => {
    it('should generate correct query keys', () => {
      expect(mcpProfileKeys.all).toEqual(['mcp-profiles']);
      expect(mcpProfileKeys.lists()).toEqual(['mcp-profiles', 'list']);
      expect(mcpProfileKeys.list(true)).toEqual(['mcp-profiles', 'list', { enabled: true }]);
      expect(mcpProfileKeys.list(undefined)).toEqual(['mcp-profiles', 'list', { enabled: undefined }]);
      expect(mcpProfileKeys.details()).toEqual(['mcp-profiles', 'detail']);
      expect(mcpProfileKeys.detail('profile-1')).toEqual(['mcp-profiles', 'detail', 'profile-1']);
      expect(mcpProfileKeys.configs()).toEqual(['mcp-profiles', 'config']);
      expect(mcpProfileKeys.config('profile-1')).toEqual(['mcp-profiles', 'config', 'profile-1']);
    });
  });

  describe('useMCPProfiles', () => {
    it('should fetch profiles list', async () => {
      const response: MCPProfileListResponse = {
        profiles: mockProfiles,
        total: 2,
      };
      vi.mocked(mcpProfilesApi.list).mockResolvedValue({ data: response });

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useMCPProfiles(), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockProfiles);
      expect(mcpProfilesApi.list).toHaveBeenCalledWith(undefined);
    });

    it('should filter by enabled state', async () => {
      const response: MCPProfileListResponse = {
        profiles: [mockProfile],
        total: 1,
      };
      vi.mocked(mcpProfilesApi.list).mockResolvedValue({ data: response });

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useMCPProfiles(true), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(mcpProfilesApi.list).toHaveBeenCalledWith(true);
    });

    it('should handle fetch error', async () => {
      vi.mocked(mcpProfilesApi.list).mockRejectedValue(new Error('Network error'));

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useMCPProfiles(), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error?.message).toBe('Network error');
    });
  });

  describe('useMCPProfile', () => {
    it('should fetch single profile', async () => {
      vi.mocked(mcpProfilesApi.get).mockResolvedValue({ data: mockProfile });

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useMCPProfile('profile-1'), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockProfile);
      expect(mcpProfilesApi.get).toHaveBeenCalledWith('profile-1');
    });

    it('should not fetch when profileId is empty', async () => {
      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useMCPProfile(''), {
        wrapper: createWrapper(queryClient),
      });

      expect(result.current.fetchStatus).toBe('idle');
      expect(mcpProfilesApi.get).not.toHaveBeenCalled();
    });
  });

  describe('useMCPProfileConfig', () => {
    it('should fetch profile config', async () => {
      vi.mocked(mcpProfilesApi.getConfig).mockResolvedValue({ data: mockConfig });

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useMCPProfileConfig('profile-1'), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockConfig);
      expect(mcpProfilesApi.getConfig).toHaveBeenCalledWith('profile-1');
    });
  });

  describe('useCreateMCPProfile', () => {
    it('should create profile successfully', async () => {
      vi.mocked(mcpProfilesApi.create).mockResolvedValue({ data: mockProfile });

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useCreateMCPProfile(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        await result.current.mutateAsync({
          name: 'test-profile',
          description: 'A test profile',
          collection_ids: ['col-1', 'col-2'],
        });
      });

      expect(mcpProfilesApi.create).toHaveBeenCalledWith({
        name: 'test-profile',
        description: 'A test profile',
        collection_ids: ['col-1', 'col-2'],
      });
    });

    it('should handle create error', async () => {
      vi.mocked(mcpProfilesApi.create).mockRejectedValue(new Error('Creation failed'));

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useCreateMCPProfile(), {
        wrapper: createWrapper(queryClient),
      });

      await expect(
        act(async () => {
          await result.current.mutateAsync({
            name: 'test-profile',
            description: 'A test profile',
            collection_ids: ['col-1'],
          });
        })
      ).rejects.toThrow('Creation failed');
    });
  });

  describe('useUpdateMCPProfile', () => {
    it('should update profile successfully', async () => {
      const updatedProfile = { ...mockProfile, name: 'updated-profile' };
      vi.mocked(mcpProfilesApi.update).mockResolvedValue({ data: updatedProfile });

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useUpdateMCPProfile(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        await result.current.mutateAsync({
          profileId: 'profile-1',
          data: { name: 'updated-profile' },
        });
      });

      expect(mcpProfilesApi.update).toHaveBeenCalledWith('profile-1', { name: 'updated-profile' });
    });
  });

  describe('useDeleteMCPProfile', () => {
    it('should delete profile successfully', async () => {
      vi.mocked(mcpProfilesApi.delete).mockResolvedValue({ data: undefined });

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useDeleteMCPProfile(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        await result.current.mutateAsync({
          profileId: 'profile-1',
          profileName: 'test-profile',
        });
      });

      expect(mcpProfilesApi.delete).toHaveBeenCalledWith('profile-1');
    });
  });

  describe('useToggleMCPProfileEnabled', () => {
    it('should toggle profile enabled state', async () => {
      const toggledProfile = { ...mockProfile, enabled: false };
      vi.mocked(mcpProfilesApi.update).mockResolvedValue({ data: toggledProfile });

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useToggleMCPProfileEnabled(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        await result.current.mutateAsync({
          profileId: 'profile-1',
          enabled: false,
          profileName: 'test-profile',
        });
      });

      expect(mcpProfilesApi.update).toHaveBeenCalledWith('profile-1', { enabled: false });
    });

    it('should perform optimistic update', async () => {
      const response: MCPProfileListResponse = {
        profiles: mockProfiles,
        total: 2,
      };
      vi.mocked(mcpProfilesApi.list).mockResolvedValue({ data: response });
      vi.mocked(mcpProfilesApi.update).mockImplementation(
        () => new Promise((resolve) => setTimeout(() => resolve({ data: { ...mockProfile, enabled: false } }), 100))
      );

      const queryClient = createTestQueryClient();

      // First, populate the cache
      const { result: listResult } = renderHook(() => useMCPProfiles(), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(listResult.current.isSuccess).toBe(true);
      });

      // Now toggle
      const { result: toggleResult } = renderHook(() => useToggleMCPProfileEnabled(), {
        wrapper: createWrapper(queryClient),
      });

      act(() => {
        toggleResult.current.mutate({
          profileId: 'profile-1',
          enabled: false,
          profileName: 'test-profile',
        });
      });

      // The optimistic update should happen immediately
      await waitFor(() => {
        const cachedData = queryClient.getQueryData<MCPProfile[]>(mcpProfileKeys.list(undefined));
        expect(cachedData?.[0]?.enabled).toBe(false);
      });
    });
  });
});
