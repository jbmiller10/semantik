import { renderHook, waitFor, act } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { AxiosError } from 'axios';
import {
  useApiKeys,
  useCreateApiKey,
  useRevokeApiKey,
  apiKeyKeys,
} from '../useApiKeys';
import { apiKeysApi } from '../../services/api/v2/api-keys';
import type {
  ApiKeyResponse,
  ApiKeyListResponse,
  ApiKeyCreateResponse,
} from '../../types/api-key';

// Mock the API module
vi.mock('../../services/api/v2/api-keys', () => ({
  apiKeysApi: {
    list: vi.fn(),
    get: vi.fn(),
    create: vi.fn(),
    update: vi.fn(),
  },
}));

// Mock the UI store
const mockAddToast = vi.fn();
vi.mock('../../stores/uiStore', () => ({
  useUIStore: () => ({
    addToast: mockAddToast,
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
const mockApiKey: ApiKeyResponse = {
  id: 'key-1-uuid',
  name: 'test-key',
  is_active: true,
  permissions: null,
  last_used_at: null,
  expires_at: '2026-01-01T00:00:00Z',
  created_at: '2025-01-01T00:00:00Z',
};

const mockApiKeys: ApiKeyResponse[] = [
  mockApiKey,
  {
    ...mockApiKey,
    id: 'key-2-uuid',
    name: 'another-key',
    is_active: false,
    last_used_at: '2025-06-01T12:00:00Z',
  },
];

const mockCreateResponse: ApiKeyCreateResponse = {
  ...mockApiKey,
  api_key: 'smtk_a1b2c3d4e5f6g7h8i9j0',
};

describe('useApiKeys hooks', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockAddToast.mockClear();
  });

  describe('apiKeyKeys', () => {
    it('should generate correct query keys', () => {
      expect(apiKeyKeys.all).toEqual(['api-keys']);
      expect(apiKeyKeys.lists()).toEqual(['api-keys', 'list']);
      expect(apiKeyKeys.list()).toEqual(['api-keys', 'list']);
      expect(apiKeyKeys.details()).toEqual(['api-keys', 'detail']);
      expect(apiKeyKeys.detail('key-1')).toEqual(['api-keys', 'detail', 'key-1']);
    });
  });

  describe('useApiKeys', () => {
    it('should fetch API keys list', async () => {
      const response: ApiKeyListResponse = {
        api_keys: mockApiKeys,
        total: 2,
      };
      vi.mocked(apiKeysApi.list).mockResolvedValue({ data: response });

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useApiKeys(), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockApiKeys);
      expect(apiKeysApi.list).toHaveBeenCalled();
    });

    it('should handle fetch error', async () => {
      vi.mocked(apiKeysApi.list).mockRejectedValue(new Error('Network error'));

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useApiKeys(), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error?.message).toBe('Network error');
    });

    it('should return empty array when no keys exist', async () => {
      const response: ApiKeyListResponse = {
        api_keys: [],
        total: 0,
      };
      vi.mocked(apiKeysApi.list).mockResolvedValue({ data: response });

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useApiKeys(), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual([]);
    });
  });

  describe('useCreateApiKey', () => {
    it('should create API key successfully', async () => {
      vi.mocked(apiKeysApi.create).mockResolvedValue({ data: mockCreateResponse });

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useCreateApiKey(), {
        wrapper: createWrapper(queryClient),
      });

      let createdKey: ApiKeyCreateResponse | undefined;
      await act(async () => {
        createdKey = await result.current.mutateAsync({
          name: 'test-key',
          expires_in_days: 30,
        });
      });

      expect(apiKeysApi.create).toHaveBeenCalledWith({
        name: 'test-key',
        expires_in_days: 30,
      });
      expect(createdKey).toEqual(mockCreateResponse);
      // No toast shown on success (caller handles showing the key)
    });

    it('should create API key without expiration', async () => {
      vi.mocked(apiKeysApi.create).mockResolvedValue({ data: mockCreateResponse });

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useCreateApiKey(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        await result.current.mutateAsync({
          name: 'never-expires',
        });
      });

      expect(apiKeysApi.create).toHaveBeenCalledWith({
        name: 'never-expires',
      });
    });

    it('should handle 409 duplicate name error', async () => {
      const axiosError = new AxiosError('Conflict');
      axiosError.response = {
        status: 409,
        data: { detail: 'API key with this name already exists' },
        statusText: 'Conflict',
        headers: {},
        config: {} as never,
      };
      vi.mocked(apiKeysApi.create).mockRejectedValue(axiosError);

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useCreateApiKey(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        try {
          await result.current.mutateAsync({
            name: 'duplicate-name',
          });
        } catch {
          // Expected to throw
        }
      });

      // 409 is handled by the form, no toast shown
      expect(mockAddToast).not.toHaveBeenCalled();
    });

    it('should handle 400 limit reached error', async () => {
      const axiosError = new AxiosError('Bad Request');
      axiosError.response = {
        status: 400,
        data: { detail: 'Maximum API keys limit reached' },
        statusText: 'Bad Request',
        headers: {},
        config: {} as never,
      };
      vi.mocked(apiKeysApi.create).mockRejectedValue(axiosError);

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useCreateApiKey(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        try {
          await result.current.mutateAsync({
            name: 'too-many-keys',
          });
        } catch {
          // Expected to throw
        }
      });

      // Wait for onError callback to be processed
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          type: 'error',
          message: 'Maximum API keys limit reached',
        });
      });
    });

    it('should handle generic create error', async () => {
      vi.mocked(apiKeysApi.create).mockRejectedValue(new Error('Network error'));

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useCreateApiKey(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        try {
          await result.current.mutateAsync({
            name: 'test-key',
          });
        } catch (error) {
          expect((error as Error).message).toBe('Network error');
        }
      });

      // Wait for onError callback to be processed
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          type: 'error',
          message: expect.any(String),
        });
      });
    });
  });

  describe('useRevokeApiKey', () => {
    it('should revoke API key successfully', async () => {
      const revokedKey = { ...mockApiKey, is_active: false };
      vi.mocked(apiKeysApi.update).mockResolvedValue({ data: revokedKey });

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useRevokeApiKey(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        await result.current.mutateAsync({
          keyId: 'key-1-uuid',
          isActive: false,
          keyName: 'test-key',
        });
      });

      expect(apiKeysApi.update).toHaveBeenCalledWith('key-1-uuid', { is_active: false });
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'success',
        message: 'API key "test-key" revoked',
      });
    });

    it('should reactivate API key successfully', async () => {
      const reactivatedKey = { ...mockApiKey, is_active: true };
      vi.mocked(apiKeysApi.update).mockResolvedValue({ data: reactivatedKey });

      const queryClient = createTestQueryClient();
      const { result } = renderHook(() => useRevokeApiKey(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        await result.current.mutateAsync({
          keyId: 'key-1-uuid',
          isActive: true,
          keyName: 'test-key',
        });
      });

      expect(apiKeysApi.update).toHaveBeenCalledWith('key-1-uuid', { is_active: true });
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'success',
        message: 'API key "test-key" reactivated',
      });
    });

    it('should perform optimistic update', async () => {
      const response: ApiKeyListResponse = {
        api_keys: mockApiKeys,
        total: 2,
      };
      vi.mocked(apiKeysApi.list).mockResolvedValue({ data: response });
      vi.mocked(apiKeysApi.update).mockImplementation(
        () => new Promise((resolve) => setTimeout(() => resolve({ data: { ...mockApiKey, is_active: false } }), 100))
      );

      const queryClient = createTestQueryClient();

      // First, populate the cache
      const { result: listResult } = renderHook(() => useApiKeys(), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(listResult.current.isSuccess).toBe(true);
      });

      // Now revoke
      const { result: revokeResult } = renderHook(() => useRevokeApiKey(), {
        wrapper: createWrapper(queryClient),
      });

      act(() => {
        revokeResult.current.mutate({
          keyId: 'key-1-uuid',
          isActive: false,
          keyName: 'test-key',
        });
      });

      // The optimistic update should happen immediately
      await waitFor(() => {
        const cachedData = queryClient.getQueryData<ApiKeyResponse[]>(apiKeyKeys.list());
        expect(cachedData?.[0]?.is_active).toBe(false);
      });
    });

    it('should rollback on error', async () => {
      const response: ApiKeyListResponse = {
        api_keys: mockApiKeys,
        total: 2,
      };
      vi.mocked(apiKeysApi.list).mockResolvedValue({ data: response });
      vi.mocked(apiKeysApi.update).mockRejectedValue(new Error('Update failed'));

      const queryClient = createTestQueryClient();

      // First, populate the cache
      const { result: listResult } = renderHook(() => useApiKeys(), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(listResult.current.isSuccess).toBe(true);
      });

      // Now try to revoke (will fail)
      const { result: revokeResult } = renderHook(() => useRevokeApiKey(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        try {
          await revokeResult.current.mutateAsync({
            keyId: 'key-1-uuid',
            isActive: false,
            keyName: 'test-key',
          });
        } catch {
          // Expected to throw
        }
      });

      // Cache should be rolled back to original state
      const cachedData = queryClient.getQueryData<ApiKeyResponse[]>(apiKeyKeys.list());
      expect(cachedData?.[0]?.is_active).toBe(true);
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'error',
        message: expect.any(String),
      });
    });
  });
});
