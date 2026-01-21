import { renderHook, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';

import { useSparseReindexProgress } from '../useSparseIndex';
import { sparseIndexApi, sparseIndexKeys } from '../../services/api/v2/sparse-index';
import { createMockAxiosResponse } from '../../tests/types/test-types';

vi.mock('../../services/api/v2/sparse-index', () => ({
  sparseIndexApi: {
    getStatus: vi.fn(),
    enable: vi.fn(),
    disable: vi.fn(),
    triggerReindex: vi.fn(),
    getReindexProgress: vi.fn(),
  },
  sparseIndexKeys: {
    all: ['sparse-index'],
    status: (collectionUuid: string) => ['sparse-index', 'status', collectionUuid],
    reindexProgress: (collectionUuid: string, jobId: string) => [
      'sparse-index',
      'reindex',
      collectionUuid,
      jobId,
    ],
  },
}));

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

describe('useSparseReindexProgress', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('normalizes Celery task states to the UI reindex statuses', async () => {
    const queryClient = createTestQueryClient();

    vi.mocked(sparseIndexApi.getReindexProgress).mockResolvedValue(
      createMockAxiosResponse({
        job_id: 'job-1',
        status: 'PENDING',
        progress: 0,
      })
    );

    const { result } = renderHook(() => useSparseReindexProgress('col-1', 'job-1'), {
      wrapper: createWrapper(queryClient),
    });

    await waitFor(() => {
      expect(result.current.data?.status).toBe('pending');
    });

    const query = queryClient
      .getQueryCache()
      .find(sparseIndexKeys.reindexProgress('col-1', 'job-1'));
    expect(query).toBeTruthy();

    const refetchInterval = query?.options.refetchInterval;
    expect(typeof refetchInterval).toBe('function');
    if (typeof refetchInterval === 'function' && query) {
      expect(refetchInterval(query)).toBe(2000);
    }
  });

  it('stops polling when the job reaches SUCCESS', async () => {
    const queryClient = createTestQueryClient();

    vi.mocked(sparseIndexApi.getReindexProgress).mockResolvedValue(
      createMockAxiosResponse({
        job_id: 'job-1',
        status: 'SUCCESS',
        progress: 100,
      })
    );

    const { result } = renderHook(() => useSparseReindexProgress('col-1', 'job-1'), {
      wrapper: createWrapper(queryClient),
    });

    await waitFor(() => {
      expect(result.current.data?.status).toBe('completed');
    });

    const query = queryClient
      .getQueryCache()
      .find(sparseIndexKeys.reindexProgress('col-1', 'job-1'));
    expect(query).toBeTruthy();

    const refetchInterval = query?.options.refetchInterval;
    expect(typeof refetchInterval).toBe('function');
    if (typeof refetchInterval === 'function' && query) {
      expect(refetchInterval(query)).toBe(false);
    }
  });
});

