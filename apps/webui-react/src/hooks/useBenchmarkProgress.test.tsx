import type { ReactNode } from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act, waitFor } from '@/tests/utils/test-utils';
import { AllTheProviders } from '@/tests/utils/providers';
import { createTestQueryClient } from '@/tests/utils/queryClient';
import { useAuthStore } from '@/stores/authStore';
import { useBenchmarkProgress } from './useBenchmarkProgress';
import { useWebSocket } from './useWebSocket';
import { operationsV2Api } from '@/services/api/v2/operations';

vi.mock('./useWebSocket', () => ({
  useWebSocket: vi.fn(),
}));

vi.mock('@/services/api/v2/operations', () => ({
  operationsV2Api: {
    getGlobalWebSocketConnectionInfo: vi.fn(),
  },
}));

const createWrapper = () => {
  const queryClient = createTestQueryClient();
  const Wrapper = ({ children }: { children: ReactNode }) => (
    <AllTheProviders queryClient={queryClient}>{children}</AllTheProviders>
  );
  return { wrapper: Wrapper, queryClient };
};

describe('useBenchmarkProgress', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useAuthStore.setState({ token: 't', refreshToken: null, user: null });
    vi.mocked(operationsV2Api.getGlobalWebSocketConnectionInfo).mockReturnValue({
      url: 'ws://example.test/ws/operations',
      protocols: ['access_token.t'],
    });
  });

  it('processes benchmark progress and completion messages', async () => {
    const onComplete = vi.fn();
    const onError = vi.fn();

    let onMessage: ((event: MessageEvent) => void) | undefined;

    vi.mocked(useWebSocket).mockImplementation((_url, options) => {
      onMessage = options?.onMessage;
      return { readyState: WebSocket.OPEN } as never;
    });

    const { wrapper, queryClient } = createWrapper();
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    const { result } = renderHook(
      () => useBenchmarkProgress('bench-1', 'op-1', { onComplete, onError }),
      { wrapper }
    );

    // Ignore unrelated benchmark messages
    act(() => {
      onMessage?.(
        new MessageEvent('message', {
          data: JSON.stringify({ type: 'benchmark_progress', data: { benchmark_id: 'bench-x', total_runs: 5 } }),
        })
      );
    });
    expect(result.current.progress.totalRuns).toBe(0);

    act(() => {
      onMessage?.(
        new MessageEvent('message', {
          data: JSON.stringify({
            type: 'benchmark_progress',
            data: {
              benchmark_id: 'bench-1',
              total_runs: 2,
              completed_runs: 0,
              failed_runs: 0,
              primary_k: 10,
              k_values_for_metrics: [10],
              status: 'running',
              stage: 'evaluating',
              current_run: { run_order: 1, total_queries: 10, completed_queries: 3, config: { search_mode: 'dense' } },
              last_completed_run: {
                run_id: 'run-1',
                run_order: 0,
                config: { search_mode: 'dense', top_k: 10 },
                metrics: { mrr: 0.5, precision: { '10': 0.6 }, recall: {}, ndcg: { '10': 0.55 } },
                timing: { search_ms: 1, rerank_ms: null, total_ms: 2 },
              },
            },
          }),
        })
      );
    });

    await waitFor(() => expect(result.current.progress.totalRuns).toBe(2));
    expect(result.current.progress.currentQueries.processed).toBe(3);
    expect(result.current.progress.recentMetrics).toHaveLength(1);
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: expect.any(Array) });

    // Generic progress update uses progress_percent
    act(() => {
      onMessage?.(
        new MessageEvent('message', {
          data: JSON.stringify({ type: 'operation_progress', data: { benchmark_id: 'bench-1', progress_percent: 50 } }),
        })
      );
    });
    await waitFor(() => expect(result.current.progress.completedRuns).toBeGreaterThanOrEqual(1));

    // Completion message triggers callbacks and invalidations
    act(() => {
      onMessage?.(
        new MessageEvent('message', {
          data: JSON.stringify({ type: 'benchmark_completed', data: { benchmark_id: 'bench-1' } }),
        })
      );
    });

    await waitFor(() => expect(result.current.progress.stage).toBe('completed'));
    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(onError).not.toHaveBeenCalled();
  });

  it('handles parse errors gracefully', async () => {
    let onMessage: ((event: MessageEvent) => void) | undefined;

    vi.mocked(useWebSocket).mockImplementation((_url, options) => {
      onMessage = options?.onMessage;
      return { readyState: WebSocket.OPEN } as never;
    });

    const { wrapper } = createWrapper();
    renderHook(() => useBenchmarkProgress('bench-1', 'op-1'), { wrapper });

    act(() => {
      onMessage?.(new MessageEvent('message', { data: '{not-json' }));
    });

    expect(console.error).toHaveBeenCalled();
  });
});
