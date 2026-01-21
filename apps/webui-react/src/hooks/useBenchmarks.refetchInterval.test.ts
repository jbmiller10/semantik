import { describe, it, expect, vi, beforeEach } from 'vitest';

const useQuery = vi.hoisted(() => vi.fn((options: unknown) => options));

vi.mock('@tanstack/react-query', () => ({
  useQuery,
  useMutation: vi.fn(),
  useQueryClient: vi.fn(),
}));

import { useBenchmark, useBenchmarks } from './useBenchmarks';

describe('useBenchmarks refetchInterval options', () => {
  beforeEach(() => {
    useQuery.mockClear();
  });

  it('auto-refetches list when any benchmark is running', () => {
    useBenchmarks();
    const options = useQuery.mock.calls[0][0] as {
      refetchInterval?: (query: { state: { data?: { benchmarks?: Array<{ status: string }> } } }) => number | false;
    };

    expect(
      options.refetchInterval?.({ state: { data: { benchmarks: [{ status: 'running' }] } } })
    ).toBe(5000);
    expect(
      options.refetchInterval?.({ state: { data: { benchmarks: [{ status: 'completed' }] } } })
    ).toBe(false);
  });

  it('auto-refetches detail when benchmark is running', () => {
    useBenchmark('bench-1');
    const options = useQuery.mock.calls[0][0] as {
      refetchInterval?: (query: { state: { data?: { status?: string } } }) => number | false;
    };

    expect(options.refetchInterval?.({ state: { data: { status: 'running' } } })).toBe(3000);
    expect(options.refetchInterval?.({ state: { data: { status: 'completed' } } })).toBe(false);
  });
});
