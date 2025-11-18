import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook, waitFor } from '@testing-library/react';

import {
  TOOLTIP_DEBOUNCE_MS,
  TOOLTIP_MAX_INFLIGHT,
  useProjectionTooltip,
} from '../useProjectionTooltip';
import { projectionsV2Api } from '../../services/api/v2/projections';

vi.mock('../../services/api/v2/projections', () => ({
  projectionsV2Api: {
    select: vi.fn(),
  },
}));

describe('useProjectionTooltip', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('aborts oldest tooltip requests to enforce the in-flight limit', () => {
    vi.useFakeTimers();
    const signals: AbortSignal[] = [];

    vi.mocked(projectionsV2Api.select).mockImplementation((_, __, ___, config) => {
      const signal = config?.signal ?? new AbortController().signal;
      signals.push(signal);

      return new Promise((_resolve, reject) => {
        if (signal.aborted) {
          reject(new DOMException('Aborted', 'AbortError'));
          return;
        }

        const abortHandler = () => reject(new DOMException('Aborted', 'AbortError'));
        signal.addEventListener('abort', abortHandler, { once: true });
      });
    });

    const ids = new Int32Array([0, 1, 2, 3, 4, 5, 6, 7]);
    const { result, unmount } = renderHook(() =>
      useProjectionTooltip('collection-1', 'projection-1', ids)
    );

    for (let index = 0; index < TOOLTIP_MAX_INFLIGHT + 2; index += 1) {
      act(() => {
        result.current.handleTooltip({ x: index, y: index, index });
      });

      act(() => {
        vi.advanceTimersByTime(TOOLTIP_DEBOUNCE_MS);
      });
    }

    expect(projectionsV2Api.select).toHaveBeenCalledTimes(TOOLTIP_MAX_INFLIGHT + 2);

    const abortedCount = signals.filter((signal) => signal.aborted).length;
    const activeCount = signals.filter((signal) => !signal.aborted).length;

    expect(abortedCount).toBeGreaterThan(0);
    expect(activeCount).toBeLessThanOrEqual(TOOLTIP_MAX_INFLIGHT);

    unmount();
    vi.useRealTimers();
  });

  it('uses the cache for repeated tooltips on the same id', async () => {
    const ids = new Int32Array([42]);

    vi.mocked(projectionsV2Api.select).mockResolvedValue({
      data: {
        items: [
          {
            selected_id: 42,
            index: 0,
            document_id: 'doc-1',
            chunk_index: 3,
            content_preview: 'Hello world',
          },
        ],
        missing_ids: [],
        degraded: false,
      },
    } as Awaited<ReturnType<typeof projectionsV2Api.select>>);

    const { result, unmount } = renderHook(() =>
      useProjectionTooltip('collection-1', 'projection-1', ids)
    );

    await act(async () => {
      result.current.handleTooltip({ x: 0, y: 0, index: 0 });
      await new Promise((resolve) => setTimeout(resolve, TOOLTIP_DEBOUNCE_MS));
    });

    await waitFor(() => {
      expect(projectionsV2Api.select).toHaveBeenCalledTimes(1);
    });

    // Second hover over the same point should be served from cache
    await act(async () => {
      result.current.handleTooltip({ x: 0, y: 0, index: 0 });
      await new Promise((resolve) => setTimeout(resolve, TOOLTIP_DEBOUNCE_MS));
    });

    await waitFor(() => {
      expect(projectionsV2Api.select).toHaveBeenCalledTimes(1);
      expect(result.current.tooltipState.status).toBe('success');
      expect(result.current.tooltipState.metadata?.selectedId).toBe(42);
      expect(result.current.tooltipState.metadata?.source).toBe('cache');
    });

    unmount();
  });

  it('sets an error state when the API rejects', async () => {
    const ids = new Int32Array([7]);

    vi.mocked(projectionsV2Api.select).mockRejectedValue(new Error('boom'));

    const { result, unmount } = renderHook(() =>
      useProjectionTooltip('collection-1', 'projection-1', ids)
    );

    await act(async () => {
      result.current.handleTooltip({ x: 0, y: 0, index: 0 });
      await new Promise((resolve) => setTimeout(resolve, TOOLTIP_DEBOUNCE_MS));
    });

    await waitFor(() => {
      expect(projectionsV2Api.select).toHaveBeenCalledTimes(1);
      expect(result.current.tooltipState.status).toBe('error');
      expect(result.current.tooltipState.metadata).toBeNull();
      expect(result.current.tooltipState.error).toBe('No metadata available');
    });

    // Leaving the tooltip should reset state back to idle
    act(() => {
      result.current.handleTooltip(null);
    });

    expect(result.current.tooltipState.status).toBe('idle');
    expect(result.current.tooltipState.metadata).toBeNull();

    unmount();
  });
});
