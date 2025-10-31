import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';

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
    vi.useFakeTimers();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('aborts oldest tooltip requests to enforce the in-flight limit', () => {
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
  });
});
