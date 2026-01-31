import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

import { useIsTouchDevice } from '../useIsTouchDevice';

describe('useIsTouchDevice', () => {
  // Store original matchMedia
  const originalMatchMedia = window.matchMedia;

  // Mock matchMedia helper
  function mockMatchMedia(matches: boolean) {
    const listeners: Array<(e: MediaQueryListEvent) => void> = [];

    const mediaQueryList = {
      matches,
      media: '(pointer: coarse)',
      onchange: null,
      addListener: vi.fn(), // Deprecated
      removeListener: vi.fn(), // Deprecated
      addEventListener: vi.fn((event: string, callback: (e: MediaQueryListEvent) => void) => {
        if (event === 'change') {
          listeners.push(callback);
        }
      }),
      removeEventListener: vi.fn((event: string, callback: (e: MediaQueryListEvent) => void) => {
        if (event === 'change') {
          const index = listeners.indexOf(callback);
          if (index >= 0) listeners.splice(index, 1);
        }
      }),
      dispatchEvent: vi.fn(),
      // Helper to simulate change
      _triggerChange: (newMatches: boolean) => {
        const event = { matches: newMatches } as MediaQueryListEvent;
        listeners.forEach((listener) => listener(event));
      },
    };

    window.matchMedia = vi.fn().mockReturnValue(mediaQueryList);
    return mediaQueryList;
  }

  afterEach(() => {
    window.matchMedia = originalMatchMedia;
    vi.restoreAllMocks();
  });

  describe('initial state', () => {
    it('returns false by default (JSDOM is mouse-primary)', () => {
      mockMatchMedia(false);

      const { result } = renderHook(() => useIsTouchDevice());

      expect(result.current).toBe(false);
    });

    it('returns true when media query matches (touch device)', () => {
      mockMatchMedia(true);

      const { result } = renderHook(() => useIsTouchDevice());

      expect(result.current).toBe(true);
    });
  });

  describe('media query changes', () => {
    it('updates when media query changes to true', () => {
      const mediaQueryList = mockMatchMedia(false);

      const { result } = renderHook(() => useIsTouchDevice());

      expect(result.current).toBe(false);

      // Simulate device change (e.g., switching to tablet mode)
      act(() => {
        mediaQueryList._triggerChange(true);
      });

      expect(result.current).toBe(true);
    });

    it('updates when media query changes to false', () => {
      const mediaQueryList = mockMatchMedia(true);

      const { result } = renderHook(() => useIsTouchDevice());

      expect(result.current).toBe(true);

      // Simulate device change (e.g., connecting a mouse)
      act(() => {
        mediaQueryList._triggerChange(false);
      });

      expect(result.current).toBe(false);
    });
  });

  describe('cleanup', () => {
    it('removes event listener on unmount', () => {
      const mediaQueryList = mockMatchMedia(false);

      const { unmount } = renderHook(() => useIsTouchDevice());

      expect(mediaQueryList.addEventListener).toHaveBeenCalledWith(
        'change',
        expect.any(Function)
      );

      unmount();

      expect(mediaQueryList.removeEventListener).toHaveBeenCalledWith(
        'change',
        expect.any(Function)
      );
    });
  });

  describe('SSR/no matchMedia', () => {
    beforeEach(() => {
      // Simulate environment without matchMedia
      // @ts-expect-error - intentionally setting to undefined for test
      window.matchMedia = undefined;
    });

    it('returns false when matchMedia is not available', () => {
      const { result } = renderHook(() => useIsTouchDevice());

      expect(result.current).toBe(false);
    });
  });
});
