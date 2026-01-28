/**
 * Hook to detect if the user is on a touch-primary device.
 * Uses CSS media query '(pointer: coarse)' which indicates touch devices.
 *
 * This is used to provide alternative interaction patterns for touch devices
 * where drag-to-connect doesn't work well due to lack of hover states and
 * gesture conflicts with scrolling.
 */

import { useState, useEffect } from 'react';

/**
 * Detects if the device has a coarse pointer (touch-primary).
 * Returns true for phones and tablets, false for desktop with mouse.
 */
export function useIsTouchDevice(): boolean {
  const [isTouch, setIsTouch] = useState(false);

  useEffect(() => {
    // Check if matchMedia is available (not in SSR or some test environments)
    if (typeof window === 'undefined' || !window.matchMedia) {
      return;
    }

    const mediaQuery = window.matchMedia('(pointer: coarse)');
    setIsTouch(mediaQuery.matches);

    const handler = (e: MediaQueryListEvent) => setIsTouch(e.matches);
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  return isTouch;
}

export default useIsTouchDevice;
