/**
 * Hook to get the user's preferred data refresh interval.
 * Falls back to a default value if preferences aren't loaded yet.
 */
import { usePreferences } from './usePreferences';

/** Default refresh interval in milliseconds (30 seconds) */
export const DEFAULT_REFRESH_INTERVAL_MS = 30000;

/**
 * Get the user's preferred refresh interval for polling.
 *
 * @param activeOverride - Optional override for active operations (e.g., 5000ms for fast polling)
 * @returns The refresh interval in milliseconds
 */
export function useRefreshInterval(activeOverride?: number): number {
  const { data: preferences } = usePreferences();

  // If an override is provided (e.g., for active operations), use it
  if (activeOverride !== undefined) {
    return activeOverride;
  }

  // Use user's preference, or default if not available
  return preferences?.interface?.data_refresh_interval_ms ?? DEFAULT_REFRESH_INTERVAL_MS;
}

/**
 * Get the user's preferred refresh interval value directly.
 * Useful for components that need the value without the hook overhead.
 */
export function getRefreshIntervalFromPreferences(
  preferences: { interface?: { data_refresh_interval_ms?: number } } | undefined
): number {
  return preferences?.interface?.data_refresh_interval_ms ?? DEFAULT_REFRESH_INTERVAL_MS;
}
