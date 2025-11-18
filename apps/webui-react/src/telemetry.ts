export type TelemetryEventName =
  | 'visualize_tab_open'
  | 'visualize_recompute_start'
  | 'visualize_recompute_reuse'
  | 'visualize_selection_open'
  | 'visualize_selection_find_similar'
  | (string & {});

export interface TelemetryPayload {
  [key: string]: unknown;
}

export function trackTelemetry(event: TelemetryEventName, payload: TelemetryPayload = {}): void {
  try {
    if (typeof window !== 'undefined') {
      const anyWindow = window as any;
      const analytics = anyWindow.analytics;
      if (analytics && typeof analytics.track === 'function') {
        analytics.track(event, payload);
        return;
      }
    }
  } catch (error) {
    // eslint-disable-next-line no-console
    console.warn('Telemetry dispatch failed', { event, error });
  }

  // Fallback: structured console log for local debugging
  // eslint-disable-next-line no-console
  console.log('telemetry_event', { event, ...payload });
}

