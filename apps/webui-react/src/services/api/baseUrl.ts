type RuntimeEnv = typeof window & {
  __API_BASE_URL__?: string;
  __SEMANTIK_API_BASE_URL__?: string;
  API_BASE_URL?: string;
  __env__?: { API_BASE_URL?: string };
  __ENV__?: { API_BASE_URL?: string };
};

/**
 * Resolve the API base URL from runtime configuration or fall back to the current origin.
 * Supports values injected at deploy-time (window.__API_BASE_URL__/__SEMANTIK_API_BASE_URL__),
 * Vite env (VITE_API_BASE_URL), or the browser origin as a last resort.
 */
export function getApiBaseUrl(): string {
  if (typeof window === 'undefined') return '';

  const runtime = window as RuntimeEnv;

  const viteBase = typeof import.meta !== 'undefined' && import.meta.env
    ? import.meta.env.VITE_API_BASE_URL
    : undefined;

  const configuredBase =
    runtime.__API_BASE_URL__ ??
    runtime.__SEMANTIK_API_BASE_URL__ ??
    runtime.API_BASE_URL ??
    runtime.__env__?.API_BASE_URL ??
    runtime.__ENV__?.API_BASE_URL ??
    viteBase;

  // Allow relative prefixes (e.g., '/semantik') by attaching them to the current origin
  const base = configuredBase
    ? configuredBase.startsWith('http') || configuredBase.startsWith('ws')
      ? configuredBase
      : `${window.location.origin}${configuredBase.startsWith('/') ? '' : '/'}${configuredBase}`
    : window.location.origin;

  return base.replace(/\/$/, '');
}

/**
 * Build a websocket URL that respects the configured API base (host + optional path prefix).
 * Note: Token is now passed via Sec-WebSocket-Protocol header for security.
 */
export function buildWebSocketUrl(path: string): string | null {
  const baseUrl = getApiBaseUrl();
  if (!baseUrl) return null;

  try {
    const normalizedBase = baseUrl.startsWith('http') || baseUrl.startsWith('ws')
      ? baseUrl
      : `${window.location.origin}${baseUrl.startsWith('/') ? '' : '/'}${baseUrl}`;

    const url = new URL(normalizedBase);

    if (url.protocol === 'http:') {
      url.protocol = 'ws:';
    } else if (url.protocol === 'https:') {
      url.protocol = 'wss:';
    } else if (!url.protocol.startsWith('ws')) {
      url.protocol = 'ws:';
    }

    const normalizedPath = path.startsWith('/') ? path : `/${path}`;
    const basePath = url.pathname.endsWith('/') ? url.pathname.slice(0, -1) : url.pathname;
    url.pathname = `${basePath}${normalizedPath}`.replace(/\/\//g, '/');

    return url.toString();
  } catch (error) {
    console.error('Failed to build WebSocket URL', error);
    return null;
  }
}

/**
 * Build WebSocket protocols array with authentication token.
 * Uses Sec-WebSocket-Protocol header which is more secure than query params.
 */
export function buildWebSocketProtocols(token?: string | null): string[] {
  if (!token) return [];
  // Format: access_token.<jwt_token>
  return [`access_token.${token}`];
}

/**
 * Retrieve the persisted auth token (Zustand store is persisted in localStorage) with an
 * optional explicit override. Returns an empty string when unavailable so callers can decide
 * whether to initiate a connection.
 */
export function getAuthToken(fallbackToken?: string | null): string {
  if (fallbackToken) return fallbackToken;

  if (typeof window === 'undefined') return '';

  try {
    const authStorage = window.localStorage?.getItem('auth-storage');
    if (!authStorage) return '';

    const authState = JSON.parse(authStorage);
    return authState.state?.token || '';
  } catch (error) {
    console.error('Failed to read auth token from storage:', error);
    return '';
  }
}
