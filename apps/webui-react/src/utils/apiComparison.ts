/**
 * API Comparison Utility
 * 
 * This utility helps compare API calls between the vanilla JS and React versions
 * to ensure all endpoints are being called with the correct parameters.
 */

export interface APICall {
  method: string;
  url: string;
  headers?: Record<string, string>;
  body?: BodyInit | null;
  timestamp: number;
}

export class APIMonitor {
  private calls: APICall[] = [];
  private originalFetch: typeof fetch;

  constructor() {
    this.originalFetch = window.fetch;
  }

  start() {
    // Intercept fetch calls
    window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input.toString();
      const method = init?.method || 'GET';
      
      // Log the API call
      const call: APICall = {
        method,
        url,
        headers: init?.headers as Record<string, string>,
        body: init?.body,
        timestamp: Date.now(),
      };
      
      this.calls.push(call);
      
      // Make the actual call
      return this.originalFetch(input, init);
    };

    // Also intercept XMLHttpRequest for WebSocket upgrade requests
    const originalXHROpen = XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open = function(method: string, url: string, ...args: unknown[]) {
      console.log('XHR:', method, url);
      return originalXHROpen.apply(this, [method, url, ...args] as Parameters<typeof originalXHROpen>);
    };
  }

  stop() {
    window.fetch = this.originalFetch;
  }

  getCalls() {
    return this.calls;
  }

  getCallsByEndpoint(endpoint: string) {
    return this.calls.filter(call => call.url.includes(endpoint));
  }

  exportCalls() {
    const data = {
      version: 'react',
      timestamp: new Date().toISOString(),
      calls: this.calls,
      summary: this.getSummary(),
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `api-calls-react-${Date.now()}.json`;
    a.click();
  }

  getSummary() {
    const endpoints = new Set(this.calls.map(call => {
      const url = new URL(call.url, window.location.origin);
      return `${call.method} ${url.pathname}`;
    }));

    return {
      totalCalls: this.calls.length,
      uniqueEndpoints: Array.from(endpoints),
      callsByMethod: this.calls.reduce((acc, call) => {
        acc[call.method] = (acc[call.method] || 0) + 1;
        return acc;
      }, {} as Record<string, number>),
    };
  }

  compareWithVanilla(vanillaData: { summary: { uniqueEndpoints: string[] } }) {
    const vanillaEndpoints = new Set<string>(vanillaData.summary.uniqueEndpoints);
    const reactEndpoints = new Set(this.getSummary().uniqueEndpoints);

    const comparison = {
      missingInReact: Array.from(vanillaEndpoints).filter((e: string) => !reactEndpoints.has(e)),
      extraInReact: Array.from(reactEndpoints).filter(e => !vanillaEndpoints.has(e)),
      common: Array.from(reactEndpoints).filter(e => vanillaEndpoints.has(e)),
    };

    return comparison;
  }
}

// Expected API endpoints from v2 implementation
export const EXPECTED_ENDPOINTS = {
  auth: [
    'GET /api/auth/me',
    'POST /api/auth/login',
    'POST /api/auth/logout',
    'POST /api/auth/register',
    'POST /api/auth/refresh',
  ],
  collections: [
    'GET /api/v2/collections',
    'POST /api/v2/collections',
    'GET /api/v2/collections/{id}',
    'PUT /api/v2/collections/{id}',
    'DELETE /api/v2/collections/{id}',
    'POST /api/v2/collections/{id}/add',
    'POST /api/v2/collections/{id}/reindex',
  ],
  search: [
    'POST /api/search',
    'POST /api/hybrid_search',
    'GET /api/search/collections',
  ],
  documents: [
    'GET /api/documents/{collectionId}/{docId}/info',
    'GET /api/documents/{collectionId}/{docId}',
  ],
  models: [
    'GET /api/models',
  ],
  metrics: [
    'GET /api/metrics',
  ],
  operations: [
    'GET /api/operations',
  ],
};

// Verification function
export function verifyEndpointCoverage(monitor: APIMonitor) {
  const summary = monitor.getSummary();
  const allExpectedEndpoints = Object.values(EXPECTED_ENDPOINTS).flat();
  
  const coverage = {
    total: allExpectedEndpoints.length,
    covered: 0,
    missing: [] as string[],
    percentage: 0,
  };

  allExpectedEndpoints.forEach(endpoint => {
    const [method, path] = endpoint.split(' ');
    const found = summary.uniqueEndpoints.some(e => {
      const [m, p] = e.split(' ');
      // Simple pattern matching for parameterized URLs
      const pathPattern = path.replace(/{[^}]+}/g, '[^/]+');
      const regex = new RegExp(`^${pathPattern}$`);
      return m === method && regex.test(p);
    });
    
    if (found) {
      coverage.covered++;
    } else {
      coverage.missing.push(endpoint);
    }
  });

  coverage.percentage = Math.round((coverage.covered / coverage.total) * 100);
  
  return coverage;
}