import { describe, it, expect, vi, beforeEach } from 'vitest';

const mockGet = vi.fn();
const mockPost = vi.fn();
const mockDefaults = { baseURL: 'http://api.example' };

vi.mock('../client', () => ({
  default: {
    get: mockGet,
    post: mockPost,
    defaults: mockDefaults,
  },
}));

const mockGetState = vi.fn();
vi.mock('../../../../stores/authStore', () => ({
  useAuthStore: {
    getState: mockGetState,
  },
}));

import { documentsV2Api } from '../documents';

describe('documentsV2Api', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockDefaults.baseURL = 'http://api.example';
  });

  it('builds content URL with auth header when token exists', () => {
    mockGetState.mockReturnValue({ token: 'token-123' });

    const result = documentsV2Api.getContent('col-1', 'doc-1');

    expect(result.url).toBe('http://api.example/api/v2/collections/col-1/documents/doc-1/content');
    expect(result.headers).toEqual({ Authorization: 'Bearer token-123' });
  });

  it('builds content URL without auth header when token missing', () => {
    mockGetState.mockReturnValue({ token: null });

    const result = documentsV2Api.getContent('col-1', 'doc-1');

    expect(result.url).toBe('http://api.example/api/v2/collections/col-1/documents/doc-1/content');
    expect(result.headers).toEqual({});
  });

  it('get() calls the document metadata endpoint', () => {
    documentsV2Api.get('col-1', 'doc-1');
    expect(mockGet).toHaveBeenCalledWith('/api/v2/collections/col-1/documents/doc-1');
  });

  it('retry() calls the retry endpoint', () => {
    documentsV2Api.retry('col-1', 'doc-1');
    expect(mockPost).toHaveBeenCalledWith('/api/v2/collections/col-1/documents/doc-1/retry');
  });

  it('retryFailed() calls the bulk retry endpoint', () => {
    documentsV2Api.retryFailed('col-1');
    expect(mockPost).toHaveBeenCalledWith('/api/v2/collections/col-1/documents/retry-failed');
  });

  it('getFailedCount() passes retryable_only param', () => {
    documentsV2Api.getFailedCount('col-1', true);
    expect(mockGet).toHaveBeenCalledWith(
      '/api/v2/collections/col-1/documents/failed/count',
      { params: { retryable_only: true } }
    );
  });
});
