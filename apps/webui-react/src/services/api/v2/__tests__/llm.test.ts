import { describe, it, expect, vi, beforeEach } from 'vitest';
import { llmApi } from '../llm';

vi.mock('../client', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
  },
}));

import apiClient from '../client';

describe('llmApi', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('refreshModels uses POST body instead of query params', async () => {
    vi.mocked(apiClient.post).mockResolvedValueOnce({ data: { models: [] } });

    await llmApi.refreshModels('anthropic', 'sk-ant-test-key');

    expect(apiClient.post).toHaveBeenCalledWith('/api/v2/llm/models/refresh', {
      provider: 'anthropic',
      api_key: 'sk-ant-test-key',
    });
  });
});
