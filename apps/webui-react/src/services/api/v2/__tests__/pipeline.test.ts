import { describe, it, expect, vi, beforeEach } from 'vitest';
import { pipelineApi } from '../pipeline';
import type { RoutePreviewResponse } from '@/types/routePreview';
import type { PipelineDAG } from '@/types/pipeline';

// Mock apiClient
vi.mock('../client', () => ({
  default: {
    post: vi.fn(),
    get: vi.fn(),
  }
}));

import apiClient from '../client';

describe('pipelineApi', () => {
  const mockDAG: PipelineDAG = {
    id: 'test-dag',
    version: '1.0',
    nodes: [
      { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
      { id: 'chunker1', type: 'chunker', plugin_id: 'recursive', config: {} },
    ],
    edges: [
      { from_node: '_source', to_node: 'parser1', when: null },
      { from_node: 'parser1', to_node: 'chunker1', when: null },
    ],
  };

  const mockResponse: RoutePreviewResponse = {
    file_info: {
      filename: 'test.txt',
      extension: '.txt',
      mime_type: 'text/plain',
      size_bytes: 1024,
      uri: 'file:///test.txt',
    },
    sniff_result: {
      is_code: false,
      is_structured_data: false,
    },
    routing_stages: [
      {
        stage: 'entry',
        from_node: '_source',
        evaluated_edges: [],
        selected_node: 'parser1',
        metadata_snapshot: {},
      },
    ],
    path: ['_source', 'parser1', 'chunker1'],
    parsed_metadata: null,
    total_duration_ms: 150,
    warnings: [],
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('previewRoute', () => {
    it('sends file and DAG as FormData', async () => {
      vi.mocked(apiClient.post).mockResolvedValueOnce({ data: mockResponse });

      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
      await pipelineApi.previewRoute(file, mockDAG);

      expect(apiClient.post).toHaveBeenCalledTimes(1);
      const [url, formData] = vi.mocked(apiClient.post).mock.calls[0];

      expect(url).toBe('/api/v2/pipeline/preview-route');
      expect(formData).toBeInstanceOf(FormData);
      expect((formData as FormData).get('file')).toBeTruthy();
      expect((formData as FormData).get('dag')).toBe(JSON.stringify(mockDAG));
    });

    it('includes include_parser_metadata flag as "true" by default', async () => {
      vi.mocked(apiClient.post).mockResolvedValueOnce({ data: mockResponse });

      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
      await pipelineApi.previewRoute(file, mockDAG);

      const [, formData] = vi.mocked(apiClient.post).mock.calls[0];
      expect((formData as FormData).get('include_parser_metadata')).toBe('true');
    });

    it('includes include_parser_metadata flag as "false" when specified', async () => {
      vi.mocked(apiClient.post).mockResolvedValueOnce({ data: mockResponse });

      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
      await pipelineApi.previewRoute(file, mockDAG, false);

      const [, formData] = vi.mocked(apiClient.post).mock.calls[0];
      expect((formData as FormData).get('include_parser_metadata')).toBe('false');
    });

    it('returns the route preview response', async () => {
      vi.mocked(apiClient.post).mockResolvedValueOnce({ data: mockResponse });

      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
      const result = await pipelineApi.previewRoute(file, mockDAG);

      expect(result).toEqual(mockResponse);
      expect(result.path).toEqual(['_source', 'parser1', 'chunker1']);
      expect(result.file_info.filename).toBe('test.txt');
      expect(result.total_duration_ms).toBe(150);
    });

    it('handles warnings in response', async () => {
      const responseWithWarnings = {
        ...mockResponse,
        warnings: ['Parser timeout', 'Metadata extraction failed'],
      };
      vi.mocked(apiClient.post).mockResolvedValueOnce({ data: responseWithWarnings });

      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
      const result = await pipelineApi.previewRoute(file, mockDAG);

      expect(result.warnings).toHaveLength(2);
      expect(result.warnings).toContain('Parser timeout');
      expect(result.warnings).toContain('Metadata extraction failed');
    });

    it('handles parsed metadata in response', async () => {
      const responseWithMetadata = {
        ...mockResponse,
        parsed_metadata: {
          title: 'Test Document',
          author: 'Test Author',
          page_count: 5,
        },
      };
      vi.mocked(apiClient.post).mockResolvedValueOnce({ data: responseWithMetadata });

      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
      const result = await pipelineApi.previewRoute(file, mockDAG);

      expect(result.parsed_metadata).not.toBeNull();
      expect(result.parsed_metadata!.title).toBe('Test Document');
      expect(result.parsed_metadata!.page_count).toBe(5);
    });

    it('handles empty path (no route found)', async () => {
      const noRouteResponse = {
        ...mockResponse,
        path: ['_source'],
        routing_stages: [],
      };
      vi.mocked(apiClient.post).mockResolvedValueOnce({ data: noRouteResponse });

      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
      const result = await pipelineApi.previewRoute(file, mockDAG);

      expect(result.path).toEqual(['_source']);
    });

    it('throws on server error', async () => {
      const error = {
        response: {
          status: 500,
          data: { detail: 'Internal server error' },
        },
      };
      vi.mocked(apiClient.post).mockRejectedValueOnce(error);

      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });

      await expect(pipelineApi.previewRoute(file, mockDAG)).rejects.toEqual(error);
    });

    it('throws on validation error', async () => {
      const error = {
        response: {
          status: 422,
          data: { detail: 'Invalid DAG configuration' },
        },
      };
      vi.mocked(apiClient.post).mockRejectedValueOnce(error);

      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });

      await expect(pipelineApi.previewRoute(file, mockDAG)).rejects.toEqual(error);
    });

    it('posts to correct endpoint', async () => {
      vi.mocked(apiClient.post).mockResolvedValueOnce({ data: mockResponse });

      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
      await pipelineApi.previewRoute(file, mockDAG);

      const [url] = vi.mocked(apiClient.post).mock.calls[0];
      expect(url).toBe('/api/v2/pipeline/preview-route');
    });

    it('sets correct content-type header', async () => {
      vi.mocked(apiClient.post).mockResolvedValueOnce({ data: mockResponse });

      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
      await pipelineApi.previewRoute(file, mockDAG);

      const [, , config] = vi.mocked(apiClient.post).mock.calls[0];
      expect(config).toEqual(
        expect.objectContaining({
          headers: { 'Content-Type': 'multipart/form-data' },
        })
      );
    });

    it('handles sniff_result with various fields', async () => {
      const responseWithSniffResult = {
        ...mockResponse,
        sniff_result: {
          is_code: true,
          is_structured_data: false,
          structured_format: null,
          is_scanned_pdf: false,
          detected_language: 'python',
        },
      };
      vi.mocked(apiClient.post).mockResolvedValueOnce({ data: responseWithSniffResult });

      const file = new File(['def main(): pass'], 'test.py', { type: 'text/x-python' });
      const result = await pipelineApi.previewRoute(file, mockDAG);

      expect(result.sniff_result).toEqual({
        is_code: true,
        is_structured_data: false,
        structured_format: null,
        is_scanned_pdf: false,
        detected_language: 'python',
      });
    });

    it('handles multiple routing stages', async () => {
      const responseWithMultipleStages = {
        ...mockResponse,
        routing_stages: [
          {
            stage: 'entry',
            from_node: '_source',
            evaluated_edges: [
              { from_node: '_source', to_node: 'parser1', status: 'matched', predicate: null, matched: true, field_evaluations: null },
            ],
            selected_node: 'parser1',
            metadata_snapshot: {},
          },
          {
            stage: 'parser_output',
            from_node: 'parser1',
            evaluated_edges: [
              { from_node: 'parser1', to_node: 'chunker1', status: 'matched', predicate: null, matched: true, field_evaluations: null },
            ],
            selected_node: 'chunker1',
            metadata_snapshot: { parsed: { title: 'Test' } },
          },
        ],
      };
      vi.mocked(apiClient.post).mockResolvedValueOnce({ data: responseWithMultipleStages });

      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
      const result = await pipelineApi.previewRoute(file, mockDAG);

      expect(result.routing_stages).toHaveLength(2);
      expect(result.routing_stages[1].metadata_snapshot).toEqual({ parsed: { title: 'Test' } });
    });
  });
});
