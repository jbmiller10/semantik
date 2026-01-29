import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useRoutePreview } from '../useRoutePreview';
import { pipelineApi } from '@/services/api/v2/pipeline';
import type { PipelineDAG } from '@/types/pipeline';
import type { RoutePreviewResponse } from '@/types/routePreview';

// Mock the pipeline API
vi.mock('@/services/api/v2/pipeline', () => ({
  pipelineApi: {
    previewRoute: vi.fn(),
  },
}));

describe('useRoutePreview', () => {
  const mockDAG: PipelineDAG = {
    id: 'test-dag',
    version: '1.0',
    nodes: [
      { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
    ],
    edges: [
      { from_node: '_source', to_node: 'parser1', when: null },
    ],
  };

  const mockFile = new File(['test content'], 'test.txt', { type: 'text/plain' });

  const mockResponse: RoutePreviewResponse = {
    file_info: {
      filename: 'test.txt',
      extension: '.txt',
      size_bytes: 12,
      mime_type: 'text/plain',
    },
    sniff_result: {
      is_code: false,
      is_structured_data: false,
      structured_format: null,
      is_scanned_pdf: null,
    },
    path: ['_source', 'parser1'],
    routing_stages: [],
    parsed_metadata: null,
    timing_ms: 10,
    warnings: [],
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('initial state', () => {
    it('has all null/false initial values', () => {
      const { result } = renderHook(() => useRoutePreview());

      expect(result.current.isLoading).toBe(false);
      expect(result.current.error).toBeNull();
      expect(result.current.result).toBeNull();
      expect(result.current.file).toBeNull();
    });
  });

  describe('previewFile', () => {
    it('sets loading state and clears error when starting', async () => {
      // Make API call take some time
      vi.mocked(pipelineApi.previewRoute).mockImplementation(
        () => new Promise((resolve) => setTimeout(() => resolve(mockResponse), 100))
      );

      const { result } = renderHook(() => useRoutePreview());

      // Start the preview
      act(() => {
        result.current.previewFile(mockFile, mockDAG);
      });

      // Check immediately after starting
      expect(result.current.isLoading).toBe(true);
      expect(result.current.error).toBeNull();
      expect(result.current.file).toBe(mockFile);

      // Wait for completion
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });
    });

    it('updates result on success', async () => {
      vi.mocked(pipelineApi.previewRoute).mockResolvedValueOnce(mockResponse);

      const { result } = renderHook(() => useRoutePreview());

      await act(async () => {
        await result.current.previewFile(mockFile, mockDAG);
      });

      expect(result.current.isLoading).toBe(false);
      expect(result.current.result).toEqual(mockResponse);
      expect(result.current.error).toBeNull();
    });

    it('passes includeParserMetadata to API', async () => {
      vi.mocked(pipelineApi.previewRoute).mockResolvedValueOnce(mockResponse);

      const { result } = renderHook(() => useRoutePreview());

      await act(async () => {
        await result.current.previewFile(mockFile, mockDAG, false);
      });

      expect(pipelineApi.previewRoute).toHaveBeenCalledWith(mockFile, mockDAG, false);
    });

    it('defaults includeParserMetadata to true', async () => {
      vi.mocked(pipelineApi.previewRoute).mockResolvedValueOnce(mockResponse);

      const { result } = renderHook(() => useRoutePreview());

      await act(async () => {
        await result.current.previewFile(mockFile, mockDAG);
      });

      expect(pipelineApi.previewRoute).toHaveBeenCalledWith(mockFile, mockDAG, true);
    });
  });

  describe('error extraction', () => {
    it('extracts message from Error instance', async () => {
      vi.mocked(pipelineApi.previewRoute).mockRejectedValueOnce(
        new Error('Test error message')
      );

      const { result } = renderHook(() => useRoutePreview());

      await act(async () => {
        await result.current.previewFile(mockFile, mockDAG);
      });

      expect(result.current.error).toBe('Test error message');
      expect(result.current.isLoading).toBe(false);
    });

    it('extracts detail from Axios response (err.response.data.detail)', async () => {
      const axiosError = {
        response: {
          data: {
            detail: 'Validation error: file too large',
          },
        },
      };
      vi.mocked(pipelineApi.previewRoute).mockRejectedValueOnce(axiosError);

      const { result } = renderHook(() => useRoutePreview());

      await act(async () => {
        await result.current.previewFile(mockFile, mockDAG);
      });

      expect(result.current.error).toBe('Validation error: file too large');
    });

    it('falls back to "Preview failed" for Axios error without detail', async () => {
      const axiosError = {
        response: {
          data: {},
        },
      };
      vi.mocked(pipelineApi.previewRoute).mockRejectedValueOnce(axiosError);

      const { result } = renderHook(() => useRoutePreview());

      await act(async () => {
        await result.current.previewFile(mockFile, mockDAG);
      });

      expect(result.current.error).toBe('Preview failed');
    });

    it('falls back to "Preview failed" for unknown error shapes', async () => {
      vi.mocked(pipelineApi.previewRoute).mockRejectedValueOnce('string error');

      const { result } = renderHook(() => useRoutePreview());

      await act(async () => {
        await result.current.previewFile(mockFile, mockDAG);
      });

      expect(result.current.error).toBe('Preview failed');
    });

    it('handles null/undefined errors', async () => {
      vi.mocked(pipelineApi.previewRoute).mockRejectedValueOnce(null);

      const { result } = renderHook(() => useRoutePreview());

      await act(async () => {
        await result.current.previewFile(mockFile, mockDAG);
      });

      expect(result.current.error).toBe('Preview failed');
    });
  });

  describe('clearPreview', () => {
    it('resets all state to initial values', async () => {
      // First, get a successful result
      vi.mocked(pipelineApi.previewRoute).mockResolvedValueOnce(mockResponse);

      const { result } = renderHook(() => useRoutePreview());

      await act(async () => {
        await result.current.previewFile(mockFile, mockDAG);
      });

      // Verify we have a result
      expect(result.current.result).not.toBeNull();
      expect(result.current.file).not.toBeNull();

      // Clear
      act(() => {
        result.current.clearPreview();
      });

      // Verify all reset
      expect(result.current.isLoading).toBe(false);
      expect(result.current.error).toBeNull();
      expect(result.current.result).toBeNull();
      expect(result.current.file).toBeNull();
    });

    it('clears error state', async () => {
      // First, get an error
      vi.mocked(pipelineApi.previewRoute).mockRejectedValueOnce(new Error('Failed'));

      const { result } = renderHook(() => useRoutePreview());

      await act(async () => {
        await result.current.previewFile(mockFile, mockDAG);
      });

      expect(result.current.error).toBe('Failed');

      // Clear
      act(() => {
        result.current.clearPreview();
      });

      expect(result.current.error).toBeNull();
    });
  });

  describe('multiple calls', () => {
    it('overwrites previous result with new result', async () => {
      const firstResponse = { ...mockResponse, timing_ms: 10 };
      const secondResponse = { ...mockResponse, timing_ms: 20 };

      vi.mocked(pipelineApi.previewRoute)
        .mockResolvedValueOnce(firstResponse)
        .mockResolvedValueOnce(secondResponse);

      const { result } = renderHook(() => useRoutePreview());

      await act(async () => {
        await result.current.previewFile(mockFile, mockDAG);
      });

      expect(result.current.result?.timing_ms).toBe(10);

      await act(async () => {
        await result.current.previewFile(mockFile, mockDAG);
      });

      expect(result.current.result?.timing_ms).toBe(20);
    });
  });
});
