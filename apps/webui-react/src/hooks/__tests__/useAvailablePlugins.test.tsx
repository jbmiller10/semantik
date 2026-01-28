import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';

import { useAvailablePlugins } from '../useAvailablePlugins';
import { usePipelinePlugins } from '../usePlugins';
import type { PipelinePluginInfo } from '@/types/plugin';

// Mock usePlugins
vi.mock('../usePlugins', () => ({
  usePipelinePlugins: vi.fn(),
}));

const mockUsePipelinePlugins = usePipelinePlugins as ReturnType<typeof vi.fn>;

// Create a wrapper with QueryClient
function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

// Sample plugin data for testing
const mockPlugins: PipelinePluginInfo[] = [
  {
    id: 'pdf-parser',
    type: 'parser',
    display_name: 'PDF Parser',
    description: 'Parses PDF documents',
    source: 'builtin',
    enabled: true,
  },
  {
    id: 'html-parser',
    type: 'parser',
    display_name: 'HTML Parser',
    description: 'Parses HTML documents',
    source: 'builtin',
    enabled: true,
  },
  {
    id: 'disabled-parser',
    type: 'parser',
    display_name: 'Disabled Parser',
    description: 'A disabled parser',
    source: 'external',
    enabled: false,
  },
];

const mockChunkingPlugins: PipelinePluginInfo[] = [
  {
    id: 'sentence-chunker',
    type: 'chunking',
    display_name: 'Sentence Chunker',
    description: 'Chunks by sentences',
    source: 'builtin',
    enabled: true,
  },
];

const mockExtractorPlugins: PipelinePluginInfo[] = [
  {
    id: 'entity-extractor',
    type: 'extractor',
    display_name: 'Entity Extractor',
    description: 'Extracts entities',
    source: 'builtin',
    enabled: true,
  },
];

const mockEmbeddingPlugins: PipelinePluginInfo[] = [
  {
    id: 'openai-embedding',
    type: 'embedding',
    display_name: 'OpenAI Embedding',
    description: 'Uses OpenAI embeddings',
    source: 'builtin',
    enabled: true,
  },
];

describe('useAvailablePlugins', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('node type to plugin type mapping', () => {
    it('maps parser node type to parser plugin type', () => {
      mockUsePipelinePlugins.mockReturnValue({
        data: mockPlugins,
        isLoading: false,
        error: null,
      });

      renderHook(() => useAvailablePlugins('parser'), { wrapper: createWrapper() });

      expect(mockUsePipelinePlugins).toHaveBeenCalledWith({
        plugin_type: 'parser',
      });
    });

    it('maps chunker node type to chunking plugin type', () => {
      mockUsePipelinePlugins.mockReturnValue({
        data: mockChunkingPlugins,
        isLoading: false,
        error: null,
      });

      renderHook(() => useAvailablePlugins('chunker'), { wrapper: createWrapper() });

      expect(mockUsePipelinePlugins).toHaveBeenCalledWith({
        plugin_type: 'chunking',
      });
    });

    it('maps extractor node type to extractor plugin type', () => {
      mockUsePipelinePlugins.mockReturnValue({
        data: mockExtractorPlugins,
        isLoading: false,
        error: null,
      });

      renderHook(() => useAvailablePlugins('extractor'), { wrapper: createWrapper() });

      expect(mockUsePipelinePlugins).toHaveBeenCalledWith({
        plugin_type: 'extractor',
      });
    });

    it('maps embedder node type to embedding plugin type', () => {
      mockUsePipelinePlugins.mockReturnValue({
        data: mockEmbeddingPlugins,
        isLoading: false,
        error: null,
      });

      renderHook(() => useAvailablePlugins('embedder'), { wrapper: createWrapper() });

      expect(mockUsePipelinePlugins).toHaveBeenCalledWith({
        plugin_type: 'embedding',
      });
    });
  });

  describe('enabled filter', () => {
    it('only returns enabled plugins', () => {
      mockUsePipelinePlugins.mockReturnValue({
        data: mockPlugins, // Contains 2 enabled and 1 disabled
        isLoading: false,
        error: null,
      });

      const { result } = renderHook(() => useAvailablePlugins('parser'), {
        wrapper: createWrapper(),
      });

      // Should only have the 2 enabled plugins
      expect(result.current.plugins).toHaveLength(2);
      expect(result.current.plugins.every((p) => p.id !== 'disabled-parser')).toBe(true);
    });

    it('returns empty array when all plugins are disabled', () => {
      const allDisabled: PipelinePluginInfo[] = [
        { ...mockPlugins[0], enabled: false },
        { ...mockPlugins[1], enabled: false },
      ];

      mockUsePipelinePlugins.mockReturnValue({
        data: allDisabled,
        isLoading: false,
        error: null,
      });

      const { result } = renderHook(() => useAvailablePlugins('parser'), {
        wrapper: createWrapper(),
      });

      expect(result.current.plugins).toHaveLength(0);
    });
  });

  describe('mapping to simplified format', () => {
    it('maps plugin data to AvailablePluginOption format', () => {
      mockUsePipelinePlugins.mockReturnValue({
        data: mockPlugins,
        isLoading: false,
        error: null,
      });

      const { result } = renderHook(() => useAvailablePlugins('parser'), {
        wrapper: createWrapper(),
      });

      expect(result.current.plugins[0]).toEqual({
        id: 'pdf-parser',
        name: 'PDF Parser',
        description: 'Parses PDF documents',
      });
    });

    it('maps display_name to name field', () => {
      mockUsePipelinePlugins.mockReturnValue({
        data: mockPlugins,
        isLoading: false,
        error: null,
      });

      const { result } = renderHook(() => useAvailablePlugins('parser'), {
        wrapper: createWrapper(),
      });

      // Verify all plugins have 'name' (not 'display_name')
      result.current.plugins.forEach((plugin) => {
        expect(plugin).toHaveProperty('name');
        expect(plugin).not.toHaveProperty('display_name');
      });
    });
  });

  describe('loading state passthrough', () => {
    it('passes through isLoading true', () => {
      mockUsePipelinePlugins.mockReturnValue({
        data: undefined,
        isLoading: true,
        error: null,
      });

      const { result } = renderHook(() => useAvailablePlugins('parser'), {
        wrapper: createWrapper(),
      });

      expect(result.current.isLoading).toBe(true);
    });

    it('passes through isLoading false', () => {
      mockUsePipelinePlugins.mockReturnValue({
        data: mockPlugins,
        isLoading: false,
        error: null,
      });

      const { result } = renderHook(() => useAvailablePlugins('parser'), {
        wrapper: createWrapper(),
      });

      expect(result.current.isLoading).toBe(false);
    });
  });

  describe('error state passthrough', () => {
    it('passes through error when present', () => {
      const testError = new Error('Failed to fetch plugins');
      mockUsePipelinePlugins.mockReturnValue({
        data: undefined,
        isLoading: false,
        error: testError,
      });

      const { result } = renderHook(() => useAvailablePlugins('parser'), {
        wrapper: createWrapper(),
      });

      expect(result.current.error).toBe(testError);
    });

    it('passes through null error when no error', () => {
      mockUsePipelinePlugins.mockReturnValue({
        data: mockPlugins,
        isLoading: false,
        error: null,
      });

      const { result } = renderHook(() => useAvailablePlugins('parser'), {
        wrapper: createWrapper(),
      });

      expect(result.current.error).toBeNull();
    });
  });

  describe('empty/null data handling', () => {
    it('returns empty array when data is undefined', () => {
      mockUsePipelinePlugins.mockReturnValue({
        data: undefined,
        isLoading: false,
        error: null,
      });

      const { result } = renderHook(() => useAvailablePlugins('parser'), {
        wrapper: createWrapper(),
      });

      expect(result.current.plugins).toEqual([]);
    });

    it('returns empty array when data is null', () => {
      mockUsePipelinePlugins.mockReturnValue({
        data: null,
        isLoading: false,
        error: null,
      });

      const { result } = renderHook(() => useAvailablePlugins('parser'), {
        wrapper: createWrapper(),
      });

      expect(result.current.plugins).toEqual([]);
    });

    it('returns empty array when data is empty array', () => {
      mockUsePipelinePlugins.mockReturnValue({
        data: [],
        isLoading: false,
        error: null,
      });

      const { result } = renderHook(() => useAvailablePlugins('parser'), {
        wrapper: createWrapper(),
      });

      expect(result.current.plugins).toEqual([]);
    });
  });

  describe('memoization', () => {
    it('returns same plugins array reference when data unchanged', () => {
      mockUsePipelinePlugins.mockReturnValue({
        data: mockPlugins,
        isLoading: false,
        error: null,
      });

      const { result, rerender } = renderHook(() => useAvailablePlugins('parser'), {
        wrapper: createWrapper(),
      });

      const firstPlugins = result.current.plugins;
      rerender();
      const secondPlugins = result.current.plugins;

      // Should be the same reference due to useMemo
      expect(firstPlugins).toBe(secondPlugins);
    });
  });
});
