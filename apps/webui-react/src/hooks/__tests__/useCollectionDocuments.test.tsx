import { renderHook, waitFor, act } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import {
  useCollectionDocuments,
  usePrefetchDocuments,
  useSourceDirectories,
  documentKeys,
} from '../useCollectionDocuments';
import { collectionsV2Api } from '../../services/api/v2/collections';
import type { DocumentListResponse, DocumentItem } from '../../services/api/v2/types';

// Mock the API module
vi.mock('../../services/api/v2/collections', () => ({
  collectionsV2Api: {
    listDocuments: vi.fn(),
  },
}));

// Test helpers
const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        staleTime: 0,
      },
    },
  });

const createWrapper = (queryClient: QueryClient) => {
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

// Mock data
const mockDocuments: DocumentItem[] = [
  {
    id: 'doc-1',
    collection_id: 'col-1',
    source_path: '/data/docs',
    file_path: '/data/docs/document1.pdf',
    file_type: 'pdf',
    file_size: 1024000,
    metadata: {
      title: 'Document 1',
      pages: 10,
    },
    status: 'indexed',
    vector_count: 50,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  },
  {
    id: 'doc-2',
    collection_id: 'col-1',
    source_path: '/data/docs',
    file_path: '/data/docs/document2.txt',
    file_type: 'txt',
    file_size: 5000,
    metadata: {},
    status: 'indexed',
    vector_count: 10,
    created_at: '2024-01-01T01:00:00Z',
    updated_at: '2024-01-01T01:00:00Z',
  },
  {
    id: 'doc-3',
    collection_id: 'col-1',
    source_path: '/data/images',
    file_path: '/data/images/photo1.jpg',
    file_type: 'jpg',
    file_size: 2048000,
    metadata: {
      width: 1920,
      height: 1080,
    },
    status: 'indexed',
    vector_count: 5,
    created_at: '2024-01-01T02:00:00Z',
    updated_at: '2024-01-01T02:00:00Z',
  },
];

const mockDocumentResponse: DocumentListResponse = {
  documents: mockDocuments,
  total: 3,
  page: 1,
  per_page: 50,
  total_pages: 1,
};

describe('useCollectionDocuments', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    vi.clearAllMocks();
    queryClient = createTestQueryClient();
  });

  describe('useCollectionDocuments hook', () => {
    it('should fetch documents with default pagination', async () => {
      vi.mocked(collectionsV2Api.listDocuments).mockResolvedValue({
        data: mockDocumentResponse,
      } as any);

      const { result } = renderHook(() => useCollectionDocuments('col-1'), {
        wrapper: createWrapper(queryClient),
      });

      expect(result.current.isLoading).toBe(true);

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockDocumentResponse);
      expect(collectionsV2Api.listDocuments).toHaveBeenCalledWith('col-1', {
        page: 1,
        limit: 50,
      });
    });

    it('should fetch documents with custom pagination', async () => {
      const customResponse = {
        ...mockDocumentResponse,
        page: 2,
        per_page: 20,
      };

      vi.mocked(collectionsV2Api.listDocuments).mockResolvedValue({
        data: customResponse,
      } as any);

      const { result } = renderHook(
        () => useCollectionDocuments('col-1', { page: 2, limit: 20 }),
        { wrapper: createWrapper(queryClient) }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(collectionsV2Api.listDocuments).toHaveBeenCalledWith('col-1', {
        page: 2,
        limit: 20,
      });
    });

    it('should not fetch when collection ID is empty', () => {
      const { result } = renderHook(() => useCollectionDocuments(''), {
        wrapper: createWrapper(queryClient),
      });

      // When enabled is false, query should be in fetching: 'idle' state
      expect(result.current.isLoading).toBe(false);
      expect(result.current.fetchStatus).toBe('idle');
      expect(collectionsV2Api.listDocuments).not.toHaveBeenCalled();
    });

    it('should respect enabled option', () => {
      const { result } = renderHook(
        () => useCollectionDocuments('col-1', { enabled: false }),
        { wrapper: createWrapper(queryClient) }
      );

      // When enabled is false, query should be in fetching: 'idle' state
      expect(result.current.isLoading).toBe(false);
      expect(result.current.fetchStatus).toBe('idle');
      expect(collectionsV2Api.listDocuments).not.toHaveBeenCalled();
    });

    it('should keep previous data while fetching new page', async () => {
      // First page
      vi.mocked(collectionsV2Api.listDocuments).mockResolvedValue({
        data: mockDocumentResponse,
      } as any);

      const { result, rerender } = renderHook(
        ({ page }) => useCollectionDocuments('col-1', { page }),
        {
          wrapper: createWrapper(queryClient),
          initialProps: { page: 1 },
        }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      const firstPageData = result.current.data;

      // Second page - mock different data
      const secondPageResponse = {
        ...mockDocumentResponse,
        page: 2,
        documents: [
          {
            ...mockDocuments[0],
            id: 'doc-4',
            file_path: '/data/docs/document4.pdf',
          },
        ],
      };

      vi.mocked(collectionsV2Api.listDocuments).mockResolvedValue({
        data: secondPageResponse,
      } as any);

      // Change to page 2
      rerender({ page: 2 });

      // Should still have previous data while loading
      expect(result.current.data).toEqual(firstPageData);
      expect(result.current.isPlaceholderData).toBe(true);

      await waitFor(() => {
        expect(result.current.data).toEqual(secondPageResponse);
      });
    });

    it('should handle API errors', async () => {
      const error = new Error('Failed to fetch documents');
      vi.mocked(collectionsV2Api.listDocuments).mockRejectedValue(error);

      const { result } = renderHook(() => useCollectionDocuments('col-1'), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error).toBe(error);
    });
  });

  describe('usePrefetchDocuments hook', () => {
    it('should prefetch next page when not on last page', async () => {
      const prefetchSpy = vi.spyOn(queryClient, 'prefetchQuery');

      const { result } = renderHook(() => usePrefetchDocuments(), {
        wrapper: createWrapper(queryClient),
      });

      vi.mocked(collectionsV2Api.listDocuments).mockResolvedValue({
        data: mockDocumentResponse,
      } as any);

      await act(async () => {
        await result.current('col-1', 1, 50, 3); // Current page 1 of 3
      });

      expect(prefetchSpy).toHaveBeenCalled();
      expect(collectionsV2Api.listDocuments).toHaveBeenCalledWith('col-1', {
        page: 2,
        limit: 50,
      });
    });

    it('should not prefetch when on last page', async () => {
      const prefetchSpy = vi.spyOn(queryClient, 'prefetchQuery');

      const { result } = renderHook(() => usePrefetchDocuments(), {
        wrapper: createWrapper(queryClient),
      });

      await act(async () => {
        await result.current('col-1', 3, 50, 3); // Current page 3 of 3
      });

      expect(prefetchSpy).not.toHaveBeenCalled();
      expect(collectionsV2Api.listDocuments).not.toHaveBeenCalled();
    });

    it('should use custom limit for prefetch', async () => {
      const { result } = renderHook(() => usePrefetchDocuments(), {
        wrapper: createWrapper(queryClient),
      });

      vi.mocked(collectionsV2Api.listDocuments).mockResolvedValue({
        data: mockDocumentResponse,
      } as any);

      await act(async () => {
        await result.current('col-1', 1, 25); // Custom limit of 25
      });

      expect(collectionsV2Api.listDocuments).toHaveBeenCalledWith('col-1', {
        page: 2,
        limit: 25,
      });
    });
  });

  describe('useSourceDirectories utility', () => {
    it('should aggregate documents by source directory', () => {
      const { result } = renderHook(() => useSourceDirectories(mockDocumentResponse), {
        wrapper: createWrapper(queryClient),
      });

      expect(result.current).toHaveLength(2); // Two unique source paths
      
      const docsDir = result.current.find(dir => dir.path === '/data/docs');
      expect(docsDir).toEqual({
        path: '/data/docs',
        document_count: 2,
      });

      const imagesDir = result.current.find(dir => dir.path === '/data/images');
      expect(imagesDir).toEqual({
        path: '/data/images',
        document_count: 1,
      });
    });

    it('should handle empty or undefined data', () => {
      const { result: undefinedResult } = renderHook(() => useSourceDirectories(undefined), {
        wrapper: createWrapper(queryClient),
      });

      expect(undefinedResult.current).toEqual([]);

      const { result: emptyResult } = renderHook(
        () => useSourceDirectories({ ...mockDocumentResponse, documents: [] }),
        { wrapper: createWrapper(queryClient) }
      );

      expect(emptyResult.current).toEqual([]);
    });

    it('should count documents correctly for each source', () => {
      const documentsWithSameSource: DocumentItem[] = [
        { ...mockDocuments[0], id: 'doc-a', source_path: '/data/shared' },
        { ...mockDocuments[1], id: 'doc-b', source_path: '/data/shared' },
        { ...mockDocuments[2], id: 'doc-c', source_path: '/data/shared' },
        { ...mockDocuments[0], id: 'doc-d', source_path: '/data/other' },
      ];

      const response: DocumentListResponse = {
        documents: documentsWithSameSource,
        total: 4,
        page: 1,
        per_page: 50,
        total_pages: 1,
      };

      const { result } = renderHook(() => useSourceDirectories(response), {
        wrapper: createWrapper(queryClient),
      });

      expect(result.current).toHaveLength(2);

      const sharedDir = result.current.find(dir => dir.path === '/data/shared');
      expect(sharedDir?.document_count).toBe(3);

      const otherDir = result.current.find(dir => dir.path === '/data/other');
      expect(otherDir?.document_count).toBe(1);
    });
  });

  describe('query key generation', () => {
    it('should generate correct query keys', () => {
      expect(documentKeys.all).toEqual(['documents']);
      expect(documentKeys.lists()).toEqual(['documents', 'list']);
      expect(documentKeys.list('col-1')).toEqual(['documents', 'list', 'col-1', { page: 1, limit: 50 }]);
      expect(documentKeys.list('col-1', 2, 25)).toEqual(['documents', 'list', 'col-1', { page: 2, limit: 25 }]);
    });
  });

  describe('stale time configuration', () => {
    it('should use 30 second stale time', async () => {
      vi.mocked(collectionsV2Api.listDocuments).mockResolvedValue({
        data: mockDocumentResponse,
      } as any);

      const { result } = renderHook(() => useCollectionDocuments('col-1'), {
        wrapper: createWrapper(queryClient),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      const query = queryClient.getQueryCache().find(documentKeys.list('col-1'));
      expect(query?.options.staleTime).toBe(30000); // 30 seconds
    });
  });

  describe('loading states', () => {
    it('should properly handle loading transitions', async () => {
      let resolvePromise: (value: any) => void;
      const promise = new Promise((resolve) => {
        resolvePromise = resolve;
      });

      vi.mocked(collectionsV2Api.listDocuments).mockReturnValue(promise as any);

      const { result } = renderHook(() => useCollectionDocuments('col-1'), {
        wrapper: createWrapper(queryClient),
      });

      // Initial loading state
      expect(result.current.isLoading).toBe(true);
      expect(result.current.data).toBeUndefined();

      // Resolve the promise
      act(() => {
        resolvePromise!({ data: mockDocumentResponse });
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      // Final success state
      expect(result.current.isLoading).toBe(false);
      expect(result.current.data).toEqual(mockDocumentResponse);
    });
  });
});