import { useQuery } from '@tanstack/react-query';
import { collectionsV2Api } from '../services/api/v2/collections';
import type { DocumentsResponse } from '../services/api/v2/types';

// Query key factory for documents
export const documentKeys = {
  all: ['documents'] as const,
  lists: () => [...documentKeys.all, 'list'] as const,
  list: (collectionId: string, page: number = 1, limit: number = 50) => 
    [...documentKeys.lists(), collectionId, { page, limit }] as const,
};

interface UseCollectionDocumentsOptions {
  page?: number;
  limit?: number;
  enabled?: boolean;
}

// Hook to fetch documents for a collection with pagination
export function useCollectionDocuments(
  collectionId: string, 
  options?: UseCollectionDocumentsOptions
) {
  const { 
    page = 1, 
    limit = 50, 
    enabled = true 
  } = options || {};

  return useQuery({
    queryKey: documentKeys.list(collectionId, page, limit),
    queryFn: async () => {
      const response = await collectionsV2Api.listDocuments(collectionId, { 
        page,
        limit 
      });
      return response.data;
    },
    enabled: !!collectionId && enabled,
    staleTime: 30000, // Consider data stale after 30 seconds
    keepPreviousData: true, // Keep previous page data while fetching new page
  });
}

// Hook to prefetch the next page of documents
export function usePrefetchDocuments() {
  const queryClient = useQueryClient();
  
  return async (
    collectionId: string, 
    currentPage: number, 
    limit: number = 50,
    totalPages?: number
  ) => {
    // Don't prefetch if we're already on the last page
    if (totalPages && currentPage >= totalPages) return;
    
    const nextPage = currentPage + 1;
    
    await queryClient.prefetchQuery({
      queryKey: documentKeys.list(collectionId, nextPage, limit),
      queryFn: async () => {
        const response = await collectionsV2Api.listDocuments(collectionId, { 
          page: nextPage,
          limit 
        });
        return response.data;
      },
      staleTime: 30000,
    });
  };
}

// Utility hook to aggregate source directories from documents
export function useSourceDirectories(documentsData?: DocumentsResponse) {
  if (!documentsData) return [];
  
  const sourceMap = documentsData.documents.reduce((acc, doc) => {
    if (!acc.has(doc.source_path)) {
      acc.set(doc.source_path, { 
        path: doc.source_path, 
        document_count: 0 
      });
    }
    acc.get(doc.source_path)!.document_count++;
    return acc;
  }, new Map<string, { path: string; document_count: number }>());
  
  return Array.from(sourceMap.values());
}