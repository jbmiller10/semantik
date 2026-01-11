import { useQuery, keepPreviousData } from '@tanstack/react-query';
import { collectionsV2Api } from '../services/api/v2/collections';

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
    placeholderData: keepPreviousData, // Keep previous page data while fetching new page
  });
}

