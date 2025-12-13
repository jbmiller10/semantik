import { useQuery, useMutation } from '@tanstack/react-query';
import { connectorsApi } from '../services/api/v2/connectors';
import type {
  ConnectorCatalog,
  GitPreviewRequest,
  GitPreviewResponse,
  ImapPreviewRequest,
  ImapPreviewResponse,
} from '../types/connector';

/**
 * Query key factory for connector queries
 */
export const connectorKeys = {
  all: ['connectors'] as const,
  catalog: () => [...connectorKeys.all, 'catalog'] as const,
  definition: (type: string) => [...connectorKeys.all, 'definition', type] as const,
};

/**
 * Hook to fetch the connector catalog
 * Cached for 5 minutes since it rarely changes
 */
export function useConnectorCatalog() {
  return useQuery({
    queryKey: connectorKeys.catalog(),
    queryFn: async (): Promise<ConnectorCatalog> => {
      const response = await connectorsApi.getCatalog();
      return response.data.connectors;
    },
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes
    gcTime: 10 * 60 * 1000, // Keep in cache for 10 minutes
  });
}

/**
 * Hook to fetch a specific connector definition
 */
export function useConnectorDefinition(type: string) {
  return useQuery({
    queryKey: connectorKeys.definition(type),
    queryFn: async () => {
      const response = await connectorsApi.getConnector(type);
      return response.data.definition;
    },
    enabled: !!type,
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Hook for Git repository preview/validation
 * Returns available refs (branches/tags) if connection succeeds
 */
export function useGitPreview() {
  return useMutation<GitPreviewResponse, Error, GitPreviewRequest>({
    mutationFn: async (data: GitPreviewRequest) => {
      const response = await connectorsApi.previewGit(data);
      return response.data;
    },
  });
}

/**
 * Hook for IMAP connection preview/validation
 * Returns available mailboxes if connection succeeds
 */
export function useImapPreview() {
  return useMutation<ImapPreviewResponse, Error, ImapPreviewRequest>({
    mutationFn: async (data: ImapPreviewRequest) => {
      const response = await connectorsApi.previewImap(data);
      return response.data;
    },
  });
}
