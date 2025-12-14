import apiClient from './client';
import type {
  ConnectorCatalog,
  ConnectorDefinition,
  GitPreviewRequest,
  GitPreviewResponse,
  ImapPreviewRequest,
  ImapPreviewResponse,
} from '../../../types/connector';

/**
 * Response wrapper for connector catalog
 */
interface ConnectorCatalogResponse {
  connectors: ConnectorCatalog;
}

/**
 * Response wrapper for single connector definition
 */
interface ConnectorDefinitionResponse {
  type: string;
  definition: ConnectorDefinition;
}

/**
 * V2 Connectors API client
 * Provides connector catalog and preview/validation endpoints
 */
export const connectorsApi = {
  /**
   * Get the full connector catalog with all available connector types
   */
  getCatalog: () =>
    apiClient.get<ConnectorCatalogResponse>('/api/v2/connectors'),

  /**
   * Get definition for a specific connector type
   */
  getConnector: (type: string) =>
    apiClient.get<ConnectorDefinitionResponse>(`/api/v2/connectors/${type}`),

  /**
   * Preview/validate a Git repository connection
   * Tests connectivity and returns available refs (branches/tags)
   */
  previewGit: (data: GitPreviewRequest) =>
    apiClient.post<GitPreviewResponse>('/api/v2/connectors/preview/git', data),

  /**
   * Preview/validate an IMAP connection
   * Tests connectivity and returns available mailboxes
   */
  previewImap: (data: ImapPreviewRequest) =>
    apiClient.post<ImapPreviewResponse>('/api/v2/connectors/preview/imap', data),
};
