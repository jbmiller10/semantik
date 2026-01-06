/**
 * MCP Profile type definitions for the MCP profile management UI.
 * These types match the backend MCP profile schemas.
 */

/**
 * Minimal collection info for MCP profile responses
 */
export interface CollectionSummary {
  id: string;
  name: string;
}

/**
 * Search types available for MCP profiles
 */
export type MCPSearchType = 'semantic' | 'hybrid' | 'keyword' | 'question' | 'code';

/**
 * Full MCP profile information from the API
 */
export interface MCPProfile {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  search_type: MCPSearchType;
  result_count: number;
  use_reranker: boolean;
  score_threshold: number | null;
  hybrid_alpha: number | null;
  collections: CollectionSummary[];
  created_at: string;
  updated_at: string;
}

/**
 * Request schema for creating an MCP profile
 */
export interface MCPProfileCreate {
  name: string;
  description: string;
  collection_ids: string[];
  enabled?: boolean;
  search_type?: MCPSearchType;
  result_count?: number;
  use_reranker?: boolean;
  score_threshold?: number | null;
  hybrid_alpha?: number | null;
}

/**
 * Request schema for updating an MCP profile
 */
export interface MCPProfileUpdate {
  name?: string;
  description?: string;
  collection_ids?: string[];
  enabled?: boolean;
  search_type?: MCPSearchType;
  result_count?: number;
  use_reranker?: boolean;
  score_threshold?: number | null;
  hybrid_alpha?: number | null;
}

/**
 * Response from listing MCP profiles
 */
export interface MCPProfileListResponse {
  profiles: MCPProfile[];
  total: number;
}

/**
 * Claude Desktop / MCP client configuration snippet
 */
export interface MCPClientConfig {
  server_name: string;
  command: string;
  args: string[];
  env: Record<string, string>;
}

/**
 * Form data state for create/edit profile forms
 */
export interface MCPProfileFormData {
  name: string;
  description: string;
  collection_ids: string[];
  enabled: boolean;
  search_type: MCPSearchType;
  result_count: number;
  use_reranker: boolean;
  score_threshold: number | null;
  hybrid_alpha: number | null;
}

/**
 * Search type display labels for the UI
 */
export const SEARCH_TYPE_LABELS: Record<MCPSearchType, string> = {
  semantic: 'Semantic',
  hybrid: 'Hybrid',
  keyword: 'Keyword',
  question: 'Question',
  code: 'Code',
};

/**
 * Search type descriptions for help text
 */
export const SEARCH_TYPE_DESCRIPTIONS: Record<MCPSearchType, string> = {
  semantic: 'Find results based on meaning and context',
  hybrid: 'Combine semantic and keyword search',
  keyword: 'Traditional keyword matching',
  question: 'Optimized for question-answering',
  code: 'Optimized for code search',
};

/**
 * Default form values for creating a new profile
 */
export const DEFAULT_PROFILE_FORM_DATA: MCPProfileFormData = {
  name: '',
  description: '',
  collection_ids: [],
  enabled: true,
  search_type: 'semantic',
  result_count: 10,
  use_reranker: true,
  score_threshold: null,
  hybrid_alpha: null,
};
