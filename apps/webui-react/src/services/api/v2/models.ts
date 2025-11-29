import apiClient from './client';

/**
 * Embedding model configuration from the API.
 */
export interface EmbeddingModel {
  model_name: string;
  dimension: number;
  description: string;
  provider: string;
  supports_quantization?: boolean;
  recommended_quantization?: string;
  memory_estimate?: Record<string, number>;
  max_sequence_length?: number;
  is_asymmetric?: boolean;
  query_prefix?: string;
  document_prefix?: string;
  default_query_instruction?: string;
}

/**
 * Response shape from /api/models endpoint.
 * Maintains backward compatibility with legacy consumers.
 */
export interface ModelsResponse {
  models: Record<string, EmbeddingModel>;
  current_device: string;
  using_real_embeddings: boolean;
}

/**
 * Models API service for fetching available embedding models.
 */
export const modelsApi = {
  /**
   * Fetch all available embedding models including plugin models.
   * Models are returned as a dict keyed by model_name.
   */
  async getModels(): Promise<ModelsResponse> {
    const response = await apiClient.get<ModelsResponse>('/api/models');
    return response.data;
  },
};
