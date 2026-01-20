/**
 * API client for Model Manager endpoints.
 * Note: This is separate from models.ts which handles legacy /api/models endpoint.
 */
import apiClient from './client';
import type {
  ModelType,
  ModelListResponse,
  TaskProgressResponse,
  ModelUsageResponse,
  TaskResponse,
} from '../../../types/model-manager';

export interface ListModelsParams {
  model_type?: ModelType;
  installed_only?: boolean;
  include_cache_size?: boolean;
  force_refresh_cache?: boolean;
}

/**
 * Model Manager API service for curated model management.
 */
export const modelManagerApi = {
  /**
   * List all curated models.
   * GET /api/v2/models
   */
  async listModels(params?: ListModelsParams): Promise<ModelListResponse> {
    const response = await apiClient.get<ModelListResponse>('/api/v2/models', {
      params,
    });
    return response.data;
  },

  /**
   * Get task progress for a download/delete operation.
   * GET /api/v2/models/tasks/{task_id}
   */
  async getTaskProgress(taskId: string): Promise<TaskProgressResponse> {
    const response = await apiClient.get<TaskProgressResponse>(
      `/api/v2/models/tasks/${encodeURIComponent(taskId)}`
    );
    return response.data;
  },

  /**
   * Get model usage info (preflight check before deletion).
   * GET /api/v2/models/usage
   * Note: Stub for Phase 2C
   */
  async getModelUsage(modelId: string): Promise<ModelUsageResponse> {
    const response = await apiClient.get<ModelUsageResponse>('/api/v2/models/usage', {
      params: { model_id: modelId },
    });
    return response.data;
  },

  /**
   * Start a model download.
   * POST /api/v2/models/download
   * Note: Stub for Phase 2B
   */
  async startDownload(modelId: string): Promise<TaskResponse> {
    const response = await apiClient.post<TaskResponse>('/api/v2/models/download', {
      model_id: modelId,
    });
    return response.data;
  },

  /**
   * Delete a model from cache.
   * DELETE /api/v2/models/cache
   * Note: Stub for Phase 2C
   */
  async deleteModel(modelId: string, confirm?: boolean): Promise<TaskResponse> {
    const response = await apiClient.delete<TaskResponse>('/api/v2/models/cache', {
      params: { model_id: modelId, confirm },
    });
    return response.data;
  },
};
