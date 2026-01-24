import apiClient from './client';
import type {
  TemplateSummary,
  TemplateDetail,
  TemplateListResponse,
} from '../../../types/template';

/**
 * V2 Templates API client
 * Provides endpoints for pipeline template discovery and retrieval
 */
export const templatesApi = {
  /**
   * List all available pipeline templates
   * Returns summary information suitable for selection UI
   */
  list: () =>
    apiClient.get<TemplateListResponse>('/api/v2/templates'),

  /**
   * Get full details for a specific template
   * @param templateId The template identifier
   * Returns complete template including pipeline DAG and tunable parameters
   */
  get: (templateId: string) =>
    apiClient.get<TemplateDetail>(`/api/v2/templates/${templateId}`),
};

export default templatesApi;
