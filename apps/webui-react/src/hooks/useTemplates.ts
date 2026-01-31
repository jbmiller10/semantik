import { useQuery } from '@tanstack/react-query';
import { templatesApi } from '../services/api/v2/templates';
import type { TemplateSummary, TemplateDetail } from '../types/template';

/**
 * Query key factory for template queries
 */
export const templateKeys = {
  all: ['templates'] as const,
  list: () => [...templateKeys.all, 'list'] as const,
  detail: (templateId: string) => [...templateKeys.all, 'detail', templateId] as const,
};

/**
 * Hook to fetch all available pipeline templates
 * Returns summary information for display in selection UI
 */
export function useTemplates() {
  return useQuery({
    queryKey: templateKeys.list(),
    queryFn: async (): Promise<TemplateSummary[]> => {
      const response = await templatesApi.list();
      return response.data.templates;
    },
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes (templates are static)
    gcTime: 30 * 60 * 1000, // Keep in cache for 30 minutes
  });
}

/**
 * Hook to fetch full details for a specific template
 * @param templateId The template identifier
 * @param enabled Whether to enable the query (default: true)
 */
export function useTemplate(templateId: string | null, enabled = true) {
  return useQuery({
    queryKey: templateKeys.detail(templateId ?? ''),
    queryFn: async (): Promise<TemplateDetail> => {
      if (!templateId) {
        throw new Error('Template ID is required');
      }
      const response = await templatesApi.get(templateId);
      return response.data;
    },
    enabled: enabled && !!templateId,
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes
    gcTime: 30 * 60 * 1000, // Keep in cache for 30 minutes
  });
}
