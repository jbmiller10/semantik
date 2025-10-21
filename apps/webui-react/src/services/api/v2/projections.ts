import apiClient from './client';
import type {
  ProjectionData,
  ProjectionListResponse,
  ProjectionMetadata,
  StartProjectionRequest,
  StartProjectionResponse,
} from '../../../types/projection';

export const projectionsV2Api = {
  list: (collectionId: string) =>
    apiClient.get<ProjectionListResponse>(
      `/api/v2/collections/${collectionId}/projections`
    ),

  getMetadata: (collectionId: string, projectionId: string) =>
    apiClient.get<ProjectionMetadata>(
      `/api/v2/collections/${collectionId}/projections/${projectionId}`
    ),

  getData: (collectionId: string, projectionId: string) =>
    apiClient.get<ProjectionData>(
      `/api/v2/collections/${collectionId}/projections/${projectionId}/array`
    ),

  start: (collectionId: string, payload: StartProjectionRequest) =>
    apiClient.post<StartProjectionResponse>(
      `/api/v2/collections/${collectionId}/projections`,
      payload
    ),

  delete: (collectionId: string, projectionId: string) =>
    apiClient.delete<void>(
      `/api/v2/collections/${collectionId}/projections/${projectionId}`
    ),
};
