import type { AxiosRequestConfig } from 'axios';
import apiClient from './client';
import type {
  ProjectionArtifactName,
  ProjectionListResponse,
  ProjectionMetadata,
  ProjectionSelectionResponse,
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

  getArtifact: (collectionId: string, projectionId: string, artifactName: ProjectionArtifactName) =>
    apiClient.get<ArrayBuffer>(
      `/api/v2/collections/${collectionId}/projections/${projectionId}/arrays/${artifactName}`,
      { responseType: 'arraybuffer' }
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

  select: (
    collectionId: string,
    projectionId: string,
    ids: number[],
    config?: AxiosRequestConfig
  ) =>
    apiClient.post<ProjectionSelectionResponse>(
      `/api/v2/collections/${collectionId}/projections/${projectionId}/select`,
      { ids },
      config
    ),
};
