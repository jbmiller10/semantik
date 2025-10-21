export type ProjectionReducer = 'umap' | 'tsne' | 'pca';

export type ProjectionStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface ProjectionMetadata {
  id: string;
  collection_id: string;
  status: ProjectionStatus | string;
  reducer: ProjectionReducer | string;
  dimensionality: number;
  created_at?: string | null;
  message?: string | null;
  operation_id?: string | null;
  operation_status?: string | null;
}

export interface ProjectionListResponse {
  projections: ProjectionMetadata[];
}

export interface StartProjectionRequest {
  reducer?: ProjectionReducer;
  dimensionality?: number;
  config?: Record<string, unknown>;
}

export type StartProjectionResponse = ProjectionMetadata;

export interface ProjectionPoint {
  id: string;
  x: number;
  y: number;
  z?: number;
  label?: string;
  color?: string;
  cluster?: string | number;
}

export interface ProjectionData {
  projection: ProjectionMetadata;
  points: ProjectionPoint[];
  metadata?: Record<string, unknown>;
}
