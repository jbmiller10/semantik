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
  config?: Record<string, unknown> | null;
  meta?: Record<string, unknown> | null;
}

export interface ProjectionLegendItem {
  index: number;
  label: string;
  count?: number;
}

export type ProjectionArtifactName = 'x' | 'y' | 'cat' | 'ids';

export interface ProjectionListResponse {
  projections: ProjectionMetadata[];
}

export interface StartProjectionRequest {
  reducer?: ProjectionReducer;
  dimensionality?: number;
  config?: Record<string, unknown>;
  color_by?: string;
}

export interface ProjectionSelectionItem {
  selected_id: number;
  index: number;
  original_id?: string | null;
  chunk_id?: number | null;
  document_id?: string | null;
  chunk_index?: number | null;
  content_preview?: string | null;
  document?: Record<string, unknown> | null;
}

export interface ProjectionSelectionResponse {
  projection_id: string;
  items: ProjectionSelectionItem[];
  missing_ids: number[];
  degraded?: boolean;
}

export type StartProjectionResponse = ProjectionMetadata;
