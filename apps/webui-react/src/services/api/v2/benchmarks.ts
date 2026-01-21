/**
 * Benchmark API client for datasets and benchmarks management.
 * Follows patterns from collections.ts
 */

import apiClient from './client';
import type {
  BenchmarkDataset,
  DatasetListResponse,
  DatasetUploadRequest,
  DatasetMapping,
  MappingCreateRequest,
  MappingResolveResponse,
  Benchmark,
  BenchmarkListResponse,
  BenchmarkCreateRequest,
  BenchmarkStartResponse,
  BenchmarkResultsResponse,
  RunQueryResultsResponse,
  BenchmarkPaginationParams,
} from '../../../types/benchmark';

/**
 * Benchmark Datasets API client
 */
export const benchmarkDatasetsApi = {
  /**
   * List all benchmark datasets with pagination
   */
  list: (params?: BenchmarkPaginationParams) =>
    apiClient.get<DatasetListResponse>('/api/v2/benchmark-datasets', { params }),

  /**
   * Get a specific dataset by ID
   */
  get: (id: string) =>
    apiClient.get<BenchmarkDataset>(`/api/v2/benchmark-datasets/${id}`),

  /**
   * Upload a new benchmark dataset
   * @param data Dataset metadata
   * @param file JSON file containing dataset queries
   */
  upload: (data: DatasetUploadRequest, file: File) => {
    const formData = new FormData();
    formData.append('name', data.name);
    if (data.description) {
      formData.append('description', data.description);
    }
    formData.append('file', file);

    return apiClient.post<BenchmarkDataset>('/api/v2/benchmark-datasets', formData);
  },

  /**
   * Delete a benchmark dataset
   */
  delete: (id: string) =>
    apiClient.delete<void>(`/api/v2/benchmark-datasets/${id}`),

  /**
   * Create a mapping between a dataset and a collection
   */
  createMapping: (datasetId: string, data: MappingCreateRequest) =>
    apiClient.post<DatasetMapping>(`/api/v2/benchmark-datasets/${datasetId}/mappings`, data),

  /**
   * List all mappings for a dataset
   */
  listMappings: (datasetId: string) =>
    apiClient.get<DatasetMapping[]>(`/api/v2/benchmark-datasets/${datasetId}/mappings`),

  /**
   * Resolve document references for a mapping
   */
  resolveMapping: (datasetId: string, mappingId: number) =>
    apiClient.post<MappingResolveResponse>(
      `/api/v2/benchmark-datasets/${datasetId}/mappings/${mappingId}/resolve`
    ),
};

/**
 * Benchmarks API client
 */
export const benchmarksApi = {
  /**
   * List all benchmarks with pagination
   */
  list: (params?: BenchmarkPaginationParams) =>
    apiClient.get<BenchmarkListResponse>('/api/v2/benchmarks', { params }),

  /**
   * Get a specific benchmark by ID
   */
  get: (id: string) =>
    apiClient.get<Benchmark>(`/api/v2/benchmarks/${id}`),

  /**
   * Create a new benchmark
   */
  create: (data: BenchmarkCreateRequest) =>
    apiClient.post<Benchmark>('/api/v2/benchmarks', data),

  /**
   * Delete a benchmark
   */
  delete: (id: string) =>
    apiClient.delete<void>(`/api/v2/benchmarks/${id}`),

  /**
   * Start benchmark execution
   */
  start: (id: string) =>
    apiClient.post<BenchmarkStartResponse>(`/api/v2/benchmarks/${id}/start`),

  /**
   * Cancel a running benchmark
   */
  cancel: (id: string) =>
    apiClient.post<Benchmark>(`/api/v2/benchmarks/${id}/cancel`),

  /**
   * Get benchmark results
   */
  getResults: (id: string, params?: BenchmarkPaginationParams) =>
    apiClient.get<BenchmarkResultsResponse>(`/api/v2/benchmarks/${id}/results`, { params }),

  /**
   * Get per-query results for a specific run
   */
  getQueryResults: (benchmarkId: string, runId: string, params?: BenchmarkPaginationParams) =>
    apiClient.get<RunQueryResultsResponse>(
      `/api/v2/benchmarks/${benchmarkId}/runs/${runId}/queries`,
      { params }
    ),
};
