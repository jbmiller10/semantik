/**
 * React Query hooks for benchmark management.
 * Follows patterns from useCollections.ts
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { benchmarkDatasetsApi, benchmarksApi } from '../services/api/v2/benchmarks';
import { handleApiError } from '../services/api/v2/collections';
import { useUIStore } from '../stores/uiStore';
import type {
  DatasetUploadRequest,
  MappingCreateRequest,
  Benchmark,
  BenchmarkCreateRequest,
  BenchmarkPaginationParams,
} from '../types/benchmark';

// =============================================================================
// Query Key Factories
// =============================================================================

/**
 * Query key factory for benchmark datasets
 */
export const datasetKeys = {
  all: ['benchmark-datasets'] as const,
  lists: () => [...datasetKeys.all, 'list'] as const,
  list: (filters?: BenchmarkPaginationParams) => [...datasetKeys.lists(), filters] as const,
  details: () => [...datasetKeys.all, 'detail'] as const,
  detail: (id: string) => [...datasetKeys.details(), id] as const,
  mappings: (id: string) => [...datasetKeys.all, 'mappings', id] as const,
};

/**
 * Query key factory for benchmarks
 */
export const benchmarkKeys = {
  all: ['benchmarks'] as const,
  lists: () => [...benchmarkKeys.all, 'list'] as const,
  list: (filters?: BenchmarkPaginationParams) => [...benchmarkKeys.lists(), filters] as const,
  details: () => [...benchmarkKeys.all, 'detail'] as const,
  detail: (id: string) => [...benchmarkKeys.details(), id] as const,
  results: (id: string) => [...benchmarkKeys.all, 'results', id] as const,
  queryResults: (benchmarkId: string, runId: string, params?: BenchmarkPaginationParams) =>
    [
      ...benchmarkKeys.all,
      'query-results',
      benchmarkId,
      runId,
      {
        page: params?.page ?? 1,
        per_page: params?.per_page ?? 50,
      },
    ] as const,
};

// =============================================================================
// Dataset Hooks
// =============================================================================

/**
 * Hook to fetch all benchmark datasets
 */
export function useBenchmarkDatasets(params?: BenchmarkPaginationParams) {
  return useQuery({
    queryKey: datasetKeys.list(params),
    queryFn: async () => {
      const response = await benchmarkDatasetsApi.list(params);
      return response.data;
    },
    staleTime: 30000, // 30 seconds
  });
}

/**
 * Hook to fetch a single benchmark dataset
 */
export function useBenchmarkDataset(id: string) {
  return useQuery({
    queryKey: datasetKeys.detail(id),
    queryFn: async () => {
      const response = await benchmarkDatasetsApi.get(id);
      return response.data;
    },
    enabled: !!id,
    staleTime: 30000,
  });
}

/**
 * Hook to upload a new benchmark dataset
 */
export function useUploadDataset() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async ({ data, file }: { data: DatasetUploadRequest; file: File }) => {
      const response = await benchmarkDatasetsApi.upload(data, file);
      return response.data;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: datasetKeys.lists() });
      addToast({
        type: 'success',
        message: `Dataset "${data.name}" uploaded successfully`,
      });
    },
    onError: (error) => {
      const errorMessage = handleApiError(error);
      addToast({ type: 'error', message: errorMessage });
    },
  });
}

/**
 * Hook to delete a benchmark dataset
 */
export function useDeleteDataset() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async (id: string) => {
      await benchmarkDatasetsApi.delete(id);
      return id;
    },
    onMutate: async (id) => {
      await queryClient.cancelQueries({ queryKey: datasetKeys.lists() });

      const previousData = queryClient.getQueryData(datasetKeys.lists());

      return { previousData, deletedId: id };
    },
    onError: (error, _id, context) => {
      if (context?.previousData) {
        queryClient.setQueryData(datasetKeys.lists(), context.previousData);
      }
      const errorMessage = handleApiError(error);
      addToast({ type: 'error', message: errorMessage });
    },
    onSuccess: () => {
      addToast({
        type: 'success',
        message: 'Dataset deleted successfully',
      });
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: datasetKeys.lists() });
    },
  });
}

/**
 * Hook to fetch mappings for a dataset
 */
export function useDatasetMappings(datasetId: string) {
  return useQuery({
    queryKey: datasetKeys.mappings(datasetId),
    queryFn: async () => {
      const response = await benchmarkDatasetsApi.listMappings(datasetId);
      return response.data;
    },
    enabled: !!datasetId,
    staleTime: 30000,
  });
}

/**
 * Hook to create a dataset-collection mapping
 */
export function useCreateMapping() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async ({
      datasetId,
      data,
    }: {
      datasetId: string;
      data: MappingCreateRequest;
    }) => {
      const response = await benchmarkDatasetsApi.createMapping(datasetId, data);
      return response.data;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: datasetKeys.mappings(data.dataset_id) });
      addToast({
        type: 'success',
        message: 'Mapping created successfully',
      });
    },
    onError: (error) => {
      const errorMessage = handleApiError(error);
      addToast({ type: 'error', message: errorMessage });
    },
  });
}

/**
 * Hook to resolve a dataset mapping
 */
export function useResolveMapping() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async ({
      datasetId,
      mappingId,
    }: {
      datasetId: string;
      mappingId: number;
    }) => {
      const response = await benchmarkDatasetsApi.resolveMapping(datasetId, mappingId);
      return { ...response.data, datasetId };
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: datasetKeys.mappings(data.datasetId) });
      if (data.operation_uuid) {
        addToast({
          type: 'info',
          message: 'Mapping resolution queued. Progress will stream live while it runs.',
        });
      } else {
        const message =
          data.mapping_status === 'resolved'
            ? `Mapping resolved: ${data.mapped_count}/${data.total_count} documents matched`
            : `Mapping partially resolved: ${data.mapped_count}/${data.total_count} documents matched`;
        addToast({
          type: data.mapping_status === 'resolved' ? 'success' : 'warning',
          message,
        });
      }
    },
    onError: (error) => {
      const errorMessage = handleApiError(error);
      addToast({ type: 'error', message: errorMessage });
    },
  });
}

// =============================================================================
// Benchmark Hooks
// =============================================================================

/**
 * Hook to fetch all benchmarks
 */
export function useBenchmarks(params?: BenchmarkPaginationParams) {
  return useQuery({
    queryKey: benchmarkKeys.list(params),
    queryFn: async () => {
      const response = await benchmarksApi.list(params);
      return response.data;
    },
    staleTime: 10000, // 10 seconds - benchmarks may update frequently
    refetchInterval: (query) => {
      // Auto-refetch if there are running benchmarks
      const hasRunning = query.state.data?.benchmarks?.some(
        (b: Benchmark) => b.status === 'running'
      );
      return hasRunning ? 5000 : false;
    },
  });
}

/**
 * Hook to fetch a single benchmark
 */
export function useBenchmark(id: string) {
  return useQuery({
    queryKey: benchmarkKeys.detail(id),
    queryFn: async () => {
      const response = await benchmarksApi.get(id);
      return response.data;
    },
    enabled: !!id,
    staleTime: 10000,
    refetchInterval: (query) => {
      // Auto-refetch if benchmark is running
      const isRunning = query.state.data?.status === 'running';
      return isRunning ? 3000 : false;
    },
  });
}

/**
 * Hook to create a new benchmark
 */
export function useCreateBenchmark() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async (data: BenchmarkCreateRequest) => {
      const response = await benchmarksApi.create(data);
      return response.data;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: benchmarkKeys.lists() });
      addToast({
        type: 'success',
        message: `Benchmark "${data.name}" created successfully`,
      });
    },
    onError: (error) => {
      const errorMessage = handleApiError(error);
      addToast({ type: 'error', message: errorMessage });
    },
  });
}

/**
 * Hook to delete a benchmark
 */
export function useDeleteBenchmark() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async (id: string) => {
      await benchmarksApi.delete(id);
      return id;
    },
    onMutate: async (id) => {
      await queryClient.cancelQueries({ queryKey: benchmarkKeys.lists() });

      const previousData = queryClient.getQueryData(benchmarkKeys.lists());

      return { previousData, deletedId: id };
    },
    onError: (error, _id, context) => {
      if (context?.previousData) {
        queryClient.setQueryData(benchmarkKeys.lists(), context.previousData);
      }
      const errorMessage = handleApiError(error);
      addToast({ type: 'error', message: errorMessage });
    },
    onSuccess: () => {
      addToast({
        type: 'success',
        message: 'Benchmark deleted successfully',
      });
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: benchmarkKeys.lists() });
    },
  });
}

/**
 * Hook to start a benchmark
 */
export function useStartBenchmark() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async (id: string) => {
      const response = await benchmarksApi.start(id);
      return response.data;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: benchmarkKeys.detail(data.id) });
      queryClient.invalidateQueries({ queryKey: benchmarkKeys.lists() });
      addToast({
        type: 'success',
        message: data.message,
      });
    },
    onError: (error) => {
      const errorMessage = handleApiError(error);
      addToast({ type: 'error', message: errorMessage });
    },
  });
}

/**
 * Hook to cancel a running benchmark
 */
export function useCancelBenchmark() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async (id: string) => {
      const response = await benchmarksApi.cancel(id);
      return response.data;
    },
    onSuccess: (benchmark) => {
      // Update cache with returned benchmark data
      queryClient.setQueryData(benchmarkKeys.detail(benchmark.id), benchmark);
      queryClient.invalidateQueries({ queryKey: benchmarkKeys.lists() });
      queryClient.invalidateQueries({ queryKey: benchmarkKeys.results(benchmark.id) });
      addToast({
        type: 'info',
        message: 'Benchmark cancelled',
      });
    },
    onError: (error) => {
      const errorMessage = handleApiError(error);
      addToast({ type: 'error', message: errorMessage });
    },
  });
}

/**
 * Hook to fetch benchmark results
 */
export function useBenchmarkResults(id: string, params?: BenchmarkPaginationParams) {
  return useQuery({
    queryKey: benchmarkKeys.results(id),
    queryFn: async () => {
      const response = await benchmarksApi.getResults(id, params);
      return response.data;
    },
    enabled: !!id,
    staleTime: 30000,
  });
}

/**
 * Hook to fetch per-query results for a benchmark run
 */
export function useBenchmarkQueryResults(
  benchmarkId: string,
  runId: string,
  params?: BenchmarkPaginationParams
) {
  return useQuery({
    queryKey: benchmarkKeys.queryResults(benchmarkId, runId, params),
    queryFn: async () => {
      const response = await benchmarksApi.getQueryResults(benchmarkId, runId, params);
      return response.data;
    },
    enabled: !!benchmarkId && !!runId,
    staleTime: 60000, // 1 minute - query results don't change once computed
  });
}
