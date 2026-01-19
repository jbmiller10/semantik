import type { ReactNode } from 'react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type {
  Benchmark,
  BenchmarkDataset,
  BenchmarkResultsResponse,
  DatasetListResponse,
  DatasetMapping,
  MappingResolveResponse,
  RunQueryResultsResponse,
} from '../types/benchmark';
import {
  useBenchmarkDatasets,
  useDatasetMappings,
  useUploadDataset,
  useDeleteDataset,
  useCreateMapping,
  useResolveMapping,
  useCreateBenchmark,
  useDeleteBenchmark,
  useStartBenchmark,
  useCancelBenchmark,
  useBenchmarkResults,
  useBenchmarkQueryResults,
  datasetKeys,
  benchmarkKeys,
} from './useBenchmarks';
import { benchmarkDatasetsApi, benchmarksApi } from '../services/api/v2/benchmarks';
import { useUIStore } from '../stores/uiStore';

vi.mock('../services/api/v2/benchmarks', () => ({
  benchmarkDatasetsApi: {
    list: vi.fn(),
    get: vi.fn(),
    upload: vi.fn(),
    delete: vi.fn(),
    createMapping: vi.fn(),
    listMappings: vi.fn(),
    resolveMapping: vi.fn(),
  },
  benchmarksApi: {
    list: vi.fn(),
    get: vi.fn(),
    create: vi.fn(),
    delete: vi.fn(),
    start: vi.fn(),
    cancel: vi.fn(),
    getResults: vi.fn(),
    getQueryResults: vi.fn(),
  },
}));

vi.mock('../services/api/v2/collections', () => ({
  handleApiError: vi.fn((error: unknown) => (error instanceof Error ? error.message : 'API Error')),
}));

vi.mock('../stores/uiStore', () => ({
  useUIStore: vi.fn(),
}));

type MockAxiosResponse<T> = { data: T };

const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        staleTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  });

const createWrapper = (queryClient: QueryClient) => {
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('useBenchmarks hooks', () => {
  const addToast = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useUIStore).mockReturnValue({ addToast } as never);
  });

  it('uploads a dataset and resolves a mapping', async () => {
    const queryClient = createTestQueryClient();
    const wrapper = createWrapper(queryClient);
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    const dataset: BenchmarkDataset = {
      id: 'ds-1',
      name: 'Test Dataset',
      description: null,
      owner_id: 1,
      query_count: 1,
      schema_version: '1.0',
      created_at: '2025-01-01T00:00:00Z',
      updated_at: null,
    };
    vi.mocked(benchmarkDatasetsApi.upload).mockResolvedValue({ data: dataset } as MockAxiosResponse<BenchmarkDataset>);
    vi.mocked(benchmarkDatasetsApi.list).mockResolvedValue({
      data: { datasets: [dataset], total: 1, page: 1, per_page: 50 },
    } as MockAxiosResponse<DatasetListResponse>);

    const mapping: DatasetMapping = {
      id: 10,
      dataset_id: dataset.id,
      collection_id: 'col-1',
      mapping_status: 'pending',
      mapped_count: 0,
      total_count: 2,
      created_at: '2025-01-01T00:00:00Z',
      resolved_at: null,
    };
    vi.mocked(benchmarkDatasetsApi.createMapping).mockResolvedValue({ data: mapping } as MockAxiosResponse<DatasetMapping>);
    vi.mocked(benchmarkDatasetsApi.listMappings).mockResolvedValue({ data: [mapping] } as MockAxiosResponse<DatasetMapping[]>);
    vi.mocked(benchmarkDatasetsApi.resolveMapping).mockResolvedValue({
      data: {
        id: mapping.id,
        operation_uuid: null,
        mapping_status: 'resolved',
        mapped_count: 2,
        total_count: 2,
        unresolved: [],
      },
    } as MockAxiosResponse<MappingResolveResponse>);

    const { result: upload } = renderHook(() => useUploadDataset(), { wrapper });
    const datasetFile = new File([JSON.stringify({ schema_version: '1.0', queries: [] })], 'dataset.json', {
      type: 'application/json',
    });

    let uploaded: BenchmarkDataset | undefined;
    await act(async () => {
      uploaded = await upload.current.mutateAsync({ data: { name: dataset.name }, file: datasetFile });
    });

    expect(uploaded?.id).toBe(dataset.id);
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: datasetKeys.lists() });
    expect(addToast).toHaveBeenCalledWith(expect.objectContaining({ type: 'success' }));

    const { result: createMapping } = renderHook(() => useCreateMapping(), { wrapper });
    let created: DatasetMapping | undefined;
    await act(async () => {
      created = await createMapping.current.mutateAsync({
        datasetId: dataset.id,
        data: { collection_id: mapping.collection_id },
      });
    });
    expect(created?.dataset_id).toBe(dataset.id);
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: datasetKeys.mappings(dataset.id) });

    const { result: resolveMapping } = renderHook(() => useResolveMapping(), { wrapper });
    let resolved: { mapping_status: string; datasetId: string } | undefined;
    await act(async () => {
      resolved = await resolveMapping.current.mutateAsync({ datasetId: dataset.id, mappingId: mapping.id });
    });
    expect(resolved?.mapping_status).toBe('resolved');
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: datasetKeys.mappings(dataset.id) });

    const { result: datasets } = renderHook(() => useBenchmarkDatasets(), { wrapper });
    await waitFor(() => expect(datasets.current.isSuccess).toBe(true));
    expect(datasets.current.data?.datasets.some((d) => d.id === dataset.id)).toBe(true);

    const { result: mappings } = renderHook(() => useDatasetMappings(dataset.id), { wrapper });
    await waitFor(() => expect(mappings.current.isSuccess).toBe(true));
    expect(mappings.current.data?.some((m) => m.id === mapping.id)).toBe(true);
  });

  it('creates a benchmark, starts it, cancels it, and fetches results', async () => {
    const queryClient = createTestQueryClient();
    const wrapper = createWrapper(queryClient);

    const benchmark: Benchmark = {
      id: 'bench-1',
      name: 'Bench 1',
      description: null,
      owner_id: 1,
      mapping_id: 1,
      status: 'pending',
      total_runs: 1,
      completed_runs: 0,
      failed_runs: 0,
      created_at: '2025-01-01T00:00:00Z',
      started_at: null,
      completed_at: null,
      operation_uuid: null,
    };
    vi.mocked(benchmarksApi.create).mockResolvedValue({ data: benchmark } as MockAxiosResponse<Benchmark>);
    vi.mocked(benchmarksApi.start).mockResolvedValue({
      data: { id: benchmark.id, status: 'running', operation_uuid: 'op-1', message: 'Benchmark execution started' },
    } as MockAxiosResponse<{ id: string; status: string; operation_uuid: string; message: string }>);

    const cancelled: Benchmark = { ...benchmark, status: 'cancelled', completed_at: '2025-01-02T00:00:00Z' };
    vi.mocked(benchmarksApi.cancel).mockResolvedValue({ data: cancelled } as MockAxiosResponse<Benchmark>);

    const runId = 'run-1';
    const results: BenchmarkResultsResponse = {
      benchmark_id: benchmark.id,
      primary_k: 10,
      k_values_for_metrics: [10],
      runs: [
        {
          id: runId,
          run_order: 0,
          config_hash: 'cfg-0',
          config: { search_mode: 'dense', use_reranker: false, top_k: 10, rrf_k: 60, score_threshold: null },
          status: 'completed',
          error_message: null,
          metrics: { mrr: 0.5, precision: {}, recall: {}, ndcg: {} },
          metrics_flat: {},
          timing: { indexing_ms: null, evaluation_ms: null, total_ms: null },
        },
      ],
      summary: { total_runs: 1, completed_runs: 1, failed_runs: 0 },
      total_runs: 1,
    };
    vi.mocked(benchmarksApi.getResults).mockResolvedValue({ data: results } as MockAxiosResponse<BenchmarkResultsResponse>);

    const perQuery: RunQueryResultsResponse = { run_id: runId, results: [], total: 0, page: 1, per_page: 50 };
    vi.mocked(benchmarksApi.getQueryResults).mockResolvedValue({
      data: perQuery,
    } as MockAxiosResponse<RunQueryResultsResponse>);

    const { result: createBenchmark } = renderHook(() => useCreateBenchmark(), { wrapper });
    let benchmarkId: string | undefined;
    await act(async () => {
      const created = await createBenchmark.current.mutateAsync({
        name: 'Bench 1',
        mapping_id: 1,
        config_matrix: {
          search_modes: ['dense'],
          use_reranker: [false],
          top_k_values: [10],
          rrf_k_values: [60],
          score_thresholds: [null],
        },
        top_k: 10,
        metrics_to_compute: ['mrr'],
      });
      benchmarkId = created.id;
    });
    expect(benchmarkId).toBe(benchmark.id);

    const { result: startBenchmark } = renderHook(() => useStartBenchmark(), { wrapper });
    await act(async () => {
      await startBenchmark.current.mutateAsync(benchmarkId as string);
    });

    const { result: cancelBenchmark } = renderHook(() => useCancelBenchmark(), { wrapper });
    let cancelledResult: Benchmark | undefined;
    await act(async () => {
      cancelledResult = await cancelBenchmark.current.mutateAsync(benchmarkId as string);
    });
    expect(cancelledResult?.status).toBe('cancelled');

    const { result: resultsHook } = renderHook(() => useBenchmarkResults(benchmarkId as string), { wrapper });
    await waitFor(() => expect(resultsHook.current.isSuccess).toBe(true));
    expect(resultsHook.current.data?.benchmark_id).toBe(benchmarkId);

    const { result: queryResultsHook } = renderHook(
      () => useBenchmarkQueryResults(benchmarkId as string, runId),
      { wrapper }
    );
    await waitFor(() => expect(queryResultsHook.current.isSuccess).toBe(true));
    expect(queryResultsHook.current.data?.run_id).toBe(runId);

    expect(queryClient.getQueryData(benchmarkKeys.results(benchmarkId as string))).toBeDefined();
  });

  it('fetches results after starting a benchmark', async () => {
    const queryClient = createTestQueryClient();
    const wrapper = createWrapper(queryClient);

    const benchmark: Benchmark = {
      id: 'bench-auto',
      name: 'Bench auto',
      description: null,
      owner_id: 1,
      mapping_id: 1,
      status: 'pending',
      total_runs: 1,
      completed_runs: 0,
      failed_runs: 0,
      created_at: '2025-01-01T00:00:00Z',
      started_at: null,
      completed_at: null,
      operation_uuid: null,
    };
    vi.mocked(benchmarksApi.create).mockResolvedValue({ data: benchmark } as MockAxiosResponse<Benchmark>);
    vi.mocked(benchmarksApi.start).mockResolvedValue({
      data: { id: benchmark.id, status: 'running', operation_uuid: 'op-auto', message: 'Benchmark execution started' },
    } as MockAxiosResponse<{ id: string; status: string; operation_uuid: string; message: string }>);

    const results: BenchmarkResultsResponse = {
      benchmark_id: benchmark.id,
      primary_k: 10,
      k_values_for_metrics: [10],
      runs: [],
      summary: { total_runs: 0, completed_runs: 0, failed_runs: 0 },
      total_runs: 0,
    };
    vi.mocked(benchmarksApi.getResults).mockResolvedValue({ data: results } as MockAxiosResponse<BenchmarkResultsResponse>);

    const { result: createBenchmark } = renderHook(() => useCreateBenchmark(), { wrapper });
    let benchmarkId: string | undefined;
    await act(async () => {
      benchmarkId = (await createBenchmark.current.mutateAsync({
        name: benchmark.name,
        mapping_id: 1,
        config_matrix: {
          search_modes: ['dense'],
          use_reranker: [false],
          top_k_values: [10],
          rrf_k_values: [60],
          score_thresholds: [null],
        },
        top_k: 10,
        metrics_to_compute: ['mrr'],
      })).id;
    });

    const { result: startBenchmark } = renderHook(() => useStartBenchmark(), { wrapper });
    await act(async () => {
      await startBenchmark.current.mutateAsync(benchmarkId as string);
    });

    const { result: resultsHook } = renderHook(() => useBenchmarkResults(benchmarkId as string), { wrapper });
    await waitFor(() => expect(resultsHook.current.isSuccess).toBe(true));
    expect(resultsHook.current.data?.benchmark_id).toBe(benchmarkId);
  });

  it('shows queued toast when mapping resolution returns an operation_uuid', async () => {
    const queryClient = createTestQueryClient();
    const wrapper = createWrapper(queryClient);

    vi.mocked(benchmarkDatasetsApi.resolveMapping).mockResolvedValue({
      data: {
        id: 1,
        operation_uuid: 'op-queued',
        mapping_status: 'pending',
        mapped_count: 0,
        total_count: 10,
        unresolved: [],
      },
    } as MockAxiosResponse<MappingResolveResponse>);

    const { result: resolveMapping } = renderHook(() => useResolveMapping(), { wrapper });
    await act(async () => {
      await resolveMapping.current.mutateAsync({ datasetId: 'ds-1', mappingId: 1 });
    });

    expect(addToast).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'info', message: expect.stringMatching(/queued/i) })
    );
  });

  it('restores cached list data when dataset deletion fails', async () => {
    const queryClient = createTestQueryClient();
    const wrapper = createWrapper(queryClient);

    const previous = { datasets: [], total: 0, page: 1, per_page: 50 };
    queryClient.setQueryData(datasetKeys.lists(), previous);
    const setQueryDataSpy = vi.spyOn(queryClient, 'setQueryData');

    vi.mocked(benchmarkDatasetsApi.delete).mockRejectedValue(new Error('nope'));

    const { result } = renderHook(() => useDeleteDataset(), { wrapper });

    await act(async () => {
      await result.current.mutateAsync('ds-1').catch(() => undefined);
    });

    expect(setQueryDataSpy).toHaveBeenCalledWith(datasetKeys.lists(), previous);
    expect(addToast).toHaveBeenCalledWith(expect.objectContaining({ type: 'error' }));
  });

  it('restores cached list data when benchmark deletion fails', async () => {
    const queryClient = createTestQueryClient();
    const wrapper = createWrapper(queryClient);

    const previous = { benchmarks: [], total: 0, page: 1, per_page: 50 };
    queryClient.setQueryData(benchmarkKeys.lists(), previous);
    const setQueryDataSpy = vi.spyOn(queryClient, 'setQueryData');

    vi.mocked(benchmarksApi.delete).mockRejectedValue(new Error('nope'));

    const { result } = renderHook(() => useDeleteBenchmark(), { wrapper });

    await act(async () => {
      await result.current.mutateAsync('bench-1').catch(() => undefined);
    });

    expect(setQueryDataSpy).toHaveBeenCalledWith(benchmarkKeys.lists(), previous);
    expect(addToast).toHaveBeenCalledWith(expect.objectContaining({ type: 'error' }));
  });
});
