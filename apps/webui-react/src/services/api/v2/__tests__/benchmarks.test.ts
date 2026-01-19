import { describe, it, expect, vi, beforeEach } from 'vitest'

const { mockGet, mockPost, mockDelete, mockDefaults } = vi.hoisted(() => ({
  mockGet: vi.fn(),
  mockPost: vi.fn(),
  mockDelete: vi.fn(),
  mockDefaults: { baseURL: 'http://api.example' },
}))

vi.mock('../client', () => ({
  default: {
    get: mockGet,
    post: mockPost,
    delete: mockDelete,
    defaults: mockDefaults,
  },
}))

import { benchmarkDatasetsApi, benchmarksApi } from '../benchmarks'

describe('benchmarksApi + benchmarkDatasetsApi', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockDefaults.baseURL = 'http://api.example'
  })

  it('lists datasets with pagination params', () => {
    benchmarkDatasetsApi.list({ page: 2, per_page: 25 })
    expect(mockGet).toHaveBeenCalledWith('/api/v2/benchmark-datasets', { params: { page: 2, per_page: 25 } })
  })

  it('gets a dataset by id', () => {
    benchmarkDatasetsApi.get('ds-1')
    expect(mockGet).toHaveBeenCalledWith('/api/v2/benchmark-datasets/ds-1')
  })

  it('uploads a dataset as multipart form data', () => {
    const file = new File([JSON.stringify({ schema_version: '1.0', queries: [] })], 'dataset.json', { type: 'application/json' })
    benchmarkDatasetsApi.upload({ name: 'My Dataset', description: 'desc' }, file)

    const [url, body, config] = mockPost.mock.calls[0]
    expect(url).toBe('/api/v2/benchmark-datasets')
    expect(body).toBeInstanceOf(FormData)
    expect((body as FormData).get('name')).toBe('My Dataset')
    expect((body as FormData).get('description')).toBe('desc')
    expect((body as FormData).get('file')).toBe(file)
    expect(config).toEqual({ headers: { 'Content-Type': 'multipart/form-data' } })
  })

  it('deletes a dataset', () => {
    benchmarkDatasetsApi.delete('ds-1')
    expect(mockDelete).toHaveBeenCalledWith('/api/v2/benchmark-datasets/ds-1')
  })

  it('creates and lists mappings and resolves mapping', () => {
    benchmarkDatasetsApi.createMapping('ds-1', { collection_id: 'col-1' })
    expect(mockPost).toHaveBeenCalledWith('/api/v2/benchmark-datasets/ds-1/mappings', { collection_id: 'col-1' })

    benchmarkDatasetsApi.listMappings('ds-1')
    expect(mockGet).toHaveBeenCalledWith('/api/v2/benchmark-datasets/ds-1/mappings')

    benchmarkDatasetsApi.resolveMapping('ds-1', 123)
    expect(mockPost).toHaveBeenCalledWith('/api/v2/benchmark-datasets/ds-1/mappings/123/resolve')
  })

  it('lists, gets, creates, starts, cancels, and deletes benchmarks', () => {
    benchmarksApi.list({ page: 1, per_page: 10 })
    expect(mockGet).toHaveBeenCalledWith('/api/v2/benchmarks', { params: { page: 1, per_page: 10 } })

    benchmarksApi.get('bench-1')
    expect(mockGet).toHaveBeenCalledWith('/api/v2/benchmarks/bench-1')

    benchmarksApi.create({
      name: 'Bench',
      mapping_id: 1,
      config_matrix: {
        search_modes: ['dense'],
        use_reranker: [false],
        top_k_values: [10],
        rrf_k_values: [60],
        score_thresholds: [null],
      },
    })
    expect(mockPost).toHaveBeenCalledWith('/api/v2/benchmarks', expect.any(Object))

    benchmarksApi.start('bench-1')
    expect(mockPost).toHaveBeenCalledWith('/api/v2/benchmarks/bench-1/start')

    benchmarksApi.cancel('bench-1')
    expect(mockPost).toHaveBeenCalledWith('/api/v2/benchmarks/bench-1/cancel')

    benchmarksApi.delete('bench-1')
    expect(mockDelete).toHaveBeenCalledWith('/api/v2/benchmarks/bench-1')
  })

  it('gets benchmark results and per-query results', () => {
    benchmarksApi.getResults('bench-1', { page: 2, per_page: 5 })
    expect(mockGet).toHaveBeenCalledWith('/api/v2/benchmarks/bench-1/results', { params: { page: 2, per_page: 5 } })

    benchmarksApi.getQueryResults('bench-1', 'run-1', { page: 3, per_page: 50 })
    expect(mockGet).toHaveBeenCalledWith(
      '/api/v2/benchmarks/bench-1/runs/run-1/queries',
      { params: { page: 3, per_page: 50 } }
    )
  })
})

