import { http, HttpResponse } from 'msw'
import type { Collection, Operation } from '../../types/collection'
import type { ApiKeyResponse, ApiKeyCreateResponse, ApiKeyListResponse } from '../../types/api-key'
import type {
  Benchmark,
  BenchmarkDataset,
  BenchmarkListResponse,
  BenchmarkResultsResponse,
  BenchmarkRun,
  BenchmarkRunStatus,
  DatasetListResponse,
  DatasetMapping,
  MappingResolveResponse,
  RunQueryResultsResponse,
} from '../../types/benchmark'

// =============================================================================
// Benchmark in-memory mock state (used by contract-level tests)
// =============================================================================

type BenchmarkMockState = {
  benchmarkDatasetsState: BenchmarkDataset[];
  datasetTotalRefsById: Record<string, number>;
  datasetMappingsState: DatasetMapping[];
  benchmarksState: Benchmark[];
  benchmarkRunsByBenchmarkId: Record<string, BenchmarkRun[]>;
  runQueryResultsByRunId: Record<string, RunQueryResultsResponse>;
  autoCompleteBenchmarksOnStart: boolean;
}

const benchmarkMockState: BenchmarkMockState = (() => {
  const globalState = globalThis as unknown as { __semantikBenchmarkMockState?: BenchmarkMockState }
  if (!globalState.__semantikBenchmarkMockState) {
    globalState.__semantikBenchmarkMockState = {
      benchmarkDatasetsState: [],
      datasetTotalRefsById: {},
      datasetMappingsState: [],
      benchmarksState: [],
      benchmarkRunsByBenchmarkId: {},
      runQueryResultsByRunId: {},
      autoCompleteBenchmarksOnStart: false,
    }
  }
  return globalState.__semantikBenchmarkMockState
})()

export function resetBenchmarkMocks(): void {
  benchmarkMockState.benchmarkDatasetsState = []
  benchmarkMockState.datasetTotalRefsById = {}
  benchmarkMockState.datasetMappingsState = []
  benchmarkMockState.benchmarksState = []
  benchmarkMockState.benchmarkRunsByBenchmarkId = {}
  benchmarkMockState.runQueryResultsByRunId = {}
  benchmarkMockState.autoCompleteBenchmarksOnStart = false
}

export function seedBenchmarkDataset(overrides: Partial<BenchmarkDataset> & { total_refs?: number } = {}): BenchmarkDataset {
  const now = new Date().toISOString()
  const dataset: BenchmarkDataset = {
    id: overrides.id ?? `ds-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    name: overrides.name ?? 'Seed Dataset',
    description: overrides.description ?? null,
    owner_id: overrides.owner_id ?? 1,
    query_count: overrides.query_count ?? 1,
    schema_version: overrides.schema_version ?? '1.0',
    created_at: overrides.created_at ?? now,
    updated_at: overrides.updated_at ?? null,
  }

  benchmarkMockState.benchmarkDatasetsState = [dataset, ...benchmarkMockState.benchmarkDatasetsState]
  benchmarkMockState.datasetTotalRefsById[dataset.id] = overrides.total_refs ?? dataset.query_count
  return dataset
}

export function seedDatasetMapping(
  overrides: Partial<DatasetMapping> & { dataset_id: string; collection_id: string }
): DatasetMapping {
  const now = new Date().toISOString()
  const mapping: DatasetMapping = {
    id: overrides.id ?? Math.floor(Math.random() * 100000),
    dataset_id: overrides.dataset_id,
    collection_id: overrides.collection_id,
    mapping_status: overrides.mapping_status ?? 'pending',
    mapped_count: overrides.mapped_count ?? 0,
    total_count: overrides.total_count ?? benchmarkMockState.datasetTotalRefsById[overrides.dataset_id] ?? 0,
    created_at: overrides.created_at ?? now,
    resolved_at: overrides.resolved_at ?? null,
  }
  benchmarkMockState.datasetMappingsState = [mapping, ...benchmarkMockState.datasetMappingsState]
  return mapping
}

export function setBenchmarkAutoCompleteOnStart(enabled: boolean): void {
  benchmarkMockState.autoCompleteBenchmarksOnStart = enabled
}

function _markBenchmarkCompleted(benchmarkId: string): void {
  const index = benchmarkMockState.benchmarksState.findIndex((b) => b.id === benchmarkId)
  if (index < 0) return

  const now = new Date().toISOString()
  const runs = (benchmarkMockState.benchmarkRunsByBenchmarkId[benchmarkId] ?? []).map((run, idx) => ({
    ...run,
    status: 'completed' as BenchmarkRunStatus,
    metrics: {
      mrr: 0.5 + idx * 0.05,
      precision: { '10': 0.6 + idx * 0.02 },
      recall: { '10': 0.4 + idx * 0.02 },
      ndcg: { '10': 0.55 + idx * 0.03 },
    },
    timing: {
      indexing_ms: null,
      evaluation_ms: null,
      total_ms: 200 + idx * 50,
    },
  }))

  benchmarkMockState.benchmarkRunsByBenchmarkId[benchmarkId] = runs
  const updated: Benchmark = {
    ...benchmarkMockState.benchmarksState[index],
    status: 'completed',
    completed_at: now,
    completed_runs: runs.length,
    failed_runs: 0,
  }
  benchmarkMockState.benchmarksState = [
    ...benchmarkMockState.benchmarksState.slice(0, index),
    updated,
    ...benchmarkMockState.benchmarksState.slice(index + 1),
  ]
}

function _paginate<T>(items: T[], pageRaw: string | null, perPageRaw: string | null) {
  const page = Math.max(1, Number(pageRaw ?? 1))
  const perPage = Math.min(100, Math.max(1, Number(perPageRaw ?? 50)))
  const start = (page - 1) * perPage
  const end = start + perPage
  return { page, perPage, total: items.length, items: items.slice(start, end) }
}

function _countTotalRefsFromDatasetFile(raw: unknown): { schemaVersion: string; queryCount: number; totalRefs: number } {
  if (!raw || typeof raw !== 'object') return { schemaVersion: '1.0', queryCount: 0, totalRefs: 0 }
  const obj = raw as Record<string, unknown>
  const schemaVersion = typeof obj.schema_version === 'string' ? obj.schema_version : '1.0'
  const queries = Array.isArray(obj.queries) ? obj.queries : []

  let totalRefs = 0
  for (const q of queries) {
    if (!q || typeof q !== 'object') continue
    const qObj = q as Record<string, unknown>
    const relevantDocs = Array.isArray(qObj.relevant_docs) ? qObj.relevant_docs : []
    totalRefs += relevantDocs.length
  }

  return { schemaVersion, queryCount: queries.length, totalRefs }
}

async function _readMultipartFileText(file: unknown): Promise<string> {
  if (!file || typeof file !== 'object') {
    throw new Error('Invalid file')
  }

  const maybeText = (file as { text?: unknown }).text
  if (typeof maybeText === 'function') {
    return await (maybeText as () => Promise<string>)()
  }

  const maybeArrayBuffer = (file as { arrayBuffer?: unknown }).arrayBuffer
  if (typeof maybeArrayBuffer === 'function') {
    const buffer = await (maybeArrayBuffer as () => Promise<ArrayBuffer>)()
    return new TextDecoder().decode(buffer)
  }

  throw new Error('Invalid file')
}

export const handlers = [
  // Auth endpoints
  http.post('/api/auth/login', async ({ request }) => {
    const { username, password } = await request.json() as { username: string; password: string }
    
    if (username === 'testuser' && password === 'testpass') {
      return HttpResponse.json({
        access_token: 'mock-jwt-token',
        refresh_token: 'mock-refresh-token',
        user: {
          id: 1,
          username: 'testuser',
          email: 'test@example.com',
          full_name: 'Test User',
          is_active: true,
          created_at: new Date().toISOString(),
        }
      })
    }
    
    return HttpResponse.json(
      { detail: 'Invalid credentials' },
      { status: 401 }
    )
  }),

  http.get('/api/auth/me', () => {
    return HttpResponse.json({
      id: 1,
      username: 'testuser',
      email: 'test@example.com',
      full_name: 'Test User',
      is_active: true,
      created_at: new Date().toISOString(),
    })
  }),

  http.post('/api/auth/logout', () => {
    return HttpResponse.json({ message: 'Logged out successfully' })
  }),

  // Collections endpoints
  http.get('/api/collections', () => {
    return HttpResponse.json([
      {
        name: 'test-collection',
        total_files: 100,
        total_vectors: 500,
        model_name: 'Qwen/Qwen3-Embedding-0.6B',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      }
    ])
  }),

  http.get('/api/collections/:name', ({ params }) => {
    return HttpResponse.json({
      name: params.name,
      document_count: 100,
      total_chunks: 500,
      avg_chunks_per_doc: 5,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    })
  }),

  http.put('/api/collections/:name', async ({ params, request }) => {
    const body = await request.json() as { new_name: string }
    const { new_name } = body
    
    return HttpResponse.json({
      message: `Collection ${params.name} renamed to ${new_name}`,
      collection: {
        name: body.new_name,
        document_count: 100,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      }
    })
  }),

  http.delete('/api/collections/:name', ({ params }) => {
    return HttpResponse.json({
      message: `Collection ${params.name} deleted successfully`
    })
  }),

  // Search endpoints
  http.post('/api/search', async ({ request }) => {
    const body = await request.json() as { query: string; collection?: string; search_type?: string }
    
    return HttpResponse.json({
      results: [
        {
          id: '1',
          chunk_id: 'chunk-1',
          content: 'Test search result content',
          metadata: {
            file_path: '/test/document.txt',
            chunk_index: 0,
            doc_id: 'doc-1',
          },
          score: 0.95,
        }
      ],
      query: body.query,
      collection: body.collection,
      search_type: body.search_type || 'vector',
      total_results: 1,
    })
  }),

  // Models endpoints - returns dict of models with provider metadata
  http.get('/api/models', () => {
    return HttpResponse.json({
      models: {
        'BAAI/bge-large-en-v1.5': {
          model_name: 'BAAI/bge-large-en-v1.5',
          dimension: 1024,
          description: 'BGE Large English v1.5 (High quality)',
          provider: 'dense_local',
          supports_quantization: true,
          recommended_quantization: 'float16',
          is_asymmetric: true,
        },
        'Qwen/Qwen3-Embedding-0.6B': {
          model_name: 'Qwen/Qwen3-Embedding-0.6B',
          dimension: 1024,
          description: 'Qwen3-Embedding-0.6B (Default - Fast)',
          provider: 'dense_local',
          supports_quantization: true,
          recommended_quantization: 'float16',
          is_asymmetric: true,
        },
        'sentence-transformers/all-MiniLM-L6-v2': {
          model_name: 'sentence-transformers/all-MiniLM-L6-v2',
          dimension: 384,
          description: 'All-MiniLM-L6-v2 (Lightweight)',
          provider: 'dense_local',
          supports_quantization: true,
          recommended_quantization: 'float32',
          is_asymmetric: false,
        },
        'test-plugin/model-v1': {
          model_name: 'test-plugin/model-v1',
          dimension: 768,
          description: 'Test Plugin Model',
          provider: 'test_plugin',
          supports_quantization: false,
          is_asymmetric: false,
        },
      },
      current_device: 'cuda:0',
      using_real_embeddings: true,
    })
  }),

  // Settings endpoints
  http.get('/api/settings/stats', () => {
    return HttpResponse.json({
      file_count: 100,
      database_size_mb: 50,
      parquet_files_count: 10,
      parquet_size_mb: 25,
    })
  }),

  http.post('/api/settings/reset-database', () => {
    return HttpResponse.json({ message: 'Database reset successfully' })
  }),

  // V2 API endpoints
  // System info
  http.get('*/api/v2/system/info', () => {
    return HttpResponse.json({
      version: '0.8.0',
      environment: 'development',
      python_version: '3.11.0',
      rate_limits: {
        chunking_preview: '10/minute',
        plugin_install: '5/minute',
        llm_test: '3/minute',
      },
    })
  }),

  // System health
  http.get('*/api/v2/system/health', () => {
    return HttpResponse.json({
      postgres: { status: 'healthy', message: 'Connected' },
      redis: { status: 'healthy', message: 'Connected' },
      qdrant: { status: 'healthy', message: 'Connected' },
      vecpipe: { status: 'healthy', message: 'Connected' },
    })
  }),

  // System status
  http.get('*/api/v2/system/status', () => {
    return HttpResponse.json({
      healthy: true,
      version: '0.8.0',
      services: {
        database: 'healthy',
        redis: 'healthy',
        qdrant: 'healthy',
      },
      reranking_available: true,
      gpu_available: true,
      gpu_memory_mb: 8192,
      cuda_device_name: 'NVIDIA GeForce RTX 4090',
      cuda_device_count: 1,
      available_reranking_models: ['Qwen/Qwen3-Reranker-0.6B'],
    })
  }),

  // Collections
  http.get('*/api/v2/collections', () => {
    return HttpResponse.json({
      collections: [
        {
          id: '123e4567-e89b-12d3-a456-426614174000',
          name: 'Test Collection 1',
          description: 'Test collection',
          owner_id: 1,
          vector_store_name: 'test_collection_vectors',
          embedding_model: 'Qwen/Qwen3-Embedding-0.6B',
          quantization: 'float32',
          chunk_size: 1000,
          chunk_overlap: 200,
          is_public: false,
          status: 'ready',
          document_count: 10,
          vector_count: 100,
          total_size_bytes: 1048576,
          created_at: '2025-01-01T00:00:00Z',
          updated_at: '2025-01-01T00:00:00Z',
        },
        {
          id: '456e7890-e89b-12d3-a456-426614174001',
          name: 'Test Collection 2',
          description: 'Another test collection',
          owner_id: 1,
          vector_store_name: 'test_collection_2_vectors',
          embedding_model: 'BAAI/bge-small-en-v1.5',
          quantization: 'float32',
          chunk_size: 512,
          chunk_overlap: 50,
          is_public: false,
          status: 'ready',
          document_count: 20,
          vector_count: 200,
          total_size_bytes: 2097152,
          created_at: '2025-01-01T00:00:00Z',
          updated_at: '2025-01-01T00:00:00Z',
        }
      ],
      total: 2,
      page: 1,
      page_size: 10,
    })
  }),

  http.get('*/api/v2/collections/:uuid', ({ params }) => {
    const collection: Collection = {
      id: params.uuid as string,
      name: 'Test Collection',
      description: 'Test description',
      owner_id: 1,
      vector_store_name: 'test_collection_vectors',
      embedding_model: 'text-embedding-ada-002',
      quantization: 'float32',
      chunk_size: 1000,
      chunk_overlap: 200,
      is_public: false,
      status: 'ready',
      document_count: 150,
      vector_count: 2500,
      total_size_bytes: 1048576,
      created_at: '2025-01-01T00:00:00Z',
      updated_at: '2025-01-01T00:00:00Z',
    }
    return HttpResponse.json(collection)
  }),

  http.post('*/api/v2/collections/:uuid/reindex', ({ params }) => {
    const operation: Operation = {
      id: 'op-' + Date.now(),
      collection_id: params.uuid as string,
      operation_type: 'reindex',
      status: 'pending',
      started_at: new Date().toISOString(),
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      config: {},
      progress: {
        current: 0,
        total: 100,
        percentage: 0,
        message: 'Starting reindex operation',
      },
    }
    return HttpResponse.json(operation)
  }),

  http.delete('*/api/v2/collections/:uuid', () => {
    return HttpResponse.json({ message: 'Collection deleted successfully' })
  }),

  // Sparse index endpoints
  http.get('*/api/v2/collections/:uuid/sparse-index', () => {
    return HttpResponse.json({
      enabled: false,
      plugin_id: null,
      plugin_config: null,
      indexed_documents: 0,
      total_documents: 150,
      last_indexed_at: null,
    })
  }),

  http.post('*/api/v2/collections/:uuid/sparse-index', async ({ request }) => {
    const body = await request.json() as { plugin_id: string; config?: Record<string, unknown> }
    return HttpResponse.json({
      enabled: true,
      plugin_id: body.plugin_id,
      plugin_config: body.config || { k1: 1.2, b: 0.75 },
      indexed_documents: 0,
      total_documents: 150,
      last_indexed_at: null,
    })
  }),

  http.delete('*/api/v2/collections/:uuid/sparse-index', () => {
    return HttpResponse.json({
      enabled: false,
      plugin_id: null,
      plugin_config: null,
      indexed_documents: 0,
      total_documents: 150,
      last_indexed_at: null,
    })
  }),

  http.post('*/api/v2/collections/:uuid/sparse-index/reindex', () => {
    return HttpResponse.json({
      job_id: 'mock-reindex-job-' + Date.now(),
      status: 'pending',
      message: 'Sparse reindex job queued',
    })
  }),

  http.get('*/api/v2/collections/:uuid/sparse-index/reindex/:jobId', () => {
    return HttpResponse.json({
      job_id: 'mock-reindex-job',
      status: 'completed',
      progress: 100,
      processed_documents: 150,
      total_documents: 150,
      started_at: new Date(Date.now() - 60000).toISOString(),
      completed_at: new Date().toISOString(),
    })
  }),

  // Connectors catalog endpoint
  http.get('*/api/v2/connectors', () => {
    return HttpResponse.json({
      directory: {
        name: 'Directory',
        description: 'Index files from a local directory',
        icon: 'folder',
        fields: [
          {
            name: 'path',
            type: 'text',
            label: 'Directory Path',
            required: true,
            placeholder: '/path/to/documents',
          },
        ],
        secrets: [],
        supports_sync: true,
      },
      git: {
        name: 'Git Repository',
        description: 'Clone and index a Git repository',
        icon: 'git-branch',
        fields: [
          {
            name: 'url',
            type: 'text',
            label: 'Repository URL',
            required: true,
            placeholder: 'https://github.com/user/repo.git',
          },
          {
            name: 'branch',
            type: 'text',
            label: 'Branch',
            required: false,
            placeholder: 'main',
            default: 'main',
          },
        ],
        secrets: [
          {
            name: 'token',
            label: 'Personal Access Token',
            required: false,
            placeholder: 'ghp_xxxx (optional for public repos)',
          },
        ],
        supports_sync: true,
      },
      imap: {
        name: 'Email (IMAP)',
        description: 'Index emails from an IMAP server',
        icon: 'mail',
        fields: [
          {
            name: 'host',
            type: 'text',
            label: 'IMAP Server',
            required: true,
            placeholder: 'imap.gmail.com',
          },
          {
            name: 'username',
            type: 'text',
            label: 'Username',
            required: true,
            placeholder: 'user@example.com',
          },
          {
            name: 'folder',
            type: 'text',
            label: 'Folder',
            required: false,
            placeholder: 'INBOX',
            default: 'INBOX',
          },
        ],
        secrets: [
          {
            name: 'password',
            label: 'Password',
            required: true,
            placeholder: 'App password',
          },
        ],
        supports_sync: true,
      },
    })
  }),

  // Search endpoint
  http.post('*/api/v2/search', async ({ request }) => {
    const body = await request.json() as {
      use_reranker?: boolean;
      use_hyde?: boolean;
      collection_uuids?: string[];
      rerank_model?: string;
    }

    return HttpResponse.json({
      results: [
        {
          document_id: 'doc_1',
          chunk_id: 'chunk_1',
          score: body.use_reranker ? 0.95 : 0.85,
          text: body.use_reranker ? 'Test result with reranking' : 'Test result without reranking',
          file_path: '/test.txt',
          file_name: 'test.txt',
          collection_id: body.collection_uuids?.[0] || '123e4567-e89b-12d3-a456-426614174000',
          collection_name: 'Test Collection 1',
        }
      ],
      total_results: 1,
      reranking_used: body.use_reranker || false,
      reranker_model: body.use_reranker ? (body.rerank_model || 'Qwen/Qwen3-Reranker-0.6B') : null,
      reranking_time_ms: body.use_reranker ? 50 : undefined,
      search_time_ms: body.use_reranker ? 100 : 50,
      total_time_ms: body.use_reranker ? 150 : 50,
      partial_failure: false,
      hyde_used: body.use_hyde || false,
      hyde_info: body.use_hyde ? {
        expanded_query: 'This is a hypothetical document generated for the query.',
        generation_time_ms: 250,
        tokens_used: 150,
        provider: 'anthropic',
        model: 'claude-3-5-haiku-20241022',
      } : null,
    })
  }),

  // LLM Settings endpoints
  http.get('*/api/v2/llm/settings', () => {
    return HttpResponse.json({
      high_quality_provider: 'anthropic',
      high_quality_model: 'claude-opus-4-5-20251101',
      low_quality_provider: 'anthropic',
      low_quality_model: 'claude-3-5-haiku-20241022',
      anthropic_has_key: true,
      openai_has_key: false,
      default_temperature: 0.7,
      default_max_tokens: 4096,
      created_at: '2025-01-01T00:00:00Z',
      updated_at: '2025-01-01T00:00:00Z',
    })
  }),

  http.put('*/api/v2/llm/settings', async ({ request }) => {
    const body = await request.json() as Record<string, unknown>
    return HttpResponse.json({
      high_quality_provider: body.high_quality_provider ?? 'anthropic',
      high_quality_model: body.high_quality_model ?? 'claude-opus-4-5-20251101',
      low_quality_provider: body.low_quality_provider ?? 'anthropic',
      low_quality_model: body.low_quality_model ?? 'claude-3-5-haiku-20241022',
      anthropic_has_key: !!body.anthropic_api_key || true,
      openai_has_key: !!body.openai_api_key || false,
      default_temperature: body.default_temperature ?? 0.7,
      default_max_tokens: body.default_max_tokens ?? 4096,
      created_at: '2025-01-01T00:00:00Z',
      updated_at: new Date().toISOString(),
    })
  }),

  http.get('*/api/v2/llm/models', () => {
    return HttpResponse.json({
      models: [
        {
          id: 'claude-opus-4-5-20251101',
          name: 'Opus 4.5',
          display_name: 'Claude - Opus 4.5',
          provider: 'anthropic',
          tier_recommendation: 'high',
          context_window: 200000,
          description: 'Most capable Claude model for complex tasks',
          is_curated: true,
        },
        {
          id: 'claude-sonnet-4-20250514',
          name: 'Sonnet 4',
          display_name: 'Claude - Sonnet 4',
          provider: 'anthropic',
          tier_recommendation: 'high',
          context_window: 200000,
          description: 'Balanced performance and cost',
          is_curated: true,
        },
        {
          id: 'claude-3-5-haiku-20241022',
          name: 'Haiku 3.5',
          display_name: 'Claude - Haiku 3.5',
          provider: 'anthropic',
          tier_recommendation: 'low',
          context_window: 200000,
          description: 'Fast and cost-effective for simple tasks',
          is_curated: true,
        },
        {
          id: 'gpt-4o',
          name: 'GPT-4o',
          display_name: 'OpenAI - GPT-4o',
          provider: 'openai',
          tier_recommendation: 'high',
          context_window: 128000,
          description: 'OpenAI flagship model with vision',
          is_curated: true,
        },
        {
          id: 'gpt-4o-mini',
          name: 'GPT-4o Mini',
          display_name: 'OpenAI - GPT-4o Mini',
          provider: 'openai',
          tier_recommendation: 'low',
          context_window: 128000,
          description: 'Fast and affordable for simple tasks',
          is_curated: true,
        },
      ],
    })
  }),

  http.post('*/api/v2/llm/models/refresh', async ({ request }) => {
    const payload = (await request.json()) as { provider?: string }
    const provider = payload.provider

    if (provider === 'anthropic') {
      return HttpResponse.json({
        models: [
          {
            id: 'claude-opus-4-5-20251101',
            name: 'Opus 4.5',
            display_name: 'Claude - Opus 4.5',
            provider: 'anthropic',
            tier_recommendation: 'high',
            context_window: 200000,
            description: 'Claude model: claude-opus-4-5-20251101',
            is_curated: false,
          },
          {
            id: 'claude-sonnet-4-20250514',
            name: 'Sonnet 4',
            display_name: 'Claude - Sonnet 4',
            provider: 'anthropic',
            tier_recommendation: 'high',
            context_window: 200000,
            description: 'Claude model: claude-sonnet-4-20250514',
            is_curated: false,
          },
        ],
      })
    }

    return HttpResponse.json({
      models: [
        {
          id: 'gpt-4o',
          name: 'GPT-4o',
          display_name: 'OpenAI - GPT-4o',
          provider: 'openai',
          tier_recommendation: 'high',
          context_window: 128000,
          description: 'OpenAI model: gpt-4o',
          is_curated: false,
        },
      ],
    })
  }),

  http.post('*/api/v2/llm/test', async ({ request }) => {
    const body = await request.json() as { provider: string; api_key: string }

    // Simulate invalid API key
    if (body.api_key === 'invalid-key') {
      return HttpResponse.json({
        success: false,
        message: 'Invalid API key',
        model_tested: null,
      })
    }

    return HttpResponse.json({
      success: true,
      message: `Successfully connected to ${body.provider} API`,
      model_tested: body.provider === 'anthropic' ? 'claude-3-5-haiku-20241022' : 'gpt-4o-mini',
    })
  }),

  http.get('*/api/v2/llm/usage', () => {
    return HttpResponse.json({
      total_input_tokens: 58023,
      total_output_tokens: 30245,
      total_tokens: 88268,
      by_feature: {
        hyde: { input_tokens: 12345, output_tokens: 6789, total_tokens: 19134 },
        summary: { input_tokens: 45678, output_tokens: 23456, total_tokens: 69134 },
      },
      by_provider: {
        anthropic: { input_tokens: 50000, output_tokens: 25000, total_tokens: 75000 },
        openai: { input_tokens: 8023, output_tokens: 5245, total_tokens: 13268 },
      },
      event_count: 156,
      period_days: 30,
    })
  }),

  // User Preferences endpoints
  http.get('*/api/v2/preferences', () => {
    return HttpResponse.json({
      search: {
        top_k: 10,
        mode: 'dense',
        use_reranker: false,
        rrf_k: 60,
        similarity_threshold: null,
        use_hyde: false,
        hyde_quality_tier: 'low',
        hyde_timeout_seconds: 10,
      },
      collection_defaults: {
        embedding_model: null,
        quantization: 'float16',
        chunking_strategy: 'recursive',
        chunk_size: 1024,
        chunk_overlap: 200,
        enable_sparse: false,
        sparse_type: 'bm25',
        enable_hybrid: false,
      },
      interface: {
        data_refresh_interval_ms: 30000,
        visualization_sample_limit: 200000,
        animation_enabled: true,
      },
      created_at: '2025-01-01T00:00:00Z',
      updated_at: '2025-01-01T00:00:00Z',
    })
  }),

  http.put('*/api/v2/preferences', async ({ request }) => {
    const body = await request.json() as Record<string, unknown>
    return HttpResponse.json({
      search: {
        top_k: (body.search as Record<string, unknown>)?.top_k ?? 10,
        mode: (body.search as Record<string, unknown>)?.mode ?? 'dense',
        use_reranker: (body.search as Record<string, unknown>)?.use_reranker ?? false,
        rrf_k: (body.search as Record<string, unknown>)?.rrf_k ?? 60,
        similarity_threshold: (body.search as Record<string, unknown>)?.similarity_threshold ?? null,
        use_hyde: (body.search as Record<string, unknown>)?.use_hyde ?? false,
        hyde_quality_tier: (body.search as Record<string, unknown>)?.hyde_quality_tier ?? 'low',
        hyde_timeout_seconds: (body.search as Record<string, unknown>)?.hyde_timeout_seconds ?? 10,
      },
      collection_defaults: {
        embedding_model: (body.collection_defaults as Record<string, unknown>)?.embedding_model ?? null,
        quantization: (body.collection_defaults as Record<string, unknown>)?.quantization ?? 'float16',
        chunking_strategy: (body.collection_defaults as Record<string, unknown>)?.chunking_strategy ?? 'recursive',
        chunk_size: (body.collection_defaults as Record<string, unknown>)?.chunk_size ?? 1024,
        chunk_overlap: (body.collection_defaults as Record<string, unknown>)?.chunk_overlap ?? 200,
        enable_sparse: (body.collection_defaults as Record<string, unknown>)?.enable_sparse ?? false,
        sparse_type: (body.collection_defaults as Record<string, unknown>)?.sparse_type ?? 'bm25',
        enable_hybrid: (body.collection_defaults as Record<string, unknown>)?.enable_hybrid ?? false,
      },
      interface: {
        data_refresh_interval_ms: (body.interface as Record<string, unknown>)?.data_refresh_interval_ms ?? 30000,
        visualization_sample_limit: (body.interface as Record<string, unknown>)?.visualization_sample_limit ?? 200000,
        animation_enabled: (body.interface as Record<string, unknown>)?.animation_enabled ?? true,
      },
      created_at: '2025-01-01T00:00:00Z',
      updated_at: new Date().toISOString(),
    })
  }),

  http.post('*/api/v2/preferences/reset/search', () => {
    return HttpResponse.json({
      search: {
        top_k: 10,
        mode: 'dense',
        use_reranker: false,
        rrf_k: 60,
        similarity_threshold: null,
        use_hyde: false,
        hyde_quality_tier: 'low',
        hyde_timeout_seconds: 10,
      },
      collection_defaults: {
        embedding_model: null,
        quantization: 'float16',
        chunking_strategy: 'recursive',
        chunk_size: 1024,
        chunk_overlap: 200,
        enable_sparse: false,
        sparse_type: 'bm25',
        enable_hybrid: false,
      },
      interface: {
        data_refresh_interval_ms: 30000,
        visualization_sample_limit: 200000,
        animation_enabled: true,
      },
      created_at: '2025-01-01T00:00:00Z',
      updated_at: new Date().toISOString(),
    })
  }),

  http.post('*/api/v2/preferences/reset/collection-defaults', () => {
    return HttpResponse.json({
      search: {
        top_k: 10,
        mode: 'dense',
        use_reranker: false,
        rrf_k: 60,
        similarity_threshold: null,
        use_hyde: false,
        hyde_quality_tier: 'low',
        hyde_timeout_seconds: 10,
      },
      collection_defaults: {
        embedding_model: null,
        quantization: 'float16',
        chunking_strategy: 'recursive',
        chunk_size: 1024,
        chunk_overlap: 200,
        enable_sparse: false,
        sparse_type: 'bm25',
        enable_hybrid: false,
      },
      interface: {
        data_refresh_interval_ms: 30000,
        visualization_sample_limit: 200000,
        animation_enabled: true,
      },
      created_at: '2025-01-01T00:00:00Z',
      updated_at: new Date().toISOString(),
    })
  }),

  http.post('*/api/v2/preferences/reset/interface', () => {
    return HttpResponse.json({
      search: {
        top_k: 10,
        mode: 'dense',
        use_reranker: false,
        rrf_k: 60,
        similarity_threshold: null,
        use_hyde: false,
        hyde_quality_tier: 'low',
        hyde_timeout_seconds: 10,
      },
      collection_defaults: {
        embedding_model: null,
        quantization: 'float16',
        chunking_strategy: 'recursive',
        chunk_size: 1024,
        chunk_overlap: 200,
        enable_sparse: false,
        sparse_type: 'bm25',
        enable_hybrid: false,
      },
      interface: {
        data_refresh_interval_ms: 30000,
        visualization_sample_limit: 200000,
        animation_enabled: true,
      },
      created_at: '2025-01-01T00:00:00Z',
      updated_at: new Date().toISOString(),
    })
  }),

  // System Settings endpoints (admin-only)
  http.get('*/api/v2/system-settings', () => {
    return HttpResponse.json({
      settings: {
        max_collections_per_user: { value: 10, updated_at: null, updated_by: null },
        max_storage_gb_per_user: { value: 50, updated_at: null, updated_by: null },
        max_document_size_mb: { value: 100, updated_at: null, updated_by: null },
      },
    })
  }),

  http.get('*/api/v2/system-settings/effective', () => {
    return HttpResponse.json({
      settings: {
        // GPU & Memory settings
        gpu_memory_max_percent: 0.9,
        cpu_memory_max_percent: 0.5,
        enable_cpu_offload: true,
        eviction_idle_threshold_seconds: 120,
        // Search & Rerank settings
        rerank_candidate_multiplier: 5,
        rerank_min_candidates: 20,
        rerank_max_candidates: 200,
        rerank_hybrid_weight: 0.3,
        // Resource limits
        max_collections_per_user: 10,
        max_storage_gb_per_user: 50,
        max_document_size_mb: 100,
        // Performance settings
        cache_ttl_seconds: 300,
        model_unload_timeout_seconds: 300,
      },
    })
  }),

  http.get('*/api/v2/system-settings/defaults', () => {
    return HttpResponse.json({
      defaults: {
        // GPU & Memory settings
        gpu_memory_max_percent: 0.9,
        cpu_memory_max_percent: 0.5,
        enable_cpu_offload: true,
        eviction_idle_threshold_seconds: 120,
        // Search & Rerank settings
        rerank_candidate_multiplier: 5,
        rerank_min_candidates: 20,
        rerank_max_candidates: 200,
        rerank_hybrid_weight: 0.3,
        // Resource limits
        max_collections_per_user: 10,
        max_storage_gb_per_user: 50,
        max_document_size_mb: 100,
        // Performance settings
        cache_ttl_seconds: 300,
        model_unload_timeout_seconds: 300,
      },
    })
  }),

  http.patch('*/api/v2/system-settings', async ({ request }) => {
    const body = await request.json() as { settings: Record<string, unknown> }
    const keys = Object.keys(body.settings || {})
    return HttpResponse.json({
      updated: keys,
      settings: keys.reduce((acc, key) => ({
        ...acc,
        [key]: { value: body.settings[key], updated_at: new Date().toISOString(), updated_by: 1 },
      }), {}),
    })
  }),

  // API Keys endpoints
  http.get('*/api/v2/api-keys', () => {
    const mockApiKeys: ApiKeyResponse[] = [
      {
        id: 'key-1-uuid',
        name: 'Development Key',
        is_active: true,
        permissions: null,
        last_used_at: '2025-01-10T12:00:00Z',
        expires_at: '2026-01-01T00:00:00Z',
        created_at: '2025-01-01T00:00:00Z',
      },
      {
        id: 'key-2-uuid',
        name: 'CI/CD Key',
        is_active: true,
        permissions: null,
        last_used_at: null,
        expires_at: null,
        created_at: '2025-01-05T00:00:00Z',
      },
      {
        id: 'key-3-uuid',
        name: 'Revoked Key',
        is_active: false,
        permissions: null,
        last_used_at: '2025-01-08T09:00:00Z',
        expires_at: '2026-06-01T00:00:00Z',
        created_at: '2025-01-02T00:00:00Z',
      },
    ]
    const response: ApiKeyListResponse = {
      api_keys: mockApiKeys,
      total: mockApiKeys.length,
    }
    return HttpResponse.json(response)
  }),

  http.get('*/api/v2/api-keys/:keyId', ({ params }) => {
    const key: ApiKeyResponse = {
      id: params.keyId as string,
      name: 'Test Key',
      is_active: true,
      permissions: null,
      last_used_at: '2025-01-10T12:00:00Z',
      expires_at: '2026-01-01T00:00:00Z',
      created_at: '2025-01-01T00:00:00Z',
    }
    return HttpResponse.json(key)
  }),

  http.post('*/api/v2/api-keys', async ({ request }) => {
    const body = await request.json() as { name: string; expires_in_days?: number | null }

    // Simulate duplicate name error
    if (body.name === 'duplicate-name') {
      return HttpResponse.json(
        { detail: 'API key with this name already exists' },
        { status: 409 }
      )
    }

    // Simulate limit reached error
    if (body.name === 'limit-reached') {
      return HttpResponse.json(
        { detail: 'Maximum API keys limit reached (10)' },
        { status: 400 }
      )
    }

    const expiresAt = body.expires_in_days
      ? new Date(Date.now() + body.expires_in_days * 24 * 60 * 60 * 1000).toISOString()
      : null

    const response: ApiKeyCreateResponse = {
      id: 'new-key-uuid-' + Date.now(),
      name: body.name,
      is_active: true,
      permissions: null,
      last_used_at: null,
      expires_at: expiresAt,
      created_at: new Date().toISOString(),
      api_key: 'smtk_' + Math.random().toString(36).substring(2, 34),
    }
    return HttpResponse.json(response, { status: 201 })
  }),

  http.patch('*/api/v2/api-keys/:keyId', async ({ params, request }) => {
    const body = await request.json() as { is_active: boolean }
    const response: ApiKeyResponse = {
      id: params.keyId as string,
      name: 'Updated Key',
      is_active: body.is_active,
      permissions: null,
      last_used_at: '2025-01-10T12:00:00Z',
      expires_at: '2026-01-01T00:00:00Z',
      created_at: '2025-01-01T00:00:00Z',
    }
    return HttpResponse.json(response)
  }),

  // =============================================================================
  // Benchmark Datasets + Benchmarks (v2)
  // =============================================================================

  http.get('*/api/v2/benchmark-datasets', ({ request }) => {
    const { page, perPage, total, items } = _paginate(
      benchmarkMockState.benchmarkDatasetsState,
      request.url.searchParams.get('page'),
      request.url.searchParams.get('per_page'),
    )

    const response: DatasetListResponse = {
      datasets: items,
      total,
      page,
      per_page: perPage,
    }
    return HttpResponse.json(response)
  }),

  http.get('*/api/v2/benchmark-datasets/:datasetId', ({ params }) => {
    const dataset = benchmarkMockState.benchmarkDatasetsState.find((d) => d.id === String(params.datasetId))
    if (!dataset) {
      return HttpResponse.json({ detail: 'Dataset not found' }, { status: 404 })
    }
    return HttpResponse.json(dataset)
  }),

  http.post('*/api/v2/benchmark-datasets', async ({ request }) => {
    const form = await request.formData()
    const name = String(form.get('name') ?? '').trim()
    const descriptionRaw = form.get('description')
    const description = typeof descriptionRaw === 'string' && descriptionRaw.trim() ? descriptionRaw : null
    const file = form.get('file')

    if (!name) {
      return HttpResponse.json({ detail: 'name is required' }, { status: 422 })
    }
    if (!file) {
      return HttpResponse.json({ detail: 'file is required' }, { status: 422 })
    }

    let parsed: unknown = null
    try {
      parsed = JSON.parse(await _readMultipartFileText(file))
    } catch {
      return HttpResponse.json({ detail: 'Invalid JSON' }, { status: 400 })
    }

    const { schemaVersion, queryCount, totalRefs } = _countTotalRefsFromDatasetFile(parsed)
    if (queryCount <= 0) {
      return HttpResponse.json({ detail: 'Dataset must contain at least one query' }, { status: 400 })
    }

    const now = new Date().toISOString()
    const dataset: BenchmarkDataset = {
      id: `ds-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      name,
      description,
      owner_id: 1,
      query_count: queryCount,
      schema_version: schemaVersion,
      created_at: now,
      updated_at: null,
    }

    benchmarkMockState.benchmarkDatasetsState = [dataset, ...benchmarkMockState.benchmarkDatasetsState]
    benchmarkMockState.datasetTotalRefsById[dataset.id] = totalRefs

    return HttpResponse.json(dataset, { status: 201 })
  }),

  http.delete('*/api/v2/benchmark-datasets/:datasetId', ({ params }) => {
    const datasetId = String(params.datasetId)
    benchmarkMockState.benchmarkDatasetsState = benchmarkMockState.benchmarkDatasetsState.filter((d) => d.id !== datasetId)
    benchmarkMockState.datasetMappingsState = benchmarkMockState.datasetMappingsState.filter((m) => m.dataset_id !== datasetId)
    delete benchmarkMockState.datasetTotalRefsById[datasetId]
    return new HttpResponse(null, { status: 204 })
  }),

  http.post('*/api/v2/benchmark-datasets/:datasetId/mappings', async ({ params, request }) => {
    const datasetId = String(params.datasetId)
    const dataset = benchmarkMockState.benchmarkDatasetsState.find((d) => d.id === datasetId)
    if (!dataset) {
      return HttpResponse.json({ detail: 'Dataset not found' }, { status: 404 })
    }

    const body = await request.json() as { collection_id: string }
    const existing = benchmarkMockState.datasetMappingsState.find(
      (m) => m.dataset_id === datasetId && m.collection_id === body.collection_id
    )
    if (existing) {
      return HttpResponse.json({ detail: 'Mapping already exists' }, { status: 409 })
    }

    const now = new Date().toISOString()
    const mapping: DatasetMapping = {
      id: Math.floor(Math.random() * 100000),
      dataset_id: datasetId,
      collection_id: body.collection_id,
      mapping_status: 'pending',
      mapped_count: 0,
      total_count: benchmarkMockState.datasetTotalRefsById[datasetId] ?? dataset.query_count,
      created_at: now,
      resolved_at: null,
    }
    benchmarkMockState.datasetMappingsState = [mapping, ...benchmarkMockState.datasetMappingsState]
    return HttpResponse.json(mapping, { status: 201 })
  }),

  http.get('*/api/v2/benchmark-datasets/:datasetId/mappings', ({ params }) => {
    const datasetId = String(params.datasetId)
    const mappings = benchmarkMockState.datasetMappingsState.filter((m) => m.dataset_id === datasetId)
    return HttpResponse.json(mappings)
  }),

  http.post('*/api/v2/benchmark-datasets/:datasetId/mappings/:mappingId/resolve', ({ params }) => {
    const mappingId = Number(params.mappingId)
    const mappingIndex = benchmarkMockState.datasetMappingsState.findIndex((m) => m.id === mappingId)
    if (mappingIndex < 0) {
      return HttpResponse.json({ detail: 'Mapping not found' }, { status: 404 })
    }

    const mapping = benchmarkMockState.datasetMappingsState[mappingIndex]
    const resolved: DatasetMapping = {
      ...mapping,
      mapping_status: 'resolved',
      mapped_count: mapping.total_count,
      resolved_at: new Date().toISOString(),
    }
    benchmarkMockState.datasetMappingsState = [
      ...benchmarkMockState.datasetMappingsState.slice(0, mappingIndex),
      resolved,
      ...benchmarkMockState.datasetMappingsState.slice(mappingIndex + 1),
    ]

    const response: MappingResolveResponse = {
      id: resolved.id,
      operation_uuid: null,
      mapping_status: resolved.mapping_status,
      mapped_count: resolved.mapped_count,
      total_count: resolved.total_count,
      unresolved: [],
    }
    return HttpResponse.json(response)
  }),

  http.get('*/api/v2/benchmarks', ({ request }) => {
    const statusFilter = request.url.searchParams.get('status_filter')
    const filtered = statusFilter
      ? benchmarkMockState.benchmarksState.filter((b) => b.status === statusFilter)
      : benchmarkMockState.benchmarksState

    const { page, perPage, total, items } = _paginate(
      filtered,
      request.url.searchParams.get('page'),
      request.url.searchParams.get('per_page'),
    )

    const response: BenchmarkListResponse = {
      benchmarks: items,
      total,
      page,
      per_page: perPage,
    }
    return HttpResponse.json(response)
  }),

  http.get('*/api/v2/benchmarks/:benchmarkId', ({ params }) => {
    const benchmark = benchmarkMockState.benchmarksState.find((b) => b.id === String(params.benchmarkId))
    if (!benchmark) {
      return HttpResponse.json({ detail: 'Benchmark not found' }, { status: 404 })
    }
    return HttpResponse.json(benchmark)
  }),

  http.post('*/api/v2/benchmarks', async ({ request }) => {
    const body = await request.json() as {
      name: string
      description?: string
      mapping_id: number
      config_matrix: {
        search_modes: string[]
        use_reranker: boolean[]
        top_k_values?: number[]
        rrf_k_values?: number[]
        score_thresholds?: Array<number | null>
        primary_k?: number
        k_values_for_metrics?: number[]
      }
      top_k?: number
      metrics_to_compute?: string[]
    }

    const now = new Date().toISOString()
    const benchmarkId = `bench-${Date.now()}-${Math.random().toString(16).slice(2)}`

    const benchmark: Benchmark = {
      id: benchmarkId,
      name: body.name,
      description: body.description ?? null,
      owner_id: 1,
      mapping_id: body.mapping_id,
      status: 'pending',
      total_runs: 0,
      completed_runs: 0,
      failed_runs: 0,
      created_at: now,
      started_at: null,
      completed_at: null,
      operation_uuid: null,
    }

    const searchModes = Array.isArray(body.config_matrix?.search_modes) ? body.config_matrix.search_modes : ['dense']
    const useReranker = Array.isArray(body.config_matrix?.use_reranker) ? body.config_matrix.use_reranker : [false]
    const topKValues = body.config_matrix?.top_k_values?.length ? body.config_matrix.top_k_values : [body.top_k ?? 10]
    const rrfKValues = body.config_matrix?.rrf_k_values?.length ? body.config_matrix.rrf_k_values : [60]
    const scoreThresholds = body.config_matrix?.score_thresholds?.length ? body.config_matrix.score_thresholds : [null]

    const runs: BenchmarkRun[] = []
    let runOrder = 0
    for (const searchMode of searchModes) {
      for (const reranker of useReranker) {
        for (const topK of topKValues) {
          for (const rrfK of rrfKValues) {
            for (const scoreThreshold of scoreThresholds) {
              const status: BenchmarkRunStatus = 'pending'
              runs.push({
                id: `run-${benchmarkId}-${runOrder}`,
                run_order: runOrder,
                config_hash: `cfg-${runOrder}`,
                config: {
                  search_mode: searchMode,
                  use_reranker: reranker,
                  top_k: topK,
                  rrf_k: rrfK,
                  score_threshold: scoreThreshold,
                },
                status,
                error_message: null,
                metrics: { mrr: null, precision: {}, recall: {}, ndcg: {} },
                metrics_flat: {},
                timing: { indexing_ms: null, evaluation_ms: null, total_ms: null },
              })
              runOrder += 1
            }
          }
        }
      }
    }

    benchmark.total_runs = runs.length
    benchmarkMockState.benchmarksState = [benchmark, ...benchmarkMockState.benchmarksState]
    benchmarkMockState.benchmarkRunsByBenchmarkId[benchmarkId] = runs
    benchmarkMockState.runQueryResultsByRunId = {
      ...benchmarkMockState.runQueryResultsByRunId,
      ...Object.fromEntries(runs.map((r) => [r.id, { run_id: r.id, results: [], total: 0, page: 1, per_page: 50 }])),
    }

    return HttpResponse.json(benchmark, { status: 201 })
  }),

  http.post('*/api/v2/benchmarks/:benchmarkId/start', ({ params }) => {
    const benchmarkId = String(params.benchmarkId)
    const index = benchmarkMockState.benchmarksState.findIndex((b) => b.id === benchmarkId)
    if (index < 0) {
      return HttpResponse.json({ detail: 'Benchmark not found' }, { status: 404 })
    }

    const operationUuid = `op-${Date.now()}-${Math.random().toString(16).slice(2)}`
    const updated: Benchmark = {
      ...benchmarkMockState.benchmarksState[index],
      status: 'running',
      started_at: new Date().toISOString(),
      operation_uuid: operationUuid,
    }
    benchmarkMockState.benchmarksState = [
      ...benchmarkMockState.benchmarksState.slice(0, index),
      updated,
      ...benchmarkMockState.benchmarksState.slice(index + 1),
    ]

    if (benchmarkMockState.autoCompleteBenchmarksOnStart) {
      _markBenchmarkCompleted(benchmarkId)
    }

    return HttpResponse.json({
      id: benchmarkId,
      status: 'running',
      operation_uuid: operationUuid,
      message: 'Benchmark execution started',
    })
  }),

  http.post('*/api/v2/benchmarks/:benchmarkId/cancel', ({ params }) => {
    const benchmarkId = String(params.benchmarkId)
    const index = benchmarkMockState.benchmarksState.findIndex((b) => b.id === benchmarkId)
    if (index < 0) {
      return HttpResponse.json({ detail: 'Benchmark not found' }, { status: 404 })
    }

    const updated: Benchmark = {
      ...benchmarkMockState.benchmarksState[index],
      status: 'cancelled',
      completed_at: new Date().toISOString(),
    }
    benchmarkMockState.benchmarksState = [
      ...benchmarkMockState.benchmarksState.slice(0, index),
      updated,
      ...benchmarkMockState.benchmarksState.slice(index + 1),
    ]
    return HttpResponse.json(updated)
  }),

  http.get('*/api/v2/benchmarks/:benchmarkId/results', ({ params }) => {
    const benchmarkId = String(params.benchmarkId)
    const benchmark = benchmarkMockState.benchmarksState.find((b) => b.id === benchmarkId)
    if (!benchmark) {
      return HttpResponse.json({ detail: 'Benchmark not found' }, { status: 404 })
    }

    const runs = benchmarkMockState.benchmarkRunsByBenchmarkId[benchmarkId] ?? []
    const response: BenchmarkResultsResponse = {
      benchmark_id: benchmarkId,
      primary_k: 10,
      k_values_for_metrics: [10],
      runs,
      summary: {
        total_runs: runs.length,
        completed_runs: benchmark.completed_runs,
        failed_runs: benchmark.failed_runs,
      },
      total_runs: runs.length,
    }
    return HttpResponse.json(response)
  }),

  http.get('*/api/v2/benchmarks/:benchmarkId/runs/:runId/queries', ({ params, request }) => {
    const runId = String(params.runId)
    const existing = benchmarkMockState.runQueryResultsByRunId[runId]
    if (!existing) {
      return HttpResponse.json({ detail: 'Run not found' }, { status: 404 })
    }

    const { page, perPage } = _paginate(
      existing.results,
      request.url.searchParams.get('page'),
      request.url.searchParams.get('per_page'),
    )

    const start = (page - 1) * perPage
    const end = start + perPage
    const response: RunQueryResultsResponse = {
      run_id: runId,
      results: existing.results.slice(start, end),
      total: existing.total,
      page,
      per_page: perPage,
    }
    return HttpResponse.json(response)
  }),

  http.delete('*/api/v2/benchmarks/:benchmarkId', ({ params }) => {
    const benchmarkId = String(params.benchmarkId)
    benchmarkMockState.benchmarksState = benchmarkMockState.benchmarksState.filter((b) => b.id !== benchmarkId)
    delete benchmarkMockState.benchmarkRunsByBenchmarkId[benchmarkId]
    return new HttpResponse(null, { status: 204 })
  }),

  // =============================================================================
  // Templates (v2)
  // =============================================================================

  http.get('*/api/v2/templates', () => {
    return HttpResponse.json({
      templates: [
        {
          id: 'academic-papers',
          name: 'Academic Papers',
          description: 'Optimized for academic papers and research documents',
          suggested_for: ['PDF papers', 'research documents', 'academic publications'],
        },
        {
          id: 'codebase',
          name: 'Codebase',
          description: 'Optimized for source code repositories',
          suggested_for: ['source code', 'git repositories', 'documentation'],
        },
        {
          id: 'documentation',
          name: 'Documentation',
          description: 'Optimized for technical documentation',
          suggested_for: ['markdown files', 'technical docs', 'wikis'],
        },
        {
          id: 'email-archive',
          name: 'Email Archive',
          description: 'Optimized for email archives with attachments',
          suggested_for: ['emails', 'IMAP archives', 'mbox files'],
        },
        {
          id: 'mixed-documents',
          name: 'Mixed Documents',
          description: 'Balanced configuration for varied file types',
          suggested_for: ['general documents', 'mixed file types'],
        },
      ],
      total: 5,
    })
  }),

  http.get('*/api/v2/templates/:templateId', ({ params }) => {
    const templateId = params.templateId as string

    const templates: Record<string, object> = {
      'academic-papers': {
        id: 'academic-papers',
        name: 'Academic Papers',
        description: 'Optimized for academic papers and research documents',
        suggested_for: ['PDF papers', 'research documents', 'academic publications'],
        pipeline: {
          id: 'academic-papers-pipeline',
          version: '1.0',
          nodes: [
            { id: 'parser', type: 'parser', plugin_id: 'unstructured', config: { strategy: 'hi_res' } },
            { id: 'chunker', type: 'chunker', plugin_id: 'semantic', config: { max_tokens: 512 } },
            { id: 'extractor', type: 'extractor', plugin_id: 'keyword_extractor', config: {} },
            { id: 'embedder', type: 'embedder', plugin_id: 'dense_local', config: {} },
          ],
          edges: [
            { from_node: '_source', to_node: 'parser', when: null },
            { from_node: 'parser', to_node: 'chunker', when: null },
            { from_node: 'chunker', to_node: 'extractor', when: null },
            { from_node: 'extractor', to_node: 'embedder', when: null },
          ],
        },
        tunable: [
          { path: 'nodes.chunker.config.max_tokens', description: 'Maximum tokens per chunk', default: 512, range: [256, 1024], options: null },
        ],
      },
      'codebase': {
        id: 'codebase',
        name: 'Codebase',
        description: 'Optimized for source code repositories',
        suggested_for: ['source code', 'git repositories', 'documentation'],
        pipeline: {
          id: 'codebase-pipeline',
          version: '1.0',
          nodes: [
            { id: 'parser', type: 'parser', plugin_id: 'text', config: {} },
            { id: 'chunker', type: 'chunker', plugin_id: 'recursive', config: { chunk_size: 1000, chunk_overlap: 200 } },
            { id: 'embedder', type: 'embedder', plugin_id: 'dense_local', config: {} },
          ],
          edges: [
            { from_node: '_source', to_node: 'parser', when: null },
            { from_node: 'parser', to_node: 'chunker', when: null },
            { from_node: 'chunker', to_node: 'embedder', when: null },
          ],
        },
        tunable: [
          { path: 'nodes.chunker.config.chunk_size', description: 'Chunk size in characters', default: 1000, range: [500, 2000], options: null },
          { path: 'nodes.chunker.config.chunk_overlap', description: 'Overlap between chunks', default: 200, range: [50, 500], options: null },
        ],
      },
      'documentation': {
        id: 'documentation',
        name: 'Documentation',
        description: 'Optimized for technical documentation',
        suggested_for: ['markdown files', 'technical docs', 'wikis'],
        pipeline: {
          id: 'documentation-pipeline',
          version: '1.0',
          nodes: [
            { id: 'parser', type: 'parser', plugin_id: 'text', config: {} },
            { id: 'chunker', type: 'chunker', plugin_id: 'markdown', config: {} },
            { id: 'extractor', type: 'extractor', plugin_id: 'keyword_extractor', config: {} },
            { id: 'embedder', type: 'embedder', plugin_id: 'dense_local', config: {} },
          ],
          edges: [
            { from_node: '_source', to_node: 'parser', when: null },
            { from_node: 'parser', to_node: 'chunker', when: null },
            { from_node: 'chunker', to_node: 'extractor', when: null },
            { from_node: 'extractor', to_node: 'embedder', when: null },
          ],
        },
        tunable: [],
      },
      'email-archive': {
        id: 'email-archive',
        name: 'Email Archive',
        description: 'Optimized for email archives with attachments',
        suggested_for: ['emails', 'IMAP archives', 'mbox files'],
        pipeline: {
          id: 'email-archive-pipeline',
          version: '1.0',
          nodes: [
            { id: 'parser', type: 'parser', plugin_id: 'text', config: {} },
            { id: 'chunker', type: 'chunker', plugin_id: 'recursive', config: { chunk_size: 800, chunk_overlap: 100 } },
            { id: 'embedder', type: 'embedder', plugin_id: 'dense_local', config: {} },
          ],
          edges: [
            { from_node: '_source', to_node: 'parser', when: null },
            { from_node: 'parser', to_node: 'chunker', when: null },
            { from_node: 'chunker', to_node: 'embedder', when: null },
          ],
        },
        tunable: [],
      },
      'mixed-documents': {
        id: 'mixed-documents',
        name: 'Mixed Documents',
        description: 'Balanced configuration for varied file types',
        suggested_for: ['general documents', 'mixed file types'],
        pipeline: {
          id: 'mixed-documents-pipeline',
          version: '1.0',
          nodes: [
            { id: 'parser', type: 'parser', plugin_id: 'text', config: {} },
            { id: 'chunker', type: 'chunker', plugin_id: 'recursive', config: { chunk_size: 1000, chunk_overlap: 200 } },
            { id: 'embedder', type: 'embedder', plugin_id: 'dense_local', config: {} },
          ],
          edges: [
            { from_node: '_source', to_node: 'parser', when: null },
            { from_node: 'parser', to_node: 'chunker', when: null },
            { from_node: 'chunker', to_node: 'embedder', when: null },
          ],
        },
        tunable: [],
      },
    }

    const template = templates[templateId]
    if (!template) {
      return HttpResponse.json({ detail: `Template '${templateId}' not found` }, { status: 404 })
    }

    return HttpResponse.json(template)
  }),

  // =============================================================================
  // Agent Conversation Endpoints
  // =============================================================================

  // Create a new agent conversation
  http.post('*/api/v2/agent/conversations', async ({ request }) => {
    const body = await request.json() as { source_id: number }
    const now = new Date().toISOString()

    return HttpResponse.json({
      id: 'conv-test-123',
      status: 'active',
      source_id: body.source_id,
      collection_id: null,
      current_pipeline: null,
      source_analysis: {
        total_files: 247,
        total_size_bytes: 47185920,
        file_types: { '.pdf': 150, '.txt': 97 },
        sample_files: ['/docs/file1.pdf', '/docs/file2.txt'],
        warnings: [],
      },
      uncertainties: [],
      messages: [],
      summary: null,
      created_at: now,
      updated_at: now,
    })
  }),

  // Get agent conversation by ID
  http.get('*/api/v2/agent/conversations/:id', ({ params }) => {
    const now = new Date().toISOString()

    return HttpResponse.json({
      id: params.id,
      status: 'active',
      source_id: 42,
      collection_id: null,
      current_pipeline: {
        embedding_model: 'Qwen/Qwen3-Embedding-0.6B',
        quantization: 'float16',
        chunking_strategy: 'semantic',
        chunking_config: { max_tokens: 512, overlap_tokens: 50 },
      },
      source_analysis: {
        total_files: 247,
        total_size_bytes: 47185920,
        file_types: { '.pdf': 150, '.txt': 97 },
        sample_files: ['/docs/file1.pdf', '/docs/file2.txt'],
        warnings: [],
      },
      uncertainties: [
        {
          id: 'unc-1',
          severity: 'notable',
          message: 'Some PDF files appear to be scanned images',
          resolved: false,
          context: { affected_files: 5 },
        },
      ],
      messages: [
        {
          role: 'user',
          content: 'Help me set up a pipeline for my documents',
          timestamp: now,
        },
        {
          role: 'assistant',
          content: 'I found 247 files in your source. Let me analyze them...',
          timestamp: now,
        },
      ],
      summary: null,
      created_at: now,
      updated_at: now,
    })
  }),

  // List agent conversations
  http.get('*/api/v2/agent/conversations', () => {
    const now = new Date().toISOString()

    return HttpResponse.json({
      conversations: [
        {
          id: 'conv-test-123',
          status: 'active',
          source_id: 42,
          created_at: now,
        },
        {
          id: 'conv-test-456',
          status: 'applied',
          source_id: 43,
          created_at: now,
        },
      ],
      total: 2,
    })
  }),

  // Apply pipeline
  http.post('*/api/v2/agent/conversations/:id/apply', async ({ request }) => {
    const body = await request.json() as { collection_name: string; force?: boolean }

    return HttpResponse.json({
      collection_id: 'coll-new-123',
      collection_name: body.collection_name,
      operation_id: 'op-index-123',
      status: 'indexing',
    })
  }),

  // Abandon conversation
  http.patch('*/api/v2/agent/conversations/:id/status', ({ params }) => {
    const now = new Date().toISOString()

    return HttpResponse.json({
      id: params.id,
      status: 'abandoned',
      source_id: 42,
      created_at: now,
    })
  }),

  // =============================================================================
  // Assisted Flow (v2) - SDK-based pipeline configuration
  // =============================================================================

  http.post('*/api/v2/assisted-flow/start', async ({ request }) => {
    const body = await request.json() as { source_id?: number; inline_source?: { source_type: string; source_config: Record<string, unknown> } }

    // Generate session ID based on source or inline config
    let sessionId: string
    let sourceName: string

    if (body.source_id) {
      sessionId = `session_${body.source_id}_mock123`
      sourceName = `Source ${body.source_id}`
    } else if (body.inline_source) {
      sessionId = `session_inline_${body.inline_source.source_type}_mock123`
      const config = body.inline_source.source_config
      sourceName = (config.path as string) || (config.repo_url as string) || `New ${body.inline_source.source_type} Source`
    } else {
      return HttpResponse.json(
        { detail: 'Must specify either source_id or inline_source' },
        { status: 422 }
      )
    }

    return HttpResponse.json({
      session_id: sessionId,
      source_name: sourceName,
    })
  }),

  http.post('*/api/v2/assisted-flow/:sessionId/messages/stream', async () => {
    // Return a simple SSE stream for testing
    const encoder = new TextEncoder()
    const stream = new ReadableStream({
      start(controller) {
        // Send a text event
        controller.enqueue(encoder.encode('event: text\ndata: {"type":"text","content":"Analyzing your source..."}\n\n'))

        // Send a tool use event
        controller.enqueue(encoder.encode('event: tool_use\ndata: {"type":"tool_use","tool_name":"list_plugins"}\n\n'))

        // Send a tool result event
        controller.enqueue(encoder.encode('event: tool_result\ndata: {"type":"tool_result","tool_name":"list_plugins","success":true}\n\n'))

        // Send done event
        controller.enqueue(encoder.encode('event: done\ndata: {"status":"complete"}\n\n'))

        controller.close()
      },
    })

    return new HttpResponse(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    })
  }),

  // =============================================================================
  // Pipeline Preview (v2)
  // =============================================================================

  http.post('*/api/v2/pipeline/available-predicate-fields', async () => {
    // Return mock predicate fields based on source/detected/parsed categories
    // NOTE: Source fields are top-level FileReference attributes
    return HttpResponse.json({
      fields: [
        // Source metadata (from connector) - top-level FileReference attributes
        { value: 'mime_type', label: 'MIME Type', category: 'source' },
        { value: 'extension', label: 'Extension', category: 'source' },
        { value: 'source_type', label: 'Source Type', category: 'source' },
        { value: 'content_type', label: 'Content Type', category: 'source' },
        // Detected metadata (from pre-routing sniff)
        { value: 'metadata.detected.is_scanned_pdf', label: 'Is Scanned PDF', category: 'detected' },
        { value: 'metadata.detected.is_code', label: 'Is Code', category: 'detected' },
        { value: 'metadata.detected.is_structured_data', label: 'Is Structured Data', category: 'detected' },
        // Parsed metadata (for mid-pipeline routing)
        { value: 'metadata.parsed.detected_language', label: 'Detected Language', category: 'parsed' },
        { value: 'metadata.parsed.approx_token_count', label: 'Token Count', category: 'parsed' },
        { value: 'metadata.parsed.has_tables', label: 'Has Tables', category: 'parsed' },
        { value: 'metadata.parsed.has_images', label: 'Has Images', category: 'parsed' },
        { value: 'metadata.parsed.has_code_blocks', label: 'Has Code Blocks', category: 'parsed' },
        { value: 'metadata.parsed.page_count', label: 'Page Count', category: 'parsed' },
      ],
    })
  }),

  http.post('*/api/v2/pipeline/preview-route', async ({ request }) => {
    const formData = await request.formData()
    const file = formData.get('file') as File | null
    const dagString = formData.get('dag') as string | null
    const includeParserMetadata = formData.get('include_parser_metadata') === 'true'

    if (!file || !dagString) {
      return HttpResponse.json(
        { detail: 'file and dag are required' },
        { status: 422 }
      )
    }

    // Parse the DAG to get nodes for path computation
    let dag: { nodes?: Array<{ id: string }> }
    try {
      dag = JSON.parse(dagString)
    } catch {
      return HttpResponse.json(
        { detail: 'Invalid DAG JSON' },
        { status: 422 }
      )
    }

    // Build a basic path from _source through the nodes
    const path = ['_source', ...(dag.nodes?.map(n => n.id) ?? [])]

    return HttpResponse.json({
      file_info: {
        filename: file.name,
        extension: file.name.includes('.') ? '.' + file.name.split('.').pop() : null,
        mime_type: file.type || 'application/octet-stream',
        size_bytes: file.size,
        uri: `file:///${file.name}`,
      },
      sniff_result: {
        is_code: false,
        is_structured_data: false,
        structured_format: null,
        is_scanned_pdf: null,
      },
      routing_stages: [
        {
          stage: 'entry',
          from_node: '_source',
          evaluated_edges: [],
          selected_node: path[1] ?? null,
          metadata_snapshot: {},
        },
      ],
      path,
      parsed_metadata: includeParserMetadata ? { mock_field: 'mock_value' } : null,
      total_duration_ms: Math.floor(Math.random() * 100) + 50,
      warnings: [],
    })
  }),
]
