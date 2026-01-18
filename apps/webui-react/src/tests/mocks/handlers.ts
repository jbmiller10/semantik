import { http, HttpResponse } from 'msw'
import type { Collection, Operation } from '../../types/collection'
import type { ApiKeyResponse, ApiKeyCreateResponse, ApiKeyListResponse } from '../../types/api-key'

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
  http.get('/api/v2/system/info', () => {
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
  http.get('/api/v2/system/health', () => {
    return HttpResponse.json({
      postgres: { status: 'healthy', message: 'Connected' },
      redis: { status: 'healthy', message: 'Connected' },
      qdrant: { status: 'healthy', message: 'Connected' },
      vecpipe: { status: 'healthy', message: 'Connected' },
    })
  }),

  // System status
  http.get('/api/v2/system/status', () => {
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
  http.get('/api/v2/collections', () => {
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

  http.get('/api/v2/collections/:uuid', ({ params }) => {
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

  http.post('/api/v2/collections/:uuid/reindex', ({ params }) => {
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

  http.delete('/api/v2/collections/:uuid', () => {
    return HttpResponse.json({ message: 'Collection deleted successfully' })
  }),

  // Sparse index endpoints
  http.get('/api/v2/collections/:uuid/sparse-index', () => {
    return HttpResponse.json({
      enabled: false,
      plugin_id: null,
      plugin_config: null,
      indexed_documents: 0,
      total_documents: 150,
      last_indexed_at: null,
    })
  }),

  http.post('/api/v2/collections/:uuid/sparse-index', async ({ request }) => {
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

  http.delete('/api/v2/collections/:uuid/sparse-index', () => {
    return HttpResponse.json({
      enabled: false,
      plugin_id: null,
      plugin_config: null,
      indexed_documents: 0,
      total_documents: 150,
      last_indexed_at: null,
    })
  }),

  http.post('/api/v2/collections/:uuid/sparse-index/reindex', () => {
    return HttpResponse.json({
      job_id: 'mock-reindex-job-' + Date.now(),
      status: 'pending',
      message: 'Sparse reindex job queued',
    })
  }),

  http.get('/api/v2/collections/:uuid/sparse-index/reindex/:jobId', () => {
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
  http.get('/api/v2/connectors', () => {
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
  http.post('/api/v2/search', async ({ request }) => {
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
  http.get('/api/v2/llm/settings', () => {
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

  http.put('/api/v2/llm/settings', async ({ request }) => {
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

  http.get('/api/v2/llm/models', () => {
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

  http.get('/api/v2/llm/models/refresh', ({ request }) => {
    const url = new URL(request.url)
    const provider = url.searchParams.get('provider')

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

  http.post('/api/v2/llm/test', async ({ request }) => {
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

  http.get('/api/v2/llm/usage', () => {
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
  http.get('/api/v2/preferences', () => {
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

  http.put('/api/v2/preferences', async ({ request }) => {
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

  http.post('/api/v2/preferences/reset/search', () => {
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

  http.post('/api/v2/preferences/reset/collection-defaults', () => {
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

  http.post('/api/v2/preferences/reset/interface', () => {
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
  http.get('/api/v2/system-settings', () => {
    return HttpResponse.json({
      settings: {
        max_collections_per_user: { value: 10, updated_at: null, updated_by: null },
        max_storage_gb_per_user: { value: 50, updated_at: null, updated_by: null },
        max_document_size_mb: { value: 100, updated_at: null, updated_by: null },
      },
    })
  }),

  http.get('/api/v2/system-settings/effective', () => {
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

  http.get('/api/v2/system-settings/defaults', () => {
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

  http.patch('/api/v2/system-settings', async ({ request }) => {
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
  http.get('/api/v2/api-keys', () => {
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

  http.get('/api/v2/api-keys/:keyId', ({ params }) => {
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

  http.post('/api/v2/api-keys', async ({ request }) => {
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

  http.patch('/api/v2/api-keys/:keyId', async ({ params, request }) => {
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
]
