import { http, HttpResponse } from 'msw'
import type { Collection, Operation } from '../../types/collection'

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
  // System status
  http.get('/api/v2/system/status', () => {
    return HttpResponse.json({
      healthy: true,
      version: '2.0.0',
      services: {
        database: 'healthy',
        redis: 'healthy',
        qdrant: 'healthy',
      },
      reranking_available: true,
      gpu_available: true,
      gpu_memory_mb: 8192,
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
    const body = await request.json() as { use_reranker?: boolean }
    
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
    })
  }),
]