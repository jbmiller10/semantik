import { http, HttpResponse } from 'msw'

export const handlers = [
  // Auth endpoints
  http.post('/api/auth/login', async ({ request }) => {
    const body = await request.json() as any
    const { username, password } = body as { username: string; password: string }
    
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
    const body = await request.json() as any
    const { new_name } = body as { new_name: string }
    
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
    const body = await request.json() as any
    
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

  // Models endpoints
  http.get('/api/models', () => {
    return HttpResponse.json({
      models: [
        {
          name: 'Qwen/Qwen3-Embedding-0.6B',
          size: '600M',
          description: 'Small efficient embedding model',
          is_downloaded: true,
        }
      ]
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
]