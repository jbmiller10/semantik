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

  // Jobs endpoints
  http.get('/api/jobs', () => {
    return HttpResponse.json([
      {
        id: '1',
        name: 'Test Collection',
        directory_path: '/test/path',
        collection_name: 'test-collection',
        status: 'completed',
        progress: 100,
        total_files: 10,
        total_documents: 10,
        processed_files: 10,
        processed_documents: 10,
        failed_files: 0,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      },
      {
        id: '2',
        name: 'Test Collection 2',
        directory_path: '/test/path2',
        collection_name: 'test-collection-2',
        status: 'completed',
        progress: 100,
        total_files: 20,
        total_documents: 20,
        processed_files: 20,
        processed_documents: 20,
        failed_files: 0,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      }
    ])
  }),

  http.post('/api/jobs', async ({ request }) => {
    const body = await request.json() as any
    
    return HttpResponse.json({
      id: '2',
      name: body.name || 'New Job',
      directory_path: body.directory_path,
      collection_name: body.name,
      status: 'pending',
      progress: 0,
      total_files: 0,
      total_documents: 0,
      processed_files: 0,
      processed_documents: 0,
      failed_files: 0,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    })
  }),

  http.delete('/api/jobs/:jobId', ({ params }) => {
    return HttpResponse.json({ 
      message: `Job ${params.jobId} deleted successfully` 
    })
  }),

  http.post('/api/jobs/:jobId/cancel', ({ params }) => {
    return HttpResponse.json({ 
      message: `Job ${params.jobId} cancellation requested` 
    })
  }),

  // Collections endpoints
  http.get('/api/collections', () => {
    return HttpResponse.json({
      collections: [
        {
          name: 'test-collection',
          document_count: 100,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        }
      ]
    })
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
      total_collections: 1,
      total_documents: 100,
      total_chunks: 500,
      database_size: '100MB',
      vector_store_size: '200MB',
    })
  }),
  
  // Collections status endpoint
  http.get('/api/jobs/collections-status', () => {
    return HttpResponse.json({
      '1': {
        exists: true,
        point_count: 100,
        status: 'completed',
      },
      '2': {
        exists: true,
        point_count: 200,
        status: 'completed',
      },
    })
  }),

  // Settings endpoints
  http.get('/api/settings/stats', () => {
    return HttpResponse.json({
      job_count: 10,
      file_count: 100,
      database_size_mb: 50,
      parquet_files_count: 10,
      parquet_size_mb: 25,
    })
  }),

  http.post('/api/settings/reset', () => {
    return HttpResponse.json({ message: 'Database reset successfully' })
  }),
]