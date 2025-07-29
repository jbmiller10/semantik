import { http, HttpResponse } from 'msw'
import { server } from './server'
import type { Operation } from '../../types/collection'

/**
 * Utility functions for mocking API responses in tests
 */

// Helper to mock a specific error response for reindex endpoint
export function mockReindexError(collectionId: string, status: number, detail?: string) {
  server.use(
    http.post(`/api/v2/collections/${collectionId}/reindex`, () => {
      const errorResponse: { detail?: string } = {}
      
      if (detail) {
        errorResponse.detail = detail
      }
      
      return HttpResponse.json(errorResponse, { status })
    })
  )
}

// Helper to mock a successful reindex response
export function mockReindexSuccess(collectionId: string, operation?: Partial<Operation>) {
  server.use(
    http.post(`/api/v2/collections/${collectionId}/reindex`, () => {
      const defaultOperation: Operation = {
        id: 'op-' + Date.now(),
        collection_id: collectionId,
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
      
      return HttpResponse.json({ ...defaultOperation, ...operation })
    })
  )
}

// Helper to mock search API errors
export function mockSearchError(status: number, detail?: string | { message: string; suggestion: string }) {
  server.use(
    http.post('/api/v2/search', () => {
      if (status === 507 && detail) {
        return HttpResponse.json({ detail }, { status })
      }
      
      return HttpResponse.json(
        { detail: detail || 'Search failed' },
        { status }
      )
    })
  )
}

// Helper to mock search API success with custom response
export function mockSearchSuccess(response: { results: unknown[] }) {
  server.use(
    http.post('/api/v2/search', () => {
      return HttpResponse.json(response)
    })
  )
}