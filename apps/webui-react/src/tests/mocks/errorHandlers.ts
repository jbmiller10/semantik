import { http, HttpResponse, delay } from 'msw'

/**
 * Utility to create error handlers for common HTTP error scenarios
 */
export const createErrorHandler = (
  method: 'get' | 'post' | 'put' | 'delete' | 'patch',
  path: string,
  status: number,
  response?: { detail?: string; message?: string } | any
) => {
  return http[method](path, () => {
    const defaultResponses: Record<number, any> = {
      400: { detail: 'Bad request' },
      401: { detail: 'Unauthorized' },
      403: { detail: 'Access denied' },
      404: { detail: 'Not found' },
      409: { detail: 'Conflict' },
      429: { detail: 'Too many requests' },
      500: { detail: 'Internal server error' },
      503: { detail: 'Service unavailable' }
    }

    return HttpResponse.json(
      response || defaultResponses[status] || { detail: `Error ${status}` },
      { status }
    )
  })
}

/**
 * Create a network error handler (connection refused, etc.)
 */
export const createNetworkErrorHandler = (
  method: 'get' | 'post' | 'put' | 'delete' | 'patch',
  path: string
) => {
  return http[method](path, () => {
    return HttpResponse.error()
  })
}

/**
 * Create a timeout handler with configurable delay
 */
export const createTimeoutHandler = (
  method: 'get' | 'post' | 'put' | 'delete' | 'patch',
  path: string,
  delayMs: number = 5000
) => {
  return http[method](path, async () => {
    await delay(delayMs)
    return HttpResponse.timeout()
  })
}

/**
 * Common error scenarios for collections
 */
export const collectionErrorHandlers = {
  // Network errors
  networkError: () => [
    createNetworkErrorHandler('get', '/api/v2/collections'),
    createNetworkErrorHandler('post', '/api/v2/collections'),
    createNetworkErrorHandler('get', '/api/v2/collections/:uuid'),
    createNetworkErrorHandler('put', '/api/v2/collections/:uuid'),
    createNetworkErrorHandler('delete', '/api/v2/collections/:uuid')
  ],

  // Validation errors
  validationError: () => [
    createErrorHandler('post', '/api/v2/collections', 400, {
      detail: 'Collection with this name already exists'
    }),
    createErrorHandler('put', '/api/v2/collections/:uuid', 400, {
      detail: 'Invalid collection name: must be between 3-50 characters'
    }),
    createErrorHandler('post', '/api/v2/collections/:uuid/add-source', 400, {
      detail: 'Path does not exist or is not accessible: /nonexistent/path'
    })
  ],

  // Permission errors
  permissionError: () => [
    createErrorHandler('get', '/api/v2/collections/:uuid', 403, {
      detail: 'You do not have permission to access this collection'
    }),
    createErrorHandler('delete', '/api/v2/collections/:uuid', 403, {
      detail: 'Only the collection owner can delete this collection'
    })
  ],

  // Not found errors
  notFound: () => [
    createErrorHandler('get', '/api/v2/collections/:uuid', 404, {
      detail: 'Collection not found'
    }),
    createErrorHandler('put', '/api/v2/collections/:uuid', 404, {
      detail: 'Collection not found'
    })
  ],

  // Server errors
  serverError: () => [
    createErrorHandler('get', '/api/v2/collections', 500),
    createErrorHandler('post', '/api/v2/collections', 500),
    createErrorHandler('post', '/api/v2/collections/:uuid/reindex', 500, {
      detail: 'Failed to start reindex operation'
    })
  ],

  // Rate limiting
  rateLimited: () => [
    createErrorHandler('post', '/api/v2/collections', 429, {
      detail: 'Collection limit reached (10 max)'
    }),
    createErrorHandler('post', '/api/v2/collections/:uuid/add-source', 429, {
      detail: 'Too many operations in progress. Please wait and try again.'
    })
  ]
}

/**
 * Common error scenarios for operations
 */
export const operationErrorHandlers = {
  networkError: () => [
    createNetworkErrorHandler('get', '/api/v2/operations'),
    createNetworkErrorHandler('get', '/api/v2/operations/:id')
  ],

  serverError: () => [
    createErrorHandler('get', '/api/v2/operations', 500),
    createErrorHandler('get', '/api/v2/operations/:id', 500)
  ]
}

/**
 * Common error scenarios for search
 */
export const searchErrorHandlers = {
  networkError: () => [
    createNetworkErrorHandler('post', '/api/v2/search')
  ],

  serverError: () => [
    createErrorHandler('post', '/api/v2/search', 500, {
      detail: 'Search service unavailable'
    })
  ],

  partialFailure: () => [
    http.post('/api/v2/search', () => {
      return HttpResponse.json({
        results: [
          {
            collection_id: 'coll-1',
            collection_name: 'Working Collection',
            chunk_id: 1,
            content: 'Test result',
            score: 0.9,
            file_path: '/test/doc.txt'
          }
        ],
        total_results: 1,
        partial_failure: true,
        failed_collections: [
          {
            collection_id: 'coll-2',
            collection_name: 'Failed Collection',
            error: 'Vector index corrupted'
          },
          {
            collection_id: 'coll-3',
            collection_name: 'Another Failed',
            error: 'Timeout during search'
          }
        ],
        timing: {
          total: 2.5
        }
      })
    })
  ]
}

/**
 * Common error scenarios for authentication
 */
export const authErrorHandlers = {
  unauthorized: () => [
    createErrorHandler('get', '/api/auth/me', 401),
    createErrorHandler('post', '/api/auth/refresh', 401, {
      detail: 'Token expired'
    })
  ],

  networkError: () => [
    createNetworkErrorHandler('post', '/api/auth/login'),
    createNetworkErrorHandler('post', '/api/auth/refresh')
  ]
}

/**
 * Common error scenarios for documents
 */
export const documentErrorHandlers = {
  notFound: () => [
    createErrorHandler('get', '/api/documents/:collectionId/*', 404, {
      detail: 'Document not found'
    })
  ],

  permissionError: () => [
    createErrorHandler('get', '/api/documents/:collectionId/*', 403, {
      detail: 'Access denied to document'
    })
  ]
}

/**
 * Combine multiple error handler sets
 */
export const combineErrorHandlers = (...handlerSets: (() => any[])[]) => {
  return handlerSets.flatMap(handlerSet => handlerSet())
}

/**
 * Simulate degraded service (slow responses)
 */
export const createSlowResponseHandler = (
  method: 'get' | 'post' | 'put' | 'delete' | 'patch',
  path: string,
  delayMs: number,
  response: any
) => {
  return http[method](path, async () => {
    await delay(delayMs)
    return HttpResponse.json(response)
  })
}