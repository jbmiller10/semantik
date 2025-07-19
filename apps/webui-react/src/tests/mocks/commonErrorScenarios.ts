import { 
  collectionErrorHandlers, 
  operationErrorHandlers, 
  searchErrorHandlers,
  authErrorHandlers,
  documentErrorHandlers,
  combineErrorHandlers
} from './errorHandlers'

/**
 * Common error scenarios for testing
 */

// Network failure scenarios
export const networkFailureScenario = () => combineErrorHandlers(
  collectionErrorHandlers.networkError,
  operationErrorHandlers.networkError,
  searchErrorHandlers.networkError,
  authErrorHandlers.networkError
)

// Server error (500) scenarios
export const serverErrorScenario = () => combineErrorHandlers(
  collectionErrorHandlers.serverError,
  operationErrorHandlers.serverError,
  searchErrorHandlers.serverError
)

// Permission denied scenarios
export const permissionDeniedScenario = () => combineErrorHandlers(
  collectionErrorHandlers.permissionError,
  documentErrorHandlers.permissionError
)

// Resource not found scenarios
export const notFoundScenario = () => combineErrorHandlers(
  collectionErrorHandlers.notFound,
  documentErrorHandlers.notFound
)

// Authentication failure scenarios
export const authFailureScenario = () => combineErrorHandlers(
  authErrorHandlers.unauthorized
)

// Validation error scenarios
export const validationErrorScenario = () => combineErrorHandlers(
  collectionErrorHandlers.validationError
)

// Rate limiting scenarios
export const rateLimitScenario = () => combineErrorHandlers(
  collectionErrorHandlers.rateLimited
)

// Mixed error scenario (some succeed, some fail)
export const partialFailureScenario = () => [
  ...searchErrorHandlers.partialFailure(),
  // Some collections load, others fail
  {
    handler: collectionErrorHandlers.serverError()[0],
    predicate: (req: Request) => req.url.includes('failing-collection')
  }
]

/**
 * Scenario configurations for specific test cases
 */
export const errorScenarios = {
  // Collection creation errors
  collectionCreation: {
    duplicateName: collectionErrorHandlers.validationError()[0],
    networkError: collectionErrorHandlers.networkError()[1],
    serverError: collectionErrorHandlers.serverError()[1],
    rateLimited: collectionErrorHandlers.rateLimited()[0]
  },

  // Collection loading errors
  collectionLoading: {
    networkError: collectionErrorHandlers.networkError()[0],
    serverError: collectionErrorHandlers.serverError()[0],
    unauthorized: authErrorHandlers.unauthorized()[0]
  },

  // Search errors
  search: {
    networkError: searchErrorHandlers.networkError()[0],
    serverError: searchErrorHandlers.serverError()[0],
    partialFailure: searchErrorHandlers.partialFailure()[0]
  },

  // Operation errors
  operations: {
    networkError: operationErrorHandlers.networkError()[0],
    serverError: operationErrorHandlers.serverError()[0]
  },

  // Reindex errors
  reindex: {
    serverError: collectionErrorHandlers.serverError()[2],
    validationError: collectionErrorHandlers.validationError()[1]
  }
}

/**
 * WebSocket error scenarios
 */
export const webSocketScenarios = {
  // Connection fails immediately
  connectionFailure: {
    url: '/ws/operations/fail-connection',
    expectedClose: { code: 1006, reason: 'Connection failed' }
  },

  // Connection drops after some messages
  connectionDrop: {
    url: '/ws/operations/test-id',
    messages: [
      { type: 'progress', progress: 25, message: 'Processing...' },
      { type: 'progress', progress: 50, message: 'Halfway there...' }
    ],
    dropAfterMessages: 2,
    expectedClose: { code: 1006, reason: 'Connection lost' }
  },

  // Invalid auth token
  authFailure: {
    url: '/ws/operations/test-id?token=invalid',
    expectedClose: { code: 4401, reason: 'Authentication failed' }
  },

  // Permission denied
  permissionDenied: {
    url: '/ws/operations/other-user-operation',
    expectedClose: { code: 4403, reason: 'Permission denied' }
  },

  // Malformed messages
  malformedMessage: {
    url: '/ws/operations/test-id',
    messages: [
      'invalid json',
      { invalidStructure: true },
      { type: 'unknown-type' }
    ]
  },

  // Reconnection scenario
  reconnection: {
    url: '/ws/operations/test-id',
    disconnectAfter: 1000,
    reconnectAfter: 2000,
    messages: {
      beforeDisconnect: [
        { type: 'progress', progress: 25, message: 'Processing...' }
      ],
      afterReconnect: [
        { type: 'progress', progress: 75, message: 'Almost done...' },
        { type: 'completed', progress: 100, message: 'Completed' }
      ]
    }
  }
}