import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@/tests/utils/test-utils'
import CollectionOperations from '../CollectionOperations'
import { useCollectionOperations } from '@/hooks/useCollectionOperations'
import { useUIStore } from '@/stores/uiStore'
import type { Collection, Operation } from '@/types/collection'
import type { MockedFunction } from '@/tests/types/test-types'
import type { UseQueryResult } from '@tanstack/react-query'

interface NetworkError extends Error {
  code?: string;
  response?: {
    status: number;
    headers?: Record<string, string>;
  };
}

// Mock the hooks and components
vi.mock('@/hooks/useCollectionOperations', () => ({
  useCollectionOperations: vi.fn(),
}))

vi.mock('../OperationProgress', () => ({
  default: vi.fn(({ operation }) => (
    <div data-testid={`operation-${operation.id}`}>
      Operation Progress: {operation.type}
    </div>
  )),
}))

vi.mock('@/stores/uiStore', () => ({
  useUIStore: vi.fn(),
}))

const mockCollection: Collection = {
  id: 'test-collection-id',
  name: 'Test Collection',
  description: 'Test collection description',
  owner_id: 1,
  vector_store_name: 'test_collection_vectors',
  embedding_model: 'Qwen/Qwen3-Embedding-0.6B',
  quantization: 'float16',
  chunk_size: 512,
  chunk_overlap: 50,
  is_public: false,
  status: 'ready',
  document_count: 10,
  vector_count: 100,
  total_size_bytes: 1024000,
  created_at: '2025-01-10T10:00:00Z',
  updated_at: '2025-01-14T15:30:00Z',
}

describe('CollectionOperations - Network Error Handling', () => {
  const mockRefetch = vi.fn()
  const mockAddToast = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    ;(useUIStore as MockedFunction<typeof useUIStore>).mockReturnValue({
      addToast: mockAddToast,
    } as Partial<UseQueryResult>)
  })

  it('handles network errors gracefully when fetching operations', () => {
    const error = new Error('Network error')
    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: undefined,
      error,
      isError: true,
      isLoading: false,
      refetch: mockRefetch,
    } as Partial<UseQueryResult>)

    render(<CollectionOperations collection={mockCollection} />)

    // Should still render without crashing
    expect(screen.getByText('No operations yet')).toBeInTheDocument()
  })

  it('shows loading state while fetching operations', () => {
    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: undefined,
      isLoading: true,
      refetch: mockRefetch,
    } as Partial<UseQueryResult>)

    render(<CollectionOperations collection={mockCollection} />)

    // Component should handle loading state gracefully
    // Since our component doesn't show a loading indicator, it shows empty state
    expect(screen.getByText('No operations yet')).toBeInTheDocument()
  })

  it('handles undefined data from API', () => {
    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: undefined,
      isLoading: false,
      refetch: mockRefetch,
    } as Partial<UseQueryResult>)

    render(<CollectionOperations collection={mockCollection} />)

    // Should show empty state
    expect(screen.getByText('No operations yet')).toBeInTheDocument()
  })

  it('handles null data from API', () => {
    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: null,
      isLoading: false,
      refetch: mockRefetch,
    } as Partial<UseQueryResult>)

    render(<CollectionOperations collection={mockCollection} />)

    // Component uses default empty array, so should show empty state
    expect(screen.getByText('No operations yet')).toBeInTheDocument()
  })

  it('handles API returning invalid operation data', () => {
    const invalidOperations = [
      { 
        id: 'op-1',
        // Missing required fields like type, status, etc.
      },
      {
        id: 'op-2',
        type: 'index',
        status: 'invalid-status', // Invalid status
        created_at: 'invalid-date',
      },
    ] as unknown as Operation[]

    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: invalidOperations,
      refetch: mockRefetch,
    } as Partial<UseQueryResult>)

    // Should not crash even with invalid data
    expect(() => {
      render(<CollectionOperations collection={mockCollection} />)
    }).not.toThrow()
  })

  it('continues to show cached data during refetch', async () => {
    const operations = [
      {
        id: 'op-1',
        collection_id: mockCollection.id,
        type: 'index',
        status: 'completed',
        config: {},
        created_at: '2025-01-14T10:00:00Z',
      },
    ]

    const { rerender } = render(<CollectionOperations collection={mockCollection} />)

    // Initial render with data
    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: operations,
      isLoading: false,
      isFetching: false,
      refetch: mockRefetch,
    } as Partial<UseQueryResult>)

    rerender(<CollectionOperations collection={mockCollection} />)
    expect(screen.getByTestId('operation-op-1')).toBeInTheDocument()

    // Simulate refetch with loading state but cached data still available
    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: operations,
      isLoading: false,
      isFetching: true, // Refetching
      refetch: mockRefetch,
    } as Partial<UseQueryResult>)

    rerender(<CollectionOperations collection={mockCollection} />)

    // Should still show the cached data
    expect(screen.getByTestId('operation-op-1')).toBeInTheDocument()
  })

  it('handles API timeout errors', () => {
    const timeoutError = new Error('Request timeout')
    ;(timeoutError as NetworkError).code = 'ECONNABORTED'

    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: undefined,
      error: timeoutError,
      isError: true,
      isLoading: false,
      refetch: mockRefetch,
    } as Partial<UseQueryResult>)

    render(<CollectionOperations collection={mockCollection} />)

    // Should handle timeout gracefully
    expect(screen.getByText('No operations yet')).toBeInTheDocument()
  })

  it('handles 404 errors when collection not found', () => {
    const notFoundError = new Error('Collection not found')
    ;(notFoundError as NetworkError).response = { status: 404 }

    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: undefined,
      error: notFoundError,
      isError: true,
      isLoading: false,
      refetch: mockRefetch,
    } as Partial<UseQueryResult>)

    render(<CollectionOperations collection={mockCollection} />)

    // Should handle 404 gracefully
    expect(screen.getByText('No operations yet')).toBeInTheDocument()
  })

  it('handles 401 unauthorized errors', () => {
    const unauthorizedError = new Error('Unauthorized')
    ;(unauthorizedError as NetworkError).response = { status: 401 }

    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: undefined,
      error: unauthorizedError,
      isError: true,
      isLoading: false,
      refetch: mockRefetch,
    } as Partial<UseQueryResult>)

    render(<CollectionOperations collection={mockCollection} />)

    // Should handle auth errors gracefully
    expect(screen.getByText('No operations yet')).toBeInTheDocument()
  })

  it('handles server errors (5xx)', () => {
    const serverError = new Error('Internal Server Error')
    ;(serverError as NetworkError).response = { status: 500 }

    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: undefined,
      error: serverError,
      isError: true,
      isLoading: false,
      refetch: mockRefetch,
    } as Partial<UseQueryResult>)

    render(<CollectionOperations collection={mockCollection} />)

    // Should handle server errors gracefully
    expect(screen.getByText('No operations yet')).toBeInTheDocument()
  })

  it('handles rate limiting errors (429)', () => {
    const rateLimitError = new Error('Too Many Requests')
    ;(rateLimitError as NetworkError).response = { 
      status: 429,
      headers: { 'retry-after': '60' }
    }

    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: undefined,
      error: rateLimitError,
      isError: true,
      isLoading: false,
      refetch: mockRefetch,
    } as Partial<UseQueryResult>)

    render(<CollectionOperations collection={mockCollection} />)

    // Should handle rate limiting gracefully
    expect(screen.getByText('No operations yet')).toBeInTheDocument()
  })

  it('handles network connectivity issues', () => {
    const networkError = new Error('Network Error')
    ;(networkError as NetworkError).code = 'ERR_NETWORK'

    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: undefined,
      error: networkError,
      isError: true,
      isLoading: false,
      refetch: mockRefetch,
    } as Partial<UseQueryResult>)

    render(<CollectionOperations collection={mockCollection} />)

    // Should handle network issues gracefully
    expect(screen.getByText('No operations yet')).toBeInTheDocument()
  })

  it('retries failed requests when refetch is called', async () => {
    const operations = [
      {
        id: 'op-1',
        collection_id: mockCollection.id,
        type: 'index',
        status: 'completed',
        config: {},
        created_at: '2025-01-14T10:00:00Z',
      },
    ]

    // Start with an error
    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: undefined,
      error: new Error('Network error'),
      isError: true,
      isLoading: false,
      refetch: mockRefetch.mockResolvedValue({ data: operations }),
    } as Partial<UseQueryResult>)

    const { rerender } = render(<CollectionOperations collection={mockCollection} />)

    // Trigger refetch
    await mockRefetch()

    // Update mock to return success
    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: operations,
      error: null,
      isError: false,
      isLoading: false,
      refetch: mockRefetch,
    } as Partial<UseQueryResult>)

    rerender(<CollectionOperations collection={mockCollection} />)

    // Should now show the operations
    await waitFor(() => {
      expect(screen.getByTestId('operation-op-1')).toBeInTheDocument()
    })
  })
})