import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@/tests/utils/test-utils'
import CollectionOperations from '../CollectionOperations'
import { useCollectionOperations } from '@/hooks/useCollectionOperations'
import type { Collection, Operation } from '@/types/collection'
import type { MockedFunction } from '@/tests/types/test-types'
import type { UseQueryResult } from '@tanstack/react-query'

// Mock the hooks and components
vi.mock('@/hooks/useCollectionOperations', () => ({
  useCollectionOperations: vi.fn(),
}))

vi.mock('../OperationProgress', () => ({
  default: vi.fn(({ operation }) => {
    return (
      <div data-testid={`operation-${operation.id}`}>
        <span data-testid={`operation-${operation.id}-type`}>Operation: {operation.type}</span>
        <span data-testid={`operation-${operation.id}-status`}>Status: {operation.status}</span>
        <span data-testid={`operation-${operation.id}-progress`}>Progress: {operation.progress || 0}%</span>
      </div>
    )
  }),
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

const createMockOperation = (
  id: string,
  status: Operation['status'] = 'pending',
  progress: number = 0
): Operation => ({
  id,
  collection_id: mockCollection.id,
  type: 'index',
  status,
  config: { source_path: '/data/documents' },
  created_at: '2025-01-14T10:00:00Z',
  progress,
})

describe('CollectionOperations - WebSocket Real-time Updates', () => {
  const mockRefetch = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('simulates real-time progress updates', () => {
    // Start with operation at 25% progress
    const operation = createMockOperation('op-1', 'processing', 25)
    
    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: [operation],
      refetch: mockRefetch,
    } as UseQueryResult<Operation[]>)

    const { rerender } = render(<CollectionOperations collection={mockCollection} />)
    expect(screen.getByTestId('operation-op-1-progress')).toHaveTextContent('Progress: 25%')

    // Simulate progress update to 50%
    const updatedOperation = { ...operation, progress: 50 }
    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: [updatedOperation],
      refetch: mockRefetch,
    } as UseQueryResult<Operation[]>)

    rerender(<CollectionOperations collection={mockCollection} />)
    expect(screen.getByTestId('operation-op-1-progress')).toHaveTextContent('Progress: 50%')
  })

  it('shows different operation statuses correctly', () => {
    // Start with pending operation
    const operation = createMockOperation('op-1', 'pending', 0)
    
    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: [operation],
      refetch: mockRefetch,
    } as UseQueryResult<Operation[]>)

    const { rerender } = render(<CollectionOperations collection={mockCollection} />)
    expect(screen.getByTestId('operation-op-1-status')).toHaveTextContent('Status: pending')

    // Update to processing
    const processingOp = { ...operation, status: 'processing' as const, progress: 50 }
    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: [processingOp],
      refetch: mockRefetch,
    } as UseQueryResult<Operation[]>)
    
    rerender(<CollectionOperations collection={mockCollection} />)
    expect(screen.getByTestId('operation-op-1-status')).toHaveTextContent('Status: processing')
    expect(screen.getByTestId('operation-op-1-progress')).toHaveTextContent('Progress: 50%')

    // Update to completed
    const completedOp = { ...operation, status: 'completed' as const, progress: 100 }
    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: [completedOp],
      refetch: mockRefetch,
    } as UseQueryResult<Operation[]>)
    
    rerender(<CollectionOperations collection={mockCollection} />)
    expect(screen.getByTestId('operation-op-1-status')).toHaveTextContent('Status: completed')
    expect(screen.getByTestId('operation-op-1-progress')).toHaveTextContent('Progress: 100%')
  })

  it('handles multiple operations with different progress states', () => {
    const operations = [
      createMockOperation('op-1', 'processing', 45),
      createMockOperation('op-2', 'processing', 80),
      createMockOperation('op-3', 'completed', 100),
    ]

    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: operations,
      refetch: mockRefetch,
    } as UseQueryResult<Operation[]>)

    render(<CollectionOperations collection={mockCollection} />)

    // Check that all operations show their progress
    expect(screen.getByTestId('operation-op-1-progress')).toHaveTextContent('Progress: 45%')
    expect(screen.getByTestId('operation-op-2-progress')).toHaveTextContent('Progress: 80%')
    expect(screen.getByTestId('operation-op-3-progress')).toHaveTextContent('Progress: 100%')
  })

  it('handles operation failure states', () => {
    const operation = createMockOperation('op-1', 'failed', 45)
    
    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: [operation],
      refetch: mockRefetch,
    } as UseQueryResult<Operation[]>)

    render(<CollectionOperations collection={mockCollection} />)

    expect(screen.getByTestId('operation-op-1-status')).toHaveTextContent('Status: failed')
    // Progress remains at last known value
    expect(screen.getByTestId('operation-op-1-progress')).toHaveTextContent('Progress: 45%')
  })

  it('simulates operation completion', () => {
    // Start with processing operation
    const operation = createMockOperation('op-1', 'processing', 90)
    
    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: [operation],
      refetch: mockRefetch,
    } as UseQueryResult<Operation[]>)

    const { rerender } = render(<CollectionOperations collection={mockCollection} />)
    expect(screen.getByTestId('operation-op-1-status')).toHaveTextContent('Status: processing')

    // Update to completed
    const completedOp = { ...operation, status: 'completed' as const, progress: 100 }
    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: [completedOp],
      refetch: mockRefetch,
    } as UseQueryResult<Operation[]>)

    rerender(<CollectionOperations collection={mockCollection} />)
    expect(screen.getByTestId('operation-op-1-status')).toHaveTextContent('Status: completed')
    expect(screen.getByTestId('operation-op-1-progress')).toHaveTextContent('Progress: 100%')
  })

  it('handles empty operations list', () => {
    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: [],
      refetch: mockRefetch,
    } as UseQueryResult<Operation[]>)

    render(<CollectionOperations collection={mockCollection} />)

    expect(screen.getByText('No operations yet')).toBeInTheDocument()
    expect(screen.getByText('Operations will appear here when you add data or re-index')).toBeInTheDocument()
  })

  it('maintains UI consistency during rapid updates', () => {
    const operation = createMockOperation('op-1', 'processing', 0)
    
    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: [operation],
      refetch: mockRefetch,
    } as UseQueryResult<Operation[]>)

    const { rerender } = render(<CollectionOperations collection={mockCollection} />)
    expect(screen.getByTestId('operation-op-1-progress')).toHaveTextContent('Progress: 0%')

    // Simulate rapid updates
    for (let progress of [20, 40, 60, 80, 100]) {
      const updatedOp = { ...operation, progress }
      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: [updatedOp],
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)
      
      rerender(<CollectionOperations collection={mockCollection} />)
    }

    expect(screen.getByTestId('operation-op-1-progress')).toHaveTextContent('Progress: 100%')
  })

  it('correctly displays operations for the current collection only', () => {
    const operation = createMockOperation('op-1', 'processing', 50)
    
    ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
      data: [operation],
      refetch: mockRefetch,
    } as UseQueryResult<Operation[]>)

    render(<CollectionOperations collection={mockCollection} />)

    // Only the operation for the current collection should be shown
    expect(screen.getByTestId('operation-op-1')).toBeInTheDocument()
    expect(screen.queryByTestId('operation-op-2')).not.toBeInTheDocument()
  })
})